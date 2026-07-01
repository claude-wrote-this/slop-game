"""Roaming Poisson-disk terrain renderer over a drifting cloud background.

Generation (worker thread): continuous, Bridson-style Poisson-disk sampling that
roams. An active list darts at poisson_r..2*poisson_r around active points; a
sparse spatial-hash grid (dict of int cells, populated only where points exist)
enforces the minimum spacing. Two radii, kept separate:

  * poisson_r — minimum spacing between points (local density / fill pattern).
  * kernel_r  — radius of the kept disc around the elastic kernel center (reach).
                A constant (bounds load); the scene clamps zoom so the screen
                stays within the disc.

The kernel center springs toward the view's world center. Points beyond kernel_r
are evicted (squared-distance in/out test, kept in sync with the grid). When
spacing is satisfied across the disc the active list drains and the worker idles;
kernel motion re-arms only the frontier (the disc's leading-edge ring), so work is
O(kernel motion), never O(whole disc). Each accepted point is stamped with its
fade completion and its colour sampled from the terrain function.

Handoff: the worker owns the point store + grid + active list and publishes an
immutable (pos, colour, completion) snapshot under a short lock; the main thread
never sees a torn store and the worker never stalls on it.

Screen (main thread): the settled points are cached on a screen-size colorkey
terrain layer, reprojected each frame (scroll for pan, scale for zoom, rebuilt on
drift). Fading points (completion in the future) are drawn fresh on top. Zoom is a
draw-time scale only; the cloud background is a precomputed tiled scroll-blit.
TerrainHeight (z/colour) is unchanged.
"""
import math
import sys
import threading
import time

import numpy as np
import pygame

# The generation worker is Python-heavy; shorten the GIL switch interval so the
# main draw thread gets the lock promptly each frame instead of waiting out a full
# ~5ms slice while the worker darts. Lets the worker run at full throughput.
sys.setswitchinterval(0.0008)


_CKEY = (255, 0, 255)   # colorkey for opaque kernels; never in the green palette
_TWO_PI = 2.0 * math.pi


def _make_kernel(radius):
    # SRCALPHA solid circle, ~1px antialiased edge. Used for FADING points, which
    # need real per-pixel alpha; settled points use the faster opaque colorkey one.
    d = radius * 2 + 1
    yy, xx = np.mgrid[0:d, 0:d]
    dist = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2)
    alpha = (np.clip(radius + 0.5 - dist, 0.0, 1.0) * 255).astype(np.uint8)
    surf = pygame.Surface((d, d), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = alpha.T
    del a
    return surf


def _make_cloud_tex(size, seed):
    # tileable fractal cloud texture via a power-law spectrum (FFT -> periodic)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((size, size))
    F = np.fft.fft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    radius[0, 0] = 1e-6
    F *= radius ** (-2.4)
    tex = np.fft.ifft2(F).real
    tex = (tex - tex.min()) / np.ptp(tex)
    return tex.astype(np.float32)


class Renderer:
    def __init__(self, screen_w, screen_h, *, terrain, tile=16,
                 poisson_r=20.0, kernel_r=800.0, dart_k=30, spring=0.11,
                 seed=0, bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85,
                 fade_duration=3.0, fade_jitter=0.9, fade_sat=0.5,
                 fade_near=0.35, sat_dist=1.0, haze=None,
                 cloud_front=True, front_colour=None):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg
        self.terrain = terrain

        # --- generation tunables ---
        self.poisson_r = float(poisson_r)     # minimum spacing (density)
        self.kernel_r = float(kernel_r)       # kept-disc radius (reach); constant
        self.dart_k = int(dart_k)             # Bridson attempts per active point
        self.spring = spring
        self.fade_duration = fade_duration    # fade length at the disc edge
        self.fade_jitter = fade_jitter
        self.fade_sat = fade_sat              # grow+fade in [0,fade_sat); saturate after
        self.fade_near = fade_near            # fade length near centre, as a fraction
        self.sat_dist = sat_dist              # extra distance-weight on saturation only
        # haze colour a fading point starts at — match the cloud so terrain seems
        # to resolve out of it rather than sparkle in as white.
        if haze is None:
            d = cloud_depth
            haze = ((int(255 - 0.5 * d), int(255 - 0.5 * d), int(255 - 0.35 * d))
                    if cloud else (220, 220, 225))
        self._haze = haze

        # --- view (main thread writes, worker reads) ---
        self._cam = (0.0, 0.0)        # world top-left of the view
        self.zoom = 1.0

        # --- worker-owned point store (slots) + sparse grid + active list -----
        cap = 1024
        self._cap = cap
        self._X = np.zeros(cap); self._Y = np.zeros(cap)
        self._CK = np.zeros(cap, np.uint32); self._COMP = np.zeros(cap)
        self._DUR = np.zeros(cap)     # per-point total fade length (grows w/ distance)
        self._GROW = np.zeros(cap)    # per-point grow-phase length (< _DUR)
        self._alive = np.zeros(cap, bool)
        self._free = list(range(cap - 1, -1, -1))
        self._grid = {}               # (cx, cy) -> [slot, ...], cell = poisson_r
        self._active = []             # slots that may still have room around them
        self._kc = None               # kernel center (world), springs to view
        self._rearm_kc = (0.0, 0.0)   # kernel center at the last frontier re-arm
        self._max_comp = 0.0          # latest completion stamped (gates fade path)
        self._count = 0               # live point count (loading progress)
        self._saturated = False       # disc filled (active drained) -> loading done
        self._rng = np.random.default_rng(seed)

        # --- handoff: the published snapshot main draws from ---
        self._snap = (np.zeros((0, 2)), np.zeros(0, np.uint32),
                      np.zeros(0), np.zeros(0), np.zeros(0))
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._work, name="poisson", daemon=True)
        self._thread.start()

        # --- draw kernels (main thread only) ---
        self.radius = max(1, int(tile * 0.95))
        self._zoom_bucket = None
        self._scaled_r = self.radius
        self._solid = {}
        self._fade = {}
        self._FADE_LEVELS = 10

        # --- terrain layer (screen-size colorkey cache; main thread only) ---
        self._layer = None
        self._layer_cam = (0.0, 0.0)
        self._layer_zoom = 1.0
        self._prev_zoom = 1.0
        self._prev_now = time.monotonic()
        self._P, self._K, self._C, self._D, self._G = self._snap

        # --- cloud background ---
        self.cloud = cloud
        if cloud:
            self.cloud_tex = _make_cloud_tex(512, cloud_seed)
            self.cloud_seed = cloud_seed
            self.cloud_scale = cloud_scale
            self.cloud_drift = cloud_drift
            self.cloud_depth = cloud_depth
            self.cloud_t = 0.0
            self._cloud_big, self._cloud_B = self._build_cloud_big()
            # one shared cloud palette (blue-white by cloud shade, billow 0..1) that
            # the background, the puffs and the new-point haze all draw from, so they
            # match rather than being independently tinted.
            N = 8
            self._cloud_pal = [
                (int(255 - (1.0 - bb) * cloud_depth),
                 int(255 - (1.0 - bb) * cloud_depth),
                 int(255 - (1.0 - bb) * cloud_depth * 0.7))
                for bb in (k / (N - 1) for k in range(N))]

        # --- ephemeral cloud-front puffs (main thread; no sampling/Poisson) ---
        # Off-white soft discs spawned along the leading screen edge as the view
        # pans, world-anchored so they drift in with the terrain, scaling+fading
        # in then back out. Pure decoration: a cloud front rolling ahead of the
        # haze-coloured terrain that resolves behind it. Capped + cached -> cheap.
        self.cloud_front = cloud_front
        if cloud_front:
            hr, hg, hb = self._haze
            if front_colour is None:                 # lift the haze toward white
                front_colour = (int(hr + 0.6 * (255 - hr)),
                                int(hg + 0.6 * (255 - hg)),
                                int(hb + 0.6 * (255 - hb)))
            self._front_colour = front_colour
            self._front_rng = np.random.default_rng(seed + 12345)
            self._front = []                 # (wx, wy, birth, life, r_px, cidx)
            self._front_cache = {}           # (r_px, alpha, cidx) -> soft disc surface
            self._front_alpha = 255          # grains reach full opacity mid-life
            self._front_white = 350          # target white cover: white pts + puffs
            self._front_white_rf = 0.5       # a point counts as white while rf > this
            self._front_fill = 260.0         # max puffs/sec ramp toward the target
            self._front_cap = 150
            # puffs draw from the shared cloud palette, indexed by the cloud shade
            # sampled under each, so they vary exactly the way the background does.
            self._front_pal = self._cloud_pal if cloud else [front_colour]
            self._front_K = len(self._front_pal)

    # --- view API (main thread) -------------------------------------------
    def set_camera(self, x, y):
        self._cam = (float(x), float(y))

    def set_zoom(self, z):
        self.zoom = float(z)

    def _view_center(self):
        cx, cy = self._cam
        return (cx + self.w * 0.5, cy + self.h * 0.5)

    def pending_points(self):
        return 0                          # generation streams in-game via the fade

    def render(self):
        pass

    def shutdown(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # --- worker: roaming Bridson Poisson-disk sampling --------------------
    def _cell(self, x, y):
        pr = self.poisson_r
        return (int(math.floor(x / pr)), int(math.floor(y / pr)))

    def _grow(self):
        old = self._cap; new = old * 2
        def ext(a, dt):
            b = np.zeros(new, dt); b[:old] = a; return b
        self._X = ext(self._X, float); self._Y = ext(self._Y, float)
        self._CK = ext(self._CK, np.uint32); self._COMP = ext(self._COMP, float)
        self._DUR = ext(self._DUR, float); self._GROW = ext(self._GROW, float)
        al = np.zeros(new, bool); al[:old] = self._alive; self._alive = al
        self._free.extend(range(new - 1, old - 1, -1))
        self._cap = new

    def _insert(self, x, y, ck, comp, dur, grow):
        if not self._free:
            self._grow()
        i = self._free.pop()
        self._X[i] = x; self._Y[i] = y; self._CK[i] = ck
        self._COMP[i] = comp; self._DUR[i] = dur; self._GROW[i] = grow
        self._alive[i] = True
        self._grid.setdefault(self._cell(x, y), []).append(i)
        self._active.append(i)
        self._count += 1
        return i

    def _remove(self, i):
        self._alive[i] = False
        self._count -= 1
        c = self._cell(self._X[i], self._Y[i])
        lst = self._grid.get(c)
        if lst is not None:
            try:
                lst.remove(i)
            except ValueError:
                pass
            if not lst:
                del self._grid[c]
        self._free.append(i)

    def _spacing_ok(self, x, y):
        pr2 = self.poisson_r * self.poisson_r
        cx, cy = self._cell(x, y)
        g = self._grid; X = self._X; Y = self._Y; al = self._alive
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                lst = g.get((gx, gy))
                if lst:
                    for j in lst:
                        if al[j] and (X[j] - x) ** 2 + (Y[j] - y) ** 2 < pr2:
                            return False
        return True

    def _evict(self, kc):
        kr2 = self.kernel_r * self.kernel_r
        idx = np.nonzero(self._alive)[0]
        if idx.size == 0:
            return
        dx = self._X[idx] - kc[0]; dy = self._Y[idx] - kc[1]
        far = idx[(dx * dx + dy * dy) > kr2]
        for i in far.tolist():
            self._remove(int(i))

    def _rearm(self, kc, motion):
        """Re-activate the leading-edge ring so darts refill the newly-exposed
        crescent (emergent — no lune math). Never touches the settled interior."""
        kr = self.kernel_r; pr = self.poisson_r
        inner = (kr - 2 * pr) ** 2; outer = kr * kr
        idx = np.nonzero(self._alive)[0]
        if idx.size == 0:
            return
        dx = self._X[idx] - kc[0]; dy = self._Y[idx] - kc[1]
        d2 = dx * dx + dy * dy
        ring = (d2 >= inner) & (d2 <= outer)
        mx, my = motion
        if mx * mx + my * my > 1e-9:              # only the leading half
            ring &= (dx * mx + dy * my) > 0.0
        self._active.extend(idx[ring].tolist())

    def _assign_colours(self, slots):
        sl = np.array(slots, dtype=np.intp)
        _, col = self.terrain.sample_points(self._X[sl] / self.tile,
                                            self._Y[sl] / self.tile)
        self._CK[sl] = self._colour_keys(np.ascontiguousarray(col, dtype=np.uint8))
        mc = float(self._COMP[sl].max())
        if mc > self._max_comp:
            self._max_comp = mc

    def _fade_times(self, dist2):
        """(total, grow) fade lengths for a point at squared-distance dist2 from
        the kernel centre. Both lengthen with distance (fade_near near the centre,
        full at the edge). The saturation phase lengthens *extra* by sat_dist*w, so
        distant points hold their haze markedly longer before resolving to terrain
        colour than they take to grow in — saturation is doubly distance-weighted.
        Jitter rides the saturation tail (a raggeder resolve edge)."""
        w = min(1.0, math.sqrt(dist2) / self.kernel_r)
        dw = self.fade_near + (1.0 - self.fade_near) * w
        grow = self.fade_duration * self.fade_sat * dw
        sat = self.fade_duration * (1.0 - self.fade_sat) * dw * (1.0 + self.sat_dist * w)
        sat += self._rng.random() * self.fade_jitter * w
        return grow + sat, grow

    def _seed(self, at):
        dur, grow = self._fade_times(0.0)
        s = self._insert(at[0], at[1], 0, time.monotonic() + dur, dur, grow)
        self._assign_colours([s])

    def _dart(self, kc, budget):
        kr2 = self.kernel_r * self.kernel_r
        pr = self.poisson_r; k = self.dart_k
        active = self._active; rng = self._rng
        now = time.monotonic()
        new_slots = []
        visits = 0
        while active and visits < budget:
            visits += 1
            ai = int(rng.integers(0, len(active)))
            i = active[ai]
            if not self._alive[i]:
                active[ai] = active[-1]; active.pop(); continue
            ox = self._X[i]; oy = self._Y[i]; placed = False
            for _ in range(k):
                ang = rng.random() * _TWO_PI
                rad = pr * (1.0 + rng.random())          # standard Bridson: r .. 2r
                x = ox + math.cos(ang) * rad
                y = oy + math.sin(ang) * rad
                d2 = (x - kc[0]) ** 2 + (y - kc[1]) ** 2
                if d2 > kr2:
                    continue                              # outside the kept disc
                if self._spacing_ok(x, y):
                    dur, grow = self._fade_times(d2)
                    new_slots.append(self._insert(x, y, 0, now + dur, dur, grow))
                    placed = True
                    break
            if not placed:
                active[ai] = active[-1]; active.pop()     # exhausted -> deactivate
        if new_slots:
            self._assign_colours(new_slots)
        return len(new_slots)

    def _publish(self):
        idx = np.nonzero(self._alive)[0]
        pos = np.stack([self._X[idx], self._Y[idx]], axis=1)
        snap = (pos, self._CK[idx].copy(), self._COMP[idx].copy(),
                self._DUR[idx].copy(), self._GROW[idx].copy())
        with self._lock:
            self._snap = snap

    def _work(self):
        while self._running:
            vc = self._view_center()
            if self._kc is None:
                self._kc = vc; self._rearm_kc = vc
                self._seed(vc)
            else:
                kx, ky = self._kc
                self._kc = (kx + (vc[0] - kx) * self.spring,
                            ky + (vc[1] - ky) * self.spring)
            mx = self._kc[0] - self._rearm_kc[0]
            my = self._kc[1] - self._rearm_kc[1]
            work = False
            # Re-arm at half-spacing granularity: smaller, more frequent frontier
            # refreshes spread the eviction/dart work across more ticks, so the
            # generation reads as continuous rather than stepping each full poisson_r.
            step = 0.5 * self.poisson_r
            if mx * mx + my * my > step * step:
                self._evict(self._kc)
                self._rearm(self._kc, (mx, my))
                self._rearm_kc = self._kc
                work = True
            if self._active and self._dart(self._kc, 48):
                work = True
            # saturated once the active list has drained over a filled disc
            self._saturated = (not self._active) and self._count > 0
            if work:
                self._publish()
                time.sleep(0)             # yield; short switchinterval keeps main fed
            else:
                time.sleep(0.005)         # saturated + stationary -> idle

    # --- culling (main thread, against the published snapshot) ------------
    @staticmethod
    def _colour_keys(cols):
        cb = cols & 0xF8
        return ((cb[:, 0].astype(np.uint32) << 16)
                | (cb[:, 1].astype(np.uint32) << 8)
                | cb[:, 2].astype(np.uint32))

    def _cull(self, cam, zoom, x0, y0, x1, y1):
        P = self._P
        if P.shape[0] == 0:
            e = np.zeros(0)
            return e, e, self._K[:0], self._C[:0], self._D[:0], self._G[:0]
        cx = cam[0] + self.w * 0.5
        cy = cam[1] + self.h * 0.5
        r = self._scaled_r
        wx0 = cx + (x0 - r - self.w * 0.5) / zoom
        wx1 = cx + (x1 + r - self.w * 0.5) / zoom
        wy0 = cy + (y0 - r - self.h * 0.5) / zoom
        wy1 = cy + (y1 + r - self.h * 0.5) / zoom
        m = (P[:, 0] >= wx0) & (P[:, 0] < wx1) & (P[:, 1] >= wy0) & (P[:, 1] < wy1)
        px = P[m]
        sx = self.w * 0.5 + (px[:, 0] - cx) * zoom
        sy = self.h * 0.5 + (px[:, 1] - cy) * zoom
        return sx, sy, self._K[m], self._C[m], self._D[m], self._G[m]

    # --- draw kernels -----------------------------------------------------
    def _ensure_kernels(self, zoom):
        zb = round(zoom, 2)
        if zb == self._zoom_bucket:
            return
        self._zoom_bucket = zb
        self._scaled_r = max(1, int(round(self.radius * zoom)))
        self._solid = {}
        self._fade = {}

    def _blit_settled(self, target, sx, sy, keys, r):
        cache = self._solid
        for k in np.unique(keys).tolist():
            if k not in cache:
                cache[k] = self._make_solid_kernel(r, k)
        xs = (sx - r).astype(np.int32).tolist()
        ys = (sy - r).astype(np.int32).tolist()
        kers = [cache[k] for k in keys.tolist()]
        seq = list(zip(kers, zip(xs, ys)))
        if seq:
            target.blits(seq, doreturn=False)

    def _make_solid_kernel(self, r, ck):
        d = r * 2 + 1
        s = pygame.Surface((d, d))
        s.fill(_CKEY)
        pygame.draw.circle(s, (ck >> 16, (ck >> 8) & 0xFF, ck & 0xFF), (r, r), r)
        try:
            s = s.convert()
        except pygame.error:
            pass
        s.set_colorkey(_CKEY, pygame.RLEACCEL)
        return s

    def _blit_fading(self, target, sx, sy, keys, comp, dur, grow, now):
        # Split each point's own timeline into its grow phase then its (separately
        # distance-weighted) saturation phase and fold them into one 0..2 progress:
        # grow spans [0,1], saturation spans [1,2]. The kernel cache keys off the
        # quantised combined level so the two phases still share one lookup.
        elapsed = dur - (comp - now)
        grow = np.maximum(grow, 1e-6)
        gp = np.clip(elapsed / grow, 0.0, 1.0)                       # grow 0..1
        sp = np.clip((elapsed - grow) / np.maximum(dur - grow, 1e-6), 0.0, 1.0)
        lv = ((gp + sp) * self._FADE_LEVELS).astype(np.int32)        # 0 .. 2*LEVELS
        # Draw oldest (most-resolved) first, newest (whitest) last, so the young
        # white points stay on top rather than resolved ones poking through them.
        order = np.argsort(lv)[::-1]
        xs = sx[order].tolist(); ys = sy[order].tolist()
        keys = keys[order].tolist(); lv = lv[order].tolist()
        cache = self._fade
        for i in range(len(xs)):
            p = lv[i]
            if p <= 0:
                continue
            ck = keys[i]
            ent = cache.get((ck, p))
            if ent is None:
                ent = self._make_fade_kernel(ck, p)
                cache[(ck, p)] = ent
            ker, kr = ent
            target.blit(ker, (int(xs[i]) - kr, int(ys[i]) - kr))

    def _make_fade_kernel(self, ck, level):
        # `level` runs 0..2*FADE_LEVELS: the grow phase then the saturation phase,
        # each already sized per-point by _blit_fading. So frac2 in [0,1] grows the
        # white puff (r2 + alpha) and frac2 in [1,2] desaturates haze -> terrain,
        # reading as cloud arriving first, then resolving into ground.
        frac2 = level / self._FADE_LEVELS
        grow = min(1.0, frac2)                                   # size + alpha
        sat = max(0.0, frac2 - 1.0)                              # colour
        r2 = max(1, int(round(self._scaled_r * grow)))
        ker = _make_kernel(r2)
        r = ck >> 16; g = (ck >> 8) & 0xFF; b = ck & 0xFF
        # new points start from a cloud-palette colour (matching the background)
        # picked by a stable hash of their colour key, which is already in the cache
        # key so this adds no cache entries; then lerp haze -> terrain over sat.
        if self.cloud:
            pal = self._cloud_pal
            hr, hg, hb = pal[((ck * 2654435761) >> 16) % len(pal)]
        else:
            hr, hg, hb = self._haze
        cr = int(hr + sat * (r - hr))
        cg = int(hg + sat * (g - hg))
        cb = int(hb + sat * (b - hb))
        ker.fill((cr, cg, cb, int(255 * grow)), special_flags=pygame.BLEND_RGBA_MULT)
        return ker, r2

    # --- terrain layer ----------------------------------------------------
    def _new_layer(self):
        s = pygame.Surface((self.w, self.h))
        try:
            s = s.convert()
        except pygame.error:
            pass
        s.fill(_CKEY)
        s.set_colorkey(_CKEY)
        return s

    def _rebuild_layer(self, now, cam, zoom):
        self._layer.fill(_CKEY)
        sx, sy, keys, comp, _, _ = self._cull(cam, zoom, 0, 0, self.w, self.h)
        settled = now >= comp
        if settled.any():
            self._blit_settled(self._layer, sx[settled], sy[settled],
                               keys[settled], self._scaled_r)
        self._layer_cam = cam
        self._layer_zoom = zoom

    def _scroll_layer(self, now, cam, zoom):
        L = self._layer
        lcx, lcy = self._layer_cam
        ddx = int(round((lcx - cam[0]) * zoom))
        ddy = int(round((lcy - cam[1]) * zoom))
        if ddx == 0 and ddy == 0:
            return
        if abs(ddx) >= self.w or abs(ddy) >= self.h:
            self._rebuild_layer(now, cam, zoom)
            return
        L.scroll(ddx, ddy)
        self._layer_cam = (lcx - ddx / zoom, lcy - ddy / zoom)
        lcam = self._layer_cam
        r = self._scaled_r

        def strip(x0, x1, y0, y1):
            L.fill(_CKEY, (x0, y0, x1 - x0, y1 - y0))
            sx, sy, keys, comp, _, _ = self._cull(lcam, zoom, x0, y0, x1, y1)
            settled = now >= comp
            if settled.any():
                self._blit_settled(L, sx[settled], sy[settled], keys[settled], r)

        if ddx > 0:   strip(0, ddx, 0, self.h)
        elif ddx < 0: strip(self.w + ddx, self.w, 0, self.h)
        if ddy > 0:   strip(0, self.w, 0, ddy)
        elif ddy < 0: strip(0, self.w, self.h + ddy, self.h)

    def _commit_settles(self, now, cam, zoom):
        if self._prev_now >= self._max_comp:
            return
        sx, sy, keys, comp, _, _ = self._cull(cam, zoom, 0, 0, self.w, self.h)
        just = (comp > self._prev_now) & (comp <= now)
        if just.any():
            self._blit_settled(self._layer, sx[just], sy[just], keys[just], self._scaled_r)

    def _reproject(self, target, cam, zoom):
        L = self._layer
        scale = zoom / self._layer_zoom
        lcx, lcy = self._layer_cam
        ox = self.w * 0.5 + (lcx - cam[0]) * zoom
        oy = self.h * 0.5 + (lcy - cam[1]) * zoom
        sw = max(1, int(round(self.w * scale)))
        sh = max(1, int(round(self.h * scale)))
        scaled = pygame.transform.scale(L, (sw, sh))
        scaled.set_colorkey(_CKEY)
        target.blit(scaled, (int(ox - sw * 0.5), int(oy - sh * 0.5)))

    # --- cloud ------------------------------------------------------------
    def _build_cloud_big(self):
        # Assemble the cloud tile from overlapping circular splats — the same look
        # as the point-cloud foreground — instead of a smooth blur. The fractal
        # field still sets where the cloud is thick vs clear; a jittered grid of
        # discs coloured (and sized) by it renders that structure in the point
        # style. Preprocessed once; the per-frame path is still a tile scroll-blit.
        tex = self.cloud_tex
        size = tex.shape[0]
        d = self.cloud_depth
        billow = np.abs(tex * 2.0 - 1.0)              # 0 clear .. 1 thick cloud
        self._cloud_billow = billow                   # for tinting puffs/points
        B = int(round(size / self.cloud_scale))
        big = pygame.Surface((B, B))
        big.fill((255 - d, 255 - d, int(255 - d * 0.7)))   # clear-sky colour (billow 0)
        rng = np.random.default_rng((self.cloud_seed * 2654435761 + 1) & 0xffffffff)

        def colour(px, py):
            b = float(billow[int(py / B * size) % size, int(px / B * size) % size])
            sh = int(255 - (1.0 - b) * d)
            return (sh, sh, int(255 - (1.0 - b) * d * 0.7))

        def splat(px, py, r, col):
            for ox in (0, -B, B):                      # wrap copies near the edges
                if abs(px + ox - B * 0.5) >= B * 0.5 + r:
                    continue
                for oy in (0, -B, B):
                    if abs(py + oy - B * 0.5) >= B * 0.5 + r:
                        continue
                    pygame.draw.circle(big, col, (int(px + ox), int(py + oy)), r)

        g = max(6.0, self.radius * 1.5)
        nc = max(1, int(round(B / g)))
        g = B / nc                                     # exact divisor -> seamless wrap
        # coverage pass: medium discs on a jittered grid so there are never gaps
        cover = []
        for iy in range(nc):
            for ix in range(nc):
                px = (ix + 0.5 + (rng.random() - 0.5) * 0.8) * g
                py = (iy + 0.5 + (rng.random() - 0.5) * 0.8) * g
                cover.append((px, py, int(g * (1.15 + rng.random() * 0.5)), colour(px, py)))
        rng.shuffle(cover)                             # organic overlap, not fish-scale
        for px, py, r, col in cover:
            splat(px, py, r, col)
        # variety pass: many more discs over a full, small-skewed range of sizes
        for _ in range(int(nc * nc * 3.0)):
            px = rng.random() * B; py = rng.random() * B
            t = rng.random()
            r = int(g * (0.22 + t * t * 2.3))
            if r >= 1:
                splat(px, py, r, colour(px, py))
        try:
            big = big.convert()
        except pygame.error:
            pass
        return big, B

    def _cloud_background(self, target, cam_x, cam_y):
        B = self._cloud_B
        s = self.cloud_scale
        dx, dy = self.cloud_drift
        ox = int(round(cam_x + self.cloud_t * dx / s)) % B
        oy = int(round(cam_y + self.cloud_t * dy / s)) % B
        big = self._cloud_big
        y = -oy
        while y < self.h:
            x = -ox
            while x < self.w:
                target.blit(big, (x, y))
                x += B
            y += B

    # --- ephemeral cloud front --------------------------------------------
    def _spawn_front(self, cam, zoom, now, dt):
        # Keep a constant amount of *white cover* on screen — resolving (fading)
        # points plus puffs together — by topping the puffs up to fill whatever the
        # sampler isn't currently covering. So the puff budget runs inverse to the
        # resolving count: few puffs when lots is resolving (already white), many
        # when little is (the slow / sharp-edge case). Placement errs outward from
        # the kernel centre so the cover reaches past the saturated front rather
        # than leaving it exposed. Reads the published arrays; no sampling.
        puffs = self._front
        if (dt <= 0.0 or now >= self._max_comp        # nothing resolving -> no work
                or self._P.shape[0] == 0 or len(puffs) >= self._front_cap):
            return
        # Count only the still-*white* resolving points: a point past the first
        # part of its fade is mostly terrain colour, not white cover. (Matters now
        # the fade is slow — otherwise the many half-resolved points would zero the
        # budget.) rf = 1 at birth -> 0 when done.
        rf = (self._C - now) / np.maximum(self._D, 1e-6)
        fi = np.nonzero(rf > self._front_white_rf)[0]
        if fi.size == 0:
            return
        w, h = self.w, self.h
        cx = cam[0] + w * 0.5; cy = cam[1] + h * 0.5
        P = self._P
        rx = (P[fi, 0] - cx) * zoom; ry = (P[fi, 1] - cy) * zoom
        on = (rx > -w * 0.5) & (rx < w * 0.5) & (ry > -h * 0.5) & (ry < h * 0.5)
        ci = fi[on]
        # puffs + white resolving points -> a constant white budget
        target = self._front_white - ci.size
        target = 0 if target < 0 else min(target, self._front_cap)
        deficit = target - len(puffs)
        if deficit <= 0 or ci.size == 0:
            return
        per_frame = int(self._front_fill * dt) + 1  # ramp, no sudden bursts
        n = min(deficit, per_frame, self._front_cap - len(puffs))

        rng = self._front_rng
        kc = self._kc if self._kc is not None else (cx, cy)
        dcx = P[ci, 0] - kc[0]; dcy = P[ci, 1] - kc[1]
        dist = np.sqrt(dcx * dcx + dcy * dcy) + 1e-6
        wgt = (dist / self.kernel_r) ** 2 + 0.02    # err outward, away from centre
        wgt /= wgt.sum()
        pick = rng.choice(ci.size, size=n, p=wgt)
        pr = self.poisson_r
        for j in pick.tolist():
            i = ci[j]
            ux = dcx[j] / dist[j]; uy = dcy[j] / dist[j]        # outward unit vector
            tx = -uy; ty = ux                                   # along the edge
            off = pr * (0.4 + rng.random() * 1.2)               # push past the point
            # Spread wide *along* the edge so each sample's puffs bridge to its
            # neighbours, covering the gaps between samples that briefly expose the
            # saturated front; keep the radial jitter tight so the cover stays a band.
            tan = (rng.random() * 2.0 - 1.0) * pr * 1.7
            rad = (rng.random() * 2.0 - 1.0) * pr * 0.35
            wx = P[i, 0] + ux * (off + rad) + tx * tan
            wy = P[i, 1] + uy * (off + rad) + ty * tan
            life = 0.5 + rng.random() * 0.8
            cidx = self._cloud_tint(wx, wy)             # cloud shade under the puff
            puffs.append((wx, wy, now, life, self.radius, cidx))

    def _cloud_tint(self, wx, wy):
        """Palette index from the cloud shade under a world point, so puffs vary
        the same way the background does. 0 (whitest) when there's no cloud tile."""
        if not self.cloud:
            return 0
        bil = self._cloud_billow; s = bil.shape[0]; B = self._cloud_B
        b = float(bil[int((wy % B) / B * s) % s, int((wx % B) / B * s) % s])
        i = int(b * self._front_K)
        return i if i < self._front_K else self._front_K - 1

    def _cloud_front_pass(self, target, cam, zoom, now):
        w, h = self.w, self.h
        dt = now - self._prev_now
        dt = 0.0 if dt < 0.0 else min(dt, 0.05)      # clamp (first frame / stalls)
        self._spawn_front(cam, zoom, now, dt)

        puffs = self._front
        if not puffs:
            return
        cx = cam[0] + w * 0.5; cy = cam[1] + h * 0.5
        pal = self._front_pal
        amax = self._front_alpha
        cache = self._front_cache
        live = []; seq = []
        for p in puffs:
            wx, wy, birth, life, base_r, cidx = p
            age = now - birth
            if age >= life:
                continue                          # dead
            live.append(p)
            env = math.sin(math.pi * age / life)  # 0 -> 1 -> 0 (scale)
            if env <= 0.02:
                continue
            r = int(base_r * env)
            if r < 1:
                continue
            sx = w * 0.5 + (wx - cx) * zoom
            sy = h * 0.5 + (wy - cy) * zoom
            if sx < -r or sx > w + r or sy < -r or sy > h + r:
                continue                          # off-screen: age but don't blit
            aenv = env * 1.7                      # reach full opacity over the middle
            alpha = amax if aenv >= 1.0 else (int(amax * aenv) & ~7)  # of its life
            if alpha <= 0:
                continue
            key = (r, alpha, cidx)
            surf = cache.get(key)
            if surf is None:
                surf = _make_kernel(r)
                surf.fill((*pal[cidx], alpha), special_flags=pygame.BLEND_RGBA_MULT)
                cache[key] = surf
            seq.append((surf, (int(sx) - r, int(sy) - r)))
        self._front = live
        if seq:
            target.blits(seq, doreturn=False)

    # --- frame ------------------------------------------------------------
    def draw(self, target):
        cam = self._cam
        zoom = self.zoom
        if self.cloud:
            self._cloud_background(target, cam[0], cam[1])
        else:
            target.fill(self.bg)

        with self._lock:                  # grab the latest published snapshot
            self._P, self._K, self._C, self._D, self._G = self._snap
        now = time.monotonic()
        self._ensure_kernels(zoom)

        if self._layer is None:
            self._layer = self._new_layer()
            self._rebuild_layer(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        elif zoom == self._layer_zoom:
            self._scroll_layer(now, cam, zoom)
            self._commit_settles(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        elif zoom == self._prev_zoom:
            self._rebuild_layer(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        else:
            self._reproject(target, cam, zoom)

        if now < self._max_comp:          # any point still mid-fade?
            sx, sy, keys, comp, dur, grow = self._cull(cam, zoom, 0, 0, self.w, self.h)
            fading = now < comp
            if fading.any():
                self._blit_fading(target, sx[fading], sy[fading], keys[fading],
                                  comp[fading], dur[fading], grow[fading], now)

        if self.cloud_front:              # cloud front veils the resolving edge
            self._cloud_front_pass(target, cam, zoom, now)

        self._prev_zoom = zoom
        self._prev_now = now
        if self.cloud:
            self.cloud_t += 1.0
