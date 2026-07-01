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
                 explore_p=0.04, explore_reach=8.0,
                 seed=0, bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85,
                 fade_duration=0.6, fade_jitter=0.5):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg
        self.terrain = terrain

        # --- generation tunables ---
        self.poisson_r = float(poisson_r)     # minimum spacing (density)
        self.kernel_r = float(kernel_r)       # kept-disc radius (reach); constant
        self.dart_k = int(dart_k)             # Bridson attempts per active point
        self.spring = spring
        self.explore_p = explore_p            # chance a dart reaches far (scouting)
        self.explore_reach = explore_reach    # scout reach, in units of poisson_r
        self.fade_duration = fade_duration
        self.fade_jitter = fade_jitter

        # --- view (main thread writes, worker reads) ---
        self._cam = (0.0, 0.0)        # world top-left of the view
        self.zoom = 1.0

        # --- worker-owned point store (slots) + sparse grid + active list -----
        cap = 1024
        self._cap = cap
        self._X = np.zeros(cap); self._Y = np.zeros(cap)
        self._CK = np.zeros(cap, np.uint32); self._COMP = np.zeros(cap)
        self._alive = np.zeros(cap, bool)
        self._free = list(range(cap - 1, -1, -1))
        self._grid = {}               # (cx, cy) -> [slot, ...], cell = poisson_r
        self._active = []             # slots that may still have room around them
        self._kc = None               # kernel center (world), springs to view
        self._rearm_kc = (0.0, 0.0)   # kernel center at the last frontier re-arm
        self._max_comp = 0.0          # latest completion stamped (gates fade path)
        self._rng = np.random.default_rng(seed)

        # --- handoff: the published snapshot main draws from ---
        self._snap = (np.zeros((0, 2)), np.zeros(0, np.uint32), np.zeros(0))
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
        self._FADE_LEVELS = 6

        # --- terrain layer (screen-size colorkey cache; main thread only) ---
        self._layer = None
        self._layer_cam = (0.0, 0.0)
        self._layer_zoom = 1.0
        self._prev_zoom = 1.0
        self._prev_now = time.monotonic()
        self._P = self._snap[0]; self._K = self._snap[1]; self._C = self._snap[2]

        # --- cloud background ---
        self.cloud = cloud
        if cloud:
            self.cloud_tex = _make_cloud_tex(512, cloud_seed)
            self.cloud_scale = cloud_scale
            self.cloud_drift = cloud_drift
            self.cloud_depth = cloud_depth
            self.cloud_t = 0.0
            self._cloud_big, self._cloud_B = self._build_cloud_big()

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
        al = np.zeros(new, bool); al[:old] = self._alive; self._alive = al
        self._free.extend(range(new - 1, old - 1, -1))
        self._cap = new

    def _insert(self, x, y, ck, comp):
        if not self._free:
            self._grow()
        i = self._free.pop()
        self._X[i] = x; self._Y[i] = y; self._CK[i] = ck; self._COMP[i] = comp
        self._alive[i] = True
        self._grid.setdefault(self._cell(x, y), []).append(i)
        self._active.append(i)
        return i

    def _remove(self, i):
        self._alive[i] = False
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

    def _seed(self, at):
        comp = time.monotonic() + self.fade_duration + self._rng.random() * self.fade_jitter
        s = self._insert(at[0], at[1], 0, comp)
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
            ep = self.explore_p; er = self.explore_reach
            for _ in range(k):
                ang = rng.random() * _TWO_PI
                if rng.random() < ep:                    # occasional scout further out
                    rad = pr * (1.0 + rng.random() * er)
                else:
                    rad = pr * (1.0 + rng.random())      # normal r .. 2r
                x = ox + math.cos(ang) * rad
                y = oy + math.sin(ang) * rad
                if (x - kc[0]) ** 2 + (y - kc[1]) ** 2 > kr2:
                    continue                              # outside the kept disc
                if self._spacing_ok(x, y):
                    comp = now + self.fade_duration + rng.random() * self.fade_jitter
                    new_slots.append(self._insert(x, y, 0, comp))
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
        snap = (pos, self._CK[idx].copy(), self._COMP[idx].copy())
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
            if mx * mx + my * my > self.poisson_r * self.poisson_r:
                self._evict(self._kc)
                self._rearm(self._kc, (mx, my))
                self._rearm_kc = self._kc
                work = True
            if self._active and self._dart(self._kc, 48):
                work = True
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
            return e, e, self._K[:0], self._C[:0]
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
        return sx, sy, self._K[m], self._C[m]

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

    def _blit_fading(self, target, sx, sy, keys, comp, now):
        prog = np.clip(1.0 - (comp - now) / self.fade_duration, 0.0, 1.0)
        lv = (prog * self._FADE_LEVELS).astype(np.int32)
        xs = sx.tolist(); ys = sy.tolist()
        keys = keys.tolist(); lv = lv.tolist()
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
        frac = level / self._FADE_LEVELS
        r2 = max(1, int(round(self._scaled_r * frac)))
        ker = _make_kernel(r2)
        # A fading point grows (r2), fades in (alpha) AND desaturates from white to
        # its terrain colour, so the advancing edge reads as hazy cloud resolving
        # into ground. colour = lerp(white, terrain, frac).
        r = ck >> 16; g = (ck >> 8) & 0xFF; b = ck & 0xFF
        wr = int(255 - frac * (255 - r))
        wg = int(255 - frac * (255 - g))
        wb = int(255 - frac * (255 - b))
        ker.fill((wr, wg, wb, int(255 * frac)), special_flags=pygame.BLEND_RGBA_MULT)
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
        sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
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
            sx, sy, keys, comp = self._cull(lcam, zoom, x0, y0, x1, y1)
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
        sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
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
        billow = np.abs(self.cloud_tex * 2.0 - 1.0)
        d = self.cloud_depth
        shade = (255 - (1.0 - billow) * d).astype(np.uint8)
        blue = (255 - (1.0 - billow) * (d * 0.7)).astype(np.uint8)
        size = self.cloud_tex.shape[0]
        surf = pygame.Surface((size, size))
        rgb = pygame.surfarray.pixels3d(surf)
        rgb[:, :, 0] = shade.T
        rgb[:, :, 1] = shade.T
        rgb[:, :, 2] = blue.T
        del rgb
        B = int(round(size / self.cloud_scale))
        big = pygame.transform.scale(surf, (B, B))
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

    # --- frame ------------------------------------------------------------
    def draw(self, target):
        cam = self._cam
        zoom = self.zoom
        if self.cloud:
            self._cloud_background(target, cam[0], cam[1])
        else:
            target.fill(self.bg)

        with self._lock:                  # grab the latest published snapshot
            self._P, self._K, self._C = self._snap
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
            sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
            fading = now < comp
            if fading.any():
                self._blit_fading(target, sx[fading], sy[fading], keys[fading],
                                  comp[fading], now)

        self._prev_zoom = zoom
        self._prev_now = now
        if self.cloud:
            self.cloud_t += 1.0
