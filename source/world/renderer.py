"""Static point-cloud terrain renderer over a drifting cloud background.

TEST HARNESS for pan/zoom. Two pieces, point store -> screen, all on the main
thread:

  * Point store: the world-space points (here a static scatter sampled once
    through the terrain function; later, real generation), plus an immutable
    per-point fade-in completion stamp.
  * Screen: the settled points are cached on a screen-size colorkey terrain layer
    that is REPROJECTED each frame instead of re-blitting every point — scrolled
    for pan (refilling only the exposed edge strips, culled against the LIVE
    camera so a fast pan can't leave gaps) and scaled for zoom, rebuilt only when
    the view drifts too far. Each frame: cloud, the layer, then the fading points
    fresh on top (they change every frame), then UI; the main loop flips.

The layer is screen-size and main-thread-only — never an oversized read-from
buffer and no cross-thread surface, so no SDL locks to fight. Culling is on the
main thread because the strip-fill needs the current frame's camera; with the
layer it's only ever a small strip, so it's cheap. The terrain function
(TerrainHeight) and the cloud background are untouched.
"""
import time

import numpy as np
import pygame


_CKEY = (255, 0, 255)   # colorkey for opaque kernels; never in the green palette


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
    def __init__(self, screen_w, screen_h, *, terrain, tile=16, density=0.008,
                 area=None, seed=0, bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85,
                 fade_duration=0.6, fade_jitter=0.5):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg

        # --- the static point set (sampled once through the terrain function) --
        aw, ah = area if area else (4 * self.w, 4 * self.h)
        self.area = (aw, ah)
        self.density = density
        n = max(1, int(density * aw * ah))
        rng = np.random.default_rng(seed)
        px = rng.uniform(-aw / 2, aw / 2, n)        # world-space, centered on origin
        py = rng.uniform(-ah / 2, ah / 2, n)
        self.points = np.ascontiguousarray(np.stack([px, py], axis=1))
        _, colour = terrain.sample_points(px / tile, py / tile)
        self.colours = np.ascontiguousarray(colour, dtype=np.uint8)
        self._keys_all = self._colour_keys(self.colours)    # packed colour, static
        # Per-point fade-in: immutable completion stamp set at generation time
        # (here, construction). completion = now + duration + per-point jitter so a
        # burst doesn't fade in lockstep. duration is the denominator for progress.
        self.fade_duration = fade_duration
        self.completion = (time.monotonic() + fade_duration
                           + rng.uniform(0.0, fade_jitter, n))
        self._max_completion = float(self.completion.max())
        self.n = n

        # --- view (main thread) ---
        self._cam = (0.0, 0.0)        # world top-left of the view
        self.zoom = 1.0
        # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0   # clamp — needed eventually (scene side)

        # --- draw kernels ---
        self.radius = max(1, int(tile * 0.95))
        self._zoom_bucket = None
        self._scaled_r = self.radius
        self._solid = {}                # colour key -> opaque colorkey circle (settled)
        self._fade = {}                 # (colour key, progress level) -> SRCALPHA kernel
        self._FADE_LEVELS = 6           # quantise fade progress so kernels cache

        # --- terrain layer: a screen-size colorkey cache of the SETTLED points,
        #     reprojected each frame (see module docstring). Screen-size only;
        #     never an oversized buffer, and only this (main) thread touches it. ---
        self._layer = None
        self._layer_cam = (0.0, 0.0)
        self._layer_zoom = 1.0
        self._prev_zoom = 1.0
        self._prev_now = time.monotonic()

        # --- cloud background (untouched) ---
        self.cloud = cloud
        if cloud:
            self.cloud_tex = _make_cloud_tex(512, cloud_seed)
            self.cloud_scale = cloud_scale
            self.cloud_drift = cloud_drift
            self.cloud_depth = cloud_depth
            self.cloud_t = 0.0
            self._cloud_surf = pygame.Surface((self.w, self.h))

    # --- view API ---------------------------------------------------------
    def set_camera(self, x, y):
        self._cam = (float(x), float(y))

    def set_zoom(self, z):
        self.zoom = float(z)

    def pending_points(self):
        return 0                          # static: nothing to wait for

    def render(self):
        pass

    def shutdown(self):
        pass                              # single-threaded; nothing to stop

    # --- culling (main thread, against the live camera) -------------------
    @staticmethod
    def _colour_keys(cols):
        cb = cols & 0xF8                              # bucket colours -> packed int
        return ((cb[:, 0].astype(np.uint32) << 16)
                | (cb[:, 1].astype(np.uint32) << 8)
                | cb[:, 2].astype(np.uint32))

    def _cull(self, cam, zoom, x0, y0, x1, y1):
        """Points whose kernel touches screen-rect [x0,x1)x[y0,y1) at (cam, zoom).
        Filter in world space (cheap comparisons) then transform only the matches.
        Authoritative for THIS frame's camera — no async lag to leave strip gaps."""
        cx = cam[0] + self.w * 0.5
        cy = cam[1] + self.h * 0.5
        r = self._scaled_r
        wx0 = cx + (x0 - r - self.w * 0.5) / zoom
        wx1 = cx + (x1 + r - self.w * 0.5) / zoom
        wy0 = cy + (y0 - r - self.h * 0.5) / zoom
        wy1 = cy + (y1 + r - self.h * 0.5) / zoom
        P = self.points
        m = (P[:, 0] >= wx0) & (P[:, 0] < wx1) & (P[:, 1] >= wy0) & (P[:, 1] < wy1)
        px = P[m]
        sx = self.w * 0.5 + (px[:, 0] - cx) * zoom
        sy = self.h * 0.5 + (px[:, 1] - cy) * zoom
        return sx, sy, self._keys_all[m], self.completion[m]

    # --- draw kernels (main thread only) ----------------------------------
    def _ensure_kernels(self, zoom):
        zb = round(zoom, 2)
        if zb == self._zoom_bucket:
            return
        self._zoom_bucket = zb
        self._scaled_r = max(1, int(round(self.radius * zoom)))
        self._solid = {}                  # rebuilt crisply at the new radius
        self._fade = {}

    def _blit_settled(self, target, sx, sy, keys, r):
        """Full size, full opacity, batched. Opaque colorkey circles (RLE) — a
        plain copy with no per-pixel alpha blend (~6x faster than SRCALPHA)."""
        cache = self._solid
        for k in np.unique(keys).tolist():        # only ~one per terrain colour
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
            s = s.convert()               # match the display format for a fast blit
        except pygame.error:
            pass                          # no display yet (headless tests)
        s.set_colorkey(_CKEY, pygame.RLEACCEL)
        return s

    def _blit_fading(self, target, sx, sy, keys, comp, now):
        """The fading minority: grow 0->full and fade transparent->opaque with
        progress. Per-point scale+alpha, kernels cache by (colour, progress)."""
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
        ker.fill((ck >> 16, (ck >> 8) & 0xFF, ck & 0xFF, int(255 * frac)),
                 special_flags=pygame.BLEND_RGBA_MULT)
        return ker, r2

    # --- terrain layer ----------------------------------------------------
    def _new_layer(self):
        s = pygame.Surface((self.w, self.h))
        try:
            s = s.convert()
        except pygame.error:
            pass
        s.fill(_CKEY)
        s.set_colorkey(_CKEY)             # plain colorkey (modified each frame -> no RLE)
        return s

    def _rebuild_layer(self, now, cam, zoom):
        """Full re-render of the visible settled points onto the layer."""
        self._layer.fill(_CKEY)
        sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
        settled = now >= comp
        if settled.any():
            self._blit_settled(self._layer, sx[settled], sy[settled],
                               keys[settled], self._scaled_r)
        self._layer_cam = cam
        self._layer_zoom = zoom

    def _scroll_layer(self, now, cam, zoom):
        """Pan: shift the layer to the new camera, refilling only the exposed edge
        strips — culled against the live camera, so a fast pan can't leave gaps."""
        L = self._layer
        lcx, lcy = self._layer_cam
        ddx = int(round((lcx - cam[0]) * zoom))
        ddy = int(round((lcy - cam[1]) * zoom))
        if ddx == 0 and ddy == 0:
            return                                   # sub-pixel drift; still aligned
        if abs(ddx) >= self.w or abs(ddy) >= self.h:
            self._rebuild_layer(now, cam, zoom)      # teleport
            return
        L.scroll(ddx, ddy)
        self._layer_cam = (lcx - ddx / zoom, lcy - ddy / zoom)
        lcam = self._layer_cam                       # cull strips at the layer's own
        r = self._scaled_r                           # alignment so they meet the scroll

        def strip(x0, x1, y0, y1):                   # clear + refill one exposed band
            L.fill(_CKEY, (x0, y0, x1 - x0, y1 - y0))
            sx, sy, keys, comp = self._cull(lcam, zoom, x0, y0, x1, y1)
            settled = now >= comp
            if settled.any():
                self._blit_settled(L, sx[settled], sy[settled], keys[settled], r)

        if ddx > 0:   strip(0, ddx, 0, self.h)              # exposed on the left
        elif ddx < 0: strip(self.w + ddx, self.w, 0, self.h)
        if ddy > 0:   strip(0, self.w, 0, ddy)              # exposed on the top
        elif ddy < 0: strip(0, self.w, self.h + ddy, self.h)

    def _commit_settles(self, now, cam, zoom):
        """Stamp points that crossed completion since last frame onto the layer."""
        if self._prev_now >= self._max_completion:
            return                                   # everything settled long ago
        sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
        just = (comp > self._prev_now) & (comp <= now)
        if just.any():
            self._blit_settled(self._layer, sx[just], sy[just], keys[just], self._scaled_r)

    def _reproject(self, target, cam, zoom):
        """Active zoom: display the cached layer scaled (nearest) + offset, no
        re-blit. Blocky while zooming; a crisp rebuild lands when zoom settles."""
        L = self._layer
        scale = zoom / self._layer_zoom
        lcx, lcy = self._layer_cam
        ox = self.w * 0.5 + (lcx - cam[0]) * zoom    # screen pos of the layer centre
        oy = self.h * 0.5 + (lcy - cam[1]) * zoom
        sw = max(1, int(round(self.w * scale)))
        sh = max(1, int(round(self.h * scale)))
        scaled = pygame.transform.scale(L, (sw, sh))
        scaled.set_colorkey(_CKEY)
        target.blit(scaled, (int(ox - sw * 0.5), int(oy - sh * 0.5)))

    def _cloud_background(self, target, cam_x, cam_y):
        size = self.cloud_tex.shape[0]
        dx, dy = self.cloud_drift
        s = self.cloud_scale
        xs = (((np.arange(self.w) + cam_x) * s + self.cloud_t * dx).astype(np.int64)) % size
        ys = (((np.arange(self.h) + cam_y) * s + self.cloud_t * dy).astype(np.int64)) % size
        noise = self.cloud_tex[np.ix_(ys, xs)]
        billow = np.abs(noise * 2.0 - 1.0)            # abs of signed noise -> bulbous
        d = self.cloud_depth
        shade = (255 - (1.0 - billow) * d).astype(np.uint8)
        blue = (255 - (1.0 - billow) * (d * 0.7)).astype(np.uint8)
        rgb = pygame.surfarray.pixels3d(self._cloud_surf)
        rgb[:, :, 0] = shade.T
        rgb[:, :, 1] = shade.T
        rgb[:, :, 2] = blue.T
        del rgb
        target.blit(self._cloud_surf, (0, 0))

    def draw(self, target):
        cam = self._cam
        zoom = self.zoom
        if self.cloud:
            self._cloud_background(target, cam[0], cam[1])
        else:
            target.fill(self.bg)

        now = time.monotonic()
        self._ensure_kernels(zoom)

        if self._layer is None:                                   # first frame
            self._layer = self._new_layer()
            self._rebuild_layer(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        elif zoom == self._layer_zoom:                            # pan / static
            self._scroll_layer(now, cam, zoom)
            self._commit_settles(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        elif zoom == self._prev_zoom:                             # zoom settled
            self._rebuild_layer(now, cam, zoom)
            target.blit(self._layer, (0, 0))
        else:                                                     # active zoom
            self._reproject(target, cam, zoom)

        # fading points change every frame -> drawn fresh on top (only while any
        # point is still mid-fade; after that this is skipped entirely)
        if now < self._max_completion:
            sx, sy, keys, comp = self._cull(cam, zoom, 0, 0, self.w, self.h)
            fading = now < comp
            if fading.any():
                self._blit_fading(target, sx[fading], sy[fading], keys[fading],
                                  comp[fading], now)

        self._prev_zoom = zoom
        self._prev_now = now
        if self.cloud:
            self.cloud_t += 1.0
