"""Static point-cloud terrain renderer over a drifting cloud background.

TEST HARNESS (deliberately minimal): the renderer holds a STATIC set of
world-space points and draws them through a movable, zoomable view. There is no
generation worker, no sampler, no buffering — the point of this version is to
isolate and measure pan/zoom on a fixed point set.

  * A fixed world-space `area` is sampled once at construction with
    n = density * area points (random positions + a legible banded colour).
  * The view is a world center + zoom. Pan moves the center; zoom is a draw-time
    world->screen scale only. Soft/blurry when zoomed in is expected.

The previous elastic-kernel generator, its sampler dependency, and the triple
buffer were removed: the sampler was the wrong model and, with static data, the
buffer guards nothing. Generation will come back later against the real sampler;
the cloud background is kept untouched as the backdrop the points sit over.
"""
import numpy as np
import pygame


def _make_kernel(radius, peak):
    # soft, semi-transparent disc; tinted per point colour at draw time.
    d = radius * 2 + 1
    yy, xx = np.mgrid[0:d, 0:d]
    dist = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2) / max(1, radius)
    alpha = (np.clip(1 - dist, 0, 1) ** 1.5 * peak).astype(np.uint8)
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
    def __init__(self, screen_w, screen_h, *, tile=16, density=0.008, area=None,
                 seed=0, bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85, fade=200):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg

        # --- static point set: n = density * area, sampled once ---------------
        aw, ah = area if area else (4 * self.w, 4 * self.h)
        self.area = (aw, ah)
        self.density = density
        n = max(1, int(density * aw * ah))
        rng = np.random.default_rng(seed)
        px = rng.uniform(-aw / 2, aw / 2, n)        # world-space, centered on origin
        py = rng.uniform(-ah / 2, ah / 2, n)
        self.points = np.stack([px, py], axis=1)
        # a legible low-frequency band so pan/zoom is visually meaningful
        t = 0.5 + 0.5 * np.sin(px * 0.012) * np.cos(py * 0.012)
        lo = np.array([90, 120, 80], float); hi = np.array([170, 180, 140], float)
        self.colours = (lo * (1 - t)[:, None] + hi * t[:, None]).astype(np.uint8)
        self.n = n

        # --- view (main thread) ---
        self.cam_x = 0.0          # world-space top-left of the view at zoom 1
        self.cam_y = 0.0
        self.zoom = 1.0
        # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0   # clamp — needed eventually (scene side)

        # --- draw kernels (main thread only) ---
        self.radius = max(1, int(tile * 0.95))
        self._base_kernel = _make_kernel(self.radius, fade)
        self._zoom_bucket = None
        self._scaled_base = self._base_kernel
        self._scaled_r = self.radius
        self._tint = {}                 # colour key -> tinted scaled kernel

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
        self.cam_x, self.cam_y = float(x), float(y)

    def set_zoom(self, z):
        self.zoom = float(z)

    def pending_points(self):
        return 0                          # static: nothing to wait for

    def render(self):
        pass                              # nothing to generate

    def shutdown(self):
        pass                              # no worker to stop

    # --- draw -------------------------------------------------------------
    def _ensure_kernels(self, zoom):
        zb = round(zoom, 2)
        if zb == self._zoom_bucket:
            return
        self._zoom_bucket = zb
        r = max(1, int(round(self.radius * zoom)))
        d = r * 2 + 1
        self._scaled_r = r
        self._scaled_base = (self._base_kernel if r == self.radius
                             else pygame.transform.smoothscale(self._base_kernel, (d, d)))
        self._tint = {}                   # colours must be re-tinted at the new size

    def _cloud_background(self, target):
        size = self.cloud_tex.shape[0]
        dx, dy = self.cloud_drift
        s = self.cloud_scale
        xs = (((np.arange(self.w) + self.cam_x) * s + self.cloud_t * dx).astype(np.int64)) % size
        ys = (((np.arange(self.h) + self.cam_y) * s + self.cloud_t * dy).astype(np.int64)) % size
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
        if self.cloud:
            self._cloud_background(target)
        else:
            target.fill(self.bg)

        zoom = self.zoom
        self._ensure_kernels(zoom)
        r = self._scaled_r
        cx = self.cam_x + self.w * 0.5
        cy = self.cam_y + self.h * 0.5
        # world -> screen: translate by view center, scale by zoom, re-center
        sx = self.w * 0.5 + (self.points[:, 0] - cx) * zoom
        sy = self.h * 0.5 + (self.points[:, 1] - cy) * zoom
        on = (sx > -r) & (sx < self.w + r) & (sy > -r) & (sy < self.h + r)

        # to python lists once: per-point numpy scalar access is the slow path
        xs = (sx[on] - r).astype(np.int32).tolist()
        ys = (sy[on] - r).astype(np.int32).tolist()
        cb = self.colours[on] & 0xF8                  # bucket colours (vectorised)
        keys = ((cb[:, 0].astype(np.uint32) << 16)
                | (cb[:, 1].astype(np.uint32) << 8)
                | cb[:, 2].astype(np.uint32)).tolist()

        base, cache = self._scaled_base, self._tint
        seq = []
        for i in range(len(xs)):
            k = keys[i]
            ker = cache.get(k)
            if ker is None:
                ker = base.copy()
                ker.fill((k >> 16, (k >> 8) & 0xFF, k & 0xFF, 255),
                         special_flags=pygame.BLEND_RGBA_MULT)
                cache[k] = ker
            seq.append((ker, (xs[i], ys[i])))
        if seq:
            target.blits(seq, doreturn=False)

        if self.cloud:
            self.cloud_t += 1.0
