"""Static point-cloud terrain renderer over a drifting cloud background.

TEST HARNESS for pan/zoom. A fixed set of world-space points is sampled ONCE
from the terrain function; the work then splits across two threads:

  * Buffer thread (the "filter"): repeatedly culls the full static point set down
    to just what the current view can see and publishes that small visible subset
    (world coords + colour). All numpy, no pygame. This is the expensive O(all
    points) pass, kept off the main thread.
  * Main thread (pan / zoom / draw only): reads input into the view (world center
    + zoom), grabs the latest published subset, transforms that SMALL set with the
    live camera (so it's always exactly aligned) and blits it over the cloud.

So the heavy filter scales with the whole point set but runs on the buffer thread,
while the main thread only ever touches what's on screen. The terrain function
(TerrainHeight) is the thing being sampled and is untouched; only the sampling
STRATEGY here is the static stand-in for real generation. The cloud background is
also untouched.
"""
import threading
import time

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
    def __init__(self, screen_w, screen_h, *, terrain, tile=16, density=0.008,
                 area=None, seed=0, bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85, fade=200):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg

        # --- static point set: n = density * area, sampled once through the
        #     terrain function for colour (the function is the food; this random
        #     scatter is the sampler standing in for real generation) ----------
        aw, ah = area if area else (4 * self.w, 4 * self.h)
        self.area = (aw, ah)
        self.density = density
        n = max(1, int(density * aw * ah))
        rng = np.random.default_rng(seed)
        px = rng.uniform(-aw / 2, aw / 2, n)        # world-space, centered on origin
        py = rng.uniform(-ah / 2, ah / 2, n)
        self.points = np.stack([px, py], axis=1)
        _, colour = terrain.sample_points(px / tile, py / tile)
        self.colours = np.ascontiguousarray(colour, dtype=np.uint8)
        self.n = n

        # --- view (main thread writes, buffer thread reads) ---
        self.cam_x = 0.0          # world-space top-left of the view at zoom 1
        self.cam_y = 0.0
        self.zoom = 1.0
        # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0   # clamp — needed eventually (scene side)

        # --- the buffer: latest visible subset published by the filter thread ---
        self._ready = (self.points[:0], self.colours[:0])
        self._lock = threading.Lock()

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

        self._running = True
        self._worker = threading.Thread(target=self._filter, name="buffer", daemon=True)
        self._worker.start()

    # --- view API (main thread) -------------------------------------------
    def set_camera(self, x, y):
        self.cam_x, self.cam_y = float(x), float(y)

    def set_zoom(self, z):
        self.zoom = float(z)

    def pending_points(self):
        return 0                          # static: nothing to wait for

    def render(self):
        pass                              # the filter runs on the buffer thread

    # --- buffer thread: cull the full set to the visible subset -----------
    def _filter(self):
        last = None
        while self._running:
            cam_x, cam_y, zoom = self.cam_x, self.cam_y, self.zoom
            key = (cam_x, cam_y, zoom)
            if key == last:               # view unchanged -> nothing to refilter
                time.sleep(0.004)
                continue
            last = key
            cx = cam_x + self.w * 0.5
            cy = cam_y + self.h * 0.5
            sx = self.w * 0.5 + (self.points[:, 0] - cx) * zoom
            sy = self.h * 0.5 + (self.points[:, 1] - cy) * zoom
            m = self.radius * zoom + 1.0  # margin: a kernel straddling the edge
            on = (sx > -m) & (sx < self.w + m) & (sy > -m) & (sy < self.h + m)
            subset = (self.points[on], self.colours[on])
            with self._lock:              # publish: a tiny ref swap
                self._ready = subset

    # --- main thread: transform the small visible subset and blit ---------
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

        with self._lock:                  # O(1) grab of the latest visible subset
            pts, cols = self._ready

        if pts.shape[0]:
            zoom = self.zoom
            self._ensure_kernels(zoom)
            r = self._scaled_r
            cx = self.cam_x + self.w * 0.5
            cy = self.cam_y + self.h * 0.5
            # transform the SMALL subset with the live camera -> exact alignment
            sx = (self.w * 0.5 + (pts[:, 0] - cx) * zoom - r).astype(np.int32).tolist()
            sy = (self.h * 0.5 + (pts[:, 1] - cy) * zoom - r).astype(np.int32).tolist()
            cb = cols & 0xF8                                  # bucket colours
            keys = ((cb[:, 0].astype(np.uint32) << 16)
                    | (cb[:, 1].astype(np.uint32) << 8)
                    | cb[:, 2].astype(np.uint32)).tolist()
            base, cache = self._scaled_base, self._tint
            seq = []
            for i in range(len(sx)):
                k = keys[i]
                ker = cache.get(k)
                if ker is None:
                    ker = base.copy()
                    ker.fill((k >> 16, (k >> 8) & 0xFF, k & 0xFF, 255),
                             special_flags=pygame.BLEND_RGBA_MULT)
                    cache[k] = ker
                seq.append((ker, (sx[i], sy[i])))
            if seq:
                target.blits(seq, doreturn=False)

        if self.cloud:
            self.cloud_t += 1.0

    def shutdown(self):
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
