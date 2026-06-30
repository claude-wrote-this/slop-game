"""Static point-cloud terrain renderer over a drifting cloud background.

TEST HARNESS for pan/zoom. A fixed set of world-space points is sampled ONCE from
the terrain function. The key design point: ALL screen-space work runs on a
dedicated render thread, never the main thread.

  * Render thread (does all the blitting): reads the camera + the world-space
    point arrays and blits the whole frame — cloud, culled/transformed visible
    points, UI overlay — onto the display surface. It does NOT flip.
  * Main thread: pumps input, updates the camera (set_camera/set_zoom), and calls
    pygame.display.flip() to present. flip must stay on the main thread; blitting
    does not, which is the whole point.

Cross-thread state is just the point arrays (read-only here; numpy, not surfaces)
and the camera (a tuple + a float). There is deliberately NO inter-thread buffer
or surface handoff — a shared surface would be written and read at once and fight
pygame's locks, which is exactly what we're avoiding.

The terrain function (TerrainHeight) is untouched; only the sampling STRATEGY
here is a static stand-in for real generation. The cloud background is untouched.
"""
import threading

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
                 area=None, seed=0, fps=60, bg=(15, 17, 21), cloud=True,
                 cloud_scale=0.55, cloud_drift=(0.35, 0.14), cloud_seed=0,
                 cloud_depth=85, fade=200):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.bg = bg
        self.fps = fps

        # --- static point set: n = density * area, sampled once through the
        #     terrain function for colour ------------------------------------
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

        # --- view: main thread writes, render thread reads --------------------
        self._cam = (0.0, 0.0)        # tuple so it swaps atomically (world top-left)
        self.zoom = 1.0
        # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0   # clamp — needed eventually (scene side)

        # --- draw kernels (render thread only) ---
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

        self._screen = None
        self._overlay = None
        self._running = False
        self._thread = None

    # --- view API (main thread) -------------------------------------------
    def set_camera(self, x, y):
        self._cam = (float(x), float(y))

    def set_zoom(self, z):
        self.zoom = float(z)

    def pending_points(self):
        return 0                          # static: nothing to wait for

    def render(self):
        pass                              # the render thread owns drawing

    # --- render thread ----------------------------------------------------
    def start(self, screen, overlay=None):
        """Hand the display surface to a render thread that blits into it.
        `overlay(target)` is called each frame after the terrain to paint UI (also
        off the main thread). The main loop still owns flip(). Call once when the
        scene becomes active."""
        self._screen = screen
        self._overlay = overlay
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="render", daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self):
        clock = pygame.time.Clock()
        while self._running:
            self.draw(self._screen)       # blit only; the main loop flips
            clock.tick(self.fps)

    # --- drawing (render thread) ------------------------------------------
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

    def _draw_points(self, target, cam_x, cam_y):
        zoom = self.zoom
        self._ensure_kernels(zoom)
        r = self._scaled_r
        cx = cam_x + self.w * 0.5
        cy = cam_y + self.h * 0.5
        sx = self.w * 0.5 + (self.points[:, 0] - cx) * zoom
        sy = self.h * 0.5 + (self.points[:, 1] - cy) * zoom
        on = (sx > -r) & (sx < self.w + r) & (sy > -r) & (sy < self.h + r)

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

    def draw(self, target):
        """Render one full frame. Runs on the render thread (also callable
        directly with a target surface for headless tests)."""
        cam_x, cam_y = self._cam
        if self.cloud:
            self._cloud_background(target, cam_x, cam_y)
        else:
            target.fill(self.bg)
        self._draw_points(target, cam_x, cam_y)
        if self._overlay is not None:
            self._overlay(target)
        if self.cloud:
            self.cloud_t += 1.0
