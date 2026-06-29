"""Screen-space streaming terrain renderer over a drifting cloud background.

Model: a drifting billow cloud fills the background; terrain streams in ON TOP,
fading in (alpha) to cover the cloud where it spawns. Unexplored area shows the
moving clouds; as terrain points accumulate they fade from cloud to solid ground.

A full-speed worker thread drains the exposed area (no per-frame budget): it
scatters random points, samples each (numpy -> GIL released, so genuinely
parallel) and splats a soft, semi-transparent kernel into an alpha buffer.
Selection is biased AWAY from the nearest screen edge, so the fill creeps inward
off the frame (soft trailing edge) rather than repopulating evenly. Main draws
the cloud background, then blits the alpha buffer over it.

Object protocol: covers_screen: bool; sample_points(X, Y) -> (height, colour).

Threading: the worker owns/mutates the buffer (scroll/blank/splat); main only
reads it. A lock guards a worker SCROLL against a main read; splats run lock-free
(a pixel caught mid-write is at worst one frame stale).
"""
import threading
import time

import numpy as np
import pygame

BATCH = 400


def _make_kernel(radius, peak):
    # soft, semi-transparent disc; accumulating these alpha-composites terrain
    # up from transparent (cloud showing) to opaque -> points fade in.
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
    def __init__(self, screen_w, screen_h, *, tile=16, objects=(),
                 density=0.03, oversize=3, bg=(15, 17, 21),
                 cloud=True, cloud_scale=0.55, cloud_drift=(0.35, 0.14),
                 cloud_seed=0, cloud_depth=85, fill_bias=3, fade=200, cell=32):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.objects = list(objects)
        self.density = density
        self.fill_bias = fill_bias
        self.cell = cell
        self.bg = bg
        self.radius = max(1, int(tile * 0.95))
        self.kernel = _make_kernel(self.radius, fade)
        self._kc = {}
        self.M = oversize * tile
        self.bw = self.w + 2 * self.M
        self.bh = self.h + 2 * self.M
        self.buffer = pygame.Surface((self.bw, self.bh), pygame.SRCALPHA)
        self.buffer.fill((0, 0, 0, 0))                 # transparent -> cloud shows through
        self.buf_ox = 0
        self.buf_oy = 0
        self._rng = np.random.default_rng()
        self.cam_x = 0
        self.cam_y = 0
        self._init = False
        self._prefetch = max(self.radius + tile, self.M - self.radius)
        self.dirty = []
        self._lock = threading.Lock()
        self._edge_norm = 0.5 * min(self.w, self.h)
        self.cloud = cloud
        if cloud:
            self.cloud_tex = _make_cloud_tex(512, cloud_seed)
            self.cloud_scale = cloud_scale
            self.cloud_drift = cloud_drift
            self.cloud_depth = cloud_depth
            self.cloud_t = 0.0
            self._cloud_surf = pygame.Surface((self.w, self.h))
        self._running = True
        self._worker = threading.Thread(target=self._work, name="splat", daemon=True)
        self._worker.start()

    def set_camera(self, x, y):
        self.cam_x, self.cam_y = int(x), int(y)

    def pending_points(self):
        with self._lock:
            return sum(d[4] for d in self.dirty)

    # --- worker side ------------------------------------------------------
    def _splat(self, px, py, colour):
        r = self.radius
        for i in range(px.shape[0]):
            c = colour[i]
            key = (int(c[0]) & ~7, int(c[1]) & ~7, int(c[2]) & ~7)
            k = self._kc.get(key)
            if k is None:
                k = self.kernel.copy()
                k.fill((*key, 255), special_flags=pygame.BLEND_RGBA_MULT)
                self._kc[key] = k
            self.buffer.blit(k, (px[i] - r, py[i] - r))

    def _stream(self, budget):
        if not self.dirty:
            return
        rb = np.array([(x, y, w, h) for x, y, w, h, _, _ in self.dirty], dtype=float)
        rem_arr = np.array([d[4] for d in self.dirty], dtype=float)
        init_arr = np.array([max(1.0, d[5]) for d in self.dirty], dtype=float)
        areas = rb[:, 2] * rb[:, 3]
        total = areas.sum()
        remaining = rem_arr.sum()
        budget = min(int(budget), int(remaining))
        if total <= 0 or budget <= 0:
            if remaining <= 0:
                self.dirty = []
            return
        # oversample, then keep `budget` biased AWAY from the nearest screen edge
        # (fills in even rings -> soft trailing edge, no corner lag); Gumbel-top-k.
        n = budget * 4
        rem_arr = np.array([d[4] for d in self.dirty], dtype=float)
        frac = np.clip(rem_arr / np.maximum(1.0, self.density * areas), 0.0, 1.0) ** 2
        cidx = self._rng.choice(len(self.dirty), size=n, p=areas / total)
        csel = rb[cidx]
        cpx = csel[:, 0] + self._rng.random(n) * csel[:, 2]
        cpy = csel[:, 1] + self._rng.random(n) * csel[:, 3]
        sx = cpx - (self.cam_x - self.buf_ox)
        sy = cpy - (self.cam_y - self.buf_oy)
        edge = np.minimum(np.minimum(sx, self.w - sx),
                          np.minimum(sy, self.h - sy)) / self._edge_norm
        g = -np.log(-np.log(self._rng.random(n) + 1e-12) + 1e-12)
        # bias fades as a region fills (frac -> 0) so the final points land
        # uniformly and the edges catch up to full density -> clean settle.
        pick = np.argpartition(g + edge * (self.fill_bias * frac[cidx]),
                               n - budget)[n - budget:]
        idx = cidx[pick]
        px = cpx[pick]
        py = cpy[pick]
        X = (px + self.buf_ox) / self.tile
        Y = (py + self.buf_oy) / self.tile
        for obj in self.objects:
            if getattr(obj, "covers_screen", False):
                _, colour = obj.sample_points(X, Y)
                self._splat(px, py, colour.astype(np.int64))
                break
        counts = np.bincount(idx, minlength=len(self.dirty))
        self.dirty = [(x, y, w, h, rem - int(counts[i]), init)
                      for i, (x, y, w, h, rem, init) in enumerate(self.dirty)
                      if rem - int(counts[i]) > 0]

    def _mark(self, rect):
        x, y, w, h = rect
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.bw, x + w), min(self.bh, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        self.buffer.fill((0, 0, 0, 0), (x0, y0, x1 - x0, y1 - y0))   # back to cloud
        m = self.radius
        ex0, ey0 = max(0, x0 - m), max(0, y0 - m)
        ex1, ey1 = min(self.bw, x1 + m), min(self.bh, y1 + m)
        # subdivide into small cells so each fills to full density uniformly
        # (no under-filled fringe); the bias then orders cells, not pixels.
        c = self.cell
        cy = ey0
        while cy < ey1:
            ch = min(c, ey1 - cy)
            cx = ex0
            while cx < ex1:
                cw = min(c, ex1 - cx)
                cnt = int(self.density * cw * ch)
                if cnt > 0:
                    self.dirty.append((cx, cy, cw, ch, cnt, cnt))
                cx += c
            cy += c

    def _recenter(self):
        wx, wy = self.cam_x - self.buf_ox, self.cam_y - self.buf_oy
        p = self._prefetch
        if not (wx < p or wy < p or wx > self.bw - self.w - p or wy > self.bh - self.h - p):
            return
        to_ox, to_oy = self.cam_x - self.M, self.cam_y - self.M
        ddx, ddy = to_ox - self.buf_ox, to_oy - self.buf_oy
        if abs(ddx) >= self.bw or abs(ddy) >= self.bh:
            self.buf_ox, self.buf_oy = to_ox, to_oy
            self.dirty = []
            self._mark((0, 0, self.bw, self.bh))
            return
        self.buffer.scroll(-ddx, -ddy)
        self.buf_ox, self.buf_oy = to_ox, to_oy
        shifted = []
        for bx, by, bw, bh, rem, init in self.dirty:
            nx, ny = bx - ddx, by - ddy
            cx0, cy0 = max(0, nx), max(0, ny)
            cx1, cy1 = min(self.bw, nx + bw), min(self.bh, ny + bh)
            if cx1 > cx0 and cy1 > cy0:
                frac = ((cx1 - cx0) * (cy1 - cy0)) / (bw * bh)
                shifted.append((cx0, cy0, cx1 - cx0, cy1 - cy0,
                                int(rem * frac), max(1, int(init * frac))))
        self.dirty = shifted
        # Expose the newly-revealed L-shape WITHOUT overlapping the two strips:
        # a shared corner marked twice gets double the point budget and fills at
        # 2x density, racing ahead of its neighbours (the completed-box trail).
        if ddx > 0:
            vx0, vx1 = self.bw - ddx, self.bw
        elif ddx < 0:
            vx0, vx1 = 0, -ddx
        else:
            vx0 = vx1 = 0
        if vx1 > vx0:
            self._mark((vx0, 0, vx1 - vx0, self.bh))         # vertical strip, full height
        if ddy > 0:
            hy0, hy1 = self.bh - ddy, self.bh
        elif ddy < 0:
            hy0, hy1 = 0, -ddy
        else:
            hy0 = hy1 = 0
        if hy1 > hy0:
            hx0, hx1 = 0, self.bw                            # horizontal strip,
            if ddx > 0:
                hx1 = self.bw - ddx                          # minus the vertical strip's
            elif ddx < 0:
                hx0 = -ddx                                   # column (corner already done)
            if hx1 > hx0:
                self._mark((hx0, hy0, hx1 - hx0, hy1 - hy0))

    def _work(self):
        while self._running:
            with self._lock:
                if not self._init:
                    self.buf_ox = self.cam_x - self.M
                    self.buf_oy = self.cam_y - self.M
                    self.dirty = []
                    self._mark((0, 0, self.bw, self.bh))
                    self._init = True
                else:
                    self._recenter()
            if self.dirty:
                self._stream(BATCH)
            else:
                time.sleep(0.005)

    # --- main side --------------------------------------------------------
    def render(self):
        pass                                  # all generation is on the worker

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
        with self._lock:
            wx, wy = self.cam_x - self.buf_ox, self.cam_y - self.buf_oy
            sx0, sy0 = max(0, wx), max(0, wy)
            sx1, sy1 = min(self.bw, wx + self.w), min(self.bh, wy + self.h)
            if sx1 > sx0 and sy1 > sy0:
                target.blit(self.buffer, (sx0 - wx, sy0 - wy),
                            area=pygame.Rect(sx0, sy0, sx1 - sx0, sy1 - sy0))
        if self.cloud:
            self.cloud_t += 1.0

    def shutdown(self):
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
