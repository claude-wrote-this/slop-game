"""World-space point-store terrain renderer over a drifting cloud background.

Model (replaces the old screen-anchored scrolling buffer entirely):

  * Terrain is a flat store of world-space points (x, y, z, colour). World space
    is in pixel units; a point's cell coords are world/tile (what the sampler
    wants). The store knows nothing about the screen window.
  * The screen is a movable, zoomable VIEW: a world-space center + a zoom factor.
    Pan moves the center; zoom is a draw-time scale only (world->screen happens at
    blit time, never a resplat/regen). Soft/blurry when zoomed in is intended.

Generation (worker thread, unthrottled — never on the main thread, never
budget-limited):

  * An elastic kernel center springs toward the view center each tick by a
    fraction of the gap (near -> slow, far -> fast); `spring` is the stiffness.
  * Each tick scatters `gen_rate` points with gaussian spread `gen_sigma` around
    the kernel center, samples each through the object sampler for z/colour, and
    appends them.
  * Fixed point `cap`. At cap, each new point evicts the farthest-from-view-center
    existing point (world-space distance). Eviction is the density control: the
    store settles to the `cap` points closest to where you're looking.

Concurrency — a triple buffer with reference-rotation handoff (no full copy on
the hot path):

  * Three point buffers: write / interstitial / read. The producer owns `write`,
    promotes write<->interstitial under a short lock, and the reader grabs the
    interstitial into `read` under the same lock (read<->interstitial). A buffer's
    life is write -> interstitial -> read -> interstitial -> recycled to write.
  * The buffer recycled back to the producer is STALE. The producer reconciles it
    up to current truth by replaying its own append/evict delta (the slots it
    changed, tagged by a generation counter) — O(changes), not O(cap) — falling
    back to a single bulk copy only if the backlog ever exceeds the cap.
  * The reader's acquire is an O(1) reference grab; the producer never stalls on
    the reader, and the reader never stalls beyond the swap's tiny critical
    section (it iterates its buffer entirely outside the lock).

Object protocol (unchanged): covers_screen: bool; sample_points(X, Y) ->
(height, colour). The sampler and the cloud background are deliberately untouched.
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


class _Buf:
    """One point buffer: parallel slot arrays + the generation it last reflects.

    Slots are stable indices in [0, cap); `valid` marks which hold a live point.
    `synced_gen` is the producer generation this buffer's contents represent, used
    to decide how far it must be replayed when it cycles back to the producer.
    """
    __slots__ = ("pos", "z", "col", "valid", "synced_gen")

    def __init__(self, cap):
        self.pos = np.zeros((cap, 2), np.float64)
        self.z = np.zeros(cap, np.float64)
        self.col = np.zeros((cap, 3), np.uint8)
        self.valid = np.zeros(cap, bool)
        self.synced_gen = 0


class Renderer:
    def __init__(self, screen_w, screen_h, *, tile=16, objects=(),
                 cap=None, coverage=2.5, spring=0.08, gen_sigma=None, gen_rate=400,
                 bg=(15, 17, 21), cloud=True, cloud_scale=0.55,
                 cloud_drift=(0.35, 0.14), cloud_seed=0, cloud_depth=85, fade=200):
        self.w, self.h = screen_w, screen_h
        self.tile = tile
        self.objects = list(objects)
        # the one object that actually feeds points (covers_screen) is the sampler
        self._sampler = next((o for o in self.objects
                              if getattr(o, "covers_screen", False)), None)
        self.bg = bg

        self.cap = int(cap) if cap else int(coverage * (self.w / tile) * (self.h / tile))
        self.spring = spring
        self.gen_sigma = gen_sigma if gen_sigma is not None else 0.5 * max(self.w, self.h)
        self.gen_rate = gen_rate

        # --- view (main thread writes, worker reads) ---
        self.cam_x = 0.0          # world-space top-left of the view at zoom 1
        self.cam_y = 0.0
        self.zoom = 1.0           # draw-time scale only; set by the scene
        # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0   # clamp — needed eventually (see draw/scene)

        # --- point store: triple buffer (producer owns _W) ---
        self._W = _Buf(self.cap)        # write  (producer truth)
        self._I = _Buf(self.cap)        # interstitial (handoff slot)
        self._R = _Buf(self.cap)        # read   (reader-owned)
        self._count = 0                 # live points in truth
        self._gen = 0                   # producer generation counter
        self._log = []                  # [(gen, slots, pos, z, col, valid)] deltas
        self._dirty = []                # slot-index arrays touched this interval
        self._kc = None                 # elastic kernel center (world space)
        self._rng = np.random.default_rng()
        self._lock = threading.Lock()   # guards ONLY the buffer ref rotation

        # --- draw kernels (main thread only) ---
        self.radius = max(1, int(tile * 0.95))
        self._base_kernel = _make_kernel(self.radius, fade)
        self._zoom_bucket = None        # last zoom the scaled base was built for
        self._scaled_base = self._base_kernel
        self._scaled_r = self.radius
        self._tint = {}                 # colour-bucket -> tinted scaled kernel

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
        self._worker = threading.Thread(target=self._work, name="gen", daemon=True)
        self._worker.start()

    # --- view API (main thread) -------------------------------------------
    def set_camera(self, x, y):
        self.cam_x, self.cam_y = float(x), float(y)

    def set_zoom(self, z):
        self.zoom = float(z)

    def _view_center(self):
        return self.cam_x + self.w * 0.5, self.cam_y + self.h * 0.5

    def pending_points(self):
        # loading progress: how many points until the store first fills to cap.
        return max(0, self.cap - self._count)

    # --- worker side: generate into truth (self._W), no lock --------------
    def _gen_step(self):
        cx, cy = self._view_center()
        if self._kc is None:
            self._kc = np.array([cx, cy], dtype=float)
        else:
            # elastic spring: move a fraction of the gap (far -> fast, near -> slow)
            self._kc[0] += (cx - self._kc[0]) * self.spring
            self._kc[1] += (cy - self._kc[1]) * self.spring
        if self._sampler is None:
            return 0
        m = self.gen_rate
        gx = self._kc[0] + self._rng.standard_normal(m) * self.gen_sigma
        gy = self._kc[1] + self._rng.standard_normal(m) * self.gen_sigma
        _, colour = self._sampler.sample_points(gx / self.tile, gy / self.tile)
        z = np.zeros(m)            # height kept in the model; not used by draw yet
        return self._place(gx, gy, z, colour.astype(np.uint8), cx, cy)

    def _place(self, gx, gy, z, col, cx, cy):
        """Append points, evicting the farthest-from-center when at cap. Records
        every touched slot into self._dirty. Returns the number of changes."""
        W, cap = self._W, self.cap
        changed = 0

        free = cap - self._count
        if free > 0:
            k = int(min(free, gx.size))
            idx = np.arange(self._count, self._count + k)
            W.pos[idx, 0] = gx[:k]; W.pos[idx, 1] = gy[:k]
            W.z[idx] = z[:k]; W.col[idx] = col[:k]; W.valid[idx] = True
            self._count += k
            self._dirty.append(idx)
            changed += k
            gx, gy, z, col = gx[k:], gy[k:], z[k:], col[k:]

        m = int(gx.size)
        if m > 0 and self._count >= cap:
            if m > cap:                          # never generate more than the cap
                gx, gy, z, col, m = gx[:cap], gy[:cap], z[:cap], col[:cap], cap
            ex = W.pos[:, 0] - cx; ey = W.pos[:, 1] - cy
            dex = ex * ex + ey * ey
            # the m farthest-from-center slots are evicted; the m new points take
            # their place. Keeping the new and dropping the farthest is what holds
            # the store at a stable gaussian blob (extent ~ gen_sigma) instead of
            # collapsing it toward the center over time.
            far = np.argpartition(dex, cap - m)[cap - m:].astype(np.intp)
            W.pos[far, 0] = gx; W.pos[far, 1] = gy
            W.z[far] = z; W.col[far] = col       # valid already True at these slots
            self._dirty.append(far)
            changed += m
        return changed

    def _promote(self):
        """Close this interval's delta, rotate write<->interstitial under the
        lock, then reconcile the recycled (stale) buffer back up to truth."""
        if self._dirty:
            slots = np.unique(np.concatenate(self._dirty))
            self._dirty = []
        else:
            slots = np.empty(0, np.intp)

        W = self._W
        self._gen += 1
        W.synced_gen = self._gen
        # snapshot the final value of every slot changed this interval
        self._log.append((self._gen, slots, W.pos[slots].copy(), W.z[slots].copy(),
                          W.col[slots].copy(), W.valid[slots].copy()))

        with self._lock:                      # tiny critical section: refs only
            self._W, self._I = self._I, self._W
            new_w = self._W                   # stale, producer-exclusive now
            truth = self._I                   # freshly promoted == current truth

        self._reconcile(new_w, truth)
        self._trim_log()

    def _reconcile(self, buf, truth):
        """Bring a stale buffer up to current truth by replaying the deltas it
        missed (O(changes)); bulk-copy only if the backlog exceeds the cap."""
        sg = buf.synced_gen
        replay = [b for b in self._log if b[0] > sg]
        total = sum(int(b[1].size) for b in replay)
        if total == 0:
            buf.synced_gen = self._gen
            return
        if total >= self.cap:
            buf.pos[:] = truth.pos; buf.z[:] = truth.z
            buf.col[:] = truth.col; buf.valid[:] = truth.valid
        else:
            for _, slots, pos, z, col, valid in replay:
                buf.pos[slots] = pos; buf.z[slots] = z
                buf.col[slots] = col; buf.valid[slots] = valid
        buf.synced_gen = self._gen

    def _trim_log(self):
        lo = min(self._W.synced_gen, self._I.synced_gen, self._R.synced_gen)
        i = 0
        while i < len(self._log) and self._log[i][0] <= lo:
            i += 1
        if i:
            del self._log[:i]

    def _work(self):
        while self._running:
            changed = self._gen_step()
            self._promote()
            if not changed:
                time.sleep(0.004)        # idle only when fully settled

    # --- main side: draw --------------------------------------------------
    def render(self):
        pass                             # all generation is on the worker

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
        self._tint = {}                  # colours must be re-tinted at the new size

    def _kernel_for(self, c):
        key = (int(c[0]) & ~7, int(c[1]) & ~7, int(c[2]) & ~7)
        k = self._tint.get(key)
        if k is None:
            k = self._scaled_base.copy()
            k.fill((*key, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self._tint[key] = k
        return k

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

        with self._lock:                          # O(1) acquire: grab latest ready
            self._R, self._I = self._I, self._R
            rb = self._R                          # reader-owned until next acquire

        valid = rb.valid
        if valid.any():
            zoom = self.zoom
            # ZOOM_MIN, ZOOM_MAX = 0.05, 8.0; zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom))
            self._ensure_kernels(zoom)
            r = self._scaled_r
            cx, cy = self._view_center()
            pos = rb.pos[valid]; col = rb.col[valid]
            # world -> screen: translate by view center, scale by zoom, re-center
            sx = self.w * 0.5 + (pos[:, 0] - cx) * zoom
            sy = self.h * 0.5 + (pos[:, 1] - cy) * zoom
            on = (sx > -r) & (sx < self.w + r) & (sy > -r) & (sy < self.h + r)
            sx = sx[on] - r; sy = sy[on] - r; col = col[on]
            seq = [(self._kernel_for(col[i]), (sx[i], sy[i])) for i in range(sx.size)]
            if seq:
                target.blits(seq, doreturn=False)

        if self.cloud:
            self.cloud_t += 1.0

    def shutdown(self):
        self._running = False
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
