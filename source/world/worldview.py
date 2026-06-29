"""WorldView — the camera's window onto an unbounded chunked world.

The world is a grid of chunks; each is generated from TerrainHeight at its world
offset (seamless because the height field is a pure function of world coords),
rendered ONCE into a cached surface, and blitted at (chunk_pos - camera).

Generation runs on a BACKGROUND THREAD so panning never blocks: draw() requests
any missing visible chunk and paints a placeholder; when the worker finishes it
hands the surface back via a queue and the next frame blits it. Only off-screen
surfaces are touched off the main thread (the display surface is never passed to
the worker), which is the safe boundary for SDL.

prewarm() stays synchronous — during loading we *want* to do the work and show
progress, and it guarantees the first frame is fully painted.

Set async_gen=False to fall back to synchronous on-demand generation (one switch
if threaded surface work ever misbehaves on a given device).
"""
import math
import queue
import threading

from source.world.render import render_heightmap_chunk


class WorldView:
    def __init__(self, terrain, *, chunk=24, tile=16, layers=12, view_z=None,
                 margin=3, max_cached=80, async_gen=True,
                 placeholder=(26, 30, 28), **shade):
        self.terrain = terrain
        self.chunk = chunk                       # cells per chunk side
        self.tile = tile
        self.layers = layers
        self.view_z = view_z if view_z is not None else layers * 0.6
        self.margin = margin                     # extra cells rendered then cropped
        self.shade = shade                       # sun/base/haze/... for the renderer
        self.chunk_px = chunk * tile
        self.placeholder = placeholder
        self.max_cached = max_cached

        self.cache = {}                          # (cx, cy) -> Surface   (main thread)
        self.pending = set()                     # requested, not yet back (main thread)

        self.async_gen = async_gen
        self._req_q = queue.Queue()              # main -> worker: (cx, cy)
        self._res_q = queue.Queue()              # worker -> main: ((cx, cy), surface)
        self._worker = None
        if async_gen:
            self._worker = threading.Thread(target=self._work, name="chunkgen",
                                            daemon=True)
            self._worker.start()

    # --- generation -------------------------------------------------------
    def _render(self, cx, cy):
        # margin of neighbour cells so splats bleed across the edge (no seam) and
        # the shadow scan has context, then crop to the chunk. Off-screen only.
        import pygame
        m = self.margin
        cells = self.chunk + 2 * m
        layer, height = self.terrain.chunk(cx * self.chunk - m, cy * self.chunk - m,
                                           cells, cells)
        full = render_heightmap_chunk(
            layer, height, layers=self.layers, tile=self.tile, view_z=self.view_z,
            jitter_seed=((cx * 73856093) ^ (cy * 19349663)) & 0x7fffffff, **self.shade)
        crop = pygame.Rect(m * self.tile, m * self.tile, self.chunk_px, self.chunk_px)
        return full.subsurface(crop).copy()

    def _work(self):
        while True:
            job = self._req_q.get()
            if job is None:                      # shutdown sentinel
                return
            cx, cy = job
            try:
                surf = self._render(cx, cy)
            except Exception:
                surf = None                      # report back so pending clears
            self._res_q.put(((cx, cy), surf))

    def _drain(self):
        # main thread: move finished chunks into the cache
        while True:
            try:
                key, surf = self._res_q.get_nowait()
            except queue.Empty:
                break
            self.pending.discard(key)
            if surf is not None:
                self.cache[key] = surf

    def _request(self, cx, cy):
        if (cx, cy) not in self.pending:
            self.pending.add((cx, cy))
            self._req_q.put((cx, cy))

    # --- access -----------------------------------------------------------
    def get(self, cx, cy):
        """Synchronous: generate now if missing. Used by prewarm()."""
        s = self.cache.get((cx, cy))
        if s is None:
            s = self._render(cx, cy)
            self.cache[(cx, cy)] = s
        return s

    def visible_chunks(self, cam_x, cam_y, sw, sh):
        cx0 = math.floor(cam_x / self.chunk_px)
        cy0 = math.floor(cam_y / self.chunk_px)
        cx1 = math.floor((cam_x + sw - 1) / self.chunk_px)
        cy1 = math.floor((cam_y + sh - 1) / self.chunk_px)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                yield cx, cy

    def draw(self, surface, cam_x, cam_y):
        self._drain()                            # collect anything the worker finished
        sw, sh = surface.get_size()
        visible = set()
        for cx, cy in self.visible_chunks(cam_x, cam_y, sw, sh):
            visible.add((cx, cy))
            px = cx * self.chunk_px - cam_x
            py = cy * self.chunk_px - cam_y
            s = self.cache.get((cx, cy))
            if s is not None:
                surface.blit(s, (px, py))
            elif self.async_gen:
                self._request(cx, cy)            # ask the worker, paint a placeholder
                surface.fill(self.placeholder, (px, py, self.chunk_px, self.chunk_px))
            else:
                surface.blit(self.get(cx, cy), (px, py))   # sync fallback
        # evict off-screen chunks once the cache grows past the cap
        if len(self.cache) > self.max_cached:
            for k in [k for k in self.cache if k not in visible]:
                del self.cache[k]

    def prewarm(self, cam_x, cam_y, sw, sh):
        """Generator: render the spawn viewport synchronously, yielding
        (done, total) for the loading bar."""
        chunks = list(self.visible_chunks(cam_x, cam_y, sw, sh))
        for i, (cx, cy) in enumerate(chunks):
            self.get(cx, cy)
            yield i + 1, len(chunks)

    def shutdown(self):
        if self._worker is not None:
            self._req_q.put(None)
            self._worker = None