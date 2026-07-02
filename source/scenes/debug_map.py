"""TEMPORARY debug tool — render the terrain sampler as a flat full-screen bitmap.

This is NOT a game feature. It is a throwaway aid for eyeballing the noise/terrain
output at large scale, bypassing the Poisson/kernel/streaming renderer entirely: a
direct regular-grid sample straight through TerrainHeight, one sample per pixel,
using the colour it returns as-is.

To delete the whole tool: remove this file, the "Debug Map" button and _debug_map()
in scenes/menu.py, and the DEBUG_MAP_* block in config.py. Nothing else depends on it.
"""
import numpy as np
import pygame

from source import config
from source.scenes.base import Scene
from source.ui.widgets import Button
from source.world.heightmap import TerrainHeight

_MEM_BUDGET = 200_000            # cap on (points x slabs) per sample call -> bounded RAM


def debug_map_job():
    """LoadingScene generator: sample the terrain colour on a grid and blit it
    full-screen. Two things keep it from taking forever / eating all RAM:

      * DOWNSAMPLE: sample a 1/N grid and nearest-upscale (N=1 = full res, cost
        ~1/N^2) — bounds the total point count.
      * chunking: sample_points builds the whole (points x slabs) occupancy stack in
        memory, so a big call at a high layer count can allocate gigabytes. We sample
        in row-chunks sized so (points x slabs) stays under a fixed budget, bounding
        peak RAM regardless of resolution or layer count.

    Coordinates are sampled in the game's own terrain space (world / TERRAIN_TILE), so
    the map matches what the renderer draws; y is up (top row = highest world-y). Yields
    (progress, label); returns the finished SCREEN_W x SCREEN_H surface."""
    yield 0.0, "building terrain"
    W, H = config.SCREEN_W, config.SCREEN_H
    n = max(1, int(getattr(config, "DEBUG_MAP_DOWNSAMPLE", 1)))
    cw = max(1, (W + n - 1) // n)                         # coarse grid dims
    ch = max(1, (H + n - 1) // n)
    # Same construction as build_world, just with the debug seed and no renderer.
    terrain = TerrainHeight(config.DEBUG_MAP_SEED, layers=config.TERRAIN_LAYERS,
                            layer_dz=config.TERRAIN_LAYER_DZ,
                            scale_x=config.TERRAIN_SCALE_X,
                            scale_y=config.TERRAIN_SCALE_Y,
                            scale_z=config.TERRAIN_SCALE_Z,
                            relief_lo=config.TERRAIN_RELIEF_LO,
                            relief_hi=config.TERRAIN_RELIEF_HI)

    # Sample in cell units (world / tile) so the field matches the live renderer.
    tile = max(1, config.TERRAIN_TILE)
    cx = (config.DEBUG_MAP_CENTER[0] + (np.arange(cw) * n - W / 2.0)
          * config.DEBUG_MAP_SCALE_X) / tile
    img = np.empty((ch, cw, 3), np.uint8)                # coarse (row, col, rgb)
    # rows per chunk so points*slabs stays under budget (>=1 row); bounds peak memory.
    rpc = max(1, _MEM_BUDGET // max(1, terrain._nz * cw))
    total = (ch + rpc - 1) // rpc
    r0 = 0
    ci = 0
    while r0 < ch:
        r1 = min(ch, r0 + rpc)
        rows = np.arange(r0, r1)
        cy = (config.DEBUG_MAP_CENTER[1] + (H / 2.0 - rows * n)
              * config.DEBUG_MAP_SCALE_Y) / tile
        X = np.broadcast_to(cx[None, :], (rows.size, cw))
        Y = np.broadcast_to(cy[:, None], (rows.size, cw))
        _, colour = terrain.sample_points(X, Y)          # (chunk_rows, cw, 3), colour as-is
        img[r0:r1] = colour
        r0 = r1
        ci += 1
        yield min(1.0, r0 / ch), f"sampling {ci}/{total}"

    # surfarray wants (col, row, rgb); our grid is (row, col, rgb). Upscale to full screen.
    coarse = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
    return pygame.transform.scale(coarse, (W, H)) if n > 1 else coarse


class DebugMapScene(Scene):
    """Static full-screen blit of the sampled bitmap, with a Back button to the menu.
    No pan, no zoom — matches the Options/Load scene shape (pushed over the menu)."""

    def __init__(self, app, bitmap):
        super().__init__(app)
        self.bitmap = bitmap
        self.back = Button((config.SCREEN_W // 2 - 120, config.SCREEN_H - 90, 240, 56),
                           "Back", self.app.pop, kind="ghost")

    def handle_event(self, event):
        self.back.handle_event(event)

    def draw(self, surface):
        surface.blit(self.bitmap, (0, 0))
        self.back.draw(surface, self.app.font)
