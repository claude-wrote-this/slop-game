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

_BANDS = 30                       # horizontal sampling bands (loading-bar granularity)


def debug_map_job():
    """LoadingScene generator: sample the terrain colour on a grid and blit it
    full-screen. Sampling is heavy in bulk (every slab of every column), so it runs
    at 1/N resolution (DEBUG_MAP_DOWNSAMPLE) and nearest-upscales to the screen —
    N=1 is exact per-pixel, N>1 costs ~1/N^2 and is what keeps it usable at high
    layer counts. Split into ~30 horizontal bands so the loading bar advances; yields
    (progress, label) and returns the finished SCREEN_W x SCREEN_H surface.

    Each sample maps to a world coordinate (y is up: the top row is the highest
    world-y), and the sampler's returned top-layer colour fills the bitmap directly."""
    yield 0.0, "building terrain"
    W, H = config.SCREEN_W, config.SCREEN_H
    n = max(1, int(getattr(config, "DEBUG_MAP_DOWNSAMPLE", 1)))
    cw = max(1, (W + n - 1) // n)                         # coarse grid dims
    ch = max(1, (H + n - 1) // n)
    # Same construction as build_world, just with the debug seed and no renderer.
    terrain = TerrainHeight(config.DEBUG_MAP_SEED, layers=config.TERRAIN_LAYERS,
                            layer_dz=config.TERRAIN_LAYER_DZ,
                            relief_lo=config.TERRAIN_RELIEF_LO,
                            relief_hi=config.TERRAIN_RELIEF_HI)

    # Each coarse sample sits at the screen pixel (col*n, row*n); map that to world.
    cx = config.DEBUG_MAP_CENTER[0] + (np.arange(cw) * n - W / 2.0) * config.DEBUG_MAP_SCALE_X
    img = np.empty((ch, cw, 3), np.uint8)                # coarse (row, col, rgb)
    bands = min(_BANDS, ch)
    edges = np.linspace(0, ch, bands + 1).astype(int)
    for b in range(bands):
        r0, r1 = int(edges[b]), int(edges[b + 1])
        if r1 <= r0:
            continue
        rows = np.arange(r0, r1)
        cy = config.DEBUG_MAP_CENTER[1] + (H / 2.0 - rows * n) * config.DEBUG_MAP_SCALE_Y
        X = np.broadcast_to(cx[None, :], (rows.size, cw))
        Y = np.broadcast_to(cy[:, None], (rows.size, cw))
        _, colour = terrain.sample_points(X, Y)          # (band_rows, cw, 3), colour as-is
        img[r0:r1] = colour
        yield (b + 1) / bands, f"sampling {b + 1}/{bands}"

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
