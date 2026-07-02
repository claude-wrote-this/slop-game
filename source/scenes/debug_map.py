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
    """LoadingScene generator: sample the terrain colour one sample per screen pixel,
    in ~30 horizontal bands so the loading bar advances between bands. Yields
    (progress, label); returns the finished SCREEN_W x SCREEN_H surface.

    Each pixel maps to a world coordinate (y is up: the top row is the highest
    world-y), and the sampler's returned top-layer colour fills the bitmap directly."""
    yield 0.0, "building terrain"
    W, H = config.SCREEN_W, config.SCREEN_H
    # Same construction as build_world, just with the debug seed and no renderer.
    terrain = TerrainHeight(config.DEBUG_MAP_SEED, layers=config.TERRAIN_LAYERS,
                            layer_dz=config.TERRAIN_LAYER_DZ,
                            relief_lo=config.TERRAIN_RELIEF_LO,
                            relief_hi=config.TERRAIN_RELIEF_HI)

    cols = np.arange(W)
    world_x = config.DEBUG_MAP_CENTER[0] + (cols - W / 2.0) * config.DEBUG_MAP_SCALE_X
    img = np.empty((H, W, 3), np.uint8)                  # (row, col, rgb)
    edges = np.linspace(0, H, _BANDS + 1).astype(int)
    for b in range(_BANDS):
        r0, r1 = int(edges[b]), int(edges[b + 1])
        if r1 <= r0:
            continue
        rows = np.arange(r0, r1)
        world_y = config.DEBUG_MAP_CENTER[1] + (H / 2.0 - rows) * config.DEBUG_MAP_SCALE_Y
        X = np.broadcast_to(world_x[None, :], (rows.size, W))
        Y = np.broadcast_to(world_y[:, None], (rows.size, W))
        _, colour = terrain.sample_points(X, Y)          # (band_rows, W, 3), colour as-is
        img[r0:r1] = colour
        yield (b + 1) / _BANDS, f"sampling {b + 1}/{_BANDS}"

    # surfarray wants (col, row, rgb); our grid is (row, col, rgb).
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
    return surf


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
