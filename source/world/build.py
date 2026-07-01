"""World build job for the loading screen. Builds the terrain function and the
roaming Poisson-disk Renderer, sized to the screen so the terrain looks the same
at any resolution and the point count stays bounded.
"""
import math

from source import config
from source.world.heightmap import TerrainHeight
from source.world.renderer import Renderer

_REF_DIAG = math.hypot(450, 820)                         # design reference screen
BASE_POISSON = getattr(config, "RENDER_POISSON_R", 20.0)  # min spacing at reference
# kernel_r as a fraction of the screen diagonal. zoom_min = 3/(4*factor), so 0.9
# gives a most-zoomed-out of ~0.83 — the default zoom (1.0) is always valid, at
# any resolution, which fixes the view starting more zoomed-out than allowed.
KERNEL_FACTOR = getattr(config, "RENDER_KERNEL_FACTOR", 0.9)


def build_world(seed):
    yield 0.0, "preparing"

    diag = math.hypot(config.SCREEN_W, config.SCREEN_H)
    s = diag / _REF_DIAG                                  # scale relative to reference
    tile = max(4, int(round(config.TERRAIN_TILE * s)))   # splats + terrain features
    poisson_r = BASE_POISSON * s                          # spacing scales -> ~const count
    kernel_r = KERNEL_FACTOR * diag                       # reach scales -> const zoom_min

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    renderer = Renderer(config.SCREEN_W, config.SCREEN_H, terrain=terrain,
                        tile=tile, seed=seed, poisson_r=poisson_r, kernel_r=kernel_r)

    cam_x = -(config.SCREEN_W // 2)                  # world origin centered on screen
    cam_y = -(config.SCREEN_H // 2)
    renderer.set_camera(cam_x, cam_y)

    yield 1.0, "ready"                               # generation streams in-game
    return {"terrain": terrain, "renderer": renderer, "cam": (cam_x, cam_y)}
