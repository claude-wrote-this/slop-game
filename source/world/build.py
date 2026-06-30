"""World build job for the loading screen. For now this builds the terrain
function and a static point-cloud Renderer (a pan/zoom test harness that samples
that function once), centered on the origin. Real sampler-driven generation will
return here later.
"""
from source import config
from source.world.heightmap import TerrainHeight
from source.world.renderer import Renderer

DENSITY = getattr(config, "RENDER_DENSITY", 0.008)   # target points per world px^2
AREA = getattr(config, "RENDER_AREA", None)          # None -> a few screens wide


def build_world(seed):
    yield 0.0, "preparing"

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    renderer = Renderer(config.SCREEN_W, config.SCREEN_H, terrain=terrain,
                        tile=config.TERRAIN_TILE, seed=seed, density=DENSITY, area=AREA)

    cam_x = -(config.SCREEN_W // 2)                  # world origin centered on screen
    cam_y = -(config.SCREEN_H // 2)
    renderer.set_camera(cam_x, cam_y)

    yield 1.0, "ready"                               # static: nothing to wait for
    return {"terrain": terrain, "renderer": renderer, "cam": (cam_x, cam_y)}
