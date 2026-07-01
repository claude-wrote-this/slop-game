"""World build job for the loading screen. For now this builds the terrain
function and a static point-cloud Renderer (a pan/zoom test harness that samples
that function once), centered on the origin. Real sampler-driven generation will
return here later.
"""
from source import config
from source.world.heightmap import TerrainHeight
from source.world.renderer import Renderer

POISSON_R = getattr(config, "RENDER_POISSON_R", 20.0)   # min spacing (density)
KERNEL_R = getattr(config, "RENDER_KERNEL_R", 800.0)    # kept-disc radius (reach)


def build_world(seed):
    yield 0.0, "preparing"

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    renderer = Renderer(config.SCREEN_W, config.SCREEN_H, terrain=terrain,
                        tile=config.TERRAIN_TILE, seed=seed,
                        poisson_r=POISSON_R, kernel_r=KERNEL_R)

    cam_x = -(config.SCREEN_W // 2)                  # world origin centered on screen
    cam_y = -(config.SCREEN_H // 2)
    renderer.set_camera(cam_x, cam_y)

    yield 1.0, "ready"                               # static: nothing to wait for
    return {"terrain": terrain, "renderer": renderer, "cam": (cam_x, cam_y)}
