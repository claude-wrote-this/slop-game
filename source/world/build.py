"""World generation job for the loading screen. Builds terrain (pure data) and the
streaming Renderer (whose worker thread starts filling immediately), points the
camera at spawn, and waits for the worker to reveal the spawn viewport before
handing control to the game. In-game, newly exposed areas clouds-clear on their
own as the worker fills them.
"""
from source import config
from source.world.heightmap import TerrainHeight
from source.world.renderer import Renderer

DENSITY = getattr(config, "RENDER_DENSITY", 0.055)
OVERSIZE = getattr(config, "RENDER_OVERSIZE", 3)


def build_world(seed):
    yield 0.0, "preparing"

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    renderer = Renderer(config.SCREEN_W, config.SCREEN_H, tile=config.TERRAIN_TILE,
                        objects=[terrain], density=DENSITY, oversize=OVERSIZE)

    cam_x = -(config.SCREEN_W // 2)
    cam_y = -(config.SCREEN_H // 2)
    renderer.set_camera(cam_x, cam_y)

    total = None
    while True:                                    # worker fills spawn in parallel
        rem = renderer.pending_points()
        if rem > 0 and total is None:
            total = rem
        if total is not None and rem == 0:
            break
        prog = 0.1 if total is None else 0.1 + 0.85 * (1 - rem / total)
        yield prog, "revealing the land"

    yield 1.0, "ready"
    return {"terrain": terrain, "renderer": renderer, "cam": (cam_x, cam_y)}
