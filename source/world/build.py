"""World generation job for the loading screen. Builds terrain (pure data) and the
point-store Renderer (whose worker thread starts filling immediately), points the
view at spawn, and waits for the store to first fill to its cap before handing
control to the game. In-game, the elastic kernel keeps the store centered on the
view as the player pans.
"""
from source import config
from source.world.heightmap import TerrainHeight
from source.world.renderer import Renderer

CAP = getattr(config, "RENDER_CAP", None)            # None -> derived from screen
SPRING = getattr(config, "RENDER_SPRING", 0.08)
GEN_RATE = getattr(config, "RENDER_GEN_RATE", 400)


def build_world(seed):
    yield 0.0, "preparing"

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    renderer = Renderer(config.SCREEN_W, config.SCREEN_H, tile=config.TERRAIN_TILE,
                        objects=[terrain], cap=CAP, spring=SPRING, gen_rate=GEN_RATE)

    cam_x = -(config.SCREEN_W // 2)                  # spawn (world 0,0) centered
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
