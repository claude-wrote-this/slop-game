"""World generation job for the loading screen. Builds the terrain and a
WorldView, then pre-warms the chunks the spawn viewport covers — so the loading
bar fills per chunk and the first frame has no hitch. Returns the WorldView and
the spawn camera; GameScene just drives them.

Tuning is the literal set used elsewhere; move into config during cleanup.
"""
from source import config
from source.world.heightmap import TerrainHeight
from source.world.worldview import WorldView

CHUNK_CELLS = getattr(config, "CHUNK_CELLS", 24)


def build_world(seed):
    yield 0.0, "preparing"

    terrain = TerrainHeight(seed, layers=config.TERRAIN_LAYERS)
    wv = WorldView(terrain, chunk=CHUNK_CELLS, tile=config.TERRAIN_TILE,
                   layers=config.TERRAIN_LAYERS,
                   sun=(-1, -1), sun_slope=0.48,
                   base=(108, 128, 90), haze=(48, 58, 78),
                   haze_k=0.105, haze_max=0.60, shade_min=0.52)

    # spawn camera: world (0,0) centred on screen
    cam_x = -(config.SCREEN_W // 2)
    cam_y = -(config.SCREEN_H // 2)

    # pre-warm the spawn viewport, chunk by chunk, driving the progress bar
    for done, total in wv.prewarm(cam_x, cam_y, config.SCREEN_W, config.SCREEN_H):
        yield 0.1 + 0.85 * done / total, "shaping the land"

    yield 1.0, "ready"
    return {"world_view": wv, "cam": (cam_x, cam_y), "terrain": terrain}
