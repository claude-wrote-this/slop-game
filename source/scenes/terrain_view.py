"""A scene to look at terrain and drive the ceiling slider. Build a Field from a
seed, render the chunk once, and re-render on slider release as the ceiling moves
(rain restarts lower, so upper terrain peels away). This is a viewer/prototype,
not the game scene.

To reach it quickly, temporarily point New Game at it, e.g. in menu._new:
    from source.scenes.terrain_view import TerrainViewScene
    self.app.set_root(TerrainViewScene(self.app, seed=state.seed))
"""
import random

from source import config
from source.scenes.base import Scene
from source.ui.widgets import Button
from source.ui.slider import Slider
from source.world.field import Field
from source.world.render import render_chunk

Z_TOP = 18                                   # highest ceiling the slider allows


class TerrainViewScene(Scene):
    def __init__(self, app, seed=None):
        super().__init__(app)
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        # NOTE: these tuning numbers want to live in config — same set the game
        # scene uses, so both read identically.
        self.field = Field(self.seed, xy_scale=0.038, z_scale=0.16,
                           octaves=4, surface_z=8.0, z_bias=0.10)
        self.tile = 12
        # +1 so the grid covers the screen even when it doesn't divide evenly.
        self.cols = config.SCREEN_W // self.tile + 1
        self.rows = config.SCREEN_H // self.tile + 1
        self.ceiling = Z_TOP
        self.surf = None
        self._render()

        self.slider = Slider((40, config.SCREEN_H - 64, config.SCREEN_W - 80, 8),
                             1, Z_TOP, self.ceiling,
                             on_change=self._ceiling_live, on_commit=self._ceiling_commit,
                             label="ceiling")
        self.back = Button((config.SCREEN_W // 2 - 110, config.SCREEN_H - 132, 220, 44),
                           "Back", self.app.pop, kind="ghost")

    def _render(self):
        # The whole picture is one cached surface; we only rebuild it on a commit.
        self.surf = render_chunk(
            self.field, 0, 0, self.cols, self.rows,
            tile=self.tile, ceiling=self.ceiling,
            sun=(-1, -1), sun_slope=0.48,
            base=(108, 128, 90), haze=(48, 58, 78),
            haze_k=0.105, haze_max=0.60, shade_min=0.52)

    def _ceiling_live(self, v):
        self.ceiling = v               # cheap: just track it; handle/label update

    def _ceiling_commit(self, v):
        self.ceiling = v
        self._render()                 # expensive: re-rain at the new ceiling

    def handle_event(self, event):
        self.slider.handle_event(event)
        self.back.handle_event(event)

    def draw(self, surface):
        surface.blit(self.surf, (0, 0))        # cached terrain; no per-frame work
        label = self.app.font_small.render(f"seed {self.seed}", True, config.TEXT)
        surface.blit(label, (12, 12))
        self.back.draw(surface, self.app.font)
        self.slider.draw(surface, self.app.font_small)
        