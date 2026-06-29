"""Loading scene. Runs a generation JOB and shows progress, then hands the
result on. It knows nothing about terrain: the job is a generator that yields
(progress, label) and returns a result. World tuning lives wherever you build
the job (config), never in here.

Wiring (see menu._new): build_world(seed) is the generator job; its return value
is the built world handed to GameScene.

    self.app.set_root(LoadingScene(
        self.app, build_world(state.seed),
        on_complete=lambda w: self.app.set_root(
            GameScene(self.app, slug, state, world=w))))
"""
import math

import pygame

from source import config
from source.scenes.base import Scene


class LoadingScene(Scene):
    def __init__(self, app, job, on_complete, *, title="Loading"):
        super().__init__(app)
        self.job = job                  # generator: yields (progress, label), returns result
        self.on_complete = on_complete  # called with the generator's return value
        self.title = title
        self.progress = 0.0
        self.label = ""
        self._t = 0.0                   # animation clock for the dots
        self._primed = False

    def update(self, dt):
        self._t += dt
        # Skip the very first update so one frame paints the screen BEFORE the
        # first (possibly heavy) job step runs — otherwise the player sees a
        # frozen blank instead of the loading screen.
        if not self._primed:
            self._primed = True
            return
        try:
            # One job step per frame: the bar moves between steps. A single fat
            # step just holds one frame; many small steps animate smoothly.
            self.progress, self.label = next(self.job)
        except StopIteration as done:
            # Generator finished: its return value is the built world.
            self.on_complete(done.value)

    def draw(self, surface):
        surface.fill(config.BG)
        cx, cy = config.SCREEN_W // 2, config.SCREEN_H // 2

        title = self.app.font_big.render(self.title, True, config.TEXT)
        surface.blit(title, title.get_rect(center=(cx, cy - 80)))

        # progress bar: track, fill proportional to progress, outline
        bw, bh = min(320, config.SCREEN_W - 80), 10
        bx, by = cx - bw // 2, cy
        pygame.draw.rect(surface, config.PANEL, (bx, by, bw, bh), border_radius=5)
        fill = int(bw * max(0.0, min(1.0, self.progress)))
        if fill > 0:
            pygame.draw.rect(surface, config.ACCENT, (bx, by, fill, bh), border_radius=5)
        pygame.draw.rect(surface, config.MUTED, (bx, by, bw, bh), width=1, border_radius=5)

        if self.label:
            lab = self.app.font.render(self.label, True, config.MUTED)
            surface.blit(lab, lab.get_rect(center=(cx, cy + 30)))

        # three dots pulsing out of phase, muted -> accent, as a liveness cue
        for i in range(3):
            a = (math.sin(self._t * 3 - i * 0.5) + 1) / 2
            col = tuple(int(config.MUTED[j] + (config.ACCENT[j] - config.MUTED[j]) * a)
                        for j in range(3))
            pygame.draw.circle(surface, col, (cx - 20 + i * 20, cy + 72), 3 + int(3 * a))
            