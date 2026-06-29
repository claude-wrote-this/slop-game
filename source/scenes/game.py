"""In-game scene: a camera over the screen-space renderer. Drag to pan; the
renderer scrolls its oversized buffer and refills exposed strips. Generation
lives in build_world; this scene receives a prebuilt world or runs it inline.
"""
from source import config
from source.scenes.base import Scene
from source.ui.widgets import Button
from source.persistence import saves


class GameScene(Scene):
    def __init__(self, app, slug, state, world=None):
        super().__init__(app)
        self.slug = slug
        self.state = state
        self._flash = 0.0

        if world is None:
            from source.world.build import build_world
            job = build_world(state.seed)
            try:
                while True:
                    next(job)
            except StopIteration as done:
                world = done.value

        self.terrain = world["terrain"]
        self.renderer = world["renderer"]
        self.cam_x, self.cam_y = world["cam"]
        self._drag = None

        cx, w, h = config.SCREEN_W // 2, 240, 56
        self.buttons = [
            Button((cx - w // 2, config.SCREEN_H - 150, w, h), "Save", self._save),
            Button((cx - w // 2, config.SCREEN_H - 86, w, h), "Menu", self._menu, kind="ghost"),
        ]

    def _save(self):
        saves.save_game(self.slug, self.state)
        self._flash = 1.2

    def _menu(self):
        self.renderer.shutdown()
        from source.scenes.menu import MainMenuScene
        self.app.set_root(MainMenuScene(self.app))

    def on_exit(self):
        saves.save_game(self.slug, self.state)

    def _on_button(self, pos):
        return any(b.rect.collidepoint(pos) for b in self.buttons)

    def handle_event(self, event):
        import pygame
        for b in self.buttons:
            b.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN and not self._on_button(event.pos):
            self._drag = event.pos
        elif event.type == pygame.MOUSEMOTION and self._drag is not None:
            dx = event.pos[0] - self._drag[0]
            dy = event.pos[1] - self._drag[1]
            self.cam_x -= dx                  # grab-and-drag the map
            self.cam_y -= dy
            self._drag = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            self._drag = None

    def update(self, dt):
        self.state.tick += 1
        if self._flash > 0:
            self._flash = max(0.0, self._flash - dt)

    def draw(self, surface):
        self.renderer.set_camera(self.cam_x, self.cam_y)
        self.renderer.render()                # cheap: nothing within slack, else a strip
        self.renderer.draw(surface)
        label = self.app.font_small.render(f"seed {self.state.seed}", True, config.TEXT)
        surface.blit(label, (12, 12))
        if self._flash > 0:
            img = self.app.font_small.render("saved", True, config.ACCENT)
            surface.blit(img, img.get_rect(center=(config.SCREEN_W // 2, config.SCREEN_H - 180)))
        for b in self.buttons:
            b.draw(surface, self.app.font)
