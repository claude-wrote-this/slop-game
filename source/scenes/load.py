"""Load / delete saves. Pushed over the main menu, so Back just pops."""
from source import config
from source.scenes.base import Scene
from source.scenes.game import GameScene
from source.ui.widgets import Button
from source.persistence import saves
import pygame

class LoadScene(Scene):
    def __init__(self, app):
        super().__init__(app)
        self._build()

    def _build(self):
        self.rows = []
        w = config.SCREEN_W - 60
        y = 130
        for m in saves.list_saves():
            load_b = Button((30, y, w - 60, 64), m["name"], self._loader(m["slug"]))
            del_b = Button((config.SCREEN_W - 80, y, 50, 64), "x",
                           self._deleter(m["slug"]), kind="danger")
            self.rows.append((load_b, del_b))
            y += 76
        self._empty = not self.rows
        self.back = Button((config.SCREEN_W // 2 - 120, config.SCREEN_H - 150, 240, 56),
                           "Back", self.app.pop, kind="ghost")

    def _loader(self, slug):
        def go():
            state = saves.load_game(slug)
            self.app.set_root(GameScene(self.app, slug, state))
        return go

    def _deleter(self, slug):
        def go():
            saves.delete_game(slug)
            self._build()
        return go

    def handle_event(self, event):
        for load_b, del_b in self.rows:
            load_b.handle_event(event)
            del_b.handle_event(event)
        self.back.handle_event(event)

    def draw(self, surface):
        bg = pygame.image.load(config.BG_IMAGE)
        bg = pygame.transform.scale(bg, (config.SCREEN_W, config.SCREEN_H))
        surface.blit(bg, (0,0))
        #surface.fill(config.BG)
        title = self.app.font.render("Load Game", True, config.TEXT)
        surface.blit(title, title.get_rect(center=(config.SCREEN_W // 2, 70)))
        if self._empty:
            msg = self.app.font_small.render("No saves yet", True, config.MUTED)
            surface.blit(msg, msg.get_rect(center=(config.SCREEN_W // 2, 200)))
        for load_b, del_b in self.rows:
            load_b.draw(surface, self.app.font_small)
            del_b.draw(surface, self.app.font)
        self.back.draw(surface, self.app.font)
        footer = pygame.image.load(config.FOOTER_IMAGE)
        footer = pygame.transform.scale(footer, (820, 110))
        surface.blit(footer, footer.get_rect(center=((config.SCREEN_W // 2)+8, config.SCREEN_H-60)))