"""Main menu — the root scene the app starts on."""
from source import config
from source.scenes.base import Scene
from source.scenes.game import GameScene
from source.scenes.load import LoadScene
from source.scenes.options import OptionsScene
from source.ui.widgets import Button
from source.persistence import saves
import pygame

class MainMenuScene(Scene):
    def __init__(self, app):
        super().__init__(app)
        cx, w, h, gap = config.SCREEN_W // 2, 450, 80, 16
        top = 1000
        labels = [
            ("New Game", config.NEW_ICON, self._new, "normal"),
            ("Load Game", config.LOAD_ICON, self._load, "normal"),
            ("Options", config.OPTIONS_ICON, self._options, "normal"),
            ("Quit", config.QUIT_ICON, self.app.quit, "ghost"),
        ]
        self.buttons = [
            Button((cx - w // 2, top + i * (h + gap), w, h), text, cb, kind=kind, icon=icon, alpha=128)
            for i, (text, icon, cb, kind) in enumerate(labels)
        ]

    def _new(self):
	    slug, state = saves.create_game("New World")
	    from source.scenes.loading import LoadingScene
	    from source.world.build import build_world
	    self.app.set_root(LoadingScene(
      	  self.app, build_world(state.seed),
      	  on_complete=lambda w: self.app.set_root(
         	   GameScene(self.app, slug, state, world=w))))
            
    def _load(self):
        self.app.push(LoadScene(self.app))

    def _options(self):
        self.app.push(OptionsScene(self.app))

    def handle_event(self, event):
        for b in self.buttons:
            b.handle_event(event)

    def draw(self, surface):
        bg = pygame.image.load(config.BG_IMAGE)
        bg = pygame.transform.scale(bg, (config.SCREEN_W, config.SCREEN_H))
        surface.blit(bg, (0,0))
        banner = pygame.image.load(config.BANNER_IMAGE)
        banner = pygame.transform.scale(banner,(550, 75))
        surface.blit(banner, banner.get_rect(center=((config.SCREEN_W // 2)+3, 420)))
        title = self.app.font_big.render(config.TITLE, True, config.TEXT)
        surface.blit(title, title.get_rect(center=(config.SCREEN_W // 2, 550)))
        #subtitle = "a quiet, growing world"
        subtitle = "invenire, aedificare, florere"
        sub = self.app.font.render(subtitle, True, config.ACCENT)
        
        sub_rect = sub.get_rect(center=(config.SCREEN_W // 2, 650))
        surface.blit(sub, sub_rect)
        bookend = pygame.image.load(config.BOOKEND_IMAGE)
        surface.blit(bookend, bookend.get_rect(center=(sub_rect.left - 25, sub_rect.centery+3)))
        surface.blit(bookend, bookend.get_rect(center=(sub_rect.right + 25, sub_rect.centery+3)))
        for b in self.buttons:
            b.draw(surface, self.app.font)
        footer = pygame.image.load(config.FOOTER_IMAGE)
        footer = pygame.transform.scale(footer, (820, 110))
        surface.blit(footer, footer.get_rect(center=((config.SCREEN_W // 2)+8, config.SCREEN_H-60)))