"""The application core: owns the window, clock and fonts, and runs a scene
STACK. Two additions over the bare version: it loads saved settings at startup
(so the chosen resolution is applied before the window is created), and exposes
apply_resolution() for the Options > Display dropdown to call.
"""
import pygame
from source import config
from source.persistence.settings import Settings
from source.utils import log

class App:
    def __init__(self):
        log.info(f"Initialising {config.TITLE}")
        self.settings = Settings.load()
        self._apply_active()                  # sets config size + creates screen
        pygame.display.set_caption(config.TITLE)
        self.clock = pygame.time.Clock()
        
        log.debug(f"Setting fonts - watch out for failures")
        self.font_big   = pygame.font.Font(config.TITLE_FONT, 230)
        self.font       = pygame.font.Font(config.TEXT_FONT, 30)
        self.font_small = pygame.font.SysFont(None, 22)
        self._stack = []
        self.running = True

    # --- scene stack ---
    @property
    def scene(self):
        return self._stack[-1] if self._stack else None

    def push(self, scene):
        self._stack.append(scene)
        scene.on_enter()

    def pop(self):
        if self._stack:
            self._stack.pop().on_exit()

    def set_root(self, scene):
        while self._stack:
            self._stack.pop().on_exit()
        self.push(scene)

    def quit(self):
        self.running = False

    # --- resolution ---
    def device_resolution(self):
        """The phone/desktop's actual screen size, independent of the window."""
        try:
            sizes = pygame.display.get_desktop_sizes()
            if sizes and sizes[0][0] > 0 and sizes[0][1] > 0:
                return (int(sizes[0][0]), int(sizes[0][1]))
        except Exception:
            pass
        return (config.SCREEN_W, config.SCREEN_H)

    def _active_size(self):
        if self.settings.get("resolution_auto"):
            return self.device_resolution()
        w, h = self.settings.get("resolution", [config.SCREEN_W, config.SCREEN_H])
        return (int(w), int(h))

    def _apply_active(self):
        w, h = self._active_size()
        log.debug(f"Applying Resolution: {w} {h}")
        #config.SCREEN_W, config.SCREEN_H = w, h
        self.screen = pygame.display.set_mode((w, h))
        config.SCREEN_W, config.SCREEN_H = self.screen.get_size()

    def _rebuild_options(self):
        # resolution is only changed from the menu, so rebuilding from the menu
        # root is safe and re-lays-out every scene at the new size.
        from source.scenes.menu import MainMenuScene
        from source.scenes.options import OptionsScene
        self.set_root(MainMenuScene(self))
        self.push(OptionsScene(self, active="Display"))

    def apply_resolution(self, w, h):
        """Manual pick from the dropdown -> turns auto off."""
        self.settings.set("resolution", [w, h])
        self.settings.set("resolution_auto", False)
        self.settings.save()
        self._apply_active()
        self._rebuild_options()

    def set_auto_resolution(self, enabled):
        """Auto on -> device size; off -> restore saved manual choice."""
        self.settings.set("resolution_auto", bool(enabled))
        self.settings.save()
        self._apply_active()
        self._rebuild_options()

    # --- main loop ---
    def run(self, first_scene, max_frames=None):
        self.push(first_scene)
        frames = 0
        while self.running and self.scene:
            dt = self.clock.tick(config.FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif self.scene:
                    self.scene.handle_event(event)
            if not self.running or not self.scene:
                break
            self.scene.update(dt)
            self.scene.draw(self.screen)
            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                break
        pygame.quit()	