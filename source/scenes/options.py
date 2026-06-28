"""Options. A tab bar across the top; each tab builds its own widgets. Only
Display exists now (resolution dropdown) — add "Audio", "Controls" etc. to TABS
and give each a _tab_<name>() builder. When a tab grows past a few widgets,
promote it to its own module/class.
"""
import pygame
from source import config
from source.scenes.base import Scene
from source.ui.widgets import Button, Dropdown, Checkbox


class OptionsScene(Scene):
    TABS = ["Display"]

    def __init__(self, app, active="Display"):
        super().__init__(app)
        self.active = active if active in self.TABS else self.TABS[0]
        self.dropdown = None
        self._layout_tabs()
        self._build_tab()
        self.back = Button((config.SCREEN_W // 2 - 120, config.SCREEN_H - 150, 240, 56),
                           "Back", self.app.pop, kind="ghost")

    def _layout_tabs(self):
        self.tab_rects = []
        n = len(self.TABS)
        tw = min(160, (config.SCREEN_W - 40) // n)
        x0 = (config.SCREEN_W - tw * n) // 2
        for i, name in enumerate(self.TABS):
            self.tab_rects.append((name, pygame.Rect(x0 + i * tw, 220, tw, 48)))

    def _build_tab(self):
        self.widgets = []
        self.dropdown = None
        builder = getattr(self, f"_tab_{self.active.lower()}", None)
        if builder:
            builder()

    # --- tabs ---
    def _tab_display(self):
        auto = self.app.settings.get("resolution_auto", False)
        current = (config.SCREEN_W, config.SCREEN_H)   # == device size when auto is on
        manual = tuple(self.app.settings.get("resolution", list(current)))
        idx = next((i for i, r in enumerate(config.RESOLUTIONS) if tuple(r) == manual), 0)

        left, gap, cb_w, y, h = 40, 16, 96, 300, 50
        dd_w = config.SCREEN_W - 2 * left - gap - cb_w
        self.dropdown = Dropdown(
            (left, y, dd_w, h), config.RESOLUTIONS, idx, self._on_resolution,
            label_fn=lambda wh: f"{wh[0]} x {wh[1]}",
            enabled=not auto,
            disabled_text=(f"{current[0]} x {current[1]}" if auto else None))
        checkbox = Checkbox((left + dd_w + gap, y, cb_w, h), auto, self._on_auto, label="Auto")
        self.widgets = [self.dropdown, checkbox]

    def _on_resolution(self, wh):
        self.app.apply_resolution(wh[0], wh[1])   # rebuilds this scene

    def _on_auto(self, checked):
        self.app.set_auto_resolution(checked)     # rebuilds this scene

    # --- scene ---
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for name, r in self.tab_rects:
                if r.collidepoint(event.pos) and name != self.active:
                    self.active = name
                    self._build_tab()
                    return
        for w in self.widgets:
            w.handle_event(event)
        self.back.handle_event(event)

    def draw(self, surface):
        bg = pygame.image.load(config.BG_IMAGE)
        bg = pygame.transform.scale(bg, (config.SCREEN_W, config.SCREEN_H))
        surface.blit(bg, (0,0))
        #surface.fill(config.BG)
        title = self.app.font_big.render("Options", True, config.TEXT)
        surface.blit(title, title.get_rect(center=(config.SCREEN_W // 2, 100)))

        for name, r in self.tab_rects:
            on = (name == self.active)
            pygame.draw.rect(surface, config.PANEL if on else config.BG, r, border_radius=8)
            pygame.draw.rect(surface, config.ACCENT if on else config.MUTED, r,
                             width=2, border_radius=8)
            img = self.app.font.render(name, True, config.TEXT if on else config.MUTED)
            surface.blit(img, img.get_rect(center=r.center))

        if self.active == "Display":
            lbl = self.app.font_small.render("Resolution", True, config.MUTED)
            surface.blit(lbl, (40, 280))

        for w in self.widgets:        # dropdown drawn here; its open list overlays below
            w.draw(surface, self.app.font)
        self.back.draw(surface, self.app.font)
        footer = pygame.image.load(config.FOOTER_IMAGE)
        footer = pygame.transform.scale(footer, (820, 110))
        surface.blit(footer, footer.get_rect(center=((config.SCREEN_W // 2)+8, config.SCREEN_H-60)))