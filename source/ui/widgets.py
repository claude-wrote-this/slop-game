"""Reusable UI. Button fires on release-inside. Dropdown shows the current
option and an open list; it can be disabled (greyed, non-interactive) and show
a fixed disabled_text instead. Checkbox toggles on release; its whole row is the
tap target (kinder on touch than a tiny box).

A scene with an open Dropdown should draw it LAST so its list overlays siblings.
"""
import pygame
from source import config

_FILL = {"normal": config.PANEL, "danger": (60, 40, 42), "ghost": config.BG}
_DISABLED_EDGE = (70, 74, 82)
_DISABLED_FILL = (24, 27, 33)
	
class Button:
    def __init__(self, rect, label, on_click, *, kind="normal", icon=None, alpha=255):
        self.rect = pygame.Rect(rect)
        self.label = label
        if icon is not None:
        	icon = pygame.image.load(icon)
        self.icon = icon
        self.on_click = on_click
        self.kind = kind
        self.alpha=alpha
        self._down = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self._down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            fired = self._down and self.rect.collidepoint(event.pos)
            self._down = False
            if fired and self.on_click:
                self.on_click()
                return True
        return False

    def draw(self, surf, font):
        base = _FILL[self.kind]
        fill = tuple(min(255, c + 18) for c in base) if self._down else base
        edge = config.DANGER if self.kind == "danger" else config.ACCENT
        panel = pygame.Surface(self.rect.size, pygame.SRCALPHA)   # per-pixel alpha
        local = panel.get_rect()
        pygame.draw.rect(panel, (*fill, self.alpha), local, border_radius=10)
        pygame.draw.rect(panel, (*edge, min(255, self.alpha +55)), local, width=2, border_radius=10)
        surf.blit(panel, self.rect.topleft)
        if self.icon is not None:
        	surf.blit(self.icon, self.icon.get_rect(center=(self.rect.left+50, self.rect.centery)))
        img = font.render(self.label, True, config.TEXT)
        surf.blit(img, img.get_rect(center=self.rect.center))
        
class ContinueButton(Button):
	def draw(self, surf, font):
		base = _FILL[self.kind]
		fill = tuple(min(255, c + 18) for c in base) if self._down else base
		edge = config.DANGER if self.kind == "danger" else config.ACCENT
		panel = pygame.Surface(self.rect.size, pygame.SRCALPHA)   # per-pixel alpha
		local = panel.get_rect()
		pygame.draw.rect(panel, (*fill, self.alpha), local, border_radius=10)
		pygame.draw.rect(panel, (*edge, min(255, self.alpha +55)), local, width=2, border_radius=10)
		surf.blit(panel, self.rect.topleft)
		if self.icon is not None:
		      surf.blit(self.icon, self.icon.get_rect(center=(self.rect.left+50, self.rect.centery)))
		img = font.render(self.label, True, config.TEXT)
		surf.blit(img, (self.rect.left+10, self.rect.top+10))

class Dropdown:
    def __init__(self, rect, options, index, on_select, *, label_fn=str,
                 enabled=True, disabled_text=None):
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.index = index
        self.on_select = on_select
        self.label_fn = label_fn
        self.enabled = enabled
        self.disabled_text = disabled_text
        self.open = False

    def _option_rects(self):
        h = self.rect.height
        return [pygame.Rect(self.rect.x, self.rect.bottom + i * h, self.rect.width, h)
                for i in range(len(self.options))]

    def handle_event(self, event):
        if not self.enabled or event.type != pygame.MOUSEBUTTONDOWN:
            return False
        if self.open:
            for i, r in enumerate(self._option_rects()):
                if r.collidepoint(event.pos):
                    self.open = False
                    if i != self.index:
                        self.index = i
                        if self.on_select:
                            self.on_select(self.options[i])
                    return True
            self.open = False
            return self.rect.collidepoint(event.pos)
        if self.rect.collidepoint(event.pos):
            self.open = True
            return True
        return False

    def draw(self, surf, font):
        edge = config.ACCENT if self.enabled else _DISABLED_EDGE
        fill = config.PANEL if self.enabled else _DISABLED_FILL
        txt = config.TEXT if self.enabled else config.MUTED
        pygame.draw.rect(surf, fill, self.rect, border_radius=8)
        pygame.draw.rect(surf, edge, self.rect, width=2, border_radius=8)

        if not self.enabled and self.disabled_text is not None:
            label = self.disabled_text
        else:
            label = self.label_fn(self.options[self.index])
        img = font.render(label, True, txt)
        surf.blit(img, img.get_rect(midleft=(self.rect.x + 12, self.rect.centery)))

        cy, cx = self.rect.centery, self.rect.right - 20
        pygame.draw.polygon(surf, edge, [(cx - 6, cy - 3), (cx + 6, cy - 3), (cx, cy + 4)])

        if self.open and self.enabled:
            for i, r in enumerate(self._option_rects()):
                hot = (i == self.index)
                pygame.draw.rect(surf, (44, 50, 60) if hot else config.PANEL, r)
                pygame.draw.rect(surf, config.MUTED, r, width=1)
                oi = font.render(self.label_fn(self.options[i]), True, config.TEXT)
                surf.blit(oi, oi.get_rect(midleft=(r.x + 12, r.centery)))


class Checkbox:
    BOX = 28

    def __init__(self, rect, checked, on_toggle, *, label=None, enabled=True):
        self.rect = pygame.Rect(rect)          # full clickable row
        self.checked = checked
        self.on_toggle = on_toggle
        self.label = label
        self.enabled = enabled
        self._down = False

    def _box(self):
        return pygame.Rect(self.rect.x, self.rect.centery - self.BOX // 2,
                           self.BOX, self.BOX)

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self._down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            fired = self._down and self.rect.collidepoint(event.pos)
            self._down = False
            if fired:
                self.checked = not self.checked
                if self.on_toggle:
                    self.on_toggle(self.checked)
                return True
        return False

    def draw(self, surf, font):
        box = self._box()
        edge = config.ACCENT if self.enabled else _DISABLED_EDGE
        pygame.draw.rect(surf, config.PANEL, box, border_radius=5)
        pygame.draw.rect(surf, edge, box, width=2, border_radius=5)
        if self.checked:
            pygame.draw.rect(surf, edge, box.inflate(-10, -10), border_radius=3)
        if self.label:
            col = config.TEXT if self.enabled else config.MUTED
            img = font.render(self.label, True, col)
            surf.blit(img, (box.right + 10, self.rect.centery - img.get_height() // 2))