"""Slider widget. on_change fires continuously while dragging (cheap: move the
handle, update a label); on_commit fires once on release (for expensive work
like re-rendering terrain, so a drag doesn't trigger dozens of re-renders).
Kept in its own file so it doesn't disturb a customised widgets.py.
"""
import pygame
from source import config


class Slider:
    def __init__(self, rect, min_v, max_v, value, *, on_change=None, on_commit=None, label=None):
        self.rect = pygame.Rect(rect)        # the track; the handle rides along it
        self.min_v, self.max_v = min_v, max_v
        self.value = value
        self.on_change = on_change           # live, every drag step (keep cheap)
        self.on_commit = on_commit           # once, on release (do heavy work here)
        self.label = label
        self._drag = False

    def _value_at(self, x):
        # Map an x pixel onto the value range, clamped, rounded to an integer step.
        t = (x - self.rect.x) / max(1, self.rect.w)
        t = min(1.0, max(0.0, t))
        return int(round(self.min_v + t * (self.max_v - self.min_v)))

    def _handle_x(self):
        # Inverse: value -> handle x for drawing.
        t = (self.value - self.min_v) / max(1, (self.max_v - self.min_v))
        return int(self.rect.x + t * self.rect.w)

    def _set(self, x):
        v = self._value_at(x)
        if v != self.value:                  # only fire on an actual change
            self.value = v
            if self.on_change:
                self.on_change(v)

    def handle_event(self, event):
        # inflate(0, 44): fatten the touch target vertically so the thin track is
        # easy to grab on a phone.
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.inflate(0, 44).collidepoint(event.pos):
            self._drag = True
            self._set(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self._drag:
            self._set(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and self._drag:
            self._drag = False
            if self.on_commit:               # heavy work happens once, here
                self.on_commit(self.value)

    def draw(self, surf, font):
        cy = self.rect.centery
        pygame.draw.line(surf, config.MUTED, (self.rect.x, cy), (self.rect.right, cy), 3)
        hx = self._handle_x()
        pygame.draw.circle(surf, config.ACCENT, (hx, cy), 12)
        pygame.draw.circle(surf, config.TEXT, (hx, cy), 12, 2)
        if self.label:
            img = font.render(f"{self.label}: {self.value}", True, config.TEXT)
            surf.blit(img, (self.rect.x, self.rect.y - 26))
