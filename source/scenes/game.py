"""In-game scene: a movable, zoomable view over the world-space point renderer.
Drag to pan; pinch to zoom (both in world units, so they track your fingers at
any zoom). Zoom is a draw-time scale only — the renderer never regenerates on
zoom. Generation lives in build_world; this scene receives a prebuilt world or
runs it inline.
"""
import math

import pygame

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

        self.terrain = world.get("terrain")     # unused by the static renderer
        self.renderer = world["renderer"]
        self.cam_x, self.cam_y = world["cam"]   # world-space top-left at zoom 1
        # Clamp zoom-out so the screen diagonal stays within ~2/3 of the kernel
        # disc's diameter — the view never pans/zooms past generated terrain.
        diag = math.hypot(config.SCREEN_W, config.SCREEN_H)
        self.zoom_min = 3.0 * diag / (4.0 * self.renderer.kernel_r)
        self.zoom_max = 8.0
        self.zoom = max(self.zoom_min, min(self.zoom_max, 1.0))   # start in range
        self._drag = None                       # mouse/single-finger pan anchor
        self._fingers = {}                      # finger_id -> (px, py) in pixels
        self._pinch_dist = None                 # last two-finger separation
        self._pinch_mid = None                  # last two-finger midpoint (pixels)

        cx, w, h = config.SCREEN_W // 2, 240, 56
        self.buttons = [
            Button((cx - w // 2, config.SCREEN_H - 150, w, h), "Save", self._save),
            Button((cx - w // 2, config.SCREEN_H - 86, w, h), "Menu", self._menu, kind="ghost"),
        ]

    def on_exit(self):
        self.renderer.shutdown()           # stop the generation worker first
        saves.save_game(self.slug, self.state)

    def _save(self):
        saves.save_game(self.slug, self.state)
        self._flash = 1.2

    def _menu(self):
        self.renderer.shutdown()
        from source.scenes.menu import MainMenuScene
        self.app.set_root(MainMenuScene(self.app))

    def _on_button(self, pos):
        return any(b.rect.collidepoint(pos) for b in self.buttons)

    def _pan_world(self, dx_px, dy_px):
        self.cam_x -= dx_px / self.zoom        # screen pixels -> world units
        self.cam_y -= dy_px / self.zoom

    def _world_under(self, sx, sy):
        """The world-space point currently drawn under screen pixel (sx, sy)."""
        return (self.cam_x + config.SCREEN_W / 2 + (sx - config.SCREEN_W / 2) / self.zoom,
                self.cam_y + config.SCREEN_H / 2 + (sy - config.SCREEN_H / 2) / self.zoom)

    def _pin(self, wx, wy, sx, sy):
        """Move the camera so world (wx, wy) lands under screen (sx, sy) at the
        current zoom — the inverse of _world_under."""
        self.cam_x = wx - config.SCREEN_W / 2 - (sx - config.SCREEN_W / 2) / self.zoom
        self.cam_y = wy - config.SCREEN_H / 2 - (sy - config.SCREEN_H / 2) / self.zoom

    def _clampz(self, z):
        return max(self.zoom_min, min(self.zoom_max, z))

    def _zoom_at(self, factor, fx, fy):
        """Scale zoom by `factor` keeping the world point under (fx, fy) fixed."""
        new = self._clampz(self.zoom * factor)
        if new == self.zoom:
            return
        wx, wy = self._world_under(fx, fy)
        self.zoom = new
        self._pin(wx, wy, fx, fy)

    def _pinch_state(self):
        """(separation, midpoint) of the first two active fingers, in pixels."""
        (x0, y0), (x1, y1) = list(self._fingers.values())[:2]
        return math.hypot(x1 - x0, y1 - y0), ((x0 + x1) / 2, (y0 + y1) / 2)

    def _apply_pinch(self, dist, mid):
        """One pinned similarity step: scale by the change in finger separation
        and pin the world point under the previous midpoint to the new midpoint.
        Folding scale+translation into a single pin keeps it drift-free even
        though fingers report one at a time."""
        if self._pinch_dist and self._pinch_mid is not None and dist > 0:
            wx, wy = self._world_under(*self._pinch_mid)     # world under old mid
            self.zoom = self._clampz(self.zoom * dist / self._pinch_dist)
            self._pin(wx, wy, mid[0], mid[1])                # ...to the new mid
        self._pinch_dist, self._pinch_mid = dist, mid

    def handle_event(self, event):
        for b in self.buttons:
            b.handle_event(event)
        et = event.type

        # --- touch: 1 finger pans, 2 fingers pinch-zoom (+midpoint pan) ---
        if et == pygame.FINGERDOWN:
            self._fingers[event.finger_id] = (event.x * config.SCREEN_W,
                                              event.y * config.SCREEN_H)
            if len(self._fingers) >= 2:
                self._drag = None              # cancel single-touch pan
                self._pinch_dist, self._pinch_mid = self._pinch_state()
        elif et == pygame.FINGERMOTION:
            if event.finger_id in self._fingers:
                self._fingers[event.finger_id] = (event.x * config.SCREEN_W,
                                                  event.y * config.SCREEN_H)
            if len(self._fingers) >= 2:
                self._apply_pinch(*self._pinch_state())
        elif et == pygame.FINGERUP:
            self._fingers.pop(event.finger_id, None)
            if len(self._fingers) < 2:
                self._pinch_dist = self._pinch_mid = None

        # --- mouse: pan + wheel-zoom (desktop; suppressed during a 2-finger pinch) ---
        elif et == pygame.MOUSEBUTTONDOWN and not self._on_button(event.pos):
            if len(self._fingers) < 2:
                self._drag = event.pos
        elif et == pygame.MOUSEMOTION and self._drag is not None and len(self._fingers) < 2:
            self._pan_world(event.pos[0] - self._drag[0], event.pos[1] - self._drag[1])
            self._drag = event.pos
        elif et == pygame.MOUSEBUTTONUP:
            self._drag = None
        elif et == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            self._zoom_at(1.1 ** event.y, mx, my)

    def update(self, dt):
        self.state.tick += 1
        if self._flash > 0:
            self._flash = max(0.0, self._flash - dt)

    def draw(self, surface):
        # Main thread: feed the view to the buffer, then draw the cloud + the
        # buffer's draw list + UI as one frame (App flips after) — no flashing.
        self.renderer.set_camera(self.cam_x, self.cam_y)
        self.renderer.set_zoom(self.zoom)
        self.renderer.draw(surface)
        label = self.app.font_small.render(f"seed {self.state.seed}", True, config.TEXT)
        surface.blit(label, (12, 12))
        if self._flash > 0:
            img = self.app.font_small.render("saved", True, config.ACCENT)
            surface.blit(img, img.get_rect(center=(config.SCREEN_W // 2, config.SCREEN_H - 180)))
        for b in self.buttons:
            b.draw(surface, self.app.font)
