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

        if world is None:                        # fallback build (no loading scene)
            import time
            from source.world.build import build_world
            job = build_world(state.seed)
            try:
                while True:
                    next(job)
                    time.sleep(0.002)            # let the worker thread fill the disc
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
        self.zoom_target = self.zoom            # gestures set this; zoom eases toward it
        self._zoom_focus = None                 # screen point kept fixed while zooming
        self._zoom_rate = 8.0                   # max zoom change factor per second
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
        """Aim zoom at `factor`x around (fx, fy); the ease in update() gets there."""
        self.zoom_target = self._clampz(self.zoom_target * factor)
        self._zoom_focus = (fx, fy)

    def _ease_zoom(self, dt):
        """Move zoom toward its target at a bounded rate, keeping the focus point
        fixed on screen. Rate-limiting spreads a zoom over frames so the renderer
        can update continuously instead of jumping and popping in on release."""
        if self._zoom_focus is None or self.zoom == self.zoom_target:
            return
        max_f = self._zoom_rate ** max(dt, 1e-3)
        ratio = min(max_f, max(1.0 / max_f, self.zoom_target / self.zoom))
        new = self._clampz(self.zoom * ratio)
        fx, fy = self._zoom_focus
        wx, wy = self._world_under(fx, fy)       # world under the focus, current zoom
        self.zoom = new
        self._pin(wx, wy, fx, fy)                # ...keep it under the focus at new zoom
        if abs(self.zoom - self.zoom_target) <= self.zoom_target * 1e-3:
            self.zoom = self.zoom_target

    def _pinch_state(self):
        """(separation, midpoint) of the first two active fingers, in pixels."""
        (x0, y0), (x1, y1) = list(self._fingers.values())[:2]
        return math.hypot(x1 - x0, y1 - y0), ((x0 + x1) / 2, (y0 + y1) / 2)

    def _apply_pinch(self, dist, mid):
        """Split the two-finger gesture: pan by the midpoint's movement now (so the
        view tracks the fingers directly), and aim the zoom target at the change in
        finger separation around the midpoint. update()'s ease drives the zoom, so a
        fast pinch resolves over a few frames rather than snapping."""
        if self._pinch_dist and self._pinch_mid is not None and dist > 0:
            pmx, pmy = self._pinch_mid
            self._pan_world(mid[0] - pmx, mid[1] - pmy)      # follow the midpoint
            self.zoom_target = self._clampz(self.zoom_target * dist / self._pinch_dist)
            self._zoom_focus = mid
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
        self._ease_zoom(dt)

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
