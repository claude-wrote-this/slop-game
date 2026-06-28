"""Base scene. The App runs a stack of these; override what you need."""


class Scene:
    def __init__(self, app):
        self.app = app

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def handle_event(self, event):
        pass

    def update(self, dt):
        pass

    def draw(self, surface):
        pass
