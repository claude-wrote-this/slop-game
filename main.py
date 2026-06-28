import pygame  # noqa: F401 — import in the executed file so Pydroid/Android wires up SDL; do not remove
from source.app import App
from source.scenes.menu import MainMenuScene
from source.utils import log

def main():
    pygame.init()
    app = App()
    app.run(MainMenuScene(app))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Program Failed with Exception: {e}")
