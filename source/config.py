"""Single source of truth for constants. SCREEN_W/H are the *effective* values:
they start at the defaults below but App overrides them at startup from saved
settings, and the Options > Display dropdown rewrites them at runtime.
"""
import os

# --- display ---
TITLE = "Efficio"
SCREEN_W, SCREEN_H = 450, 820   # default; may be overridden by settings.json
FPS = 60

# portrait resolutions offered in Options > Display
RESOLUTIONS = [
    (360, 800),
    (405, 900),
    (450, 1000),
    (540, 1200),
    (720, 1600),
    (1080, 2316),
    (1440, 3088)
]

# --- palette ---
BG     = (18, 20, 26)
PANEL  = (22, 26, 30)
ACCENT = (102, 114, 104)
DANGER = (170, 90, 90)
TEXT   = (232, 234, 238)
MUTED  = (140, 146, 156)

# --- terrain ---
TERRAIN_TILE = 16        # px per cell
TERRAIN_LAYERS = 100     # ceiling: max vertical layers (~a person is a few layers)
TERRAIN_LAYER_DZ = 0.015 # fixed noise-z height of one layer (slab thickness)
TERRAIN_SCALE = 0.06     # noise frequency (smaller = larger landforms)
TERRAIN_OCTAVES = 4

# --- paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "saves")
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")
FONTS_PATH = os.path.join(PROJECT_ROOT, "content/fonts")
IMAGES_PATH = os.path.join(PROJECT_ROOT, "content/images")

# --- design ---
# --- images ---
# --- main menu ---
BG_IMAGE = os.path.join(IMAGES_PATH, "menu-bg.png")
BANNER_IMAGE = os.path.join(IMAGES_PATH, "banner.png")
FOOTER_IMAGE = os.path.join(IMAGES_PATH, "footer.png")
NEW_ICON = os.path.join(IMAGES_PATH, "icons/new.png")
LOAD_ICON = os.path.join(IMAGES_PATH, "icons/load.png")
OPTIONS_ICON = os.path.join(IMAGES_PATH, "icons/options.png")
QUIT_ICON = os.path.join(IMAGES_PATH, "icons/quit.png")
BOOKEND_IMAGE =os.path.join(IMAGES_PATH, "bookend.png")
# --- fonts ---
TITLE_FONT = os.path.join(FONTS_PATH, "Jacquard12-Regular.ttf")
TEXT_FONT = os.path.join(FONTS_PATH, "PixeloidSans-lxa3y.ttf")