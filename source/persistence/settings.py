"""App-wide settings (resolution, and later audio, controls, etc.). Distinct
from save games: these belong to the install, not to any one world. Persisted
to settings.json next to the saves folder.
"""
import os
import json

from source import config

DEFAULTS = {
    "resolution": [config.SCREEN_W, config.SCREEN_H],
    "resolution_auto": False,
}


class Settings:
    def __init__(self, data=None):
        self.data = dict(DEFAULTS)
        if data:
            self.data.update(data)

    @classmethod
    def load(cls):
        try:
            with open(config.SETTINGS_PATH) as f:
                return cls(json.load(f))
        except Exception:
            return cls()          # missing or corrupt -> defaults

    def save(self):
        tmp = config.SETTINGS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, config.SETTINGS_PATH)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
