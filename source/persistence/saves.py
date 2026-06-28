"""All save-file I/O. Atomic writes (temp + rename) so a crash never corrupts a
save. list_saves tolerates bad files by skipping them.
"""
import os
import re
import json
import time

from source import config
from source.persistence.state import GameState


def _ensure_dir():
    os.makedirs(config.SAVE_DIR, exist_ok=True)


def _slug(name):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()
    return s or "save"


def _path(slug):
    return os.path.join(config.SAVE_DIR, slug + ".json")


def unique_slug(name):
    _ensure_dir()
    base = _slug(name)
    slug, i = base, 2
    while os.path.exists(_path(slug)):
        slug, i = f"{base}-{i}", i + 1
    return slug


def list_saves():
    _ensure_dir()
    out = []
    for fn in os.listdir(config.SAVE_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(config.SAVE_DIR, fn)) as f:
                d = json.load(f)
            out.append({"slug": fn[:-5], "name": d.get("name", fn[:-5]),
                        "last_played": d.get("last_played", 0), "tick": d.get("tick", 0)})
        except Exception:
            continue
    out.sort(key=lambda m: m["last_played"], reverse=True)
    return out


def create_game(name):
    slug = unique_slug(name)
    state = GameState(name=name)
    save_game(slug, state)
    return slug, state


def save_game(slug, state):
    _ensure_dir()
    state.last_played = time.time()
    tmp = _path(slug) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    os.replace(tmp, _path(slug))


def load_game(slug):
    with open(_path(slug)) as f:
        return GameState.from_dict(json.load(f))


def delete_game(slug):
    try:
        os.remove(_path(slug))
    except FileNotFoundError:
        pass
