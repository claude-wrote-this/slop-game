"""The serializable heart of a game. Everything that must survive save/load
lives here and ONLY here. `version` lets from_dict() migrate old saves forward.
"""
import time
import random

SAVE_VERSION = 1


class GameState:
    def __init__(self, name, seed=None, created_at=None, last_played=None, tick=0):
        self.name = name
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.created_at = created_at or time.time()
        self.last_played = last_played or self.created_at
        self.tick = tick

    def to_dict(self):
        return {
            "version": SAVE_VERSION,
            "name": self.name,
            "seed": self.seed,
            "created_at": self.created_at,
            "last_played": self.last_played,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"], seed=d.get("seed"),
            created_at=d.get("created_at"), last_played=d.get("last_played"),
            tick=d.get("tick", 0),
        )
