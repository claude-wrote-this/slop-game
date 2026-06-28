"""Realistic layered height field. A pure function of world (x, y), so chunks
generated independently seam perfectly — chunk (ox, oy) just samples the same
global function at its world coordinates. NEVER normalise per chunk (neighbours
would disagree); the transfer from noise to height is fixed.

Realism = a few cheap tricks stacked on plain fBm:
  - domain warp: perturb sample coords by noise so features bend/braid instead of
    looking like round blobs (the biggest single win);
  - ridged noise (1 - |n|) for sharp mountain spines and carved valleys;
  - a low-frequency continent mask so mountains rise only where land is high;
  - a redistribution curve to flatten lowlands and keep peaks pointy.
Then quantise to integer layers.

Independent 2D noise channels come from one 3D Perlin sampled at well-separated
z-slices — cheaper than several noise objects, still decorrelated.
"""
import numpy as np

from source.world.field import _Perlin3

_Z_WARP_X, _Z_WARP_Y = 11.5, 23.5
_Z_BASE, _Z_MOUNT, _Z_DETAIL = 0.5, 31.5, 47.5


class TerrainHeight:
    def __init__(self, seed, *, layers=12,
                 base_freq=0.011, mount_freq=0.03, detail_freq=0.11,
                 warp_freq=0.02, warp_amp=18.0,
                 octaves=5, redistribution=1.0,
                 mount_strength=1.15, detail_strength=0.05):
        self.noise = _Perlin3(seed)
        self.layers = layers
        self.base_freq = base_freq
        self.mount_freq = mount_freq
        self.detail_freq = detail_freq
        self.warp_freq = warp_freq
        self.warp_amp = warp_amp
        self.octaves = octaves
        self.redistribution = redistribution
        self.mount_strength = mount_strength
        self.detail_strength = detail_strength

    def _fbm(self, x, y, zslice, freq, octaves=None):
        octaves = octaves or self.octaves
        total, amp, f, norm = 0.0, 1.0, freq, 0.0
        for _ in range(octaves):
            total = total + self.noise(x * f, y * f, zslice) * amp
            norm += amp
            amp *= 0.5
            f *= 2.0
        return total / norm                   # ~[-1, 1]

    def _ridged(self, x, y, zslice, freq, octaves=None):
        octaves = octaves or self.octaves
        total, amp, f, norm = 0.0, 1.0, freq, 0.0
        for _ in range(octaves):
            n = 1.0 - np.abs(self.noise(x * f, y * f, zslice))
            n = n * n                         # sharpen the crest
            total = total + n * amp
            norm += amp
            amp *= 0.5
            f *= 2.0
        return total / norm                   # [0, 1], ridges near 1

    def continuous(self, X, Y):
        """Normalised height in [0, 1] at world coords X, Y (arrays)."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)

        # 1) domain warp
        wx = self._fbm(X, Y, _Z_WARP_X, self.warp_freq, octaves=4)
        wy = self._fbm(X, Y, _Z_WARP_Y, self.warp_freq, octaves=4)
        Xw = X + wx * self.warp_amp
        Yw = Y + wy * self.warp_amp

        # 2) base continent elevation -> [0, 1]
        base = (self._fbm(Xw, Yw, _Z_BASE, self.base_freq) + 1.0) * 0.5

        # 3) ridged mountains, masked to high land
        mount = self._ridged(Xw, Yw, _Z_MOUNT, self.mount_freq)
        m = np.clip((base - 0.45) / 0.30, 0.0, 1.0)
        mask = m * m * (3.0 - 2.0 * m)        # smoothstep
        mount *= mask

        # 4) fine detail
        detail = (self._fbm(Xw, Yw, _Z_DETAIL, self.detail_freq, octaves=3) + 1.0) * 0.5

        # 5) fixed combine + redistribution (NOT per-chunk)
        h = base * 0.7 + mount * self.mount_strength + detail * self.detail_strength
        h = h / (0.7 + self.mount_strength + self.detail_strength)
        h = np.clip(h, 0.0, 1.0) ** self.redistribution
        return h

    def layered(self, X, Y):
        h = self.continuous(X, Y)
        layer = np.minimum(self.layers - 1, (h * self.layers).astype(int))
        return layer, h

    def chunk(self, ox, oy, cols, rows):
        """Layered + continuous height for a chunk at world offset (ox, oy)."""
        X = np.broadcast_to((ox + np.arange(cols))[None, :], (rows, cols))
        Y = np.broadcast_to((oy + np.arange(rows))[:, None], (rows, cols))
        return self.layered(X, Y)
        