"""Realistic layered terrain — a pure DATA object. No pygame, no renderer, no
chunks. It answers one question over a block of world cells: sample(ox, oy, cols,
rows) -> (height, colour).

Height is rounded to an integer per cell; rounding IS the layering, one global
system, not a separate concept. Colour is a per-layer ramp for visibility (a
placeholder to be replaced). The renderer reads this; the terrain never draws.

Realism in the continuous field: domain-warped fBm + ridged mountains masked to
high ground + a redistribution curve. Pure function of world (x, y), so any block
sampled anywhere tiles seamlessly with its neighbours.
"""
import numpy as np

from source.world.field import _Perlin3

_Z_WARP_X, _Z_WARP_Y = 11.5, 23.5
_Z_BASE, _Z_MOUNT, _Z_DETAIL = 0.5, 31.5, 47.5

# per-layer colour ramp anchors (dark low ground -> pale high ground)
_RAMP_ANCHORS = [(44, 54, 44), (74, 94, 60), (118, 132, 84),
                 (150, 150, 108), (188, 182, 150), (222, 216, 196)]


class TerrainHeight:
    covers_screen = True            # the renderer treats terrain as always on-screen

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
        self.colour_ramp = self._build_ramp(layers)

    @staticmethod
    def _build_ramp(layers):
        xs = np.linspace(0.0, 1.0, len(_RAMP_ANCHORS))
        ts = np.linspace(0.0, 1.0, layers)
        chans = [np.interp(ts, xs, [a[c] for a in _RAMP_ANCHORS]) for c in range(3)]
        return np.stack(chans, axis=1).astype(np.uint8)      # (layers, 3)

    def _fbm(self, x, y, zslice, freq, octaves=None):
        octaves = octaves or self.octaves
        total, amp, f, norm = 0.0, 1.0, freq, 0.0
        for _ in range(octaves):
            total = total + self.noise(x * f, y * f, zslice) * amp
            norm += amp
            amp *= 0.5
            f *= 2.0
        return total / norm

    def _ridged(self, x, y, zslice, freq, octaves=None):
        octaves = octaves or self.octaves
        total, amp, f, norm = 0.0, 1.0, freq, 0.0
        for _ in range(octaves):
            n = 1.0 - np.abs(self.noise(x * f, y * f, zslice))
            n = n * n
            total = total + n * amp
            norm += amp
            amp *= 0.5
            f *= 2.0
        return total / norm

    def continuous(self, X, Y):
        """Normalised height in [0, 1] at world coords X, Y (arrays)."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        wx = self._fbm(X, Y, _Z_WARP_X, self.warp_freq, octaves=4)
        wy = self._fbm(X, Y, _Z_WARP_Y, self.warp_freq, octaves=4)
        Xw = X + wx * self.warp_amp
        Yw = Y + wy * self.warp_amp
        base = (self._fbm(Xw, Yw, _Z_BASE, self.base_freq) + 1.0) * 0.5
        mount = self._ridged(Xw, Yw, _Z_MOUNT, self.mount_freq)
        m = np.clip((base - 0.45) / 0.30, 0.0, 1.0)
        mount *= m * m * (3.0 - 2.0 * m)
        detail = (self._fbm(Xw, Yw, _Z_DETAIL, self.detail_freq, octaves=3) + 1.0) * 0.5
        h = base * 0.7 + mount * self.mount_strength + detail * self.detail_strength
        h = h / (0.7 + self.mount_strength + self.detail_strength)
        return np.clip(h, 0.0, 1.0) ** self.redistribution

    def sample_points(self, X, Y):
        """(height int, colour) at arbitrary world positions X, Y (cell units,
        fractional ok). No grid — one value per point."""
        h01 = self.continuous(X, Y)
        height = np.clip((h01 * self.layers).astype(np.int64), 0, self.layers - 1)
        colour = self.colour_ramp[height]                    # (..., 3)
        return height, colour
