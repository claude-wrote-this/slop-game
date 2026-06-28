"""Terrain DATA. Pure Python, no pygame — this knows nothing about pixels.
Perlin noise (seeded from the world seed) -> a normalised heightmap -> discrete
integer layers. Deterministic: same seed always yields the same terrain, so
saves only need the seed, not the grid.
"""
import math
import random


class Perlin:
    """Classic improved Perlin noise, 2D. Permutation table seeded for repeatability."""
    def __init__(self, seed):
        rng = random.Random(seed)
        p = list(range(256))
        rng.shuffle(p)
        self.p = p + p                      # doubled to avoid index wrapping

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a, b, t):
        return a + t * (b - a)

    @staticmethod
    def _grad(h, x, y):
        h &= 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y):
        p = self.p
        xi, yi = int(math.floor(x)) & 255, int(math.floor(y)) & 255
        xf, yf = x - math.floor(x), y - math.floor(y)
        u, v = self._fade(xf), self._fade(yf)
        aa, ab = p[p[xi] + yi], p[p[xi] + yi + 1]
        ba, bb = p[p[xi + 1] + yi], p[p[xi + 1] + yi + 1]
        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)
        return self._lerp(x1, x2, v)        # roughly [-1, 1]


def fbm(perlin, x, y, octaves, persistence=0.5, lacunarity=2.0):
    """Fractal Brownian motion: stack octaves for natural-looking detail."""
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        total += perlin.noise(x * freq, y * freq) * amp
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm                     # back to ~[-1, 1]


class TerrainData:
    def __init__(self, seed, cols, rows, *, scale=0.06, octaves=4, layers=6):
        self.cols, self.rows, self.layers = cols, rows, layers
        perlin = Perlin(seed)
        self.height = [[0.0] * rows for _ in range(cols)]   # normalised 0..1
        self.layer = [[0] * rows for _ in range(cols)]      # integer 0..layers-1
        for gx in range(cols):
            hcol, lcol = self.height[gx], self.layer[gx]
            for gy in range(rows):
                h = (fbm(perlin, gx * scale, gy * scale, octaves) + 1.0) * 0.5
                hcol[gy] = h
                lcol[gy] = min(layers - 1, int(h * layers))

    def height_at(self, gx, gy):
        return self.height[gx][gy]

    def layer_at(self, gx, gy):
        return self.layer[gx][gy]
        