"""The 3D noise FIELD — shared source of truth for rendering and collision.

Pure numpy, no pygame. One field answers three questions over arbitrary point
ARRAYS (not a grid): value, solid, grad. The render 'rain' pass and the
collision span test both call this same object, so they can never disagree about
where the world is.

3D (not a 2D heightmap) so the world isn't monotonic in z: a column can be solid
high and air low, which is what gives overhangs, caves and floating layers for
free instead of as special cases.
"""
import numpy as np


class _Perlin3:
    """Ken Perlin's improved noise, 3D, vectorised over numpy arrays."""

    def __init__(self, seed):
        # A seeded permutation table is the whole source of determinism: same
        # seed -> same shuffle -> same noise everywhere, forever.
        rng = np.random.default_rng(seed)
        p = np.arange(256, dtype=np.int64)
        rng.shuffle(p)
        # Doubled so that p[i + something] never runs off the end (max index
        # used below is 255+255+1 = 511), avoiding a modulo on every lookup.
        self.perm = np.concatenate([p, p])

    @staticmethod
    def _fade(t):
        # Perlin's quintic smootherstep: zero 1st & 2nd derivatives at 0 and 1,
        # so cells join without visible creases.
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _grad(h, x, y, z):
        # Dot of the distance vector with one of 12 edge-gradient directions
        # chosen by the low bits of the hash. Branch-free via np.where so it runs
        # on whole arrays at once.
        h = h & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def __call__(self, x, y, z):
        # Broadcast so callers can pass any mix of scalars/arrays (e.g. arrays of
        # x,y with a scalar z during a rain pass).
        x, y, z = np.broadcast_arrays(np.asarray(x, float),
                                      np.asarray(y, float),
                                      np.asarray(z, float))
        x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
        # Integer cube corner (wrapped into the table) + fractional position.
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        zi = z0.astype(np.int64) & 255
        xf, yf, zf = x - x0, y - y0, z - z0
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        p = self.perm

        # Hash the 8 corners of the cube the point sits in.
        A = p[xi] + yi
        AA, AB = p[A] + zi, p[A + 1] + zi
        B = p[xi + 1] + yi
        BA, BB = p[B] + zi, p[B + 1] + zi
        g = self._grad

        def lerp(a, b, t):
            return a + t * (b - a)

        # Trilinear blend of the 8 corner gradients, weighted by the faded
        # fractionals. x* = blend along x; y* = along y; final = along z.
        x1 = lerp(g(p[AA],     xf, yf,     zf),     g(p[BA],     xf - 1, yf,     zf),     u)
        x2 = lerp(g(p[AB],     xf, yf - 1, zf),     g(p[BB],     xf - 1, yf - 1, zf),     u)
        x3 = lerp(g(p[AA + 1], xf, yf,     zf - 1), g(p[BA + 1], xf - 1, yf,     zf - 1), u)
        x4 = lerp(g(p[AB + 1], xf, yf - 1, zf - 1), g(p[BB + 1], xf - 1, yf - 1, zf - 1), u)
        y1 = lerp(x1, x2, v)
        y2 = lerp(x3, x4, v)
        return lerp(y1, y2, w)                   # roughly [-1, 1]


class Field:
    """Configured 3D density field. density > 0 is solid, < 0 is air.

    A vertical bias makes low z ground and high z air; the fBm noise carves that
    plane into hills, banks, caves and overhangs. Surface height therefore comes
    out of the noise rather than being stored.
    """

    def __init__(self, seed, *, xy_scale=0.06, z_scale=0.12, octaves=4,
                 persistence=0.5, lacunarity=2.0, surface_z=3.0, z_bias=0.6):
        self._noise = _Perlin3(seed)
        self.xy_scale = xy_scale          # horizontal feature size (smaller = broader)
        self.z_scale = z_scale            # vertical feature size (bigger = more overhangs)
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.surface_z = surface_z        # mean ground layer
        self.z_bias = z_bias              # density cost per layer of height = "gravity"

    def _fbm(self, x, y, z):
        # Fractal sum: each octave adds finer, weaker detail. Normalised by the
        # total amplitude so the result stays in roughly [-1, 1] regardless of
        # octave count.
        total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
        for _ in range(self.octaves):
            total = total + self._noise(np.asarray(x, float) * (self.xy_scale * freq),
                                        np.asarray(y, float) * (self.xy_scale * freq),
                                        np.asarray(z, float) * (self.z_scale * freq)) * amp
            norm += amp
            amp *= self.persistence       # each octave quieter
            freq *= self.lacunarity       # each octave finer
        return total / norm

    def density(self, x, y, z):
        # The bias term is what turns blobby 3D noise into terrain: it pulls
        # density up low and down high, so a surface forms where noise ~= bias.
        # Where the noise locally beats the bias, it folds back -> an overhang.
        return self._fbm(x, y, z) + (self.surface_z - np.asarray(z, float)) * self.z_bias

    # the three queries everything shares -----------------------------------
    def value(self, x, y, z):
        return self.density(x, y, z)

    def solid(self, x, y, z, threshold=0.0):
        return self.density(x, y, z) > threshold

    def grad(self, x, y, z, eps=1e-2):
        # Central differences: robust, and only the cold paths (contour-snapping,
        # lighting normals) call it — rain and collision never do. Swap in
        # Perlin's analytic derivative here if it ever shows up in a profile.
        d = self.density
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        gx = (d(x + eps, y, z) - d(x - eps, y, z)) / (2 * eps)
        gy = (d(x, y + eps, z) - d(x, y - eps, z)) / (2 * eps)
        gz = (d(x, y, z + eps) - d(x, y, z - eps)) / (2 * eps)
        return gx, gy, gz
