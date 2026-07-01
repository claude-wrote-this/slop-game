"""The 3D Perlin noise primitive. Pure numpy, no pygame — vectorised over
arbitrary point ARRAYS (not a grid). TerrainHeight builds the world's heightmap
on top of this by sampling it at fixed z-slices.
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

    @staticmethod
    def _dfade(t):
        # derivative of the quintic fade: 30 t^2 (t-1)^2
        return 30.0 * t * t * (t - 1.0) * (t - 1.0)

    def noised(self, x, y, z):
        """3D value noise in [-1, 1] with its analytic gradient, returned as
        (value, d/dx, d/dy, d/dz). Value noise (random value per lattice corner,
        quintic-interpolated) rather than gradient noise, because its derivative is
        exact and cheap — corner values are constants, so only the fade weights
        differentiate. Feeds the derivative-damped ("erosion") fBm."""
        x, y, z = np.broadcast_arrays(np.asarray(x, float),
                                      np.asarray(y, float),
                                      np.asarray(z, float))
        x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
        xi = x0.astype(np.int64) & 255
        yi = y0.astype(np.int64) & 255
        zi = z0.astype(np.int64) & 255
        fx, fy, fz = x - x0, y - y0, z - z0
        ux, uy, uz = self._fade(fx), self._fade(fy), self._fade(fz)
        dux, duy, duz = self._dfade(fx), self._dfade(fy), self._dfade(fz)
        p = self.perm

        A = p[xi] + yi; B = p[xi + 1] + yi
        AA, AB = p[A] + zi, p[A + 1] + zi
        BA, BB = p[B] + zi, p[B + 1] + zi

        def val(idx):
            return p[idx].astype(np.float64) * (2.0 / 255.0) - 1.0

        v000 = val(AA);     v100 = val(BA);     v010 = val(AB);     v110 = val(BB)
        v001 = val(AA + 1); v101 = val(BA + 1); v011 = val(AB + 1); v111 = val(BB + 1)

        # Multilinear form value = a + b ux + c uy + d uz + e ux uy + f ux uz
        #                              + g uy uz + h ux uy uz, differentiated exactly.
        a = v000
        b = v100 - v000
        c = v010 - v000
        d = v001 - v000
        e = v000 - v100 - v010 + v110
        f = v000 - v100 - v001 + v101
        g = v000 - v010 - v001 + v011
        h = -v000 + v100 + v010 + v001 - v110 - v101 - v011 + v111

        value = (a + b * ux + c * uy + d * uz + e * ux * uy + f * ux * uz
                 + g * uy * uz + h * ux * uy * uz)
        ddx = dux * (b + e * uy + f * uz + h * uy * uz)
        ddy = duy * (c + e * ux + g * uz + h * ux * uz)
        ddz = duz * (d + f * ux + g * uy + h * ux * uy)
        return value, ddx, ddy, ddz
