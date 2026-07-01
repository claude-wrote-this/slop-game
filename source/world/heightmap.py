"""Layered terrain — a pure DATA object. No pygame, no renderer, no chunks. It
answers one question over a set of world points: sample_points(X, Y) ->
(height, colour).

Height model: per-layer 3D occupancy. The world is N layers evenly spaced up from
z=0. Each layer is an independent solid/empty test at its own z: the 3D fractal
field is sampled at (x, y, z) and compared against that layer's threshold t(z).
This yields a full vertical occupancy column — a column can read solid/empty/solid
down its height, so overhangs and caves fall out naturally. The surface the
renderer reads is the topmost solid layer per column.

The whole (x, y, z) stack is evaluated at once — z is an added numpy axis, so all N
layers are sampled in one vectorised pass with no Python z-loop, then reduced along
z to the topmost solid layer. N is still the dominant cost (every layer is a full
field evaluation over the point stack), so it is kept as the single knob and tuned
from on-device measurement.

t(z) rises with height from an always-solid floor: it climbs quickly through the
low layers and eases toward the ceiling, so ground is easy to hold low down and
progressively harder up high — most land sits low-to-mid, high ground rare and
sharp (a realistic hypsometric distribution). The `hypso` exponent is the single
knob: higher steepens the early rise, widening the flat lowlands and making peaks
rarer/sharper. (A convex "slow at the bottom" rise does the opposite for this
density, pushing the surface up, so the curve is deliberately concave.)

The fractal structure is the existing apparatus — domain-warped multi-octave fBm +
ridged mountains — carried into the per-layer evaluation: z is a live coordinate
(scaled by z_span) instead of a fixed decorrelation slice, and that vertical
variation is what gives the sub-surface its overhangs and caves. The threshold
curve controls how much land sits at each height; the fractal/ridge structure
controls whether high ground forms coherent ranges vs scattered spikes. Colour is a
per-layer ramp for visibility (a deliberate placeholder). Pure function of world
(x, y, z), so any block sampled anywhere tiles seamlessly with its neighbours.
"""
import numpy as np

from source.world.field import _Perlin3

_Z_WARP_X, _Z_WARP_Y = 11.5, 23.5
_Z_BASE, _Z_MOUNT, _Z_DETAIL = 0.5, 31.5, 47.5

# per-layer colour ramp anchors: dark green low ground -> light green high ground
_RAMP_ANCHORS = [(18, 46, 24), (40, 82, 42), (72, 120, 64),
                 (112, 162, 92), (158, 202, 134), (208, 236, 188)]


class TerrainHeight:
    covers_screen = True            # the renderer treats terrain as always on-screen

    def __init__(self, seed, *, layers=100,
                 base_freq=0.011, mount_freq=0.03, detail_freq=0.11,
                 warp_freq=0.02, warp_amp=18.0,
                 octaves=5, mount_strength=1.15, detail_strength=0.05,
                 hypso=1.6, z_span=2.5, solid_ceiling=0.85):
        self.noise = _Perlin3(seed)
        self.layers = layers
        self.base_freq = base_freq
        self.mount_freq = mount_freq
        self.detail_freq = detail_freq
        self.warp_freq = warp_freq
        self.warp_amp = warp_amp
        self.octaves = octaves
        self.mount_strength = mount_strength
        self.detail_strength = detail_strength
        self.hypso = hypso                    # threshold ease-in exponent (the knob)
        self.z_span = z_span                  # noise-z spanned over the full height
        self.solid_ceiling = solid_ceiling    # threshold at the very top layer
        self.colour_ramp = self._build_ramp(layers)

        # Per-layer constants (computed once). frac 0..1 up the stack.
        frac = np.linspace(0.0, 1.0, layers)
        self._znoise = (frac * z_span).astype(float)          # noise-z per layer
        # Concave rise: steep through the low layers, easing toward the ceiling.
        # Higher hypso -> steeper early -> wider lowlands, rarer peaks.
        thresh = solid_ceiling * frac ** (1.0 / hypso)
        thresh[0] = -1.0                                       # z=0 always solid
        self._thresh = thresh

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

    def _density(self, X, Y, Z):
        """Solidness in [0, 1] of the 3D field at world (X, Y) and noise-z Z.
        Broadcasts: with X, Y of shape (n, 1) and Z of shape (1, N) it returns the
        full (n, N) occupancy stack in one shot — every layer at once, no z-loop. Z
        varies per layer, so the warp and every field vary with height; that vertical
        variation is what gives overhangs and caves. Same warp+fBm+ridged apparatus
        as the flat model, only with a live z instead of a fixed slice."""
        wx = self._fbm(X, Y, Z + _Z_WARP_X, self.warp_freq, octaves=4)
        wy = self._fbm(X, Y, Z + _Z_WARP_Y, self.warp_freq, octaves=4)
        Xw = X + wx * self.warp_amp
        Yw = Y + wy * self.warp_amp
        base = (self._fbm(Xw, Yw, Z + _Z_BASE, self.base_freq) + 1.0) * 0.5
        mount = self._ridged(Xw, Yw, Z + _Z_MOUNT, self.mount_freq)
        m = np.clip((base - 0.45) / 0.30, 0.0, 1.0)
        mount *= m * m * (3.0 - 2.0 * m)                      # mountains on high ground
        detail = (self._fbm(Xw, Yw, Z + _Z_DETAIL, self.detail_freq, octaves=3)
                  + 1.0) * 0.5
        d = base * 0.7 + mount * self.mount_strength + detail * self.detail_strength
        d = d / (0.7 + self.mount_strength + self.detail_strength)
        return np.clip(d, 0.0, 1.0)

    def sample_points(self, X, Y):
        """(height int, colour) at arbitrary world positions X, Y (cell units,
        fractional ok). Height is the topmost solid layer of each column's 3D
        occupancy — the whole stack is evaluated vectorised and only the top is
        returned. No grid — one value per point."""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        shape = X.shape
        xf = X.reshape(-1)[:, None]                           # (n, 1)
        yf = Y.reshape(-1)[:, None]
        d = self._density(xf, yf, self._znoise[None, :])      # (n, N) whole stack
        solid = d > self._thresh[None, :]                     # (n, N), layer 0 always
        # topmost solid layer per column: first solid scanning from the top down.
        top = (self.layers - 1) - np.argmax(solid[:, ::-1], axis=1)
        height = top.reshape(shape).astype(np.int64)
        colour = self.colour_ramp[height]                     # (..., 3)
        return height, colour
