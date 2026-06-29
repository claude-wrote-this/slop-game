"""Rain/heightmap renderer. The CORE, render_heightmap(H, has, ...), takes a
continuous height grid (in layer units, NaN/has=False where open sky) and paints
a top-down chunk: horizon-scan shadows, depth haze, soft kernel splats.

Two front-ends feed the core from different sources:
  - render_heightmap_chunk: a heightmap generator (TerrainHeight) -> clamp, no rain.
  - render_chunk: the 3D field -> rain down to the first solid z, then the core.
Both share one renderer; the 2.5D path and a future span overlay can feed the
same core too.

Depth is read from light and atmosphere, not colour-by-height: one flat material,
shadows and haze do the relief.
"""
import numpy as np
import pygame


def surface_heightmap(field, ox, oy, cols, rows, ceiling, z_min=-2):
    """Field path: top solid z per cell at/under `ceiling`, refined to continuous
    height. Returns (H with NaN where open sky, has-surface)."""
    gx = (ox + np.arange(cols))[None, :] + 0.5
    gy = (oy + np.arange(rows))[:, None] + 0.5
    GX = np.broadcast_to(gx, (rows, cols)).astype(float)
    GY = np.broadcast_to(gy, (rows, cols)).astype(float)

    H = np.full((rows, cols), np.nan)
    falling = np.ones((rows, cols), bool)
    for z in range(int(ceiling), z_min - 1, -1):
        hit = falling & field.solid(GX, GY, float(z))
        H[hit] = z
        falling &= ~hit
    has = ~np.isnan(H)

    Hi = np.where(has, H, 0.0)
    d0 = field.value(GX, GY, Hi)
    d1 = field.value(GX, GY, Hi + 1.0)
    frac = np.clip(d0 / (d0 - d1 + 1e-9), 0.0, 1.0)
    H = np.where(has, Hi + frac, np.nan)
    return H, has


def _shift(A, dx, dy, fill):
    out = np.full_like(A, fill)
    rows, cols = A.shape
    sy0, sy1 = max(0, -dy), rows - max(0, dy)
    sx0, sx1 = max(0, -dx), cols - max(0, dx)
    dy0, dy1 = max(0, dy), rows - max(0, -dy)
    dx0, dx1 = max(0, dx), cols - max(0, -dx)
    out[dy0:dy1, dx0:dx1] = A[sy0:sy1, sx0:sx1]
    return out


def _smooth(A, passes=1):
    for _ in range(passes):
        up = np.empty_like(A);   up[:-1] = A[1:];   up[-1] = A[-1]
        dn = np.empty_like(A);   dn[1:] = A[:-1];   dn[0] = A[0]
        lf = np.empty_like(A);   lf[:, :-1] = A[:, 1:];  lf[:, -1] = A[:, -1]
        rt = np.empty_like(A);   rt[:, 1:] = A[:, :-1];  rt[:, 0] = A[:, 0]
        A = (A + up + dn + lf + rt) / 5.0
    return A


def cast_shadows(H, sun=(-1, -1), slope=0.7, maxdist=60, softness=1.6, smooth=2):
    """Soft shadow amount in [0, 1] via scanline horizon. Low sun = long shadows.
    Soft (a ramp over `softness` layers), not binary, so edges feather."""
    Hs = _smooth(H, smooth)
    sunx, suny = sun
    horizon = np.full_like(Hs, -1e9)
    cur = Hs.copy()
    for _ in range(1, maxdist + 1):
        cur = _shift(cur, -sunx, -suny, -1e9) - slope
        np.maximum(horizon, cur, out=horizon)
    shadow = np.clip((horizon - Hs) / max(1e-6, softness), 0.0, 1.0)
    return _smooth(shadow, 1)


def _make_kernel(radius):
    d = radius * 2 + 1
    yy, xx = np.mgrid[0:d, 0:d]
    dist = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2) / max(1, radius)
    alpha = (np.clip(1 - dist, 0, 1) ** 1.5 * 230).astype(np.uint8)
    surf = pygame.Surface((d, d), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = alpha.T
    del a
    return surf


# ---- the shared core: heightmap -> pixels ---------------------------------

def render_heightmap(H, has, *, tile=12, sun=(-1, -1), sun_slope=0.48,
                     samples_per_cell=3, base=(108, 128, 90), haze=(48, 58, 78),
                     haze_k=0, haze_max=0, shade_min=0.52,
                     view_z=None, radius=None, bg=(15, 17, 21), jitter_seed=None):
    rows, cols = H.shape
    radius = radius or int(tile * 0.95)
    H_lit = np.where(has, H, -10.0)            # holes read as very low for shadows
    shadow = cast_shadows(H_lit, sun, sun_slope)
    # haze reference: a SHARED value across chunks (passed in) so neighbouring
    # chunks don't haze differently and seam. Falls back to this chunk's own max.
    if view_z is None:
        view_z = float(np.nanmax(np.where(has, H, np.nan))) if has.any() else 0.0

    W, Hpx = cols * tile, rows * tile
    surf = pygame.Surface((W, Hpx))
    surf.fill(bg)
    kernel = _make_kernel(radius)
    cache = {}

    spc = samples_per_cell
    m, n = rows * spc, cols * spc
    rng = np.random.default_rng(0x9E3779B9 if jitter_seed is None else jitter_seed)
    px = (np.arange(n)[None, :] + rng.random((m, n))) * (tile / spc)
    py = (np.arange(m)[:, None] + rng.random((m, n))) * (tile / spc)
    cgx = np.clip((px / tile).astype(int), 0, cols - 1)
    cgy = np.clip((py / tile).astype(int), 0, rows - 1)

    z = H_lit[cgy, cgx]
    vis = has[cgy, cgx]
    shv = shadow[cgy, cgx]
    depth = np.clip((view_z - z) * haze_k, 0.0, haze_max)

    base_a, haze_a = np.array(base, float), np.array(haze, float)
    shade = (1.0 - (1.0 - shade_min) * shv)[..., None]
    col = base_a #* shade
    col = col * (1 - depth[..., None]) #+ haze_a * depth[..., None]
    col = col.astype(int)

    pxf, pyf = px.ravel(), py.ravel()
    visf = vis.ravel()
    colf = col.reshape(-1, 3)
    for i in range(pxf.size):
        if not visf[i]:
            continue
        c = colf[i]
        key = (int(c[0]) & ~7, int(c[1]) & ~7, int(c[2]) & ~7)
        k = cache.get(key)
        if k is None:
            k = kernel.copy()
            k.fill((*key, 255), special_flags=pygame.BLEND_RGBA_MULT)
            cache[key] = k
        surf.blit(k, (pxf[i] - radius, pyf[i] - radius))
    return surf


# ---- front-ends -----------------------------------------------------------

def render_heightmap_chunk(layer, height, *, layers, **kw):
    """Heightmap source (TerrainHeight.chunk -> (layer, height in [0,1])). No rain:
    the heightmap IS the surface. Scale to layer units and shade."""
    H = height * layers
    has = np.ones_like(H, dtype=bool)
    return render_heightmap(H, has, **kw)


def render_chunk(field, ox, oy, cols, rows, *, ceiling=12, **kw):
    """Field source: rain down to the first solid z, then render."""
    H, has = surface_heightmap(field, ox, oy, cols, rows, ceiling)
    return render_heightmap(H, has, **kw)
