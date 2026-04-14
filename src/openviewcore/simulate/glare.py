"""Direct-glare exposure for facade viewpoints.

For each facade viewpoint this module computes:

- **glare_hours** — hours/year of direct, unobstructed sun hitting the facade
- **irradiance_weighted_hours** — Σ(cos_i × dt) — geometry-weighted exposure
  (proportional to irradiance magnitude without needing DNI)
- **low_sun_glare_hours** — hours where the sun is below *low_sun_el* degrees
  elevation and directly lights the facade (worst-case glare geometry)
- **peak_cos_i** — maximum cos(incidence angle) in any direct-sun hour
  (1.0 = sun perfectly perpendicular to facade)
- **annual_irradiance_kwh_m2** (only when *dni* is supplied) —
  Σ(DNI × cos_i × dt) / 1000
- **discomfort_hours** (only when *dni* is supplied) — hours where
  plane-of-array irradiance exceeds *discomfort_threshold*

Algorithm
---------
Same shadow-ray approach as :mod:`~openviewcore.simulate.solar` but retains
the cosine of incidence (cos_i) as a float instead of collapsing to a bool.
This enables irradiance-weighted accumulation and the discomfort-threshold
check without any extra ray casts.

Coordinate system
-----------------
East = +X, North = +Y, Up = +Z in the projected CRS.
pvlib azimuth is clockwise from North, consistent with this convention.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .solar import _solar_dirs, _timestep_hours, facade_orientation, sun_positions


def compute_direct_glare(
    viewpoints: np.ndarray,
    scene: dict,
    lat: float,
    lon: float,
    *,
    tz: str = "UTC",
    year: int = 2024,
    freq: str = "1h",
    low_sun_el: float = 20.0,
    dni: Optional[np.ndarray] = None,
    discomfort_threshold: float = 50.0,
    ray_near: float = 0.5,
    timestep_chunk: int = 200,
    n_workers: int = 0,
    intersector=None,
) -> pd.DataFrame:
    """Compute per-viewpoint direct glare exposure metrics.

    Parameters
    ----------
    viewpoints : np.ndarray
        Structured array with ``vp_uuid``, ``vp`` (XYZ), and ``norm`` fields.
    scene : dict
        ``{'v': (N,3) float32, 'f': (M,3) int32}`` — combined scene mesh.
    lat, lon : float
        Site coordinates in WGS84 degrees.
    tz : str
        IANA timezone string, e.g. ``"America/New_York"``.
    year : int
        Calendar year to model.
    freq : str
        Time resolution for solar positions — ``"1h"`` (default).
    low_sun_el : float
        Sun elevation (degrees) below which direct hits are counted as
        ``low_sun_glare_hours``.  Default ``20°`` captures the low-angle
        problem common in winter morning/afternoon sun.
    dni : ndarray (T,), optional
        Direct Normal Irradiance (W/m²) aligned to the daylight timesteps
        returned by :func:`sun_positions` for the same *lat*, *lon*, *tz*,
        *year*, and *freq*.  When provided adds ``annual_irradiance_kwh_m2``
        and ``discomfort_hours`` to the result.
    discomfort_threshold : float
        Plane-of-array irradiance (W/m²) above which an hour is flagged as
        a discomfort hour.  Default ``50 W/m²``.  Only used when *dni* is
        supplied.
    ray_near : float
        Minimum shadow-ray hit distance (m) — avoids self-intersection.
    timestep_chunk : int
        Timesteps per worker call — controls peak memory per thread.
    n_workers : int
        Parallel threads.  ``0`` = min(cpu_count, 4).
    intersector : optional
        Pre-built ``pcu.RayMeshIntersector``.  Pass ``cfg.intersector`` to
        reuse the BVH built once for the whole pipeline.

    Returns
    -------
    DataFrame
        Indexed by ``vp_uuid``.  Always-present columns:

        ``glare_hours``
            Hours/year of direct, unobstructed sun on the facade.
        ``irradiance_weighted_hours``
            Σ(cos_i × dt) — geometry-weighted proxy for irradiance
            without needing measured DNI.
        ``low_sun_glare_hours``
            Hours of direct sun when the sun is below *low_sun_el*.
        ``peak_cos_i``
            Maximum cos(incidence angle) across all direct-sun hours.
            Ranges from 0 (grazing) to 1 (perpendicular).
        ``orientation``
            Cardinal / intercardinal facade direction.

        Optional (only when *dni* is supplied):

        ``annual_irradiance_kwh_m2``
            Plane-of-array energy in kWh/m²/year.
        ``discomfort_hours``
            Hours where POA irradiance exceeds *discomfort_threshold*.
    """
    import point_cloud_utils as pcu

    vp_xyz = np.ascontiguousarray(viewpoints["vp"],   dtype=np.float32)
    norms  = np.ascontiguousarray(viewpoints["norm"], dtype=np.float32)
    n_vps  = len(vp_xyz)

    if intersector is None:
        sv = np.ascontiguousarray(scene["v"], dtype=np.float32)
        sf = np.ascontiguousarray(scene["f"], dtype=np.int32)
        intersector = pcu.RayMeshIntersector(sv, sf)

    workers = n_workers if n_workers > 0 else min(os.cpu_count() or 1, 4)

    solpos = sun_positions(lat, lon, tz=tz, year=year, freq=freq)
    sun_d  = _solar_dirs(solpos)
    dt     = _timestep_hours(freq)
    n_t    = len(solpos)

    el           = solpos["apparent_elevation"].values   # (T,)
    low_sun_mask = el < low_sun_el                       # (T,) bool

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    glare_count   = np.zeros(n_vps, dtype=np.int32)
    irrad_accum   = np.zeros(n_vps, dtype=np.float64)
    low_sun_count = np.zeros(n_vps, dtype=np.int32)
    peak_cos_i    = np.zeros(n_vps, dtype=np.float64)

    poa_accum     = np.zeros(n_vps, dtype=np.float64) if dni is not None else None
    discomf_count = np.zeros(n_vps, dtype=np.int32)   if dni is not None else None

    # ------------------------------------------------------------------
    # Parallel shadow casting
    # ------------------------------------------------------------------
    chunks = [
        (t_start, min(t_start + timestep_chunk, n_t))
        for t_start in range(0, n_t, timestep_chunk)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _cast_glare_chunk,
                intersector, vp_xyz, norms,
                sun_d[t_start:t_end], t_start, n_vps, ray_near,
            ): (t_start, t_end)
            for t_start, t_end in chunks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Glare casting"):
            t_start, cos_i_chunk, shadowed_chunk = future.result()
            t_end = t_start + shadowed_chunk.shape[1]

            # Direct: sun faces the facade AND no obstruction
            direct = (cos_i_chunk > 0) & ~shadowed_chunk   # (V, C) bool

            glare_count += direct.sum(axis=1)

            # cos_i masked to direct hits — used for all weighted metrics
            cos_i_direct = np.where(direct, cos_i_chunk, 0.0)  # (V, C) float

            irrad_accum += cos_i_direct.sum(axis=1)

            # Low-sun subset: (T,) mask broadcasts over (V, C)
            low_sun_count += (direct & low_sun_mask[t_start:t_end]).sum(axis=1)

            # Peak cos_i: running element-wise maximum across chunks
            peak_cos_i = np.maximum(peak_cos_i, cos_i_direct.max(axis=1))

            # DNI-weighted metrics (optional)
            if poa_accum is not None:
                poa_chunk   = cos_i_direct * dni[t_start:t_end]   # (V, C) broadcast
                poa_accum  += poa_chunk.sum(axis=1)
                discomf_count += (direct & (poa_chunk > discomfort_threshold)).sum(axis=1)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "glare_hours":               glare_count   * dt,
        "irradiance_weighted_hours": irrad_accum   * dt,
        "low_sun_glare_hours":       low_sun_count * dt,
        "peak_cos_i":                peak_cos_i,
        "orientation":               facade_orientation(viewpoints).values,
    }, index=viewpoints["vp_uuid"])
    result.index.name = "vp_uuid"

    if poa_accum is not None:
        result["annual_irradiance_kwh_m2"] = poa_accum     * dt / 1000.0
        result["discomfort_hours"]         = discomf_count * dt

    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _cast_glare_chunk(
    intersector,
    vp_xyz: np.ndarray,
    norms: np.ndarray,
    sun_d_chunk: np.ndarray,
    t_start: int,
    n_vps: int,
    ray_near: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Cast shadow rays for one chunk and return per-ray cos(incidence).

    Unlike :func:`~openviewcore.simulate.solar._cast_shadow_chunk` which
    returns a bool ``facing`` mask, this function returns the raw float
    ``cos_i`` so callers can weight by incidence angle.

    Returns
    -------
    t_start : int
    cos_i : ndarray (V, C) float32
        Dot product of each facade normal with each sun direction.
        Negative values mean the sun is behind the facade.
    shadowed : ndarray (V, C) bool
        True where a shadow ray hit an obstruction.  Only rays where
        ``cos_i > 0`` are cast; all others remain False.
    """
    C = sun_d_chunk.shape[0]

    cos_i  = (norms @ sun_d_chunk.T).astype(np.float32)   # (V, C)
    facing = cos_i > 0                                      # (V, C) bool

    v_idx, t_idx = np.where(facing)

    shadowed = np.zeros((n_vps, C), dtype=bool)
    if v_idx.size > 0:
        fid, _, t_hit = intersector.intersect_rays(
            np.ascontiguousarray(vp_xyz[v_idx]),
            np.ascontiguousarray(sun_d_chunk[t_idx]),
        )
        shadowed[v_idx, t_idx] = (fid >= 0) & (t_hit > ray_near)

    return t_start, cos_i, shadowed


# ===========================================================================
# Reflective glare
# ===========================================================================

def compute_reflective_glare(
    reflectors: np.ndarray,
    observers: np.ndarray,
    scene: dict,
    lat: float,
    lon: float,
    *,
    tz: str = "UTC",
    year: int = 2024,
    freq: str = "1h",
    max_distance: float = 200.0,
    glare_cone: float = 5.0,
    reflectance: "float | dict" = 0.3,
    low_sun_el: float = 20.0,
    dni: Optional[np.ndarray] = None,
    ray_near: float = 0.5,
    timestep_chunk: int = 50,
    n_workers: int = 0,
    intersector=None,
) -> pd.DataFrame:
    """Compute per-observer reflected glare from specular building facades.

    Models sun → reflective facade → observer as a 3-point light path.
    Only (reflector, observer) pairs within *max_distance* are considered,
    and only when the reflected ray falls within *glare_cone* degrees of the
    observer's position.

    Parameters
    ----------
    reflectors : np.ndarray
        Facade viewpoints that act as reflective surfaces.  Pass a **subset**
        of building viewpoints filtered to high-reflectance facades (glass
        curtain wall, polished metal) to keep compute tractable.
        Same :data:`VP_DTYPE` as all other viewpoint arrays.
    observers : np.ndarray
        Street-level (or any) viewpoints that receive reflected glare.
        Typically the output of
        :func:`~openviewcore.viewpoints.street.generate_street_viewpoints`.
    scene : dict
        ``{'v': (N,3) float32, 'f': (M,3) int32}`` — combined scene mesh
        used for obstruction testing between reflector and observer.
    lat, lon : float
        Site coordinates in WGS84 degrees.
    tz : str
        IANA timezone string.
    year : int
        Calendar year to model.
    freq : str
        Time resolution for solar positions — ``"1h"`` (default).
    max_distance : float
        Spatial cutoff (metres).  Only reflector-observer pairs closer than
        this are tested.  Reducing this is the primary way to speed up the
        computation.  Default ``200 m``.
    glare_cone : float
        Half-angle (degrees) of the glare cone.  The reflected ray must point
        within this angle of the observer for a glare event to be recorded.
        Default ``5°`` — a conservative threshold for discomfort glare.
    reflectance : float or dict
        Fraction of incident irradiance reflected specularly.

        * ``float`` (default ``0.3``) — applied uniformly to all reflectors.
        * ``dict`` mapping ``osmid`` (int) → reflectance — lets you assign
          per-building values (e.g. ``{way_id: 0.6}`` for high-performance
          curtain wall).  Buildings not in the dict use the default of ``0.3``.
    low_sun_el : float
        Sun elevation (degrees) below which glare events are also accumulated
        into ``reflected_low_sun_hours``.  Default ``20°``.
    dni : ndarray (T,), optional
        Direct Normal Irradiance (W/m²) aligned to daylight timesteps from
        :func:`~openviewcore.simulate.solar.sun_positions`.  When supplied
        adds ``reflected_irradiance_kwh_m2``.
    ray_near : float
        Minimum hit distance (m) to skip self-intersection at the reflector.
    timestep_chunk : int
        Timesteps per worker chunk.  Smaller values reduce peak memory but
        increase thread-launch overhead.  Default ``50``.
    n_workers : int
        Parallel threads.  ``0`` = min(cpu_count, 4).
    intersector : optional
        Pre-built ``pcu.RayMeshIntersector``.  Pass ``cfg.intersector`` to
        reuse the BVH built once for the whole pipeline.

    Returns
    -------
    DataFrame
        Indexed by observer ``vp_uuid``.  Columns:

        ``reflected_glare_hours``
            Hours/year the observer received reflected glare from at least one
            unobstructed reflector within the glare cone.
        ``reflected_weighted_hours``
            Σ(cos_i × reflectance × dt) — intensity-weighted exposure
            summed across all contributing reflectors per timestep.
        ``reflected_low_sun_hours``
            Subset of ``reflected_glare_hours`` where sun elevation
            < *low_sun_el*.
        ``reflected_irradiance_kwh_m2``
            *(only when dni is provided)* Σ(DNI × cos_i × reflectance × dt)
            / 1000.
    """
    import point_cloud_utils as pcu

    ref_xyz   = np.ascontiguousarray(reflectors["vp"],   dtype=np.float32)
    ref_norms = np.ascontiguousarray(reflectors["norm"], dtype=np.float32)
    obs_xyz   = np.ascontiguousarray(observers["vp"],    dtype=np.float32)
    n_obs     = len(obs_xyz)

    if intersector is None:
        sv = np.ascontiguousarray(scene["v"], dtype=np.float32)
        sf = np.ascontiguousarray(scene["f"], dtype=np.int32)
        intersector = pcu.RayMeshIntersector(sv, sf)

    # ------------------------------------------------------------------
    # Per-reflector reflectance values
    # ------------------------------------------------------------------
    ref_values = _resolve_reflectance(reflectors, reflectance)

    # ------------------------------------------------------------------
    # Pre-compute reflector-observer pairs within max_distance (one-time)
    # ------------------------------------------------------------------
    obs_idx, ref_idx, v_ref_to_obs, pair_dist = _build_pairs(
        ref_xyz, obs_xyz, max_distance,
    )

    if len(obs_idx) == 0:
        print("  compute_reflective_glare: no reflector-observer pairs "
              f"within {max_distance} m — returning zeros.")
        result = pd.DataFrame({
            "reflected_glare_hours":    0.0,
            "reflected_weighted_hours": 0.0,
            "reflected_low_sun_hours":  0.0,
        }, index=observers["vp_uuid"])
        result.index.name = "vp_uuid"
        return result

    print(f"  Reflective glare: {len(obs_idx):,} pairs "
          f"({len(reflectors)} reflectors × {n_obs} observers, "
          f"cutoff {max_distance} m)")

    # ------------------------------------------------------------------
    # Solar positions
    # ------------------------------------------------------------------
    solpos = sun_positions(lat, lon, tz=tz, year=year, freq=freq)
    sun_d  = _solar_dirs(solpos)
    dt     = _timestep_hours(freq)
    n_t    = len(solpos)

    el           = solpos["apparent_elevation"].values
    low_sun_mask = (el < low_sun_el)

    cos_cone = float(np.cos(np.deg2rad(glare_cone)))
    workers  = n_workers if n_workers > 0 else min(os.cpu_count() or 1, 4)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    glare_count    = np.zeros(n_obs, dtype=np.int32)
    weighted_accum = np.zeros(n_obs, dtype=np.float64)
    low_sun_count  = np.zeros(n_obs, dtype=np.int32)
    poa_accum      = np.zeros(n_obs, dtype=np.float64) if dni is not None else None

    # ------------------------------------------------------------------
    # Parallel shadow + cone casting (chunked over timesteps)
    # ------------------------------------------------------------------
    chunks = [
        (t_start, min(t_start + timestep_chunk, n_t))
        for t_start in range(0, n_t, timestep_chunk)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _reflective_chunk,
                intersector,
                ref_xyz, ref_norms, ref_values,
                obs_idx, ref_idx, v_ref_to_obs, pair_dist,
                sun_d[t_start:t_end],
                low_sun_mask[t_start:t_end],
                n_obs, ray_near, cos_cone,
                dni[t_start:t_end] if dni is not None else None,
            ): (t_start, t_end)
            for t_start, t_end in chunks
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Reflective glare"):
            g, w, ls, poa = future.result()
            glare_count    += g
            weighted_accum += w
            low_sun_count  += ls
            if poa_accum is not None:
                poa_accum  += poa

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "reflected_glare_hours":    glare_count    * dt,
        "reflected_weighted_hours": weighted_accum * dt,
        "reflected_low_sun_hours":  low_sun_count  * dt,
    }, index=observers["vp_uuid"])
    result.index.name = "vp_uuid"

    if poa_accum is not None:
        result["reflected_irradiance_kwh_m2"] = poa_accum * dt / 1000.0

    return result


# ---------------------------------------------------------------------------
# Reflective glare internals
# ---------------------------------------------------------------------------

def _build_pairs(
    ref_xyz: np.ndarray,
    obs_xyz: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find all (reflector, observer) pairs within *max_distance* in XY.

    Returns
    -------
    obs_idx      : (K,) int32  — observer index per pair
    ref_idx      : (K,) int32  — reflector index per pair
    v_ref_to_obs : (K, 3) float32 — unit vector from reflector to observer
    pair_dist    : (K,) float32   — 3-D distance between the two points
    """
    from scipy.spatial import cKDTree

    tree      = cKDTree(ref_xyz[:, :2])
    pairs_raw = tree.query_ball_point(obs_xyz[:, :2], max_distance)

    obs_list: list[int] = []
    ref_list: list[int] = []
    for o, refs in enumerate(pairs_raw):
        for r in refs:
            obs_list.append(o)
            ref_list.append(r)

    if not obs_list:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty((0, 3), np.float32), np.empty(0, np.float32)

    obs_idx = np.array(obs_list, dtype=np.int32)
    ref_idx = np.array(ref_list, dtype=np.int32)

    delta     = obs_xyz[obs_idx] - ref_xyz[ref_idx]          # (K, 3)
    dist      = np.linalg.norm(delta, axis=1)                 # (K,)
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    v_to_obs  = (delta / safe_dist[:, None]).astype(np.float32)

    return obs_idx, ref_idx, v_to_obs, dist.astype(np.float32)


def _resolve_reflectance(
    reflectors: np.ndarray,
    reflectance: "float | dict",
    default: float = 0.3,
) -> np.ndarray:
    """Return a per-reflector float32 array of reflectance values.

    Parameters
    ----------
    reflectance : float or dict
        Scalar applied to all reflectors, **or** a ``{osmid: value}`` dict
        for per-building overrides.  Buildings absent from the dict use
        *default*.
    """
    n = len(reflectors)
    if isinstance(reflectance, (int, float)):
        return np.full(n, float(reflectance), dtype=np.float32)
    # dict: osmid (int) → reflectance float
    return np.array(
        [reflectance.get(int(oid), default) for oid in reflectors["osmid"]],
        dtype=np.float32,
    )


def _reflective_chunk(
    intersector,
    ref_xyz: np.ndarray,       # (R, 3) float32
    ref_norms: np.ndarray,     # (R, 3) float32
    ref_values: np.ndarray,    # (R,)   float32  reflectance per reflector
    obs_idx: np.ndarray,       # (K,)   int32
    ref_idx: np.ndarray,       # (K,)   int32
    v_ref_to_obs: np.ndarray,  # (K, 3) float32  pre-computed unit vectors
    pair_dist: np.ndarray,     # (K,)   float32  distances
    sun_d_chunk: np.ndarray,   # (C, 3) float32
    low_sun_chunk: np.ndarray, # (C,)   bool
    n_obs: int,
    ray_near: float,
    cos_cone: float,
    dni_chunk: Optional[np.ndarray],   # (C,) or None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process one timestep chunk for reflective glare.

    For each timestep in the chunk:

    1. Find reflectors facing the sun.
    2. From pre-screened pairs, keep those where the reflected ray
       falls within the glare cone of the observer.
    3. Cast an obstruction ray from reflector to observer.
    4. Accumulate per-observer counters.

    Returns ``(glare_count, weighted_accum, low_sun_count, poa_accum)``
    each of shape ``(n_obs,)``.
    """
    glare_count    = np.zeros(n_obs, dtype=np.int32)
    weighted_accum = np.zeros(n_obs, dtype=np.float64)
    low_sun_count  = np.zeros(n_obs, dtype=np.int32)
    poa_accum      = np.zeros(n_obs, dtype=np.float64)

    for c in range(len(sun_d_chunk)):
        sun_d = sun_d_chunk[c]   # (3,)

        # 1. Facing test — all reflectors at once
        cos_i_all = ref_norms @ sun_d   # (R,)
        facing_all = cos_i_all > 0      # (R,)

        # 2. Keep only pairs where the reflector faces the sun
        active = np.where(facing_all[ref_idx])[0]   # indices into K-array
        if active.size == 0:
            continue

        a_obs   = obs_idx[active]             # (Ka,)
        a_ref   = ref_idx[active]             # (Ka,)
        a_cos_i = cos_i_all[a_ref]            # (Ka,)
        a_v     = v_ref_to_obs[active]        # (Ka, 3) unit ref→obs
        a_dist  = pair_dist[active]           # (Ka,)

        # 3. Reflected ray: r = 2*(n·sun)*n - sun  (mirror reflection)
        r_rays = 2.0 * a_cos_i[:, None] * ref_norms[a_ref] - sun_d[None, :]
        r_mag  = np.linalg.norm(r_rays, axis=1, keepdims=True)
        r_rays /= np.where(r_mag > 1e-12, r_mag, 1.0)

        # 4. Angular cone test — dot(reflected_ray, direction_to_obs) > cos_cone
        cos_angle  = (a_v * r_rays).sum(axis=1)   # (Ka,)
        cone_mask  = cos_angle > cos_cone          # (Ka,) bool
        if not cone_mask.any():
            continue

        c_obs   = a_obs[cone_mask]                 # (Kc,)
        c_ref   = a_ref[cone_mask]                 # (Kc,)
        c_cos_i = a_cos_i[cone_mask]               # (Kc,)
        c_dist  = a_dist[cone_mask]                # (Kc,)
        c_v     = a_v[cone_mask]                   # (Kc, 3) unit ref→obs

        # 5. Obstruction test: cast ray from reflector toward observer
        fid, _, t_hit = intersector.intersect_rays(
            np.ascontiguousarray(ref_xyz[c_ref]),
            np.ascontiguousarray(c_v),
        )
        # Unobstructed: no hit, self-intersection, or hit beyond observer
        unobstructed = (fid < 0) | (t_hit <= ray_near) | (t_hit >= c_dist - ray_near)
        if not unobstructed.any():
            continue

        uo_obs      = c_obs[unobstructed]
        uo_cos_i    = c_cos_i[unobstructed]
        uo_refl     = ref_values[c_ref[unobstructed]]
        intensity   = uo_cos_i * uo_refl             # (Ku,)

        # glare_count: binary per observer per timestep (any source = 1)
        glare_this_t = np.zeros(n_obs, dtype=bool)
        glare_this_t[uo_obs] = True
        glare_count += glare_this_t

        # weighted: sum all contributions (multiple reflectors add up)
        np.add.at(weighted_accum, uo_obs, intensity)

        if low_sun_chunk[c]:
            ls_this_t = np.zeros(n_obs, dtype=bool)
            ls_this_t[uo_obs] = True
            low_sun_count += ls_this_t

        if dni_chunk is not None:
            np.add.at(poa_accum, uo_obs, intensity * dni_chunk[c])

    return glare_count, weighted_accum, low_sun_count, poa_accum
