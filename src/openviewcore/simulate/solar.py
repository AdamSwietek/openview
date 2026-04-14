"""Facade solar exposure: sun positions, shadow casting, irradiance integration.

For each facade viewpoint this module computes:

- **annual_solar_hours** — hours/year of direct, unobstructed sun
- **geometric_solar_hours** — hours/year the sun geometrically faces the facade
  (ignoring obstructions); upper bound for ``annual_solar_hours``
- **shadow_fraction** — fraction of geometric exposure that is blocked
- **orientation** — cardinal/intercardinal facade direction (N / NE / E / …)
- **annual_irradiance_kwh** — plane-of-array energy in kWh/m²/year
  (only when *dni* is supplied)

Algorithm
---------
1. Compute solar positions for a full year with ``pvlib``.
2. For each daylight timestep build one shadow ray per viewpoint pointing
   toward the sun.  Rays are processed in chunks (controlled by
   ``timestep_chunk``) to keep peak memory bounded.
3. Direct exposure = facing the sun **and** ray not blocked by the scene.

Coordinate system
-----------------
Viewpoints live in a projected CRS where East = +X, North = +Y, Up = +Z.
``pvlib`` azimuth is clockwise from North, which maps directly.

Dependencies
------------
``pvlib`` — ``pip install pvlib``
``point_cloud_utils`` — already a project dependency
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_solar_exposure(
    viewpoints: np.ndarray,
    scene: dict,
    lat: float,
    lon: float,
    *,
    tz: str = "UTC",
    year: int = 2024,
    solar_freq: str = "1h",
    golden_freq: str = "5min",
    golden_hour_el: float = 6.0,
    dni: Optional[np.ndarray] = None,
    ray_near: float = 0.5,
    timestep_chunk: int = 200,
    n_workers: int = 0,
    intersector=None,
) -> pd.DataFrame:
    """Compute per-viewpoint solar and golden-hour exposure in a single pass.

    Merges the former ``compute_solar_exposure`` and
    ``compute_golden_hour_exposure`` into one function, sharing the BVH
    intersector and avoiding a redundant full-year pvlib computation.

    Golden-hour pvlib positions are computed efficiently: the hourly solar
    positions are used to screen which hours fall in the golden-hour band,
    then fine timestamps (``golden_freq``) are generated only for those
    windows (~8 k timestamps vs 105 k for the full year at ``"5min"``).

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
    solar_freq : str
        Time resolution for the main solar loop — ``"1h"`` (default).
    golden_freq : str
        Time resolution for the golden-hour loop — ``"5min"`` (default).
    golden_hour_el : float
        Sun elevation threshold (degrees) separating golden-hour from
        midday.  Timesteps with ``0 < el < golden_hour_el`` accumulate
        into ``golden_morning/afternoon_hours``; those with
        ``el >= golden_hour_el`` accumulate into ``annual/midday_solar_hours``.
    dni : ndarray (T,), optional
        Direct Normal Irradiance (W/m²) aligned to the hourly daylight
        timesteps.  When provided, ``annual_irradiance_kwh`` is added.
    ray_near : float
        Minimum hit distance (m) — avoids self-intersection.
    timestep_chunk : int
        Timesteps per worker call; controls peak memory per thread.
    n_workers : int
        Parallel threads.  ``0`` = min(cpu_count, 4).
    intersector : optional
        Pre-built ``pcu.RayMeshIntersector``.  Use ``cfg.build_intersector``
        to build once and share across all simulations.

    Returns
    -------
    DataFrame
        Indexed by ``vp_uuid``.  Columns:
        ``annual_solar_hours``, ``midday_solar_hours``,
        ``geometric_solar_hours``, ``shadow_fraction``,
        ``golden_morning_hours``, ``golden_afternoon_hours``,
        ``orientation``
        (plus ``annual_irradiance_kwh`` if *dni* is supplied).
    """
    import pvlib
    import point_cloud_utils as pcu

    # ------------------------------------------------------------------
    # Shared setup
    # ------------------------------------------------------------------
    vp_xyz = np.ascontiguousarray(viewpoints["vp"],   dtype=np.float32)
    norms  = np.ascontiguousarray(viewpoints["norm"], dtype=np.float32)
    sv     = np.ascontiguousarray(scene["v"], dtype=np.float32)
    sf     = np.ascontiguousarray(scene["f"], dtype=np.int32)
    n_vps  = len(vp_xyz)

    if intersector is None:
        intersector = pcu.RayMeshIntersector(sv, sf)

    workers = n_workers if n_workers > 0 else min(os.cpu_count() or 1, 4)

    # ------------------------------------------------------------------
    # Part 1: Main solar loop (hourly)
    # ------------------------------------------------------------------
    solpos   = sun_positions(lat, lon, tz=tz, year=year, freq=solar_freq)
    sun_d    = _solar_dirs(solpos)
    dt_solar = _timestep_hours(solar_freq)
    n_t      = len(solpos)

    el        = solpos["apparent_elevation"].values
    is_midday = el >= golden_hour_el   # above golden-hour band

    annual_count = np.zeros(n_vps, dtype=np.int32)
    midday_count = np.zeros(n_vps, dtype=np.int32)
    geo_count    = np.zeros(n_vps, dtype=np.int32)
    poa_accum    = np.zeros(n_vps, dtype=np.float64) if dni is not None else None

    chunks = [
        (t_start, min(t_start + timestep_chunk, n_t))
        for t_start in range(0, n_t, timestep_chunk)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _cast_shadow_chunk,
                intersector, vp_xyz, norms, sun_d[t_start:t_end], t_start, n_vps, ray_near,
            ): (t_start, t_end)
            for t_start, t_end in chunks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Solar casting", leave=False, position=1):
            t_start, facing_chunk, shadowed_chunk = future.result()
            t_end = t_start + shadowed_chunk.shape[1]
            direct_chunk = facing_chunk & ~shadowed_chunk

            geo_count    += facing_chunk.sum(axis=1)
            annual_count += direct_chunk.sum(axis=1)
            midday_count += direct_chunk[:, is_midday[t_start:t_end]].sum(axis=1)

            if poa_accum is not None:
                cos_chunk = (norms @ sun_d[t_start:t_end].T).clip(0)
                poa_accum += (direct_chunk * cos_chunk * dni[t_start:t_end]).sum(axis=1)

    annual_solar_hours    = annual_count * dt_solar
    midday_hours          = midday_count * dt_solar
    geometric_solar_hours = geo_count    * dt_solar
    shadow_fraction = np.where(
        geometric_solar_hours > 0,
        1.0 - annual_solar_hours / geometric_solar_hours,
        np.nan,
    )

    # ------------------------------------------------------------------
    # Part 2: Golden-hour loop (fine resolution, efficient)
    #
    # Screen golden-hour windows from the hourly positions already computed,
    # then call pvlib only for fine timestamps within those windows.
    # This reduces pvlib work from ~105k to ~8k timestamps.
    # ------------------------------------------------------------------
    dt_golden = _timestep_hours(golden_freq)
    golden_morning_count   = np.zeros(n_vps, dtype=np.int32)
    golden_afternoon_count = np.zeros(n_vps, dtype=np.int32)

    golden_hour_idx = solpos.index[el < golden_hour_el]   # hourly golden windows

    if len(golden_hour_idx) > 0:
        solar_offset = pd.tseries.frequencies.to_offset(solar_freq)

        # Fine timestamps covering only the screened windows
        all_nanos = np.unique(np.concatenate([
            pd.date_range(ts, ts + solar_offset, freq=golden_freq, inclusive="left").asi8
            for ts in golden_hour_idx
        ]))
        fine_idx = pd.DatetimeIndex(all_nanos, tz=golden_hour_idx.tz)

        solpos_g = pvlib.solarposition.get_solarposition(fine_idx, lat, lon)
        solpos_g = solpos_g[
            (solpos_g["apparent_elevation"] > 0) &
            (solpos_g["apparent_elevation"] < golden_hour_el)
        ][["apparent_elevation", "azimuth"]]

        if not solpos_g.empty:
            sun_d_g = _solar_dirs(solpos_g)
            n_t_g   = len(solpos_g)
            az_g    = solpos_g["azimuth"].values
            is_morning   = az_g < 180
            is_afternoon = az_g >= 180

            chunks_g = [
                (ts, min(ts + timestep_chunk, n_t_g))
                for ts in range(0, n_t_g, timestep_chunk)
            ]

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _cast_shadow_chunk,
                        intersector, vp_xyz, norms, sun_d_g[ts:te], ts, n_vps, ray_near,
                    ): (ts, te)
                    for ts, te in chunks_g
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Golden hour casting", leave=False, position=1):
                    t_start, facing_chunk, shadowed_chunk = future.result()
                    t_end = t_start + shadowed_chunk.shape[1]
                    direct_chunk = facing_chunk & ~shadowed_chunk

                    golden_morning_count   += direct_chunk[:, is_morning[t_start:t_end]].sum(axis=1)
                    golden_afternoon_count += direct_chunk[:, is_afternoon[t_start:t_end]].sum(axis=1)

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "annual_solar_hours":    annual_solar_hours,
        "midday_solar_hours":    midday_hours,
        "geometric_solar_hours": geometric_solar_hours,
        "shadow_fraction":       shadow_fraction,
        "golden_morning_hours":  golden_morning_count * dt_golden,
        "golden_afternoon_hours": golden_afternoon_count * dt_golden,
        "orientation":           facade_orientation(viewpoints).values,
    }, index=viewpoints["vp_uuid"])
    result.index.name = "vp_uuid"

    if poa_accum is not None:
        result["annual_irradiance_kwh"] = poa_accum * dt_solar / 1000.0

    return result


def compute_golden_hour_exposure(
    viewpoints: np.ndarray,
    scene: dict,
    lat: float,
    lon: float,
    **kwargs,
) -> pd.DataFrame:
    """Deprecated: use :func:`compute_solar_exposure` which returns all metrics.

    Thin wrapper kept for backward compatibility.
    """
    result = compute_solar_exposure(viewpoints, scene, lat, lon, **kwargs)
    return result[["golden_morning_hours", "golden_afternoon_hours"]]


def sun_positions(
    lat: float,
    lon: float,
    *,
    tz: str = "UTC",
    year: int = 2024,
    freq: str = "1h",
) -> pd.DataFrame:
    """Solar positions for a full year, filtered to daylight hours only.

    Parameters
    ----------
    lat, lon : float
        Site coordinates in WGS84 degrees.
    tz : str
        IANA timezone string.
    year : int
        Calendar year.
    freq : str
        Time resolution (``"1h"`` or ``"30min"``).

    Returns
    -------
    DataFrame
        Columns ``apparent_elevation`` (degrees above horizon) and
        ``azimuth`` (degrees, clockwise from North).
        Index is a ``DatetimeTZDtype`` with the requested timezone.
        Only rows where the sun is above the horizon are included.
    """
    try:
        import pvlib
    except ImportError as exc:
        raise ImportError(
            "pvlib is required for solar analysis — pip install pvlib"
        ) from exc

    times = pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01",
        freq=freq, tz=tz, inclusive="left",
    )
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    return solpos[solpos["apparent_elevation"] > 0][["apparent_elevation", "azimuth"]]


def facade_orientation(viewpoints: np.ndarray) -> pd.Series:
    """Classify each facade viewpoint into a cardinal/intercardinal direction.

    Uses the XY components of the facade normal in a projected CRS
    (East = +X, North = +Y).

    Returns
    -------
    Series of str (``"N"``, ``"NE"``, ``"E"``, ``"SE"``, ``"S"``, ``"SW"``,
    ``"W"``, ``"NW"``) indexed by ``vp_uuid``.
    """
    nx = viewpoints["norm"][:, 0]   # East component
    ny = viewpoints["norm"][:, 1]   # North component

    # Clockwise azimuth from North: 0° = N, 90° = E, 180° = S, 270° = W
    az_deg = np.degrees(np.arctan2(nx, ny)) % 360

    # Bin edges at ±22.5° around each of the 8 compass points
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    edges  = np.arange(22.5, 360, 45)   # [22.5, 67.5, ..., 337.5]

    idx        = np.searchsorted(edges, az_deg)   # 0–7, 8 wraps to N
    idx[idx == 8] = 0

    return pd.Series(
        [labels[i] for i in idx],
        index=viewpoints["vp_uuid"],
        name="orientation",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast_shadow_chunk(
    intersector,
    vp_xyz: np.ndarray,
    norms: np.ndarray,
    sun_d_chunk: np.ndarray,
    t_start: int,
    n_vps: int,
    ray_near: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Cast shadow rays for one chunk of timesteps (runs in a worker thread).

    Returns ``(t_start, facing, shadowed)`` where both outputs have shape
    ``(V, C)``.  Only rays where the facade faces the sun are cast — this
    halves the intersection workload and avoids large repeat/tile allocations.
    """
    C = sun_d_chunk.shape[0]

    # Facing mask: (V, C) — True where sun is in front of the facade
    cos_i  = norms @ sun_d_chunk.T   # (V, C)  dot product
    facing = cos_i > 0               # (V, C)  bool

    # Indices of (viewpoint, timestep) pairs that face the sun
    v_idx, t_idx = np.where(facing)  # (K,) each — avoids repeat/tile

    shadowed = np.zeros((n_vps, C), dtype=bool)
    if v_idx.size > 0:
        fid, _, t_hit = intersector.intersect_rays(
            np.ascontiguousarray(vp_xyz[v_idx]),       # (K, 3)
            np.ascontiguousarray(sun_d_chunk[t_idx]),  # (K, 3)
        )
        shadowed[v_idx, t_idx] = (fid >= 0) & (t_hit > ray_near)

    return t_start, facing, shadowed

def _solar_dirs(solpos: pd.DataFrame) -> np.ndarray:
    """Convert pvlib solar position DataFrame to 3D unit vectors.

    In a UTM / projected CRS (East=+X, North=+Y, Up=+Z) with pvlib's
    North-clockwise azimuth convention::

        sx = sin(az) * cos(el)   # East
        sy = cos(az) * cos(el)   # North
        sz = sin(el)             # Up

    Returns
    -------
    ndarray (T, 3) float32 — unit vectors pointing FROM surface TOWARD sun.
    """
    el = np.deg2rad(solpos["apparent_elevation"].values)
    az = np.deg2rad(solpos["azimuth"].values)

    return np.column_stack([
        np.sin(az) * np.cos(el),   # East  (+X)
        np.cos(az) * np.cos(el),   # North (+Y)
        np.sin(el),                 # Up    (+Z)
    ]).astype(np.float32)


def _timestep_hours(freq: str) -> float:
    """Convert a pandas offset alias to fractional hours."""
    offset = pd.tseries.frequencies.to_offset(freq)
    return offset.nanos / 3_600_000_000_000


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_chunk_sizes(
    viewpoints: np.ndarray,
    scene: dict,
    lat: float,
    lon: float,
    chunk_sizes: list[int],
    *,
    tz: str = "UTC",
    year: int = 2024,
    freq: str = "1h",
    ray_near: float = 0.5,
) -> pd.DataFrame:
    """Time the shadow-casting loop for each value in *chunk_sizes*.

    pvlib sun positions and the BVH are built **once** and shared across
    all runs, so results isolate the effect of chunk size on intersection
    throughput.

    Parameters
    ----------
    chunk_sizes : list[int]
        Timestep chunk sizes to benchmark, e.g. ``[1, 10, 50, 200, 500]``.
        Pass ``[n_t]`` (total daylight hours) to benchmark a single batch.

    Returns
    -------
    DataFrame
        Columns: ``chunk_size``, ``n_chunks``, ``rays_per_chunk``,
        ``t_alloc_s``, ``t_intersect_s``, ``t_mask_s``, ``total_s``.
    """
    import time
    import point_cloud_utils as pcu

    solpos = sun_positions(lat, lon, tz=tz, year=year, freq=freq)
    sun_d  = _solar_dirs(solpos)
    n_t    = len(solpos)

    vp_xyz = np.ascontiguousarray(viewpoints["vp"],   dtype=np.float32)
    n_vps  = len(vp_xyz)

    sv = np.ascontiguousarray(scene["v"], dtype=np.float32)
    sf = np.ascontiguousarray(scene["f"], dtype=np.int32)
    intersector = pcu.RayMeshIntersector(sv, sf)

    print(f"{'chunk':>8}  {'n_chunks':>8}  {'rays/chunk':>12}  "
          f"{'alloc':>8}  {'intersect':>10}  {'mask':>8}  {'total':>8}")
    print("-" * 72)

    rows = []
    for chunk_size in chunk_sizes:
        t_alloc = t_intersect = t_mask = 0.0

        for t_start in range(0, n_t, chunk_size):
            t_end     = min(t_start + chunk_size, n_t)
            chunk_sun = sun_d[t_start:t_end]
            C         = chunk_sun.shape[0]

            _t = time.perf_counter()
            origins_batch = np.repeat(vp_xyz,   C,      axis=0)
            dirs_batch    = np.tile(chunk_sun, (n_vps, 1))
            t_alloc += time.perf_counter() - _t

            _t = time.perf_counter()
            fid, _, t_hit = intersector.intersect_rays(
                np.ascontiguousarray(origins_batch),
                np.ascontiguousarray(dirs_batch),
            )
            t_intersect += time.perf_counter() - _t

            _t = time.perf_counter()
            hit_mask = (fid >= 0) & (t_hit > ray_near)
            _ = hit_mask.reshape(n_vps, C)   # reshape only, no write-back needed
            t_mask += time.perf_counter() - _t

        n_chunks      = (n_t + chunk_size - 1) // chunk_size
        rays_per_chunk = n_vps * chunk_size
        total         = t_alloc + t_intersect + t_mask

        print(f"{chunk_size:>8}  {n_chunks:>8}  {rays_per_chunk:>12,}  "
              f"{t_alloc:>7.2f}s  {t_intersect:>9.2f}s  {t_mask:>7.2f}s  {total:>7.2f}s")

        rows.append({
            "chunk_size":     chunk_size,
            "n_chunks":       n_chunks,
            "rays_per_chunk": rays_per_chunk,
            "t_alloc_s":      round(t_alloc,     3),
            "t_intersect_s":  round(t_intersect, 3),
            "t_mask_s":       round(t_mask,       3),
            "total_s":        round(total,         3),
        })

    return pd.DataFrame(rows)


def aggregate_solar_to_buildings(
    solar: pd.DataFrame,
    viewpoints: np.ndarray,
    *,
    min_hours: float = 50.0,
) -> pd.DataFrame:
    """Aggregate per-viewpoint solar results to building level.

    Parameters
    ----------
    solar : DataFrame
        Output of :func:`compute_solar_exposure`, indexed by ``vp_uuid``.
    viewpoints : np.ndarray
        Structured array with ``vp_uuid`` and ``osmid`` fields.
    min_hours : float
        Annual solar hours threshold below which a building is classified
        as ``"Neither"`` in the TOD category.

    Returns
    -------
    DataFrame
        Indexed by ``osmid``.  Numeric columns are mean-aggregated across
        all facade viewpoints.  Adds ``tod_ratio`` (morning fraction) and
        ``tod_category`` (``"Sunrise"`` / ``"Sunset"`` / ``"Both"`` /
        ``"Neither"``).
    """
    vp_meta = pd.DataFrame(
        {"osmid": viewpoints["osmid"]}, index=viewpoints["vp_uuid"]
    )
    merged = solar.join(vp_meta)

    numeric_cols = solar.select_dtypes("number").columns.tolist()
    bldg = merged.groupby("osmid")[numeric_cols].mean()

    # Golden-hour TOD ratio — only if golden hour columns were joined in
    has_golden = (
        "golden_morning_hours" in bldg.columns
        and "golden_afternoon_hours" in bldg.columns
    )
    if has_golden:
        golden_total = bldg["golden_morning_hours"] + bldg["golden_afternoon_hours"]
        bldg["tod_ratio"] = np.where(
            golden_total > 0, bldg["golden_morning_hours"] / golden_total, np.nan
        )

        def _classify(row):
            if row["annual_solar_hours"] < min_hours:
                return "Neither"
            r = row["tod_ratio"]
            if np.isnan(r):
                return "Neither"
            if r > 0.6:
                return "Sunrise"
            if r < 0.4:
                return "Sunset"
            return "Both"

        bldg["tod_category"] = bldg.apply(_classify, axis=1)

    return bldg
