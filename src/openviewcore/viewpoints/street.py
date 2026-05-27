"""Generate street-level viewpoints from the OpenStreetMap road network.

Each sample location produces **two** viewpoints — one for each direction of
travel — with the ``norm`` vector parallel to the road centreline and Z at
``eye_height`` above the local terrain.

Typical usage
-------------
::

    streets = download_streets(aoi_gdf, network_type="drive")
    vps = generate_street_viewpoints(
        streets,
        eye_height=1.83,          # 6 ft — pedestrian / car eye level
        spacing=None,             # one point per street edge (default)
        terrain_mesh=scene["terrain"],
    )
    # spacing=7.62  →  one point every 25 ft (≈ 7.62 m)

The output is a structured array with :data:`VP_DTYPE` — the same dtype used
by :func:`~openviewcore.viewpoints.facade.generate_viewpoints`, so all
downstream simulation functions (solar, LULC, glare) work unchanged.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString

from ..config import DEFAULT_PROJECT_CRS, WGS84
from .facade import VP_DTYPE

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_PLACES", "threads")


# ---------------------------------------------------------------------------
# Unit constants
# ---------------------------------------------------------------------------
FT_TO_M: float = 0.3048

DEFAULT_EYE_HEIGHT: float = 6.0 * FT_TO_M   # 1.8288 m ≈ pedestrian / car
DEFAULT_SPACING_M:  float = 25.0 * FT_TO_M  # 7.620 m — reference spacing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_streets(
    gdf: gpd.GeoDataFrame,
    *,
    crs: str = DEFAULT_PROJECT_CRS,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch OSM street network edges inside *gdf* and project to *crs*.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon (any CRS) defining the area of interest.
    crs : str
        Target projected CRS for the output GeoDataFrame.
    network_type : str
        osmnx network type: ``"drive"`` (default), ``"walk"``, ``"bike"``,
        ``"all"``.
    simplify : bool
        Simplify the graph topology (merge non-intersection nodes).  ``True``
        produces longer but cleaner edges aligned with the road centreline.
    retain_all : bool
        If ``True``, keep all edges including those not strongly connected.
        Useful for pedestrian networks where dead-ends matter.

    Returns
    -------
    GeoDataFrame
        One row per directed edge.  Columns guaranteed to be present:
        ``osmid`` (int64), ``geometry`` (LineString), ``length`` (m),
        ``name``, ``highway``.  Index is reset (integer).
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)

    geom = gdf.geometry.union_all()
    G = ox.graph_from_polygon(
        geom,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
    )
    _, edges = ox.graph_to_gdfs(G)

    # Keep only useful columns (some may not exist for all network types)
    keep = [c for c in ["osmid", "geometry", "length", "name", "highway"] if c in edges.columns]
    edges = edges[keep].copy().reset_index()

    # osmid can be a list when simplified edges aggregate multiple OSM ways —
    # take the first element so it fits in the int64 structured-array field.
    edges["osmid"] = edges["osmid"].apply(
        lambda x: int(x[0]) if isinstance(x, list) else int(x)
    )

    return edges.to_crs(crs)


def generate_street_viewpoints(
    streets: gpd.GeoDataFrame,
    *,
    crs: str = DEFAULT_PROJECT_CRS,
    eye_height: float = DEFAULT_EYE_HEIGHT,
    spacing: Optional[float] = None,
    terrain_mesh: Optional[dict] = None,
    dem=None,
    min_segment_len: float = 1.0,
) -> np.ndarray:
    """Create street-level viewpoints from a street network GeoDataFrame.

    Each sample position yields **two** rows in the output — one viewpoint
    facing each direction of travel — so that simulations capture both sides
    of every street.

    Parameters
    ----------
    streets : GeoDataFrame
        Output of :func:`download_streets`, or any edge GeoDataFrame with
        ``LineString`` geometries, an ``osmid`` column, and a metric CRS.
    eye_height : float
        Height above terrain in **metres**.  Default is ``6 * 0.3048`` m
        (6 ft), appropriate for both pedestrian and driver eye levels.
    spacing : float, optional
        Sample interval in **metres** along each edge.

        * ``None`` (default) — one viewpoint pair at the midpoint of every
          edge.
        * ``float`` (e.g. ``7.62`` for 25 ft) — viewpoint pairs at
          ``spacing/2``, ``3·spacing/2``, … up to the end of each edge,
          giving approximately one pair every *spacing* metres.

        Short edges that produce no interior samples still receive the
        midpoint pair.
    terrain_mesh : dict, optional
        ``{'v': (N,3) float32, 'f': (M,3) int32}`` terrain mesh.  When
        supplied, the terrain elevation at each XY position is resolved by
        downward ray casting (most accurate).
    dem : xr.DataArray, optional
        DEM raster in the **same CRS** as *streets*.  Used when
        *terrain_mesh* is not available.  Elevation is sampled by
        nearest-neighbour lookup.
    min_segment_len : float
        Skip edges shorter than this length in metres (default 1 m).

    Returns
    -------
    np.ndarray
        Structured array with dtype :data:`~openviewcore.viewpoints.facade.VP_DTYPE`::

            vp_uuid     int64      unique viewpoint ID
            osmid       int64      OSM way ID of the parent street edge
            vp          float64[3] (X, Y, Z) position in the projected CRS
            norm        float64[3] unit vector along direction of travel
            segment_len float64    length of the parent edge (m)

        Consecutive pairs of rows share the same XY but have opposite norms.
    """
    # ------------------------------------------------------------------
    # CRS guard: street coordinates must be in a projected (metric) CRS
    # so that XY values match the building viewpoints and terrain mesh.
    # ------------------------------------------------------------------
    if streets.crs is None:
        raise ValueError(
            "streets GeoDataFrame has no CRS. "
            "Pass the output of download_streets(), or call "
            "streets.set_crs(...) / streets.to_crs(...) first."
        )
    if streets.crs.is_geographic:
        streets = streets.to_crs(crs)
        print(f"  generate_street_viewpoints: reprojected streets from "
              f"geographic CRS to {crs}.")

    records: list[dict] = []

    osmid_col = "osmid" if "osmid" in streets.columns else None

    for row in streets.itertuples(index=False):
        geom = row.geometry
        if not isinstance(geom, LineString) or geom.length < min_segment_len:
            continue

        osmid = int(getattr(row, "osmid", 0)) if osmid_col else 0
        edge_len = geom.length

        for x, y, nx, ny in _sample_line(geom, spacing, min_segment_len):
            # Two viewpoints: forward and backward along the road
            for sign in (1.0, -1.0):
                records.append({
                    "vp_uuid":     0,
                    "osmid":       osmid,
                    "vp":          [x, y, 0.0],               # Z filled below
                    "norm":        [sign * nx, sign * ny, 0.0],
                    "segment_len": edge_len,
                    "patch_area":  0.0,
                })

    if not records:
        return np.empty(0, dtype=VP_DTYPE)

    arr = np.array(
        [(d["vp_uuid"], d["osmid"], d["vp"], d["norm"], d["segment_len"], d["patch_area"])
         for d in records],
        dtype=VP_DTYPE,
    )
    arr["vp_uuid"] = np.arange(len(arr), dtype=np.int64)

    # Assign terrain elevation + eye_height
    arr = _assign_elevation(arr, eye_height, terrain_mesh, dem)

    return arr


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_line(
    line: LineString,
    spacing: Optional[float],
    min_len: float,
) -> list[tuple[float, float, float, float]]:
    """Return ``(x, y, nx, ny)`` sample points along *line*.

    *nx*, *ny* is the unit tangent pointing in the digitised direction of the
    edge.  Callers negate it to get the opposite direction of travel.

    If *spacing* is ``None``: one sample at the midpoint.
    If *spacing* is a float: samples at ``spacing/2``, ``3·spacing/2``, …
    In either case at least one sample is always returned (even for edges
    shorter than *spacing*).
    """
    if line.length < min_len:
        return []

    if spacing is None:
        pt = line.interpolate(0.5, normalized=True)
        tang = _tangent_at(line, line.length / 2.0)
        return [(pt.x, pt.y, tang[0], tang[1])]

    samples: list[tuple[float, float, float, float]] = []
    length = line.length
    d = spacing / 2.0
    while d <= length:
        pt = line.interpolate(d)
        tang = _tangent_at(line, d)
        samples.append((pt.x, pt.y, tang[0], tang[1]))
        d += spacing

    # Guarantee at least one sample for edges shorter than one spacing interval
    if not samples:
        pt = line.interpolate(0.5, normalized=True)
        tang = _tangent_at(line, line.length / 2.0)
        samples.append((pt.x, pt.y, tang[0], tang[1]))

    return samples


def _tangent_at(line: LineString, d: float) -> tuple[float, float]:
    """Unit tangent vector of *line* at arc-length *d* (metres).

    Finds the edge segment that contains the arc-length position and returns
    its direction.  Handles degenerate zero-length segments gracefully.
    """
    coords = list(line.coords)
    accumulated = 0.0

    for i in range(len(coords) - 1):
        x0, y0 = coords[i][0],     coords[i][1]
        x1, y1 = coords[i + 1][0], coords[i + 1][1]
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if accumulated + seg_len >= d - 1e-9:
            dx, dy = x1 - x0, y1 - y0
            mag = math.sqrt(dx * dx + dy * dy)
            return (dx / mag, dy / mag) if mag > 1e-12 else (1.0, 0.0)

        accumulated += seg_len

    # d is past the end of the line — use the last segment's direction
    x0, y0 = coords[-2][0], coords[-2][1]
    x1, y1 = coords[-1][0], coords[-1][1]
    dx, dy = x1 - x0, y1 - y0
    mag = math.sqrt(dx * dx + dy * dy)
    return (dx / mag, dy / mag) if mag > 1e-12 else (1.0, 0.0)


# ---------------------------------------------------------------------------
# Terrain elevation helpers
# ---------------------------------------------------------------------------

def _assign_elevation(
    arr: np.ndarray,
    eye_height: float,
    terrain_mesh: Optional[dict],
    dem,
) -> np.ndarray:
    """Fill the Z coordinate of *arr* from terrain + *eye_height*.

    Priority: terrain_mesh → dem → flat (Z = eye_height, terrain = 0).
    """
    if terrain_mesh is not None:
        return _elevation_from_mesh(arr, eye_height, terrain_mesh)
    if dem is not None:
        return _elevation_from_dem(arr, eye_height, dem)

    # Flat terrain fallback
    arr["vp"][:, 2] = eye_height
    return arr


def _elevation_from_mesh(
    arr: np.ndarray,
    eye_height: float,
    terrain_mesh: dict,
) -> np.ndarray:
    """Sample terrain Z by casting downward rays through the terrain mesh.

    Rays are fired from well above the terrain straight down; the hit distance
    gives the terrain Z at each XY position.  Points that miss the mesh
    (e.g. at the edge of the DEM coverage) fall back to Z = ``eye_height``.
    """
    import point_cloud_utils as pcu

    v = np.ascontiguousarray(terrain_mesh["v"], dtype=np.float32)
    f = np.ascontiguousarray(terrain_mesh["f"], dtype=np.int32)

    high_z = float(v[:, 2].max()) + 200.0   # well above any terrain point

    n = len(arr)
    origins = np.empty((n, 3), dtype=np.float32)
    origins[:, 0] = arr["vp"][:, 0].astype(np.float32)
    origins[:, 1] = arr["vp"][:, 1].astype(np.float32)
    origins[:, 2] = high_z

    down = np.tile(
        np.array([0.0, 0.0, -1.0], dtype=np.float32), (n, 1)
    )

    fid, _, t_hit = pcu.ray_mesh_intersection(v, f, origins, down)

    # Hit point Z = origin_z - t  (direction is -Z, so subtract t_hit)
    hit_mask   = fid >= 0
    terrain_z  = np.where(hit_mask, high_z - t_hit, 0.0)
    arr["vp"][:, 2] = terrain_z + eye_height

    missed = int((~hit_mask).sum())
    if missed:
        print(f"  Street viewpoints: {missed} points missed terrain mesh "
              f"(falling back to eye_height = {eye_height:.2f} m).")

    return arr


def _elevation_from_dem(
    arr: np.ndarray,
    eye_height: float,
    dem,
) -> np.ndarray:
    """Sample DEM elevation at each viewpoint XY (nearest-neighbour)."""
    import xarray as xr

    xs = arr["vp"][:, 0]
    ys = arr["vp"][:, 1]

    x_da = xr.DataArray(xs, dims="points")
    y_da = xr.DataArray(ys, dims="points")
    elevations = dem.sel(x=x_da, y=y_da, method="nearest").values.flatten()

    arr["vp"][:, 2] = np.nan_to_num(elevations, nan=0.0) + eye_height
    return arr
