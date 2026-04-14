"""Generate viewpoints from an arbitrary GeoDataFrame of point locations.

Use this when you have pre-defined sample points (e.g. pedestrian survey
locations, census centroids, transit stops) rather than an OSM street network.

Typical usage
-------------
::

    import geopandas as gpd
    from openviewcore.viewpoints import generate_custom_viewpoints

    pts = gpd.read_file("my_locations.gpkg")   # any point GeoDataFrame
    vps = generate_custom_viewpoints(
        gdf        = pts,
        dtm_file   = "site/site_dtm.tif",
        scene_crs  = "EPSG:32611",
        eye_height = 1.8288,
        norm       = (0.0, 0.0, 1.0),   # straight up — hemisphere rays
        id_col     = "stop_id",          # optional: use a column as osmid
    )

The output uses :data:`~openviewcore.viewpoints.facade.VP_DTYPE` so all
downstream simulation functions (solar, LULC, ray casting) work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np

from .facade import VP_DTYPE


def generate_custom_viewpoints(
    gdf: gpd.GeoDataFrame,
    dtm_file: Union[str, Path],
    scene_crs: str,
    *,
    eye_height: float = 1.8288,
    norm: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    id_col: Optional[str] = None,
) -> np.ndarray:
    """Build a VP_DTYPE structured array from an arbitrary point GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Point geometries in any CRS.  Non-point geometries are dropped.
    dtm_file : path-like
        GeoTIFF bare-earth DTM — used to sample terrain elevation at each
        point.  Points that fall outside the DTM extent are dropped.
    scene_crs : str
        Target metric CRS (e.g. ``"EPSG:32611"``).  Points are reprojected
        to this CRS before sampling the DTM.
    eye_height : float
        Height above ground in metres.  Added to the DTM elevation.
    norm : (float, float, float)
        Unit normal vector for every viewpoint.  Default ``(0,0,1)`` = looking
        straight up (hemisphere rays).  Pass a horizontal vector for facade /
        street-style simulations, or supply per-row norms via the
        ``norm_col`` parameter (not yet implemented — subclass if needed).
    id_col : str, optional
        Column in ``gdf`` to use as ``osmid`` (integer).  Falls back to the
        row index if not provided.

    Returns
    -------
    np.ndarray
        Structured array with dtype :data:`VP_DTYPE`.  Points outside the
        DTM extent (NaN elevation) are excluded.
    """
    import rioxarray
    from scipy.interpolate import RegularGridInterpolator

    # ── Load DTM ──────────────────────────────────────────────────────────────
    da = rioxarray.open_rasterio(dtm_file).squeeze("band", drop=True)
    xs_grid = da.x.values
    ys_grid = da.y.values
    zs_grid = da.values.astype(np.float64)

    if ys_grid[0] > ys_grid[-1]:          # ensure ascending y axis
        ys_grid = ys_grid[::-1]
        zs_grid = zs_grid[::-1]

    dtm = RegularGridInterpolator(
        (ys_grid, xs_grid), zs_grid,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # ── Reproject to scene CRS ────────────────────────────────────────────────
    pts = gdf[gdf.geometry.geom_type == "Point"].copy()
    if len(pts) < len(gdf):
        dropped = len(gdf) - len(pts)
        print(f"  [custom_vps] dropped {dropped} non-point geometries")

    pts_proj = pts.to_crs(scene_crs)
    xs_pt = pts_proj.geometry.x.values
    ys_pt = pts_proj.geometry.y.values

    # ── Sample terrain elevation ──────────────────────────────────────────────
    terrain_z = dtm((ys_pt, xs_pt))
    valid     = np.isfinite(terrain_z)
    print(f"  [custom_vps] {valid.sum():,} / {len(valid):,} points within DTM extent")

    xs_vp = xs_pt[valid]
    ys_vp = ys_pt[valid]
    zs_vp = terrain_z[valid] + eye_height

    # ── osmid ─────────────────────────────────────────────────────────────────
    if id_col and id_col in pts_proj.columns:
        osmid = pts_proj[id_col].values[valid].astype(np.int64)
    else:
        osmid = pts_proj.index.values[valid].astype(np.int64)

    # ── Assemble VP_DTYPE array ───────────────────────────────────────────────
    norm_arr = np.asarray(norm, dtype=np.float64)
    mag      = np.linalg.norm(norm_arr)
    if mag > 0:
        norm_arr /= mag

    n   = int(valid.sum())
    arr = np.zeros(n, dtype=VP_DTYPE)
    arr["vp_uuid"]     = np.arange(n, dtype=np.int64)
    arr["osmid"]       = osmid
    arr["vp"][:, 0]    = xs_vp
    arr["vp"][:, 1]    = ys_vp
    arr["vp"][:, 2]    = zs_vp
    arr["norm"]        = norm_arr
    arr["segment_len"] = 0.0

    return arr
