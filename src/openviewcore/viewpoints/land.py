"""Land-based viewpoints: regular grid across AOI, 8 compass directions.

Designed for open or sparsely built landscapes where facade viewpoints are
not appropriate.  Generates a uniform grid of observer positions at two
storey heights facing each of the 8 cardinal / intercardinal compass
directions.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from typing import Optional, Sequence

from .facade import VP_DTYPE


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GRID_SPACING = 50.0           # metres between grid points
DEFAULT_HEIGHTS      = [3.0, 6.0]    # 1-storey and 2-storey above terrain

# 8 compass directions as unit vectors (dx, dy, dz)
_s2 = 1.0 / 2 ** 0.5
COMPASS_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "N":  ( 0.0,  1.0,  0.0),
    "NE": ( _s2,  _s2,  0.0),
    "E":  ( 1.0,  0.0,  0.0),
    "SE": ( _s2, -_s2,  0.0),
    "S":  ( 0.0, -1.0,  0.0),
    "SW": (-_s2, -_s2,  0.0),
    "W":  (-1.0,  0.0,  0.0),
    "NW": (-_s2,  _s2,  0.0),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# NLCD classes considered water — filtered out by default
WATER_CLASSES = frozenset([11, 12])   # Open Water, Perennial Ice/Snow


def generate_land_viewpoints(
    aoi: gpd.GeoDataFrame,
    dem: xr.DataArray,
    *,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    heights: Sequence[float] = DEFAULT_HEIGHTS,
    directions: Optional[dict[str, tuple[float, float, float]]] = None,
    landcover: Optional[xr.DataArray] = None,
    water_classes: frozenset = WATER_CLASSES,
) -> np.ndarray:
    """Generate a regular grid of observer viewpoints across *aoi*.

    Parameters
    ----------
    aoi : GeoDataFrame
        Area of interest in a projected CRS (metres).
    dem : xr.DataArray
        Elevation raster aligned to the same CRS as *aoi*.
    grid_spacing : float
        Distance in metres between adjacent grid points.
    heights : sequence of float
        Observer heights above terrain surface (metres).
        Default ``[3.0, 6.0]`` corresponds to 1- and 2-storey eye level.
    directions : dict, optional
        Mapping of label → (dx, dy, dz) unit normal vector.
        Defaults to all 8 compass directions.
    landcover : xr.DataArray, optional
        NLCD (or equivalent) land-cover raster.  When provided, grid points
        whose nearest pixel belongs to *water_classes* are removed before
        viewpoints are created.  The raster may be in a different CRS —
        points are reprojected automatically before sampling.
    water_classes : frozenset of int
        Land-cover class codes treated as water.
        Default ``{11, 12}`` = NLCD Open Water + Perennial Ice/Snow.

    Returns
    -------
    np.ndarray
        Structured array with dtype :data:`~openviewcore.viewpoints.facade.VP_DTYPE`.
        The ``osmid`` field is set to the integer grid-cell index (not a
        building id).  ``segment_len`` stores the grid spacing.
    """
    if directions is None:
        directions = COMPASS_DIRECTIONS

    aoi_union = aoi.union_all()
    minx, miny, maxx, maxy = aoi.total_bounds

    # ── Regular grid over bounding box ────────────────────────────────────
    xs = np.arange(minx, maxx, grid_spacing)
    ys = np.arange(miny, maxy, grid_spacing)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    # Keep only points inside the AOI polygon
    inside = np.array([aoi_union.contains(Point(x, y)) for x, y in grid_pts])
    grid_pts = grid_pts[inside]
    n_cells  = len(grid_pts)

    if n_cells == 0:
        return np.empty(0, dtype=VP_DTYPE)

    # ── Sample terrain elevation at each grid point ───────────────────────
    x_da = xr.DataArray(grid_pts[:, 0], dims="points")
    y_da = xr.DataArray(grid_pts[:, 1], dims="points")
    elev = dem.sel(x=x_da, y=y_da, method="nearest").values.flatten()

    # Drop grid points over NaN terrain
    valid    = np.isfinite(elev)
    grid_pts = grid_pts[valid]
    elev     = elev[valid]
    n_cells  = len(grid_pts)

    # ── Land-cover water mask ─────────────────────────────────────────────
    if landcover is not None and n_cells > 0:
        lc_crs  = landcover.rio.crs
        aoi_crs = aoi.crs

        if lc_crs is not None and lc_crs != aoi_crs:
            # Reproject grid points into the landcover CRS for sampling
            pts_gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]),
                crs=aoi_crs,
            ).to_crs(lc_crs)
            lc_x = xr.DataArray(pts_gdf.geometry.x.values, dims="points")
            lc_y = xr.DataArray(pts_gdf.geometry.y.values, dims="points")
        else:
            lc_x = xr.DataArray(grid_pts[:, 0], dims="points")
            lc_y = xr.DataArray(grid_pts[:, 1], dims="points")

        lc_vals = landcover.sel(x=lc_x, y=lc_y, method="nearest").values.flatten()

        on_land  = ~np.isin(lc_vals, list(water_classes))
        n_before = n_cells
        grid_pts = grid_pts[on_land]
        elev     = elev[on_land]
        n_cells  = len(grid_pts)
        print(f"  Land mask: removed {n_before - n_cells} water points "
              f"({(n_before - n_cells) / max(n_before, 1):.1%}), "
              f"{n_cells} land points remain")

    # ── Expand: n_cells × n_heights × n_directions ────────────────────────
    n_h   = len(heights)
    n_dir = len(directions)
    dir_labels  = list(directions.keys())
    dir_vectors = np.array(list(directions.values()), dtype=np.float64)  # (n_dir, 3)

    records = []
    cell_idx = 0
    for i in range(n_cells):
        x, y, z0 = grid_pts[i, 0], grid_pts[i, 1], float(elev[i])
        for h in heights:
            z = z0 + h
            for d_vec in dir_vectors:
                records.append((
                    0,           # vp_uuid placeholder
                    cell_idx,    # osmid = grid cell index
                    (x, y, z),
                    tuple(d_vec),
                    grid_spacing,
                ))
        cell_idx += 1

    arr = np.array(records, dtype=VP_DTYPE)
    arr["vp_uuid"] = np.arange(len(arr), dtype=np.int64)
    return arr
