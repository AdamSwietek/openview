"""Drape building footprints onto a DEM — add elevation and Z coordinates."""

from __future__ import annotations

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


def elevate_footprints(
    buildings: gpd.GeoDataFrame,
    dem: xr.DataArray,
    HEIGHT_OFFSET: float = 0.0,
) -> gpd.GeoDataFrame:
    """Sample DEM elevation at each building centroid and add Z to geometry.

    Parameters
    ----------
    buildings : GeoDataFrame
        Must contain ``geometry`` (2-D polygons).
    dem : xr.DataArray
        Elevation raster in the same CRS as *buildings*.

    Returns
    -------
    GeoDataFrame
        With new ``elevation`` column and 3-D polygon geometries.
        Buildings falling in NaN elevation zones are dropped.
    """
    buildings = buildings.copy()

    centroids = buildings.geometry.centroid
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()

    # Bilinear interpolation — matches RegularGridInterpolator used in extrude_buildings
    from scipy.interpolate import RegularGridInterpolator
    dem_s = dem.sortby("y")
    rgi = RegularGridInterpolator(
        (dem_s["y"].values, dem_s["x"].values), dem_s.values,
        method="linear", bounds_error=False, fill_value=None,
    )
    elevations = rgi(np.column_stack([ys, xs]))

    buildings["elevation"] = elevations + HEIGHT_OFFSET

    buildings["geometry"] = [
        _add_z(geom, z) for geom, z in zip(buildings.geometry, elevations)
    ]

    # Drop buildings in NaN elevation zones
    buildings = buildings[buildings["elevation"].notna()].copy()

    return buildings


def _add_z(geom, z_value: float):
    """Add a constant Z coordinate to a Polygon or MultiPolygon."""
    if np.isnan(z_value):
        return geom
    if geom.geom_type == "Polygon":
        return Polygon([(x, y, z_value) for x, y in geom.exterior.coords])
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon(
            [Polygon([(x, y, z_value) for x, y in p.exterior.coords])
             for p in geom.geoms]
        )
    return geom
