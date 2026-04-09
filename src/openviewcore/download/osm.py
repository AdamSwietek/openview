"""Download building footprints from OpenStreetMap."""

from __future__ import annotations

import geopandas as gpd
import osmnx as ox
import pandas as pd

from ..config import WGS84, DEFAULT_PROJECT_CRS


def download_buildings(
    gdf: gpd.GeoDataFrame,
    *,
    crs: str = DEFAULT_PROJECT_CRS,
    meters_per_level: float = 3.0,
) -> gpd.GeoDataFrame:
    """Fetch OSM building footprints and estimate missing heights.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygon defining the download region (typically the visibility region).
    crs : str
        Target projected CRS for the output.
    meters_per_level : float
        Assumed storey height when only ``building:levels`` is available.

    Returns
    -------
    GeoDataFrame
        Buildings with ``height`` (meters), projected to *crs*.
        Points and MultiPolygons are excluded.
    """
    # osmnx requires WGS 84
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)

    geom = gdf.geometry.union_all()
    buildings = ox.features_from_polygon(geom, tags={"building": True})

    buildings = _fill_heights(buildings, meters_per_level)
    buildings = _filter_geometry(buildings)
    buildings = _clean_geometry(buildings)
    buildings = buildings.to_crs(crs)

    return buildings


def _fill_heights(
    buildings: gpd.GeoDataFrame,
    meters_per_level: float,
) -> gpd.GeoDataFrame:
    """Estimate missing height from building:levels, default 2 storey."""
    if "building:levels" not in buildings.columns:
        buildings["building:levels"] = 2
    else:
        buildings["building:levels"] = buildings["building:levels"].fillna(2)

    if "height" not in buildings.columns:
        buildings["height"] = None

    buildings = buildings.assign(
        building_levels_clean=lambda x: pd.to_numeric(
            x["building:levels"].astype(str).str.split(";").str[0],
            errors="coerce",
        ).fillna(1),
    )
    buildings["height"] = (
        buildings["height"]
        .fillna(buildings["building_levels_clean"] * meters_per_level)
    )
    buildings["height"] = pd.to_numeric(buildings["height"], errors="coerce")
    buildings = buildings.dropna(subset=["height"])
    buildings = buildings.drop(columns=["building_levels_clean"])

    return buildings


def _filter_geometry(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop Points and MultiPolygons — keep only simple Polygons."""
    geom_type = buildings.geometry.geom_type
    keep = (geom_type != "Point") & (geom_type != "MultiPolygon")
    return buildings.loc[keep].copy()


def _clean_geometry(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries by buffering with zero distance."""
    buildings["geometry"] = buildings.geometry.buffer(0)
    return buildings