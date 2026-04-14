"""Clean and normalise building footprints before meshing."""

from __future__ import annotations

import geopandas as gpd

from ..config import DEFAULT_PROJECT_CRS


def clean_buildings(
    buildings: gpd.GeoDataFrame,
    *,
    crs: str = DEFAULT_PROJECT_CRS,
    oid: str = "id",
    height_col: str = "height",
    geometry_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """Standardise columns, fix geometry, and reproject.

    Parameters
    ----------
    buildings : GeoDataFrame
        Raw building footprints (e.g. from OSM download).
    crs : str
        Target projected CRS.
    oid : str
        Column name for the unique building identifier.
    height_col : str
        Column containing building height in meters.
    geometry_col : str
        Geometry column name.

    Returns
    -------
    GeoDataFrame
        Columns: ``osmid``, ``height``, ``geometry``, ``old_geom``.
        Geometry cleaned via ``buffer(0)``; Points and MultiPolygons dropped.
    """
    data = buildings.reset_index()

    # Ensure target CRS
    if data.crs is not None and data.crs != crs:
        data = data.to_crs(crs)

    # Standardise column names
    rename_map = {}
    if oid in data.columns and oid != "osmid":
        rename_map[oid] = "osmid"
    if height_col in data.columns and height_col != "height":
        rename_map[height_col] = "height"
    if geometry_col in data.columns and geometry_col != "geometry":
        rename_map[geometry_col] = "geometry"
    if rename_map:
        data = data.rename(columns=rename_map)

    # Keep only the columns we need
    keep_cols = [c for c in ("osmid", "height", "geometry") if c in data.columns]
    data = data[keep_cols].copy()

    # Clean geometry
    cleaned_geom = _clean_polygon_geometry(data.geometry)
    geom_type = cleaned_geom.geom_type
    keep = (geom_type != "Point") & (geom_type != "MultiPolygon")

    data = data.loc[keep].copy()
    data["old_geom"] = cleaned_geom[keep]
    data["geometry"] = cleaned_geom[keep]

    return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)


def _clean_polygon_geometry(
    geom: gpd.GeoSeries,
    simplify_tolerance: float = 0.05,
) -> gpd.GeoSeries:
    """Fix common OSM polygon defects that cause sliver edges on extrusion.

    Steps applied in order:
    1. ``buffer(0)``      — repairs self-intersections and ring orientation.
    2. ``simplify()``     — removes collinear / near-duplicate vertices that
                            create zero-area sliver faces when extruded.

    Parameters
    ----------
    geom : GeoSeries
        Raw polygon geometries (projected CRS, units = meters).
    simplify_tolerance : float
        Maximum deviation allowed when removing redundant vertices (meters).
        0.05 m removes near-duplicate points without changing visible shape.
    """
    geom = geom.buffer(0)
    geom = geom.simplify(simplify_tolerance, preserve_topology=True)
    return geom
