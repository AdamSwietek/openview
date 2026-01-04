"""Download and transform scene data (terrain, buildings, etc.)."""
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box
import seamless_3dep as s3dep
from pathlib import Path
import tempfile
import rioxarray as rxr

def download_osm_buildings_from_gdf(gdf: gpd.GeoDataFrame, crs="EPSG:2229"):
    """
    Fetches building footprints from OpenStreetMap for a given polygon.

    Args:
        gdf (gpd.GeoDataFrame, optional): A GeoDataFrame defining the area of interest.
                                           Defaults to None.
        crs (str, optional): The desired output coordinate reference system.
                             Defaults to "EPSG:26911".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with building footprints and heights.
    """
    # Ensure the input GeoDataFrame is in WGS84 for osmnx
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    geom = gdf.geometry.iloc[0]
    buildings = ox.features_from_polygon(geom, tags={'building': True})

    # Fill in missing height data
    if 'building:levels' not in buildings.columns:
        buildings['building:levels'] = 1
    else:
        buildings['building:levels'] = buildings['building:levels'].fillna(1)

    if 'height' not in buildings.columns:
        buildings['height'] = None


    buildings_with_height = (
        buildings.assign(
            geom_type=lambda x: x.geometry.geom_type,
            building_levels_clean=lambda x: pd.to_numeric(
                x['building:levels'].astype(str).str.split(';').str[0],
                errors='coerce',
            ).fillna(1),
        )
        .query('geom_type != "Point" and geom_type != "MultiPolygon"')
        .assign(height=lambda x: x['height'].fillna(x['building_levels_clean'] * 3))
        .assign(height=lambda x: pd.to_numeric(x['height'], errors='coerce'))
        .dropna(subset=['height'])
        .to_crs(crs)
        .drop(columns=['building_levels_clean'])
    )

    return buildings_with_height
