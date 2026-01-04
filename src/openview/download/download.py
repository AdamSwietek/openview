"""Download and transform scene data (terrain, buildings, etc.)."""
import asyncio
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box
import seamless_3dep as s3dep
from pathlib import Path
import tempfile
import rioxarray as rxr


def _get_map_with_timeout(
    map_type,
    bbox,
    data_dir,
    resolution,
    pixel_max,
    timeout,
    limit_per_host,
):
    if timeout is None:
        return s3dep.get_map(map_type, bbox, data_dir, resolution, pixel_max)

    original_download = s3dep.terry.download

    def _download_with_timeout(urls, file_paths, **kwargs):
        return original_download(
            urls,
            file_paths,
            timeout=timeout,
            limit_per_host=limit_per_host,
            **kwargs,
        )

    s3dep.terry.download = _download_with_timeout
    try:
        return s3dep.get_map(map_type, bbox, data_dir, resolution, pixel_max)
    finally:
        s3dep.terry.download = original_download


def download_dem_from_gdf(
    gdf,
    resolution=10,
    crs="EPSG:2229",
    data_dir='cache/dem',
    *,
    timeout=300,
    pixel_max=None,
    limit_per_host=2,
    retry_pixel_max=2_000_000,
):
    """
    Downloads a DEM from the USGS National Map for a given bounding box. In/Out is WG84 CRS

    Args:
        bbox (tuple): A tuple representing the bounding box in the format
                      (west, south, east, north). 
        resolution (int, optional): The desired resolution of the DEM in meters.
                                     Defaults to 10.
        crs (str, optional): The coordinate reference system of the bounding box.
                             Defaults to "EPSG:4326".
        data_dir (str or Path, optional): A directory to cache the downloaded DEM files.
                                          If None, a temporary directory will be used.
                                          Defaults to None.
        timeout (int, optional): Per-request download timeout in seconds.
                                 Defaults to 300.
        pixel_max (int, optional): Maximum pixels per tile. Smaller values reduce
                                   per-request size at the cost of more tiles.
                                   Defaults to None (library default).
        limit_per_host (int, optional): Max concurrent connections per host.
                                        Defaults to 2.
        retry_pixel_max (int, optional): Pixel cap to retry with if a timeout
                                         occurs. Defaults to 2,000,000.

    Returns:
        xarray.DataArray: An xarray.DataArray containing the DEM data.
    """
    if data_dir is None:
        data_dir = tempfile.mkdtemp(prefix="seamless_3dep_")
    
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    bbox = tuple(gdf.total_bounds)

    data_dir = Path(data_dir)
    try:
        tiff_files = _get_map_with_timeout(
            "DEM",
            bbox,
            data_dir,
            resolution,
            pixel_max,
            timeout,
            limit_per_host,
        )
    except (asyncio.TimeoutError, TimeoutError):
        if retry_pixel_max is None:
            raise
        if pixel_max is None or pixel_max > retry_pixel_max:
            tiff_files = _get_map_with_timeout(
                "DEM",
                bbox,
                data_dir,
                resolution,
                retry_pixel_max,
                timeout,
                limit_per_host,
            )
        else:
            raise
    dem = s3dep.tiffs_to_da(tiff_files, bbox) #source file is a 'engineeringCRS', need to reassign
    dem = dem.rio.write_crs("EPSG:3857", inplace=False) # 
    dem = dem.rio.reproject(crs)
   
    return dem#, tiff_files

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
