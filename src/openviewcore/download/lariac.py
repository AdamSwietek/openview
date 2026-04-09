#external building footprint datasource
"""Download and transform scene data (terrain, buildings, etc.)."""
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box
import seamless_3dep as s3dep
from pathlib import Path
import tempfile
import rioxarray as rxr
import fiona
import shutil
from urllib.request import urlopen

def load_lariac(geom: gpd.GeoDataFrame, yr: int) -> gpd.GeoDataFrame:
    """
    Load LARIAC building footprints for a given year.

    HEIGHT loads in ft, add height in meters col
    """
    # base_url = '/home1/swietek/ondemand/elec/data/laraic_files/'
    base_url = "/Users/adamswietek/Documents/PostDoc/data/raw_la/laraic_files"
    laraic_files = {
        2014: f'{base_url}/LARIAC4_Buildings_2014.gdb',
        2017: f'{base_url}/LARIAC5_Buildings_2017.gdb',
        2020: f'{base_url}/LARIAC6_Buildings_2020.gdb',
    }
    fp = laraic_files[yr]

    import fiona
    with fiona.open(fp) as src:
        CRS = src.crs
        # print(CRS)

    if  CRS != geom.crs:
        geom = geom.to_crs(CRS)

    # bbox = tuple(geom.total_bounds)
    layers = fiona.listlayers(fp)

    data = gpd.read_file(fp, mask = geom, layer = layers[0])

    data['height_m'] = data['HEIGHT'] * 0.3048 #feet to meters conversion

    return data


def load_la_neighborhoods(crs = 'EPSG:4326'):
    
    """
    Returns a GeoDataFrame containing the neighborhoods in the
    Los Angeles area. LA TIMES map of neighborhoods

    Returns:
    GeoDataFrame: A GeoDataFrame containing the neighborhoods in the Los Angeles area.
    crs: EPSG:4326
    
    """

    fp = "https://s3-us-west-2.amazonaws.com/mappingla.com/downloads/neighborhoods/la_county.json"
    neighborhoods = gpd.read_file(fp) #crs 4326
    
    # neighborhoods = gpd.read_file('cache/la_neighborhoods.csv')
    # neighborhoods['geometry'] = neighborhoods['the_geom'].apply(wkt_loads)
    # neighborhoods = gpd.GeoDataFrame(neighborhoods) #'EPSG:2230'

    # neighborhoods.set_crs(epsg='4326', inplace=True)

    if neighborhoods.crs != crs:
        neighborhoods.to_crs(crs, inplace = True)
        
    return neighborhoods#neighborhoods.drop(columns=['the_geom'])

def download_lariac_from_bbox(
    bbox,
    *,
    out_path=None,
    base_url=(
        "https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/"
        "Countywide_Building_Outlines_(2020)/FeatureServer/0/query"
    ),
):
    """
    Download LARIAC building footprints for a bbox (WGS84).

    Args:
        bbox (tuple): (west, south, east, north) in EPSG:4326.
        out_path (str or Path, optional): If provided, save GeoJSON to this path.
        base_url (str, optional): ArcGIS REST endpoint for the layer.

    Returns:
        GeoDataFrame: Building footprints as a GeoDataFrame (EPSG:4326).
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be a 4-tuple (west, south, east, north)")

    params = {
        "where": "1=1",
        "outFields": "*",
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": 4326,
        "f": "geojson",
    }
    url = f"{base_url}?{urlencode(params)}"

    with urlopen(url) as resp:
        payload = json.load(resp)

    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True)

    return gdf



