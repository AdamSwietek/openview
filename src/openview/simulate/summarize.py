import rioxarray as rxr

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

from src.openview.constants import BBOX, WGS84_EPSG, PROJECT_CRS, PATH_TO_LULC


def raypoint_to_VSH(ray_df):
    """
    Create VSH from existing results
    results: dict, results from run_method
    bldg: geopandas.GeoDataFrame, building
    idx: int, index of building

    returns:
    vsh: pandas.DataFrame, VSH
    performance: dict, performance

    """
    
    lc_mask = (ray_df['terminal_obj'] == 'landcover') & (~ray_df.x.isna())
    ray_df.loc[lc_mask,'lcID'] = sample_landcover_at_raypoints(ray_df.loc[lc_mask,:])
                                                                                                                        
    ray_df = fill_obstruction(ray_df)

    ray_df['D'] = cut_distance(ray_df['t'])
    ray_df['O'] = map_landcover_name(ray_df['lcID'])
    ray_df['LcD'] = ray_df['lcID'].astype(int).astype(str) +'_'+ ray_df['D'].astype(str)

    total_rays_n = ray_df.ray_id.max() + 1 
    vsh = ray_df.groupby(['vp_id','LcD']).size().unstack() / total_rays_n


    return vsh

def sample_landcover(pts, FP = PATH_TO_LULC, crs = PROJECT_CRS):

    x, y = pts[:,0], pts[:1]

    with rxr.open_rasterio(Path(FP), masked=True) as da:
        # da has shape (band, y, x)
        raster_crs = da.rio.crs
        if raster_crs != crs:
            pt = gpd.points_from_xy(x = pts[:,0], y = pts[:,1])
            pt = gpd.GeoSeries(pt, crs=crs)
            pt = pt.to_crs(raster_crs)

        lulc = da.sel(x=pt.geometry.x.to_xarray(), 
                    y=pt.geometry.y.to_xarray(), method='nearest')
    return lulc.values[0].astype(int).astype(str)

def sample_ee_at_polygon(ee_url = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
                         ):
    import ee
    import geemap
    from dotenv import load_dotenv

    load_dotenv()

    ee.Authenticate()
    ee.Initialize(project="hiddenhousing")

    dataset = ee.ImageCollection(ee_url)
    img_la = dataset.filterDate("2024-01-01", "2025-01-01").filterBounds(geom)
    geoms = lariac.dropna(subset= 'HEIGHT').head(10).buffer(500).to_crs(4326)
    features = [
            ee.Feature(ee.Geometry.Polygon(g)) for g in geoms
        ]

    return data
def fill_obstruction(ray_df):
    #define facade
    facade_mask = ray_df['terminal_obj'] == "facade"
    ray_df.loc[facade_mask,'lcID'] = 9001 
    #define sky
    ray_df['lcID'] = ray_df['lcID'].fillna(0)
    return ray_df

def cut_distance(x, units = 'ft'):
    '''Split distance into pre-set categories'''
    if units == 'ft':
        bins = [0,350,3500,165000,np.inf] 

    elif units == 'm':
        bins = [0,100,1000,50000,np.inf]

    labels = ['Near','Mid','Far','Inf'] 

    return pd.cut(x, bins = bins, labels = labels)

def bin_obstruction(cellid, cell_count_dict):
    '''Split face ids into mesh bins based on cumulative max face ids.'''

    # Convert per-mesh max face ids into cumulative cutoff bins.
    cumulative_max = -1
    bins = [-1]
    for max_fid in cell_count_dict.values():
        cumulative_max += int(max_fid) + 1
        bins.append(cumulative_max)

    labels = list(cell_count_dict.keys())

    return pd.cut(cellid, bins=bins, labels=labels, include_lowest=True)