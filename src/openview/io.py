''' load geodata from cache '''

import os
import pickle
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path

import py3dep


import numpy as np

#########
# dicts
#########

def save_vf(path, vf):
    np.savez_compressed(path, v=vf["v"], f=vf["f"])
    print(f'saved to {path}')
def load_vf(path):
    data = np.load(path)
    print(f'loaded from {path}')
    return {"v": data["v"], "f": data["f"]}

#########
# TERRAIN
#########

def load_NLCD_bygeom(mask):
    """Load and clip NLCD data using a geometry mask without loading the entire raster into memory."""
    fp = Path('/Users/adamswietek/Documents/PostDoc/data/raw_national/Annual_NLCD_LndCov_2023_CU_C1V0.tif')

    # rioxarray opens raster files lazily by default, meaning the entire file is not
    # loaded into memory. Using a with statement ensures the file is properly closed.
    with rxr.open_rasterio(fp) as nlcd:
        if isinstance(mask, gpd.GeoDataFrame):
            # Ensure the CRS of the mask matches the raster's CRS
            if mask.crs != nlcd.rio.crs:
                mask = mask.to_crs(nlcd.rio.crs)
            geometries = mask.geometry
        else:
            geometries = [mask]

        # The 'from_disk=True' argument is key. It clips the raster using the
        # file on disk as the source, reading only the necessary data into memory.
        clipped_nlcd = nlcd.rio.clip(geometries, from_disk=True)
    return clipped_nlcd


def load_LACounty_terrain(scaled = False):

    if scaled:
        #read pickle
        with open('cache/meshes/LACounty_terrain_mesh_26911_scaled.pkl', 'rb') as f:
            terrain_mesh = pickle.load(f)
    else:
        #read pickle
        with open('cache/meshes/LACounty_terrain_mesh_26911.pkl', 'rb') as f:
            terrain_mesh = pickle.load(f)

    print("Loaded terrain mesh for LA County")
    return terrain_mesh




def load_mesh_data(place, target_crs, type = 'osm', region_name = None):
    ''''Get data for a place, either from cache or by fetching it from source.
    Args:
    - place: The name of the place to get data for.
    - target_crs: The target CRS to reproject the data to.
    - type: The type of data to fetch ('osm' or 'LA_neighborhood').
    Returns:
    - A tuple of data objects for the place.

    '''
    from openview.download.download import get_osm_buildings_from_poly, get_topo_raster

    # If no cached data, generate it from scratch
    print(f"Fetching OSMBldg & terrain meshes for {place} from source...")
    
    #----------------------
    # INPUT AREA OF INTEREST, LOAD BUILDINGS AND TERRAIN
    #----------------------
    # place = "Manhattan Beach, California, USA"

    buildings_with_height, city_proj = get_osm_buildings_from_poly(place, target_crs)
    topo = get_topo_raster(city_proj, geo_crs=city_proj.crs, crs=target_crs)

    #----------------------
    #Preprocess building data
    #----------------------

    #move buildings to terrain
    data = shape.move_footprint_to_terrain(buildings_with_height, topo)
    data = data[~data['elevation'].isna()] # filter out buildings in nan elevation zones

    #convert to 3d mesh
    urban_mesh = mesh.extrude_df_to_3dmesh(data) # list of pv_meshes for indexing
    # pv_urban_mesh = ov.combine_pv_meshes(urban_mesh) # list of pv_meshes for plotting
    unified_urban_mesh = mesh.join_mesh(urban_mesh) # single mesh as v,f for raycasting algo

    landscape_mesh = mesh.meshify_topo(topo)
    # pv_landscape = ov.get_pv_topo(landscape_mesh)

    # Save data to cache for future queries (without non-pickleable objects)
    io.save_to_cache(place, (data, urban_mesh, unified_urban_mesh, landscape_mesh))
    
    return data, urban_mesh, unified_urban_mesh, landscape_mesh



####
#LARIAC
####





