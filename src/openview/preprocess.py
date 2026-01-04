import geopandas as gpd
import pandas as pd

from .shape import move_footprint_to_terrain

def preprocess_footprints(data, topo_rxr,
                        #   elem_type = 'element', 
                          oid = 'id',
                          geometry = 'geometry',
                          height = 'height', 
                          elevation = 'elevation'):
    
    """ Preprocess building footprints, prep for viewpoint extraction """
    
    #reset index (e.g. osm have multi-level index)
    data = data.reset_index()

    #append Z coord for buildings footprint geom
    data = move_footprint_to_terrain(data, topo_rxr)
    data = data[data['elevation'].notna()] # filter out buildings in nan elevation zones

    #construct new GDF
    df = gpd.GeoDataFrame(
        data.loc[:, [oid, height, elevation, geometry]]
            .rename(columns={oid: 'osmid', height: 'height', elevation: 'elevation', geometry: 'geometry'}),
        geometry='geometry',
        crs=data.crs,
    )

    cleaned_geom = df.geometry.buffer(0)
    geom_type = cleaned_geom.geom_type
    keep = (geom_type != "Point") & (geom_type != "MultiPolygon")
    df = df.loc[keep].assign(old_geom=cleaned_geom[keep], geometry=cleaned_geom[keep])

    return df

