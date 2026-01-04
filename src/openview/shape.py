import math
import numpy as np
import xarray as xr
from shapely.geometry import LineString, Polygon, MultiPolygon
import pandas as pd
from shapely import wkt
import uuid

def mp_dict_to_geoseries(bldg_dict):
    from shapely import wkt
    bldg = pd.Series({
        'osmid': bldg_dict['osmid'],
        'geometry': wkt.loads(bldg_dict['geometry']) if bldg_dict['geometry'] else None,
        'old_geom': wkt.loads(bldg_dict['old_geom']) if bldg_dict['old_geom'] else None,
        'height': bldg_dict['height'],
        'elevation': bldg_dict['elevation']
    })
    return bldg

def get_viewpoints_from_footprint(bldg, units = 'ft'):
    '''
    Feed In Geoseries of building footprints
    Get viewpoints as list of (pt and norm )from the building footprint. 
    
    Good idea to feed the footprint as a buffered area 
    - to push vp out away from facade
    '''

    #chekc if padnas series or geoseries, inverse
    if not isinstance(bldg, pd.Series):
        bldg = mp_dict_to_geoseries(bldg)

    footprint = bldg.geometry.exterior # buffer by 0.5 meters so that vp are not on wall surface
    footprint_buffered = bldg.old_geom.buffer(.5).exterior # buffer by 0.5 meters so that vp are not on wall surface
    elevation = bldg.elevation
    osmid = bldg.osmid

    # Process the footprint to extract the midpoints and normal vectors of its segments
    vp_2data = process_footprint_to_vpdata(footprint_buffered)
    
    mag = .5
    if units == 'ft':
        ground_floor_h = 6
        floor_height = 10
    elif units == 'm': # OSM (+LARIAC) provide height data using meters,
        ground_floor_h = 2
        floor_height = 3
    # Create a dict of 3D points at the given height
    viewpoint_data = [{'vp_uuid':uuid.uuid4(), 'osmid':osmid, 'vp': pt_3d(pt,elevation+h,n,mag),'norm': norm_3d(n,0)} for h in range(ground_floor_h, int(bldg.height), floor_height) for pt, n in vp_2data ] 
    # add points every 3 meters up until the height of the building

    #append osmid as first dimension

    return viewpoint_data


def process_footprint_to_vpdata(line):
    '''Proccess a shapely LineString object to extract the midpoints and normal vectors of its segments.'''
    vp_data = []
    
    # Split the LineString into individual segments
    segments = split_line_into_segments(line)
    
    for segment in segments:
        # Calculate the midpoint
        midpoint = split_segment_into_points(segment)
        
        # Calculate the normal vector
        normal_vector = calculate_normal_vector(segment)
        
        vp_data.append((midpoint, normal_vector))

    #drop nan values from the list
    vp_data = [ (pt, n) for pt, n in vp_data if pt is not np.nan]
    return vp_data


# FUNC TO GET VIEWPOINTS FROM 2D BUILDING FOOTPRINT
def split_line_into_segments(line):
    ''''Split a LineString into individual segments.'''
    segments = []
    coords = list(line.coords)
    
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        segments.append(segment)
    
    return segments

def split_segment_into_points(line):
    '''Split a LineString into individual segments. Update so that we can split by more than just midpoint'''
    lenght_lim = 1
    #if lenght of line is greater than 5 else no midpoint
    if line.length > lenght_lim:
        return line.interpolate(line.length / 2)
    else:
        return np.nan
    
def calculate_normal_vector(segment):
    ''''Calculate the normal vector of a segment.'''
    # Get the start and end points of the segment
    x1, y1 = segment.coords[0]
    x2, y2 = segment.coords[1]
    
    # Calculate the direction vector (dx, dy)
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize the direction vector to a unit vector
    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length
    
    # Rotate the direction vector by 90 degrees to get the normal vector
    normal_vector = (-unit_dy, unit_dx,0)  # Rotation by 90 degrees
    
    return normal_vector

def pt_3d(point, new_z,n,mag):
    ''''Create a 3D point at the given height.'''
    # Create a 3D point at the given height
    point_3d = [point.x+n[0]*mag, point.y+n[1]*mag, new_z]
    return point_3d
def norm_3d(n, height):
    ''''Create a 3D point at the given height.'''
    # Create a 3D point at the given height
    norm_3d = [n[0], n[1], n[2]+height]
    return norm_3d


def move_footprint_to_terrain(gdf,topo):
    centroids = gdf.geometry.centroid
    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()

    # Vectorized nearest-neighbor sampling (avoids per-row xarray calls).
    x_da = xr.DataArray(xs, dims="points")
    y_da = xr.DataArray(ys, dims="points")
    elevations = topo.sel(x=x_da, y=y_da, method="nearest").values
    elevations = elevations.flatten().tolist()

    gdf["elevation"] = elevations

    gdf["geometry"] = [
        add_z_coordinate(geom, elev)
        for geom, elev in zip(gdf.geometry, elevations)
    ]
    return gdf

def add_z_coordinate(geom, z_value):
    if geom.geom_type == 'Polygon':
        # Add the Z-coordinate to each coordinate in the polygon
        return Polygon([(x, y, z_value) for x, y in geom.exterior.coords])
    elif geom.geom_type == 'MultiPolygon':
        # Handle MultiPolygons by applying the same logic to each polygon
        new_polygons = [Polygon([(x, y, z_value) for x, y in poly.exterior.coords]) for poly in geom.geoms]
        return MultiPolygon(new_polygons)
    return geom  # Return the original geometry if it's not a polygon
