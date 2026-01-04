import numpy as np
import point_cloud_utils as pcu
import time
from functools import partial#, lru_cache, 
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
import os
import warnings
from contextlib import contextmanager
import pandas as pd
import rasterio


import os
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["OMP_PLACES"] = "threads"



import rasterio
import time
import numpy as np
import pandas as pd
from functools import lru_cache

# RAY UTILS

def rotation_matrix(axis, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [cos_a + ux**2 * (1 - cos_a), ux*uy * (1 - cos_a) - uz*sin_a, ux*uz * (1 - cos_a) + uy*sin_a],
        [uy*ux * (1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a), uy*uz * (1 - cos_a) - ux*sin_a],
        [uz*ux * (1 - cos_a) - uy*sin_a, uz*uy * (1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
    ])

def create_rayends(norm, hfov=60, resolution=10, cast_type = 'grid'):
    ''' 
    Create grid of ray-end directions; vector for view cone, window etc. 
    
    Return: ray_d dim(n, 3)
    '''
    if cast_type == 'grid':
        #create window grid
        half_fov = np.deg2rad(hfov)
        r = np.tan(half_fov)  
        x, y = np.mgrid[-r:r:resolution*1j, -r:r:resolution*1j]
        # x, y = np.mgrid[-0.5:0.5:resolution*1j, -0.5:0.5:resolution*1j]
        ray_d = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=-1)
        ray_d = ray_d.astype(np.float32)
        ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)

    #assign vp direction
    new_direction = np.array([norm[0], norm[1], 0], dtype=np.float32)
    new_direction /= np.linalg.norm(new_direction)

    #assign rotation axis as Z-acis
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    rotation_axis = np.cross(z_axis, new_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    #rotate window 
    if rotation_axis_norm > 1e-6:
            rotation_axis /= rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, new_direction), -1.0, 1.0))
            R = rotation_matrix(rotation_axis, angle)
            rotated_rays = np.dot(ray_d, R.T)
    else:
        rotated_rays = ray_d

    ray_d = np.ascontiguousarray(rotated_rays, dtype=np.float32)
    return ray_d

def create_rayorigins(pt, ray_dir):
     '''
     create rayorigins to match length of rayends
     
     Return ray_o dim(n,3)
     '''
     ray_o = np.broadcast_to(
          np.ascontiguousarray(pt, dtype=np.float32)[None, :],
          (ray_dir.shape[0], 3))
     
     ray_o = np.ascontiguousarray(ray_o, dtype=np.float32)

     return ray_o

def pts_to_Raystack(PTS: np.array, resolution: int, cast_type:str):
    '''
    create a stacked array of rays for array of points:
    
    ray_o = origin[n,3], 
    ray_d = direction[n,3])
    vpid = unique id for pts
    '''
    import uuid

    all_ray_o = []
    all_ray_d = []
    all_vpid = []

    for vp_data in PTS:
        ray_o, ray_d = pt_to_Rays(vp_data, resolution, cast_type)
        vpid = uuid.uuid4()
        all_ray_o.append(ray_o)
        all_ray_d.append(ray_d)
        all_vpid.append(np.repeat(vpid, ray_o.shape[0]))

    all_ray_o = np.vstack(all_ray_o)
    all_ray_d = np.vstack(all_ray_d)
    all_vpid = np.concatenate(all_vpid)

    return all_ray_o, all_ray_d, all_vpid

def pt_to_Rays(PT: np.array, resolution: int, cast_type:str):
        '''create stacked array for single point'''
        vp, vp_d = PT['vp'], PT['norm']
        
        ray_d = create_rayends(vp_d, resolution=resolution, cast_type=cast_type)
        ray_o = create_rayorigins(vp, ray_d)

        return ray_o, ray_d


def get_mesh_intersector(v_tuple, f_tuple):
    """Cache RayMeshIntersector objects for each unique mesh.
    
    Args:
        v_tuple: Tuple of vertex coordinates (must be tuple for hashing)
        f_tuple: Tuple of face indices (must be tuple for hashing)
    """
    v = np.array(v_tuple)
    f = np.array(f_tuple)
    return pcu.RayMeshIntersector(v, f)

