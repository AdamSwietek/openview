import os
import point_cloud_utils as pcu
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..constants import PROJECT_CRS, PATH_TO_LULC
from ..mesh import join_list_of_meshes
from .raycasting import pts_to_Raystack
from .summarize import bin_obstruction, cut_distance, sample_landcover

os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["OMP_PLACES"] = "threads"


def run_windowview_sim(VP_DATA:np.array, dsm:dict, 
                     ncells: dict, resolution: int = 25, 
                     cast_type: str = 'grid',
                     landcover_path = PATH_TO_LULC, 
                     crs = PROJECT_CRS,
                     intersector = None):
    
    ''' compute '''
    
    # mask = mask_subterrain_points(vp, dsm)

    all_ray_o, all_ray_d, all_vpid = pts_to_Raystack(VP_DATA, resolution, cast_type)

    #Simulate ray mesh intersection
    if intersector:
        fid, bc, t =  intersector.intersect_rays(all_ray_o, all_ray_d, ray_near = 10)
    else:
        fid, bc, t = pcu.ray_mesh_intersection(dsm['v'], dsm['f'], 
                                       all_ray_o, all_ray_d, ray_near=10)


    obst = bin_obstruction(fid, ncells)
    dist = cut_distance(t)

    hit_mask = obst == "terrain"
    lulc = np.full(obst.shape, "_", dtype=object)

    if hit_mask.any():
        ray_intersections = pcu.interpolate_barycentric_coords(dsm["f"], fid[hit_mask], bc[hit_mask], dsm["v"])
        lulc[hit_mask] = sample_landcover(ray_intersections)

    labels = np.where(lulc == "_", obst, lulc).astype(str) + dist.astype(str)

    df = pd.DataFrame({"vpid": all_vpid, "label": labels})
    res = df.groupby("vpid")["label"].value_counts(normalize=True).unstack()

    res['osmid'] = VP_DATA['osmid'].tolist()
    res['X'] = VP_DATA['vp'][:,0].tolist()
    res['Y'] = VP_DATA['vp'][:,1].tolist()
    res['Z'] = VP_DATA['vp'][:,2].tolist()
    res['norm'] = VP_DATA['norm'].tolist()

    return res


def run_windowview_sim_chunked(VP_DATA: np.array, dsm: dict,
                             ncells: dict, resolution: int = 25,
                             cast_type: str = 'grid',
                             landcover_path=PATH_TO_LULC, crs=PROJECT_CRS,
                             chunk_size: int = 5000,
                             outpath: str = None):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    n_pts = len(VP_DATA)
    if n_pts == 0:
        return run_windowview_sim(VP_DATA, dsm, ncells, resolution, cast_type, landcover_path, crs)

    if n_pts <= chunk_size:
        return run_windowview_sim(VP_DATA, dsm, ncells, resolution, cast_type, landcover_path, crs)

    if outpath:
        os.makedirs(outpath, exist_ok=True)

    chunks = []
    written_paths = []
    for start in tqdm(range(0, n_pts, chunk_size), desc='Processing chunk: '):
        stop = min(start + chunk_size, n_pts)
        chunk = VP_DATA[start:stop]
        res = run_windowview_sim(chunk, dsm, ncells, resolution, cast_type, landcover_path, crs)
        if outpath:
            chunk_path = os.path.join(outpath, f"window_chunk_{start}_{stop}.csv")
            res.to_csv(chunk_path, index=False)
            written_paths.append(chunk_path)
        else:
            chunks.append(res)

    if outpath:
        return written_paths

    res = pd.concat(chunks, ignore_index=True, sort=False)
    label_cols = [c for c in res.columns if c not in ("osmid", "XYZ", "norm")]
    if label_cols:
        res[label_cols] = res[label_cols].fillna(0)
    return res

def mask_subterrain_points(vp, terrain_vf):
    vp_o = np.ascontiguousarray(vp['vp'], dtype=np.float32)
    vp_z = np.broadcast_to(np.array([0,0,1])[None,:], (vp_o.shape[0], 3))
    vp_z = np.ascontiguousarray(vp_z, dtype=np.float32)

    fid, bc, t = pcu.ray_mesh_intersection(terrain_vf['v'], terrain_vf['f'], 
                                        vp_o, vp_z)

    mask = fid < 0

    return mask