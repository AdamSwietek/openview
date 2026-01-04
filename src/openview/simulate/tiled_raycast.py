import numpy as np
import point_cloud_utils as pcu

from ..mesh import tile_mesh


def _ray_bounds_xy(ray_o, ray_d, ray_length):
    if ray_length is None:
        ray_min = ray_o[:, :2]
        ray_max = ray_o[:, :2]
    else:
        ray_end = ray_o + ray_d * ray_length
        ray_min = np.minimum(ray_o[:, :2], ray_end[:, :2])
        ray_max = np.maximum(ray_o[:, :2], ray_end[:, :2])
    return ray_min, ray_max


def ray_mesh_intersection_tiled(
    vf,
    ray_o,
    ray_d,
    *,
    tiles=None,
    tile_size=None,
    buffer=0.0,
    bounds=None,
    ray_length=None,
):
    """
    Ray/mesh intersection using spatial tiles to limit memory.

    Notes:
    - If tiles are not provided, tile_size is required and tiles are built via tile_mesh.
    - Tiles should include 'face_idx' to map local face ids to global face ids.
    """
    ray_o = np.asarray(ray_o, dtype=np.float32)
    ray_d = np.asarray(ray_d, dtype=np.float32)
    if ray_o.shape != ray_d.shape or ray_o.shape[1] != 3:
        raise ValueError("ray_o and ray_d must be (N,3) arrays")

    if tiles is None:
        if tile_size is None:
            raise ValueError("tile_size is required when tiles are not provided")
        tiles = tile_mesh(vf, tile_size, buffer=buffer, bounds=bounds, include_empty=False)

    n_rays = ray_o.shape[0]
    global_fid = np.full(n_rays, -1, dtype=np.int64)
    global_bc = np.full((n_rays, 3), np.nan, dtype=np.float32)
    global_t = np.full(n_rays, np.inf, dtype=np.float32)

    ray_min, ray_max = _ray_bounds_xy(ray_o, ray_d, ray_length)

    for tile in tiles:
        tile_mesh_data = tile.get("mesh")
        if tile_mesh_data is None:
            continue

        xmin, xmax, ymin, ymax = tile["bbox"]
        xmin -= buffer
        xmax += buffer
        ymin -= buffer
        ymax += buffer

        mask = (
            (ray_max[:, 0] >= xmin) & (ray_min[:, 0] <= xmax) &
            (ray_max[:, 1] >= ymin) & (ray_min[:, 1] <= ymax)
        )
        if not mask.any():
            continue

        idx = np.flatnonzero(mask)
        v = tile_mesh_data["v"]
        f = tile_mesh_data["f"]

        fid, bc, t = pcu.ray_mesh_intersection(v, f, ray_o[idx], ray_d[idx])
        hit = fid >= 0
        if not hit.any():
            continue

        if "face_idx" in tile:
            fid = tile["face_idx"][fid]

        better = hit & (t < global_t[idx])
        if not better.any():
            continue

        update_idx = idx[better]
        global_t[update_idx] = t[better]
        global_fid[update_idx] = fid[better]
        global_bc[update_idx] = bc[better]

    return global_fid, global_bc, global_t
