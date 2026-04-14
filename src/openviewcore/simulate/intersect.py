"""Ray-mesh intersection (single-shot and chunked)."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import point_cloud_utils as pcu

from ..config import DEFAULT_RAY_NEAR, DEFAULT_CHUNK_SIZE
from tqdm import tqdm
# Avoid OpenMP binding issues on macOS
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_PLACES", "threads")


def cast_rays(
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    scene: dict,
    *,
    ray_near: float = DEFAULT_RAY_NEAR,
    chunk_size: Optional[int] = None,
    intersector=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with the scene mesh.

    Parameters
    ----------
    ray_origins, ray_dirs : ndarray (N, 3)
        Ray start points and directions.
    scene : dict
        ``{'v': (V,3), 'f': (F,3)}``.
    ray_near : float
        Ignore intersections closer than this distance.
    chunk_size : int, optional
        Process rays in chunks to limit memory.  *None* = all at once.

    Returns
    -------
    fid : ndarray (N,) int   — face index of hit (-1 = miss).
    bc  : ndarray (N, 3) float — barycentric coordinates of hit.
    t   : ndarray (N,) float  — distance along ray to hit.
    """
    v = np.ascontiguousarray(scene["v"], dtype=np.float32)
    f = np.ascontiguousarray(scene["f"], dtype=np.int32)

    n = ray_origins.shape[0]
    if intersector is None:
        intersector = pcu.RayMeshIntersector(v, f)

    if chunk_size is None or n <= chunk_size:
        return intersector.intersect_rays(
            np.ascontiguousarray(ray_origins, dtype=np.float32),
            np.ascontiguousarray(ray_dirs, dtype=np.float32),
            ray_near=ray_near,
        )
        # return pcu.ray_mesh_intersection(
        #     v, f,
        #     np.ascontiguousarray(ray_origins, dtype=np.float32),
        #     np.ascontiguousarray(ray_dirs, dtype=np.float32),
        #     ray_near=ray_near,
        # )

    # Chunked path
    all_fid = np.empty(n, dtype=np.int64)
    all_bc = np.empty((n, 3), dtype=np.float32)
    all_t = np.empty(n, dtype=np.float32)

    for start in tqdm(range(0, n, chunk_size), desc="Ray Casting", leave=False, position=1):
        end = min(start + chunk_size, n)
        # fid, bc, t = pcu.ray_mesh_intersection(
        #     v, f,
        #     np.ascontiguousarray(ray_origins[start:end], dtype=np.float32),
        #     np.ascontiguousarray(ray_dirs[start:end], dtype=np.float32),
        #     ray_near=ray_near,
        # )
        fid, bc, t = intersector.intersect_rays(
            np.ascontiguousarray(ray_origins[start:end], dtype=np.float32),
            np.ascontiguousarray(ray_dirs[start:end], dtype=np.float32),
            ray_near=ray_near,
        )
        all_fid[start:end] = fid
        all_bc[start:end] = bc
        all_t[start:end] = t

    return all_fid, all_bc, all_t
