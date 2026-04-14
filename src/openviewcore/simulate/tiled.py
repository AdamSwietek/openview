"""Memory-efficient tiled ray-mesh intersection.

Use when the scene is too large to intersect all rays at once.
Spatial tiles let us process only rays that overlap each tile's bounding box.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import point_cloud_utils as pcu


def cast_rays_tiled(
    scene: dict,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    *,
    tiles: Optional[Sequence[dict]] = None,
    tile_size: Optional[float] = None,
    buffer: float = 0.0,
    ray_length: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiled ray-mesh intersection.

    Parameters
    ----------
    scene : dict
        ``{'v': (V,3), 'f': (F,3)}``.  Only used when *tiles* is None.
    ray_origins, ray_dirs : ndarray (N, 3)
    tiles : list[dict], optional
        Pre-built tiles (each with ``mesh``, ``bbox``, optional ``face_idx``).
    tile_size : float, optional
        Build tiles of this XY extent when *tiles* is None.
    buffer : float
        Expand each tile bbox by this amount.
    ray_length : float, optional
        Maximum ray travel distance (tightens bounding boxes).

    Returns
    -------
    fid, bc, t — same as :func:`~openviewcore.simulate.intersect.cast_rays`.
    """
    ray_o = np.ascontiguousarray(ray_origins, dtype=np.float32)
    ray_d = np.ascontiguousarray(ray_dirs, dtype=np.float32)

    if tiles is None:
        if tile_size is None:
            raise ValueError("Provide either `tiles` or `tile_size`.")
        tiles = _build_tiles(scene, tile_size, buffer)

    n = ray_o.shape[0]
    best_fid = np.full(n, -1, dtype=np.int64)
    best_bc = np.full((n, 3), np.nan, dtype=np.float32)
    best_t = np.full(n, np.inf, dtype=np.float32)

    ray_min, ray_max = _ray_bounds(ray_o, ray_d, ray_length)

    for tile in tiles:
        tile_mesh = tile.get("mesh")
        if tile_mesh is None:
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
        fid, bc, t = pcu.ray_mesh_intersection(
            tile_mesh["v"], tile_mesh["f"],
            ray_o[idx], ray_d[idx],
        )
        hit = fid >= 0
        if not hit.any():
            continue

        if "face_idx" in tile:
            fid = tile["face_idx"][fid]

        better = hit & (t < best_t[idx])
        update = idx[better]
        best_t[update] = t[better]
        best_fid[update] = fid[better]
        best_bc[update] = bc[better]

    return best_fid, best_bc, best_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ray_bounds(ray_o, ray_d, ray_length):
    if ray_length is None:
        return ray_o[:, :2], ray_o[:, :2]
    ray_end = ray_o + ray_d * ray_length
    return (
        np.minimum(ray_o[:, :2], ray_end[:, :2]),
        np.maximum(ray_o[:, :2], ray_end[:, :2]),
    )


def _build_tiles(scene, tile_size, buffer):
    """Naive grid-based tiling of a mesh."""
    v, f = scene["v"], scene["f"]
    xmin, ymin = v[:, 0].min() - buffer, v[:, 1].min() - buffer
    xmax, ymax = v[:, 0].max() + buffer, v[:, 1].max() + buffer

    tiles = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            bx0, bx1 = x, x + tile_size
            by0, by1 = y, y + tile_size

            centroids = v[f].mean(axis=1)
            mask = (
                (centroids[:, 0] >= bx0 - buffer) & (centroids[:, 0] < bx1 + buffer) &
                (centroids[:, 1] >= by0 - buffer) & (centroids[:, 1] < by1 + buffer)
            )
            if mask.any():
                sub_f = f[mask]
                unique_verts, new_f = np.unique(sub_f, return_inverse=True)
                tiles.append({
                    "mesh": {"v": v[unique_verts], "f": new_f.reshape(-1, 3).astype(np.int32)},
                    "bbox": (bx0, bx1, by0, by1),
                    "face_idx": np.flatnonzero(mask),
                })
            y += tile_size
        x += tile_size

    return tiles
