"""Create ray origins and directions from a viewpoint array."""

from __future__ import annotations

import numpy as np

from ..config import DEFAULT_RAY_HFOV, DEFAULT_RAY_RESOLUTION


import numpy as np

def create_rays(
    viewpoints: np.ndarray,
    *,
    hfov: float = 45.0,
    resolution: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch creates all rays for all viewpoints without any Python loops."""
    n_vps = len(viewpoints)
    n_rays_per_vp = resolution * resolution

    # 1. Generate the LOCAL grid once (R^2, 3)
    r = np.tan(np.deg2rad(hfov))
    x, y = np.mgrid[-r:r:resolution * 1j, -r:r:resolution * 1j]
    local_rays = np.stack([x.ravel(), y.ravel(), np.ones(x.size)], axis=-1).astype(np.float32)
    local_rays /= np.linalg.norm(local_rays, axis=-1, keepdims=True)

    # 2. Batch Build Orthonormal Bases (V, 3, 3)
    # Forward vector (normalized)
    fwd = viewpoints["norm"].astype(np.float32)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)

    # World Up
    world_up = np.array([0, 0, 1], dtype=np.float32)

    # Right vector: cross(world_up, fwd)
    right = np.cross(world_up, fwd)
    
    # Handle singularities (looking straight up/down)
    mag = np.linalg.norm(right, axis=-1, keepdims=True)
    right = np.where(mag < 1e-6, np.array([1, 0, 0], dtype=np.float32), right / mag)
    
    # Up vector
    up = np.cross(fwd, right)

    # Rotation Matrices: Shape (V, 3, 3)
    # R[v] = [right_v, up_v, fwd_v] as columns
    R = np.stack([right, up, fwd], axis=-1)

    # 3. Rotate all rays for all viewpoints (V, R^2, 3)
    # einsum 'vij, kj -> vki' 
    # v=viewpoint, i=row, j=col, k=ray_index
    # Effectively: For each viewpoint, multiply its 3x3 by all 100 rays
    ray_dirs = np.einsum('vij, kj -> vki', R, local_rays)

    # 4. Final Flattening and ID Generation
    # Reshape to (V * R^2, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)
    
    # Repeat origins: [o1, o1... (100x), o2, o2... (100x)]
    ray_origins = np.repeat(viewpoints["vp"], n_rays_per_vp, axis=0).astype(np.float32)
    
    # Repeat IDs
    vp_ids = np.repeat(viewpoints["vp_uuid"], n_rays_per_vp, axis=0)

    return (
        np.ascontiguousarray(ray_origins),
        np.ascontiguousarray(ray_dirs),
        vp_ids
    )

# def create_rays(
#     viewpoints: np.ndarray,
#     *,
#     hfov: float = DEFAULT_RAY_HFOV,
#     resolution: int = DEFAULT_RAY_RESOLUTION,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Build stacked ray arrays for all viewpoints.

#     Parameters
#     ----------
#     viewpoints : np.ndarray
#         Structured array with ``vp`` (origin) and ``norm`` (direction) fields.
#     hfov : float
#         Half field-of-view in degrees.
#     resolution : int
#         Grid cells per side of the view window.

#     Returns
#     -------
#     ray_origins : ndarray (N, 3) float32
#     ray_dirs    : ndarray (N, 3) float32
#     vp_ids      : ndarray (N,)   int64 (viewpoint id per ray)
#     """
#     all_o: list[np.ndarray] = []
#     all_d: list[np.ndarray] = []
#     all_id: list[np.ndarray] = []

#     for vp_data in viewpoints:
#         ray_d = _window_grid(vp_data["norm"], hfov, resolution)
#         ray_o = np.broadcast_to(
#             np.ascontiguousarray(vp_data["vp"], dtype=np.float32)[None, :],
#             ray_d.shape,
#         ).copy()

#         # rpid = uuid.uuid4()
#         vpid = vp_data['vp_uuid']
#         all_o.append(ray_o)
#         all_d.append(ray_d)
#         all_id.append(np.full(ray_o.shape[0], vpid))

#     return (
#         np.vstack(all_o).astype(np.float32),
#         np.vstack(all_d).astype(np.float32),
#         np.concatenate(all_id),
#     )


# ---------------------------------------------------------------------------
# Internal: window grid + rotation
# ---------------------------------------------------------------------------

def _window_grid(
    norm: np.ndarray,
    hfov: float,
    resolution: int,
) -> np.ndarray:
    """Create a grid of ray directions aligned to the *norm* vector."""
    # 1. Generate the initial grid on a local plane (facing +Z)
    half_fov = np.deg2rad(hfov)
    r = np.tan(half_fov)
    x, y = np.mgrid[-r:r:resolution * 1j, -r:r:resolution * 1j]
    
    # We build the grid where Z is forward, X is right, Y is up
    ray_d = np.stack(
        [x.ravel(), y.ravel(), np.ones(x.size, dtype=np.float32)], axis=-1,
    ).astype(np.float32)
    ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)

    # 2. Build an Orthonormal Basis (Camera Matrix)
    # This prevents the 'skew' by explicitly defining the horizon.
    forward = norm / np.linalg.norm(norm)
    world_up = np.array([0, 0, 1], dtype=np.float32)

    # Right vector is perpendicular to world up and forward direction
    right = np.cross(world_up, forward)
    
    # Handle the 'singularity' if the viewpoint is looking exactly Up or Down
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0], dtype=np.float32)
    else:
        right /= np.linalg.norm(right)

    # Actual UP vector for the viewcone frame
    up = np.cross(forward, right)

    # 3. Construct Rotation Matrix
    # Columns define the mapping from local (X, Y, Z) to global (Right, Up, Forward)
    R = np.stack([right, up, forward], axis=-1)

    # 4. Transform rays
    ray_d = (R @ ray_d.T).T

    return np.ascontiguousarray(ray_d, dtype=np.float32)

def create_hemisphere_rays(
    viewpoints: np.ndarray,
    *,
    n_azimuth: int = 36,
    n_zenith: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create equal-area hemisphere rays for sky-view / canopy simulation.

    Samples the upper hemisphere uniformly in solid angle by spacing azimuth
    uniformly and zenith via equal increments of cos(θ), so each ray
    represents the same solid angle patch.

    Parameters
    ----------
    viewpoints : np.ndarray
        Structured array with ``vp`` and ``norm`` fields (VP_DTYPE).
        norm should be [0, 0, 1] for upward-looking sky viewpoints.
    n_azimuth : int
        Number of azimuth divisions (default 36 → 10° steps).
    n_zenith : int
        Number of zenith divisions (default 9 → equal cos(θ) bins from
        horizon to zenith, excluding the horizon ring itself).

    Returns
    -------
    ray_origins : ndarray (N, 3) float32
    ray_dirs    : ndarray (N, 3) float32   — all Z > 0 (upper hemisphere)
    vp_ids      : ndarray (N,)  int64
    """
    # Equal-area zenith bins: uniform in cos(θ), θ ∈ (0°, 90°]
    # cos_edges from 0 (horizon) to 1 (zenith), bin centres used
    cos_edges  = np.linspace(0.0, 1.0, n_zenith + 1)
    cos_centres = 0.5 * (cos_edges[:-1] + cos_edges[1:])        # (n_zenith,)
    sin_centres = np.sqrt(1.0 - cos_centres ** 2)

    # Azimuth bin centres: uniform in [0, 2π)
    phi_centres = 2.0 * np.pi * (np.arange(n_azimuth) + 0.5) / n_azimuth  # (n_az,)

    # Build local ray directions on a Z-up frame: shape (n_zenith * n_azimuth, 3)
    sin_z, cos_z = sin_centres[:, None], cos_centres[:, None]   # (nz, 1)
    sin_p, cos_p = np.sin(phi_centres), np.cos(phi_centres)     # (naz,)

    lx = (sin_z * cos_p[None, :]).ravel()   # (nz*naz,)
    ly = (sin_z * sin_p[None, :]).ravel()
    lz = np.repeat(cos_centres, n_azimuth)

    local_rays = np.stack([lx, ly, lz], axis=-1).astype(np.float32)  # (R, 3)
    n_rays_per_vp = len(local_rays)

    # Batch-rotate into each viewpoint's frame (reuse same logic as create_rays)
    fwd = viewpoints["norm"].astype(np.float32)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(world_up, fwd)
    mag = np.linalg.norm(right, axis=-1, keepdims=True)
    safe_mag = np.where(mag < 1e-6, 1.0, mag)           # avoid divide-by-zero
    right = np.where(mag < 1e-6, np.array([1.0, 0.0, 0.0], dtype=np.float32), right / safe_mag)
    up = np.cross(fwd, right)

    R = np.stack([right, up, fwd], axis=-1)              # (V, 3, 3)
    ray_dirs = np.einsum('vij, kj -> vki', R, local_rays)  # (V, R, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)

    ray_origins = np.repeat(viewpoints["vp"], n_rays_per_vp, axis=0).astype(np.float32)
    vp_ids      = np.repeat(viewpoints["vp_uuid"], n_rays_per_vp)

    return (
        np.ascontiguousarray(ray_origins),
        np.ascontiguousarray(ray_dirs),
        vp_ids,
    )


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c + ux**2 * (1 - c),       ux*uy*(1 - c) - uz*s,  ux*uz*(1 - c) + uy*s],
        [uy*ux*(1 - c) + uz*s,      c + uy**2 * (1 - c),   uy*uz*(1 - c) - ux*s],
        [uz*ux*(1 - c) - uy*s,      uz*uy*(1 - c) + ux*s,  c + uz**2 * (1 - c)],
    ], dtype=np.float32)
