
import numpy as np


def create_window_grid(norm, resolution=100):
    # Step 1: Generate the raster grid of rays looking down the +Z axis
    ray_d = create_mgrid_cached(resolution)

    new_direction = np.array([norm[0], norm[1], 0], dtype=np.float32)
    new_direction /= np.linalg.norm(new_direction)
    
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    rotation_axis = np.cross(z_axis, new_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm > 1e-6:
        rotation_axis /= rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, new_direction), -1.0, 1.0))
        R = rotation_matrix(rotation_axis, angle)
        rotated_rays = np.dot(ray_d, R.T)
    else:
        rotated_rays = ray_d
    
    return np.ascontiguousarray(rotated_rays, dtype=np.float32)

def create_mgrid_cached(resolution):
    """Cached version of window grids with reduced resolution"""
    x, y = np.mgrid[-0.5:0.5:resolution*1j, -0.5:0.5:resolution*1j]
    
    ray_d = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())], axis=-1)
    ray_d = ray_d.astype(np.float32)
    ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)
    
    return ray_d

def rotation_matrix(axis, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [cos_a + ux**2 * (1 - cos_a), ux*uy * (1 - cos_a) - uz*sin_a, ux*uz * (1 - cos_a) + uy*sin_a],
        [uy*ux * (1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a), uy*uz * (1 - cos_a) - ux*sin_a],
        [uz*ux * (1 - cos_a) - uy*sin_a, uz*uy * (1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
    ])