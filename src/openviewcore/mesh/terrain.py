"""Convert a DEM raster to a triangle mesh, with multi-resolution DEM merging."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import xarray as xr


def meshify_terrain(
    dem: xr.DataArray,
    *,
    downscale: int = 1,
    reduce: bool = True,
    target_reduction: float = 0.9,
) -> dict:
    """Build a triangulated terrain mesh from a DEM.

    Parameters
    ----------
    dem : xr.DataArray
        Elevation raster with ``x`` and ``y`` coordinates.
    downscale : int
        Coarsen factor (1 = full resolution).
    reduce : bool
        Simplify mesh via ``fast_simplification``.
    target_reduction : float
        Fraction of faces to remove (0-1) when *reduce* is True.

    Returns
    -------
    dict
        ``{'v': ndarray(N,3), 'f': ndarray(M,3)}``
    """
    if downscale > 1:
        coarsen_dims = {}
        if "y" in dem.dims:
            coarsen_dims["y"] = downscale
        if "x" in dem.dims:
            coarsen_dims["x"] = downscale
        if coarsen_dims:
            dem = dem.coarsen(boundary="trim", **coarsen_dims).mean()

    values = dem.fillna(0).values
    x, y = np.meshgrid(dem["x"], dem["y"])
    z = np.zeros_like(x)

    grid = pv.StructuredGrid(x, y, z)
    grid["elevation"] = values.ravel(order="F")
    terrain = grid.warp_by_scalar(scalars="elevation", factor=1)
    terrain = terrain.extract_surface().triangulate()

    if reduce:
        import fast_simplification
        terrain = fast_simplification.simplify_mesh(
            terrain, target_reduction=target_reduction,
        )

    return _extract_vf(terrain)


def meshify_canopy(
    local_dem: xr.DataArray,
    canopy_height: xr.DataArray,
    *,
    min_canopy_m: float = 0.5,
    downscale: int = 1,
    reduce: bool = False,
    target_reduction: float = 0.85,
) -> dict:
    """Build a triangulated canopy-top mesh from terrain + canopy height.

    The canopy surface is defined as ``local_dem + canopy_height``.  Pixels
    where canopy height is below *min_canopy_m* are excluded (set to the
    terrain elevation so they fall below the terrain mesh and are never hit
    by rays first).

    Parameters
    ----------
    local_dem : xr.DataArray
        Absolute terrain elevation raster (metres).
    canopy_height : xr.DataArray
        Height above ground in metres (e.g. from GEE canopy height product).
        May be in a different CRS — reprojected to match *local_dem*.
    min_canopy_m : float
        Pixels below this canopy height are masked out.
    downscale : int
        Coarsen factor passed to :func:`meshify_terrain`.
    reduce : bool
        Simplify mesh via ``fast_simplification``.
    target_reduction : float
        Fraction of faces to remove when *reduce* is True.

    Returns
    -------
    dict
        ``{'v': ndarray(N,3), 'f': ndarray(M,3)}`` canopy-top mesh.
    """
    # Align canopy_height to local_dem grid
    if canopy_height.rio.crs != local_dem.rio.crs:
        canopy_height = canopy_height.rio.reproject_match(local_dem)
    else:
        canopy_height = canopy_height.interp_like(local_dem, method="nearest")

    # Optional downscale (mirrors meshify_terrain behaviour)
    if downscale > 1:
        coarsen_dims = {}
        if "y" in local_dem.dims:
            coarsen_dims["y"] = downscale
        if "x" in local_dem.dims:
            coarsen_dims["x"] = downscale
        local_dem      = local_dem.coarsen(boundary="trim", **coarsen_dims).mean()
        canopy_height  = canopy_height.coarsen(boundary="trim", **coarsen_dims).mean()

    ch       = canopy_height.fillna(0).values
    dem_vals = local_dem.fillna(0).values

    # Absolute elevation of canopy top surface
    canopy_surface = dem_vals + np.maximum(ch, 0)

    # Build structured grid with both elevation and raw canopy-height scalars
    x, y = np.meshgrid(local_dem["x"], local_dem["y"])
    z    = np.zeros_like(x)
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["elevation"] = canopy_surface.ravel(order="F")
    grid.point_data["ch"]        = ch.ravel(order="F")

    # Remove cells where ANY corner has ch < min_canopy_m.
    # all_scalars=True → a cell is kept only when every one of its vertices
    # passes the threshold, so non-canopy pixels create genuine gaps between
    # canopy islands instead of connecting them through the ground plane.
    grid_masked = grid.threshold(
        value=min_canopy_m,
        scalars="ch",
        preference="point",
        all_scalars=True,
    )

    if grid_masked.n_cells == 0:
        return {"v": np.empty((0, 3), dtype=np.float32),
                "f": np.empty((0, 3), dtype=np.int32)}

    warped  = grid_masked.warp_by_scalar(scalars="elevation", factor=1)
    surface = warped.extract_surface().triangulate()
    surface = surface.clean()       # remove orphaned vertices at island edges

    if reduce and surface.n_cells > 0:
        import fast_simplification
        surface = fast_simplification.simplify_mesh(
            surface, target_reduction=target_reduction,
        )

    return _extract_vf(surface)


def merge_terrain_meshes(
    inner: dict,
    outer: dict,
) -> dict:
    """Combine a dense inner mesh with a coarse outer mesh into one terrain.

    Three steps:
    1. Punch a hole in *outer* by removing faces whose centroid falls inside
       the *inner* mesh's XY footprint.
    2. Join the two meshes (open boundary gap between them).
    3. Stitch the gap: Delaunay-triangulate the annular zone between the
       two open boundary rings to produce a seamless surface.

    Parameters
    ----------
    inner : dict
        High-resolution mesh for the local area ``{'v', 'f'}``.
    outer : dict
        Low-resolution mesh for the wider region ``{'v', 'f'}``.

    Returns
    -------
    dict
        ``{'v': ndarray(N,3), 'f': ndarray(M,3)}`` — seamless merged terrain.
    """
    from shapely.geometry import MultiPoint

    # --- 1. Punch hole in outer --------------------------------------------
    inner_footprint = MultiPoint(inner["v"][:, :2]).convex_hull

    outer_tri_verts = outer["v"][outer["f"]]
    centroids_xy = outer_tri_verts[:, :, :2].mean(axis=1)
    inside = _contains_xy(inner_footprint, centroids_xy[:, 0], centroids_xy[:, 1])
    outer_f_trimmed = outer["f"][~inside]
    outer_v_clean, outer_f_clean = _compact_mesh(outer["v"], outer_f_trimmed)

    # --- 2. Build combined mesh (with gap) ---------------------------------
    n_inner = inner["v"].shape[0]
    combined_v = np.vstack([inner["v"], outer_v_clean])
    inner_f_combined = inner["f"]
    outer_f_combined = outer_f_clean + n_inner

    # --- 3. Find open boundary rings in combined space --------------------
    # Inner mesh boundary = its outer perimeter
    inner_bnd = _boundary_vertex_indices(inner_f_combined)
    # Outer mesh boundary (hole edge) — local to outer_f_clean, shift to combined
    outer_bnd = _boundary_vertex_indices(outer_f_combined)

    # --- 4. Delaunay stitch the gap ----------------------------------------
    stitch_f = _stitch_gap(combined_v, inner_bnd, outer_bnd, inner_footprint)

    # --- 5. Assemble -------------------------------------------------------
    all_f = np.vstack([inner_f_combined, outer_f_combined, stitch_f])

    return {
        "v": np.ascontiguousarray(combined_v, dtype=np.float32),
        "f": np.ascontiguousarray(all_f, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Gap-stitching helpers
# ---------------------------------------------------------------------------

def _boundary_vertex_indices(f: np.ndarray) -> np.ndarray:
    """Return indices of vertices that lie on boundary (open) edges.

    A boundary edge is shared by exactly one triangle.
    """
    # All directed half-edges
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)

    _, inverse, counts = np.unique(
        edges_sorted, axis=0, return_inverse=True, return_counts=True,
    )
    boundary_edge_mask = counts[inverse] == 1
    boundary_edges = edges_sorted[boundary_edge_mask]
    return np.unique(boundary_edges)


def _stitch_gap(
    combined_v: np.ndarray,
    inner_bnd: np.ndarray,
    outer_bnd: np.ndarray,
    inner_footprint,
) -> np.ndarray:
    """Delaunay-triangulate the annular gap between two boundary rings.

    Parameters
    ----------
    combined_v : ndarray (N, 3)
        All vertices in the combined mesh space.
    inner_bnd : ndarray
        Global vertex indices of the inner mesh boundary ring.
    outer_bnd : ndarray
        Global vertex indices of the outer mesh hole boundary ring.
    inner_footprint : shapely Polygon
        XY footprint of the inner mesh (used to filter out triangles that
        would fill inside the inner mesh rather than the gap).

    Returns
    -------
    ndarray (M, 3) int32
        Face index array (in combined_v space) for stitching triangles.
    """
    from scipy.spatial import Delaunay

    # Pool all boundary vertex indices and their 2-D positions
    all_bnd_idx = np.concatenate([inner_bnd, outer_bnd])
    pts_2d = combined_v[all_bnd_idx, :2]

    # Delaunay triangulation in the XY plane
    tri = Delaunay(pts_2d)
    local_tris = tri.simplices  # indices into pts_2d / all_bnd_idx

    # Filter: keep only triangles whose centroid lies in the gap
    # (outside the inner footprint — we don't want to cap the inner mesh)
    centroids = pts_2d[local_tris].mean(axis=1)
    in_inner = _contains_xy(inner_footprint, centroids[:, 0], centroids[:, 1])
    valid_tris = local_tris[~in_inner]

    # Remap local Delaunay indices -> global combined_v indices
    return all_bnd_idx[valid_tris].astype(np.int32)


def _contains_xy(geom, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon test compatible with shapely 1.x and 2.x."""
    try:
        from shapely import contains_xy  # shapely >= 2.0
        return contains_xy(geom, x, y)
    except ImportError:
        from shapely.vectorized import contains  # shapely 1.x
        return contains(geom, x, y)


def _compact_mesh(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove vertices unreferenced by *f* and reindex faces."""
    used, new_idx = np.unique(f.ravel(), return_inverse=True)
    return v[used], new_idx.reshape(f.shape).astype(np.int32)


def _extract_vf(mesh: pv.PolyData) -> dict:
    """PyVista mesh -> ``{'v': float32(N,3), 'f': int32(M,3)}``."""
    vertices = np.ascontiguousarray(mesh.points, dtype=np.float32)
    faces = mesh.faces
    if faces.size == 0:
        return {"v": vertices, "f": np.empty((0, 3), dtype=np.int32)}

    if mesh.is_all_triangles:
        faces_arr = faces.reshape(-1, 4)[:, 1:]
    else:
        faces_list, i = [], 0
        while i < len(faces):
            n = faces[i]
            faces_list.append(faces[i + 1 : i + 1 + n])
            i += n + 1
        faces_arr = np.array(faces_list)

    return {
        "v": vertices,
        "f": np.ascontiguousarray(faces_arr, dtype=np.int32),
    }
