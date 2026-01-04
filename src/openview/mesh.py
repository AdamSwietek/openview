import pyvista as pv
import numpy as np
# from scipy.spatial import Delaunay

import rasterio

import numpy as np
from tqdm import tqdm 
import pyvista as pv

def polygon_to_vf(polygon_2d):
    """
    Convert a Shapely polygon into a pyvista PolyData mesh.

    Parameters:
    - polygon_2d: A Shapely Polygon object.

    Returns:
    """
    # Extract the exterior coordinates of the polygon
    coords = np.array(polygon_2d.exterior.coords)
    
    # Create 3D points from 2D coordinates by adding a zero Z value (flat surface)
    points = np.column_stack((coords[:, 0], coords[:, 1], coords[:, 2]))
    
    # Define the face of the polygon using the point indices
    n_points = len(points)
    faces = np.hstack([[n_points] + list(range(n_points))])
    
    # Create the mesh defined by points and faces
    mesh = {'v':points, 'f':faces}
    return mesh

def extrude_fp_to_pv(footprint, height):
    mesh = polygon_to_vf(footprint)
    mesh = pv.PolyData(mesh['v'], mesh['f']) #make pv
    mesh.extrude((0,0,height), inplace=True,capping=True) # extrude pv
    mesh = mesh.triangulate() # triangulate pv
    return mesh


def extrude_vf_prism(vertices, faces, height):
    """
    Extrude a single polygon face (v/f) along +Z into a triangular prism mesh.
    
    """
    # 1. Prepare Vertices
    v = np.asarray(vertices, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError("vertices must be (N,2) or (N,3)")

    if v.shape[1] == 2:
        v = np.column_stack([v, np.zeros(len(v), dtype=np.float32)])

    n = len(v)

    # 2. Prepare Faces & SANITIZE
    f = np.asarray(faces, dtype=np.int32)
    
    # Handle the case where f might be a list of lists or 2D array
    if f.ndim == 2:
        f = f.flatten() 
        
    if len(f) > 0:
        max_idx = f.max()
        min_idx = f.min()

        # Check if indices point outside the vertex list
        if max_idx >= n:
            # Case A: 1-based indexing (e.g., 1..10 for 10 verts)
            if min_idx >= 1:
                # print("-> Detected 1-based indexing in input. Shifting to 0-based.")
                f = f - 1
            
            # Case B: Fencepost/Garbage error (e.g., 0..10 for 10 verts)
            # If fixing 1-based didn't solve it, we simply CLAMP the top.
            if f.max() >= n:
                # print("-> Input faces out of bounds. Clamping to valid range.")
                f = np.clip(f, 0, n - 1)

    poly = f
    if len(poly) < 3:
        # If clamping reduced the polygon to a line/point, return empty or raise error
        # Here we return the flat shape to prevent crash
        return {"v": v, "f": np.array([], dtype=np.int32)}

    # 3. Create 3D Geometry
    if height == 0:
        return {"v": np.ascontiguousarray(v, dtype=np.float32),
                "f": np.ascontiguousarray(poly, dtype=np.int32)}

    # Duplicate vertices for the top cap
    top = v.copy()
    top[:, 2] += height
    v_out = np.vstack([v, top])

    # 4. Triangulate
    # Bottom (Reverse order for normal)
    base_tris = []
    top_tris = []
    
    # Simple fan triangulation (assumes convex or simple polygon)
    for i in range(1, len(poly) - 1):
        base_tris.append([poly[0], poly[i + 1], poly[i]]) 
        # Offset top indices by n
        top_tris.append([poly[0] + n, poly[i] + n, poly[i + 1] + n])

    # Side faces (Quads split into 2 triangles)
    side_tris = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        
        # Triangle 1
        side_tris.append([a, b, b + n])
        # Triangle 2
        side_tris.append([a, b + n, a + n])

    f_out = np.array(base_tris + top_tris + side_tris, dtype=np.int32)
    
    return {"v": np.ascontiguousarray(v_out, dtype=np.float32),
            "f": np.ascontiguousarray(f_out, dtype=np.int32)}

def extrude_fp_to_vf(input_data):
    mesh = polygon_to_vf(input_data['geom'])
    mesh = extrude_vf_prism(mesh['v'], mesh['f'], input_data['height'])
    return mesh

def meshify_buildings(data, method = 'pv'):
    geometries = data["geometry"].to_numpy()
    heights = data["height"].to_numpy()


    if method == 'pv':
        mesh = list(tqdm(
            (extrude_fp_to_pv(geom, height) for geom, height in zip(geometries, heights)),
            desc="extruding footprints",total=len(geometries)))
        
    elif method == 'vf':
        # --- THE REQUESTED UPDATE ---
        # Create a list of dictionaries to pass to the multiprocess pool
        input_data = [{'geom': g, 'height': h} for g, h in zip(geometries, heights)]
        
        import multiprocessing as mp
        with mp.Pool(mp.cpu_count()) as pool:
            mesh = list(
                tqdm(
                    pool.imap_unordered(extrude_fp_to_vf, input_data),
                    total=len(input_data),
                    desc="extruding footprints"
                )
            )
    return mesh


def extract_vf(mesh, clean = False):
    '''Extract vertices and faces from a PyVista mesh.'''

    if clean:
    # Step 0: Make sure no nan in pointlist,only applies to triangles 
        mesh = mesh.clean(merge_tol=1e-05)
        mesh = remove_nan_from_pvmesh(mesh)
    vertices = np.ascontiguousarray(mesh.points, dtype=np.float32)
    faces = mesh.faces
    if faces.size == 0:
        return {"v": vertices, "f": np.empty((0, 3), dtype=np.int32)}

    # Fast path for common triangulated/quad meshes (avoids Python loop).
    if mesh.is_all_triangles:
        faces_array = faces.reshape(-1, 4)[:, 1:]
    elif mesh.is_all_quads:
        faces_array = faces.reshape(-1, 5)[:, 1:]
    else:
        faces_list = []
        i = 0
        while i < len(faces):
            num_verts = faces[i]
            face = faces[i + 1:i + 1 + num_verts]
            faces_list.append(face)
            i += num_verts + 1
        faces_array = np.array(faces_list)

    faces_array = np.ascontiguousarray(faces_array, dtype=np.int32)
    return {'v': vertices, 'f': faces_array}

def remove_nan_from_pvmesh(mesh):
    '''remove nan for pvmesh point list '''
    # Create a mask of indices to remove
    mask = np.isnan(mesh.points).any(axis=1)

    # Use PyVista to properly rebuild the mesh without those points
    # 'inplace=True' modifies the existing mesh
    mesh.remove_points(mask, inplace=True)
    return mesh


import numpy as np

def join_list_of_meshes(mesh_list, id_list=None):
    """
    Joins a list of mesh dictionaries into a single mesh dictionary with Building IDs.
    
    Parameters:
    - mesh_list: List of dicts {'v':..., 'f':...}
    - id_list: Optional list of IDs corresponding to each mesh. 
               If None, uses the index (0, 1, 2...)
    """
    
    # 1. Extract Vertices and Faces
    try:
        # Try treating inputs as dictionaries
        vertice_list = [mesh['v'] for mesh in mesh_list]
        faces_list = [mesh['f'] for mesh in mesh_list]
    except:
        # Fallback if inputs are PolyData (requires extract_vf helper)
        vf_list = [extract_vf(mesh) for mesh in mesh_list] 
        (vertice_list, faces_list) = zip(*vf_list)

    # 2. Handle IDs
    # Get the number of vertices in each mesh to know how many times to repeat the ID
    vertex_counts = [v.shape[0] for v in vertice_list]
    
    if id_list is None:
        id_list = np.arange(len(mesh_list))
    
    # Create the ID array aligned with vertices
    # e.g., if mesh 0 has 2 verts and mesh 1 has 3 verts -> [0, 0, 1, 1, 1]
    joined_ids = np.repeat(id_list, vertex_counts)

    # 3. Offset Faces
    # We need to shift the face indices because we are stacking the vertices
    faces_offset = np.cumsum(vertex_counts)
    faces_offset = np.insert(faces_offset, 0, 0)[:-1]

    # 4. Stack Everything
    vertices = np.vstack(vertice_list)
    faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])
    
    # Return dict with new 'id' key
    return {'v': vertices, 'f': faces, 'id': joined_ids}


def meshify_large_dem(dem, n_downscaler=1, factor=1, reduce=True):
    """Build a terrain mesh from a DEM.

    Args:
        dem: xarray.DataArray with 'x' and 'y' coords.
        n_downscaler: Integer factor to downsample the DEM (1 = full res).
        factor: Vertical scale for warp_by_scalar.
        reduce: If True, simplify the resulting mesh.
    """
    if n_downscaler and n_downscaler > 1:
        # Prefer coarsen+mean to reduce aliasing on downsampled DEMs.
        if hasattr(dem, "coarsen"):
            coarsen_dims = {}
            if "y" in getattr(dem, "dims", ()):
                coarsen_dims["y"] = n_downscaler
            if "x" in getattr(dem, "dims", ()):
                coarsen_dims["x"] = n_downscaler
            if coarsen_dims:
                dem = dem.coarsen(boundary="trim", **coarsen_dims).mean()
            else:
                dem = dem[::n_downscaler, ::n_downscaler]
        else:
            dem = dem[::n_downscaler, ::n_downscaler]

    values = dem.fillna(0).values
    # Creating meshgrid
    x, y = np.meshgrid(dem['x'], dem['y'])

    # Setting zz values
    z = np.zeros_like(x)

    # Creating Structured Grid
    mesh = pv.StructuredGrid(x, y, z)

    # Assign elevation values
    mesh["elevation"] = values.ravel(order='F')

    # 
    terrain_mesh = mesh.warp_by_scalar(scalars="elevation", factor=factor)

    #triangulate
    terrain_mesh = terrain_mesh.extract_surface().triangulate()

    if reduce:
        import fast_simplification
        terrain_mesh = fast_simplification.simplify_mesh(terrain_mesh, target_reduction=0.9)

    return terrain_mesh

def pt_to_pvpoint(arr):
    '''transform numpy arr to pv point object, easier for plotting'''
    points = np.vstack(arr['vp'])
    normals = np.vstack(arr['norm'])

    cloud = pv.PolyData(points)
    cloud['osmid'] = arr['osmid']
    cloud['normals'] = normals            # store normals if you need them
    cloud['normals'] = cloud['normals']   # also sets active normals for lighting

    return cloud
