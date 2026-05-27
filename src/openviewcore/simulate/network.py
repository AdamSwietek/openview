from typing import Generator, Optional, Tuple, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd

_EPS = 1e-12

# ---------------------------------------------------------------------------
# Viewcone filter helpers (unchanged)
# ---------------------------------------------------------------------------

def _calc_angles_vectorized(v1, v2):
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)
    dot     = np.sum(v1 * v2, axis=1)
    denom   = v1_norm * v2_norm
    valid   = denom > 1e-8
    cos_theta = np.zeros(len(v1))
    cos_theta[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _preprocess_tree_fast(i, dist_tree_i, vp_all, norm_all,
                           VIEWCONE_CUTOFF=60, NORM_CUTOFF=120):
    if len(dist_tree_i) == 0:
        return []
    dist_tree_i      = np.array(dist_tree_i)
    pt_o             = vp_all[i]
    norm_o           = norm_all[i]
    pt_e             = vp_all[dist_tree_i]
    norm_e           = norm_all[dist_tree_i]
    ray_vec          = pt_e - pt_o
    norm_o_broadcast = np.broadcast_to(norm_o, pt_e.shape)
    norm_mask        = _calc_angles_vectorized(norm_o_broadcast, norm_e)      >= NORM_CUTOFF
    viewcone_mask    = _calc_angles_vectorized(norm_o_broadcast, ray_vec)     <= VIEWCONE_CUTOFF
    return dist_tree_i[viewcone_mask & norm_mask].tolist()


# ---------------------------------------------------------------------------
# Fix 1 + 4: streaming pair generator — never accumulates all N pairs in RAM
# ---------------------------------------------------------------------------

def _iter_los_pairs(
    facade_points,
    dist: float,
    chunk_size: int,
    viewcone_cutoff: float = 60,
    norm_cutoff: float = 120,
    batch_size: int = 2000,
) -> Generator[np.ndarray, None, None]:
    """Yield (≤chunk_size, 2) int64 pair arrays as they are produced.

    KDTree queries are processed in batches of `batch_size` viewpoints.
    Python-list buffers are kept to at most chunk_size + one batch worth of
    pairs, so peak overhead is O(chunk_size) Python ints rather than O(N).
    """
    from scipy.spatial import cKDTree
    vp   = facade_points['vp']
    norm = facade_points['norm']
    n    = len(vp)
    tree = cKDTree(vp)

    buf_src: list = []
    buf_tgt: list = []

    for start in tqdm(range(0, n, batch_size), desc='KDTree + ViewCone'):
        end = min(start + batch_size, n)
        for local_i, nbrs in enumerate(tree.query_ball_point(vp[start:end], dist)):
            i = start + local_i
            filtered = _preprocess_tree_fast(i, nbrs, vp, norm, viewcone_cutoff, norm_cutoff)
            if filtered:
                buf_src.extend([i] * len(filtered))
                buf_tgt.extend(filtered)

        # Yield complete chunks as soon as the buffer is large enough
        while len(buf_src) >= chunk_size:
            yield np.column_stack([
                np.array(buf_src[:chunk_size], dtype=np.int64),
                np.array(buf_tgt[:chunk_size], dtype=np.int64),
            ])
            buf_src = buf_src[chunk_size:]
            buf_tgt = buf_tgt[chunk_size:]

    if buf_src:
        yield np.column_stack([
            np.array(buf_src, dtype=np.int64),
            np.array(buf_tgt, dtype=np.int64),
        ])


# ---------------------------------------------------------------------------
# Fix 2: build BVH once in float32 (halves memory vs float64)
# ---------------------------------------------------------------------------

def _build_intersector(scene_data):
    import point_cloud_utils as pcu
    v = np.ascontiguousarray(scene_data['v'], dtype=np.float32)
    f = np.ascontiguousarray(scene_data['f'], dtype=np.int32)
    print(f"  Building BVH ({v.shape[0]:,} verts, {f.shape[0]:,} faces)...")
    return pcu.RayMeshIntersector(v, f)


# ---------------------------------------------------------------------------
# Fix 3: raycast one chunk with a pre-built intersector
# ---------------------------------------------------------------------------

def _raycast_chunk(pairs: np.ndarray, facade_points, intersector) -> dict:
    """Raycast a single pair chunk. `intersector` is reused across chunks."""
    # print('its happening')

    vp      = facade_points['vp']
    src_idx = pairs[:, 0]
    tgt_idx = pairs[:, 1]

    src_pts = vp[src_idx].astype(np.float32)
    tgt_pts = vp[tgt_idx].astype(np.float32)
    disp    = tgt_pts - src_pts
    dist    = np.linalg.norm(disp, axis=1)
    rdir    = disp / np.maximum(dist[:, None], 1e-12).astype(np.float32)

    _, _, hit_t = intersector.intersect_rays(
        np.asfortranarray(src_pts), np.asfortranarray(rdir)
    )
    hit_t = hit_t.reshape(-1).astype(np.float64)
    vis   = hit_t >= dist.astype(np.float64) * 0.99

    return {
        'pairs':        pairs,
        'ray_dir':      rdir.astype(np.float64),
        'src_tgt_dist': dist.astype(np.float64),
        'hit_t':        hit_t,
        'vis':          vis,
    }


# ---------------------------------------------------------------------------
# Fix 4 + 5: streaming build — writes directly to parquet, no full DataFrame
# ---------------------------------------------------------------------------

def build_los_network(
    facade_points,
    scene_data,
    dist: float,
    out_path,
    chunk_size: int = 500_000,
    wind_direction: Optional[float] = None,
    vis_only: bool = True,
) -> dict:
    """Stream LOS pairs → raycast → features → parquet.

    Peak RAM = BVH + one chunk's arrays (~50 MB/chunk at 500k pairs).
    Never accumulates all N pairs simultaneously.

    Parameters
    ----------
    facade_points : structured ndarray
        Output of ``generate_viewpoints``.
    scene_data : dict
        ``{'v': verts, 'f': faces}`` scene mesh.
    dist : float
        Maximum LOS search radius (CRS units, metres recommended).
    out_path : str or Path
        Destination parquet file.  Written in streaming row-groups.
    chunk_size : int
        Pairs per chunk.  500k uses ~200 MB peak (chunk arrays + BVH).
    wind_direction : float, optional
        Meteorological wind direction (degrees clockwise from North).
    vis_only : bool
        If True (default), drop occluded pairs before writing.  Reduces
        output size by ~3–5× since most candidate pairs are blocked.

    Returns
    -------
    dict with 'n_edges' and 'out_path'.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = str(out_path)
    intersector = _build_intersector(scene_data)

    writer    = None
    n_written = 0

    for chunk_i, pairs in enumerate(_iter_los_pairs(facade_points, dist, chunk_size)):
        los_chunk = _raycast_chunk(pairs, facade_points, intersector)

        if vis_only:
            vis_mask = los_chunk['vis']
            if not np.any(vis_mask):
                del los_chunk, pairs
                continue
            los_chunk = {
                k: (v[vis_mask] if isinstance(v, np.ndarray) and len(v) == len(vis_mask) else v)
                for k, v in los_chunk.items()
            }

        df    = compute_los_edge_features(
            los_chunk, facade_points, wind_direction=wind_direction
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_written += len(df)
        print(f"  chunk {chunk_i + 1}: {len(pairs):,} pairs  |  total written: {n_written:,}",
              end='\r', flush=True)
        del df, los_chunk, pairs   # release immediately

    if writer:
        writer.close()
    print(f"\nDone — {n_written:,} edges → {out_path}")
    return {'n_edges': n_written, 'out_path': out_path}


# ---------------------------------------------------------------------------
# Feature extraction (operates on a chunk dict or full dict — unchanged API)
# ---------------------------------------------------------------------------

def _safe_dist_from_dxyz(dxyz):
    return np.linalg.norm(dxyz, axis=1)


def _safe_use_dist(dist, dxyz):
    if dist is None:
        return _safe_dist_from_dxyz(dxyz)
    dist = np.asarray(dist, dtype=float)
    bad  = ~np.isfinite(dist) | (dist <= 0)
    if np.any(bad):
        dist        = dist.copy()
        dist[bad]   = _safe_dist_from_dxyz(dxyz)[bad]
    return dist


def _unit_direction(dxyz, dist):
    return dxyz * (1.0 / np.maximum(dist, _EPS))[:, None]


def _maybe_get_field(arr, name):
    names = getattr(arr, 'dtype', None)
    if names is None:
        return None
    names = getattr(names, 'names', None)
    return arr[name] if (names and name in names) else None


def _radiative_cosines(n_recv, n_emit, rhat_12, clamp=True):
    cr = np.einsum('ij,ij->i', n_recv,  rhat_12)
    ce = np.einsum('ij,ij->i', n_emit, -rhat_12)
    if clamp:
        cr = np.clip(cr, 0.0, None)
        ce = np.clip(ce, 0.0, None)
    return cr, ce


def _kernel_from_cosines(cr, ce, dist):
    return (cr * ce) / (np.pi * np.maximum(dist, _EPS) ** 2)


def _apply_visibility(kernel, vis):
    if vis is None:
        return kernel, None
    vis = np.asarray(vis, dtype=float)
    return kernel * vis, vis


def _compute_viewing_angles(dx, dy, dz):
    hdist     = np.sqrt(dx ** 2 + dy ** 2)
    pitch_deg = np.degrees(np.arctan2(dz, np.maximum(hdist, _EPS)))
    yaw_deg   = np.degrees(np.arctan2(dy, dx))
    return pitch_deg, yaw_deg, hdist


def _compute_wind_alignment(dx, dy, wind_direction):
    wind_rad   = np.radians(90.0 - wind_direction)
    wx, wy     = np.cos(wind_rad), np.sin(wind_rad)
    hmag       = np.sqrt(dx ** 2 + dy ** 2)
    valid      = hmag > _EPS
    wind_align = np.zeros_like(dx, dtype=float)
    wind_align[valid] = (dx[valid] * wx + dy[valid] * wy) / hmag[valid]
    return wind_align, np.clip(wind_align, 0.0, None)


def _compute_slope_features(dz, hdist):
    slope_pct   = dz / np.maximum(hdist, _EPS) * 100.0
    elev_weight = np.clip(slope_pct / 100.0, 0.0, None)
    return slope_pct, elev_weight


def compute_los_edge_features(
    los_data: Dict,
    viewpoint_data: np.ndarray,
    wind_direction: Optional[float] = None,
    include_building_ids: bool = True,
    filter_valid: bool = False,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """Compute geometric LOS edge features for a pair chunk or full pair array.

    Accepts the same dict format returned by ``_raycast_chunk`` or the legacy
    ``build_los_network`` return dict — no API change required.
    """
    pairs      = np.asarray(los_data['pairs'])
    source_idx = pairs[:, 0].astype(int)
    target_idx = pairs[:, 1].astype(int)

    vp = _maybe_get_field(viewpoint_data, 'vp')
    if vp is None:
        raise ValueError("viewpoint_data must have field 'vp'.")

    src  = vp[source_idx]
    tgt  = vp[target_idx]
    dxyz = tgt - src
    dx, dy, dz = dxyz[:, 0], dxyz[:, 1], dxyz[:, 2]

    raw_dist = los_data.get('src_tgt_dist', los_data.get('dist'))
    dist     = _safe_use_dist(raw_dist, dxyz)
    rhat_12  = (los_data['ray_dir'] if 'ray_dir' in los_data
                else _unit_direction(dxyz, dist))

    pitch_deg, yaw_deg, hdist = _compute_viewing_angles(dx, dy, dz)

    if wind_direction is not None:
        wind_align, wind_weight = _compute_wind_alignment(dx, dy, wind_direction)
    else:
        wind_align  = np.full(len(dx), np.nan)
        wind_weight = np.full(len(dx), np.nan)

    slope_pct, elev_weight = _compute_slope_features(dz, hdist)

    vis     = los_data.get('vis', 1)
    normals = _maybe_get_field(viewpoint_data, 'norm')

    if normals is not None:
        cr, ce          = _radiative_cosines(normals[source_idx], normals[target_idx], rhat_12)
        kernel          = _kernel_from_cosines(cr, ce, dist)
        kernel_vis, vis_arr = _apply_visibility(kernel, vis)
    else:
        cr = ce = kernel = kernel_vis = np.full(len(dist), np.nan)
        vis_arr = np.asarray(vis, dtype=float) if vis is not None else None

    area   = _maybe_get_field(viewpoint_data, 'patch_area')
    A_emit = area[target_idx].astype(float) if area is not None else None
    A_recv = area[source_idx].astype(float) if area is not None else None

    kernel_use       = kernel_vis if (vis is not None and normals is not None) else kernel
    geom_exposure_ij = kernel_use * A_emit if A_emit is not None else None
    view_factor_ij   = (geom_exposure_ij / np.maximum(A_recv, _EPS)
                        if (geom_exposure_ij is not None and A_recv is not None) else None)

    source_uuid = viewpoint_data['vp_uuid'][source_idx]
    target_uuid = viewpoint_data['vp_uuid'][target_idx]

    out = {
        'source_uuid':    source_uuid,
        'target_uuid':    target_uuid,
        'source_idx':     source_idx,
        'target_idx':     target_idx,
        'dist':           dist,
        'inv_dist2':      1.0 / np.maximum(dist, _EPS) ** 2,
        'log_dist':       np.log(np.maximum(dist, _EPS)),
        'horizontal_dist': hdist,
        'dz':             dz,
        'pitch_deg':      pitch_deg,
        'yaw_deg':        yaw_deg,
        'wind_align':     wind_align,
        'wind_weight':    wind_weight,
        'slope_pct':      slope_pct,
        'elev_weight':    elev_weight,
        'rx':             rhat_12[:, 0],
        'ry':             rhat_12[:, 1],
        'rz':             rhat_12[:, 2],
        'vis':            vis_arr if vis_arr is not None else np.nan,
        'hit_t':          los_data['hit_t'] if 'hit_t' in los_data else np.nan,
        'cos_theta_recv': cr,
        'cos_theta_emit': ce,
        'kernel':         kernel,
        'kernel_vis':     kernel_vis if vis is not None else np.nan,
        'A_emit':         A_emit if A_emit is not None else np.nan,
        'A_recv':         A_recv if A_recv is not None else np.nan,
        'geom_exposure_ij': geom_exposure_ij if geom_exposure_ij is not None else np.nan,
        'view_factor_ij':   view_factor_ij   if view_factor_ij   is not None else np.nan,
    }

    result = pd.DataFrame(out)

    if include_building_ids:
        osmid = _maybe_get_field(viewpoint_data, 'osmid')
        if osmid is not None:
            result.insert(0, 'target_BLD_ID', osmid[target_idx])
            result.insert(0, 'source_BLD_ID', osmid[source_idx])

    if filter_valid:
        result = result[
            result['dist'].notna() & np.isfinite(result['dist']) & (result['dist'] > 0)
        ].reset_index(drop=True)

    if deduplicate:
        result = deduplicate_edges(result, source_col='source_idx', target_col='target_idx')

    return result
