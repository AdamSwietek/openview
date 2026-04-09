"""Generate facade viewpoints from building footprints.

Consolidates the old shape.py + scale.py into one module.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
from tqdm import tqdm

from ..config import (
    DEFAULT_FACADE_OFFSET,
    DEFAULT_FLOOR_HEIGHT,
    DEFAULT_GROUND_FLOOR_H,
    DEFAULT_MIN_SEGMENT_LEN,
)
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_PLACES", "threads")

# ---------------------------------------------------------------------------
# Structured array dtype shared across the codebase
# ---------------------------------------------------------------------------
VP_DTYPE = np.dtype([
    ("vp_uuid", np.int64),
    ("osmid", np.int64),
    ("vp", np.float64, (3,)),
    ("norm", np.float64, (3,)),
    ("segment_len", np.float64),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_viewpoints(
    buildings: gpd.GeoDataFrame,
    *,
    terrain_mesh: Optional[dict] = None,
    floor_height: float = DEFAULT_FLOOR_HEIGHT,
    ground_floor_h: float = DEFAULT_GROUND_FLOOR_H,
    facade_offset: float = DEFAULT_FACADE_OFFSET,
    min_segment_len: float = DEFAULT_MIN_SEGMENT_LEN,
) -> np.ndarray:
    """Create facade viewpoints for all buildings (multiprocessing).

    Parameters
    ----------
    buildings : GeoDataFrame
        Must have ``osmid``, ``height``, ``elevation``, ``geometry``,
        and ``old_geom`` columns.
    terrain_mesh : dict, optional
        ``{'v': (N,3), 'f': (M,3)}`` terrain mesh.  When provided,
        viewpoints below the terrain surface are discarded via downward
        ray casting (faster and more accurate than interpolation).
    floor_height, ground_floor_h, facade_offset, min_segment_len
        See :class:`~openviewcore.config.PipelineConfig`.

    Returns
    -------
    np.ndarray
        Structured array with dtype :data:`VP_DTYPE`.
    """
    params = {
        "floor_height": floor_height,
        "ground_floor_h": ground_floor_h,
        "facade_offset": facade_offset,
        "min_segment_len": min_segment_len,
    }

    work = [_row_to_dict(row, params) for row in buildings.itertuples()]

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(_viewpoints_for_building, work),
            total=len(work),
            desc="Generating viewpoints",
        ))

    viewpoints = _to_structured_array(results)

    if terrain_mesh is not None:
        n_before = len(viewpoints)
        viewpoints = _filter_subterrain(viewpoints, terrain_mesh)
        n_removed = n_before - len(viewpoints)
        if n_removed:
            print(f"  Removed {n_removed} sub-terrain viewpoints "
                  f"({n_removed / n_before:.1%} of total)")

    return viewpoints


# ---------------------------------------------------------------------------
# Per-building viewpoint extraction (runs in worker process)
# ---------------------------------------------------------------------------

def _viewpoints_for_building(item: dict) -> list[dict]:
    """Compute viewpoints for a single building dict (pickle-safe)."""
    old_geom = wkt.loads(item["old_geom"])

    footprint_buffered = old_geom.buffer(item["facade_offset"]).exterior
    elevation = item["elevation"]
    osmid = item["osmid"]
    height = item["height"]
    floor_h = item["floor_height"]
    ground_h = item["ground_floor_h"]
    min_seg = item["min_segment_len"]
    offset = item["facade_offset"]

    segments = _split_line_into_segments(footprint_buffered)

    viewpoints: list[dict] = []
    for seg in segments:
        midpoint, seg_len = _segment_midpoint(seg, min_seg)
        if midpoint is None:
            continue
        normal = _segment_normal(seg)

        for h in _floor_range(ground_h, height, floor_h):
            vp = [
                midpoint.x + normal[0] * offset,
                midpoint.y + normal[1] * offset,
                elevation + h,
            ]
            viewpoints.append({
                "vp_uuid": 0, #placeholder
                "osmid": osmid,
                "vp": vp,
                "norm": [normal[0], normal[1], 0.0],
                "segment_len": seg_len,
            })

    return viewpoints


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _split_line_into_segments(line: LineString) -> list[LineString]:
    coords = list(line.coords)
    return [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]


def _segment_midpoint(seg: LineString, min_len: float):
    length = seg.length
    if length < min_len:
        return None, 0.0
    return seg.interpolate(length / 2), length


def _segment_normal(seg: LineString) -> tuple[float, float, float]:
    (x1, y1), (x2, y2) = seg.coords[0][:2], seg.coords[1][:2]
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    ux, uy = dx / length, dy / length
    return (-uy, ux, 0.0)


def _floor_range(ground_h: float, building_h: float, floor_h: float):
    """Yield floor heights from ground floor up to building height."""
    h = ground_h
    while h < building_h:
        yield h
        h += floor_h


# ---------------------------------------------------------------------------
# Terrain filter
# ---------------------------------------------------------------------------

def _filter_subterrain(viewpoints: np.ndarray, terrain_mesh: dict) -> np.ndarray:
    """Drop viewpoints below the terrain surface using downward ray casting.

    Cast a -Z ray from 2 m below each viewpoint:
      fid >= 0  ->  terrain hit below origin  ->  viewpoint is above ground  ->  keep
      fid <  0  ->  no hit                   ->  viewpoint is sub-terrain   ->  discard
    """
    import point_cloud_utils as pcu

    vp_o = np.ascontiguousarray(viewpoints["vp"], dtype=np.float32)

    # Start 2 m below each viewpoint eliminate low ground clearnance pts 
    vp_o_up = vp_o.copy()
    vp_o_up[:, 2] = vp_o[:, 2] - 2.0

    ray_d = np.ascontiguousarray(
        np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (len(vp_o), 1)),
    )

    fid, _, _ = pcu.ray_mesh_intersection(
        terrain_mesh["v"], terrain_mesh["f"], vp_o_up, ray_d,
    )

    return viewpoints[fid >= 0]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row, params: dict) -> dict:
    return {
        "osmid": row.osmid,
        "geometry": row.geometry.wkt,
        "old_geom": row.old_geom.wkt,
        "height": row.height,
        "elevation": row.elevation,
        **params,
    }


def _to_structured_array(results: list[list[dict]]) -> np.ndarray:
    flat = [item for sublist in results if sublist for item in sublist]
    arr = np.array(
        [(d["vp_uuid"], d["osmid"], d["vp"], d["norm"], d["segment_len"])
         for d in flat],
        dtype=VP_DTYPE,
    )
    arr["vp_uuid"] = np.arange(len(arr), dtype=np.int64)
    return arr
