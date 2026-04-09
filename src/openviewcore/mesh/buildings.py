"""Extrude 2-D building footprints into 3-D prism meshes."""

from __future__ import annotations

import multiprocessing as mp
from shapely.geometry import Polygon, Point
from tqdm import tqdm

import numpy as np
import geopandas as gpd


def meshify_buildings(buildings: gpd.GeoDataFrame) -> list[dict]:
    """Extrude every building footprint into a 3-D triangulated mesh.

    Parameters
    ----------
    buildings : GeoDataFrame
        Must contain ``geometry`` (3-D Polygons) and ``height`` (meters).

    Returns
    -------
    list[dict]
        Each element is ``{'v': float32(N,3), 'f': int32(M,3)}``.
    """
    work = [
        {"geom": geom, "height": h}
        for geom, h in zip(buildings.geometry, buildings.height)
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        meshes = list(tqdm(
            pool.imap(_extrude_single, work),
            total=len(work),
            desc="Extruding buildings",
        ))

    return meshes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extrude_single(item: dict) -> dict:
    """Worker: polygon + height -> extruded prism mesh.

    Cap triangulation uses Delaunay + centroid-in-polygon filtering so that
    concave footprints (L, U, T shapes, etc.) are handled correctly.
    Fan triangulation only works for convex polygons and produced overhanging
    roof faces for non-convex ones.

    Side walls are unchanged from the original (per-edge quad splitting).
    """
    from scipy.spatial import Delaunay

    geom   = item["geom"]
    height = float(item["height"])

    # Base elevation stored in the 3-D polygon by elevate_footprints
    coords = list(geom.exterior.coords)
    base_z = float(coords[0][2]) if len(coords[0]) > 2 else 0.0

    # 2-D polygon (strip Z, repair if needed)
    poly2d = Polygon([(c[0], c[1]) for c in coords])
    if not poly2d.is_valid:
        poly2d = poly2d.buffer(0)

    # Ring vertices — drop the closing duplicate
    ring_xy = np.array(poly2d.exterior.coords[:-1], dtype=np.float64)
    n = len(ring_xy)

    if n < 3 or height <= 0 or poly2d.is_empty:
        return {"v": np.empty((0, 3), dtype=np.float32),
                "f": np.empty((0, 3), dtype=np.int32)}

    # ------------------------------------------------------------------
    # Cap triangulation: Delaunay over the ring vertices, then discard
    # triangles whose centroid lies outside the polygon.
    # Delaunay covers the convex hull, so exterior triangles appear for
    # any concave footprint — the centroid test removes them.
    # A small buffer avoids rejecting triangles that merely touch the
    # boundary due to floating-point rounding.
    # ------------------------------------------------------------------
    poly_buf = poly2d.buffer(0.1)   # 10 cm tolerance
    try:
        tri    = Delaunay(ring_xy)
        cap_f  = np.array(
            [s for s in tri.simplices
             if poly_buf.contains(Point(ring_xy[s].mean(axis=0)))],
            dtype=np.int32,
        )
    except Exception:
        cap_f = np.empty((0, 3), dtype=np.int32)

    if len(cap_f) == 0:
        # Fallback: fan triangulation (correct for convex polygons)
        cap_f = np.array([[0, i, i + 1] for i in range(1, n - 1)],
                         dtype=np.int32)

    # ------------------------------------------------------------------
    # 3-D vertices: bottom ring at base_z, top ring at base_z + height
    # ------------------------------------------------------------------
    xy32  = ring_xy.astype(np.float32)
    v_bot = np.column_stack([xy32, np.full(n, base_z,          dtype=np.float32)])
    v_top = np.column_stack([xy32, np.full(n, base_z + height, dtype=np.float32)])
    v_out = np.vstack([v_bot, v_top])   # 0..n-1 = bottom, n..2n-1 = top

    # Caps: flip base winding so normal points downward (-Z)
    base_tris = cap_f[:, ::-1]
    top_tris  = cap_f + n

    # Side walls: per-edge quad -> 2 triangles (unchanged from original)
    idx  = np.arange(n, dtype=np.int32)
    a, b = idx, (idx + 1) % n
    side_tris = np.vstack([
        np.column_stack([a, b, b + n]),
        np.column_stack([a, b + n, a + n]),
    ])

    f_out = np.vstack([base_tris, top_tris, side_tris]).astype(np.int32)

    return {
        "v": np.ascontiguousarray(v_out),
        "f": np.ascontiguousarray(f_out),
    }
