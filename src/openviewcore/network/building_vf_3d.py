"""Tiled, resumable construction of a finite-height 3D building VF graph.

The model samples vertical facade patches, casts one ray per unique patch pair
through an Embree BVH containing building walls/roofs and optional terrain,
then conservatively reduces reciprocal patch exchange to directional building
view factors. All internal lengths are metres and all exchange measures are
square metres.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.ops import triangulate
from tqdm.auto import tqdm

from .building_vf import _iter_tiles, _polygon_parts, _source_spec


PATCH_STRIDE_3D = 1_000_000
COLUMN_STRIDE_3D = 100_000


def _empty_frame(schema: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in schema.items()})


def _empty_patches() -> pd.DataFrame:
    return _empty_frame({
        "patch_id": "int64", "building_id": "int64",
        "column_id": "int64",
        "x_m": "float64", "y_m": "float64", "z_m": "float64",
        "nx": "float64", "ny": "float64", "nz": "float64",
        "width_m": "float64", "height_m": "float64", "area_m2": "float64",
        "z_low_m": "float64", "z_high_m": "float64",
        "home_ix": "int64", "home_iy": "int64",
    })


def _empty_contacts() -> pd.DataFrame:
    return _empty_frame({
        "patch_id": "int64", "building_id": "int64",
        "receiver_building": "int64", "contact_fraction": "float64",
        "contact_exchange_area_m2": "float64",
    })


def _empty_exchange() -> pd.DataFrame:
    return _empty_frame({
        "patch_p": "int64", "patch_q": "int64",
        "building_p": "int64", "building_q": "int64",
        "distance_m": "float64", "kernel_pq_m2_inv": "float64",
        "exchange_pq_raw_m2": "float64", "exchange_pq_prelim_m2": "float64",
    })


@dataclass(frozen=True)
class BVF3DGraphConfig:
    """Parameters for a tiled, finite-height facade-area approximation."""

    id_column: str = "OBJECTID"
    source_id_column: Optional[str] = "BLD_ID"
    layer: Optional[str] = None
    target_crs: str = "EPSG:26911"
    tile_size_m: float = 2_000.0
    max_distance_m: float = 152.4
    max_angle_deg: float = 60.0
    height_column: str = "HEIGHT"
    height_scale_to_m: float = 0.3048
    elevation_column: Optional[str] = "ELEV"
    elevation_scale_to_m: float = 0.3048
    default_height_m: float = 3.0
    default_elevation_m: float = 0.0
    max_patch_width_m: float = 20.0
    max_patch_height_m: float = 3.0
    max_vertical_patches: int = 3
    min_edge_length_m: float = 2.0
    contact_tolerance_m: float = 0.05
    ray_offset_m: float = 0.05
    hit_tolerance_m: float = 0.01
    query_batch_size: int = 500
    ray_batch_size: int = 500_000
    dem_path: Optional[str] = None
    dem_z_scale_to_m: float = 1.0
    terrain_resolution_m: float = 30.0
    use_dem_for_building_elevation: bool = True
    workers: int = 1
    resume: bool = True


@dataclass(frozen=True)
class BVF3DGraphResult:
    output_dir: Path
    patches_path: Path
    contacts_path: Path
    building_ids_path: Path
    patch_exchange_path: Path
    building_edges_path: Path
    building_metrics_path: Path
    manifest_path: Path


def build_bvf_3d_graph(
    buildings: str | Path | gpd.GeoDataFrame,
    output_dir: str | Path,
    *,
    config: Optional[BVF3DGraphConfig] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
    max_tiles: Optional[int] = None,
    selection_mask=None,
) -> BVF3DGraphResult:
    """Build a regional height-aware graph from finite facade patches.

    ``selection_mask`` controls graph nodes only. Buildings in each tile halo
    remain in the scene as occluders. If ``dem_path`` is supplied, the DEM is
    used both for building base elevations and as ray-blocking terrain.
    """
    cfg = config or BVF3DGraphConfig()
    _validate_config(cfg)
    out = Path(output_dir)
    dirs = {
        "patches": out / "patches", "contacts": out / "contacts",
        "building_ids": out / "building_ids",
        "raw": out / "patch_exchange_raw", "done": out / "tiles_done",
    }
    out.mkdir(parents=True, exist_ok=True)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    source = _source_spec(buildings, cfg, bounds, selection_mask)
    tiles = list(_iter_tiles(source["target_bounds"], cfg.tile_size_m))
    if source["selection_mask_wkb"] is not None:
        mask = shapely.from_wkb(source["selection_mask_wkb"])
        tiles = [t for t in tiles if shapely.intersects(shapely.box(*t.bounds), mask)]
    if max_tiles is not None:
        tiles = tiles[:max_tiles]
    if not tiles:
        raise ValueError("The requested bounds contain no tiles.")
    allowed_tiles = {(tile.ix, tile.iy) for tile in tiles}

    jobs = [
        (buildings, source, tile, allowed_tiles, dirs, cfg)
        for tile in tiles
        if not (cfg.resume and (dirs["done"] / f"{tile.key}.json").exists())
    ]
    completed = len(tiles) - len(jobs)
    with tqdm(
        total=len(tiles), initial=completed, desc="3D BVF tiles", unit="tile",
        dynamic_ncols=True,
    ) as tile_progress:
        if cfg.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {
                    pool.submit(_process_tile, *job, False): job[2].key for job in jobs
                }
                for future in as_completed(futures):
                    status = future.result()
                    tile_progress.set_postfix(
                        tile=status["tile"], patches=status["patches_owned"],
                        edges=status["raw_edges"], refresh=False,
                    )
                    tile_progress.update()
        else:
            for job in jobs:
                status = _process_tile(*job, True)
                tile_progress.set_postfix(
                    tile=status["tile"], patches=status["patches_owned"],
                    edges=status["raw_edges"], refresh=False,
                )
                tile_progress.update()

    result = _reduce_partitions(out, dirs)
    manifest = {
        "schema_version": "0.1", "model": "differential_area_3d",
        "input": str(buildings) if not isinstance(buildings, gpd.GeoDataFrame) else "GeoDataFrame",
        "config": asdict(cfg), "bounds": list(source["target_bounds"]),
        "selection_mask_applied": source["selection_mask_wkb"] is not None,
        "tile_count": len(tiles),
        "outputs": {name: str(path) for name, path in asdict(result).items()},
    }
    result.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return result


def _validate_config(cfg):
    positive = (
        "tile_size_m", "max_distance_m", "default_height_m",
        "max_patch_width_m", "max_patch_height_m", "ray_batch_size",
    )
    for name in positive:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if cfg.max_vertical_patches <= 0:
        raise ValueError("max_vertical_patches must be positive.")
    if cfg.dem_path is not None and not Path(cfg.dem_path).exists():
        raise FileNotFoundError(f"DEM not found: {cfg.dem_path}")


def _read_tile_buildings(buildings, source, tile, cfg):
    halo = cfg.max_distance_m + cfg.contact_tolerance_m
    x0, y0, x1, y1 = tile.bounds
    target_bbox = (x0 - halo, y0 - halo, x1 + halo, y1 + halo)
    requested = [cfg.height_column]
    if cfg.elevation_column:
        requested.append(cfg.elevation_column)
    if cfg.source_id_column and cfg.source_id_column != cfg.id_column:
        requested.append(cfg.source_id_column)
    if isinstance(buildings, gpd.GeoDataFrame):
        data = buildings.to_crs(cfg.target_crs)
        data = data.cx[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]].copy()
    else:
        import pyogrio

        transformer = Transformer.from_crs(cfg.target_crs, source["source_crs"], always_xy=True)
        source_bbox = transformer.transform_bounds(*target_bbox, densify_pts=21)
        fid_id = cfg.id_column.upper() in {"FID", "OBJECTID"}
        columns = [c for c in requested if c and not (fid_id and c == cfg.id_column)]
        if not fid_id and cfg.id_column not in columns:
            columns.append(cfg.id_column)
        data = pyogrio.read_dataframe(
            buildings, layer=cfg.layer, bbox=source_bbox,
            columns=list(dict.fromkeys(columns)), fid_as_index=fid_id,
        ).to_crs(cfg.target_crs)
    if data.empty:
        data["node_id"] = pd.Series(dtype="int64")
        data["selected"] = pd.Series(dtype="bool")
        data["height_m"] = pd.Series(dtype="float64")
        data["base_z_m"] = pd.Series(dtype="float64")
        return data
    if cfg.id_column in data:
        raw_id = data[cfg.id_column]
    elif cfg.id_column.upper() in {"FID", "OBJECTID"} and data.index.name == "fid":
        raw_id = pd.Series(data.index, index=data.index)
    else:
        raise KeyError(f"Missing building ID column {cfg.id_column!r}.")
    data = data.assign(node_id=pd.to_numeric(raw_id, errors="raise").astype(np.int64))
    if data.node_id.duplicated().any():
        raise ValueError(f"{cfg.id_column!r} must be unique per building feature.")
    data.geometry = shapely.make_valid(data.geometry.to_numpy())
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    height = pd.to_numeric(data.get(cfg.height_column), errors="coerce")
    height = height * cfg.height_scale_to_m
    data["height_m"] = height.where(np.isfinite(height) & (height > 0), cfg.default_height_m)

    if cfg.dem_path and cfg.use_dem_for_building_elevation:
        base_z = _sample_dem(cfg.dem_path, data.geometry.centroid, cfg)
    elif cfg.elevation_column and cfg.elevation_column in data:
        base_z = pd.to_numeric(data[cfg.elevation_column], errors="coerce").to_numpy(float)
        base_z *= cfg.elevation_scale_to_m
    else:
        base_z = np.full(len(data), cfg.default_elevation_m, dtype=float)
    base_z = np.where(np.isfinite(base_z), base_z, cfg.default_elevation_m)
    data["base_z_m"] = base_z
    if source["selection_mask_wkb"] is None:
        data["selected"] = True
    else:
        mask = shapely.from_wkb(source["selection_mask_wkb"])
        data["selected"] = shapely.intersects(data.geometry.to_numpy(), mask)
    return data


def _sample_dem(path, centroids, cfg):
    import rasterio
    from rasterio.vrt import WarpedVRT

    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    with rasterio.open(path) as src, WarpedVRT(src, crs=cfg.target_crs) as vrt:
        values = np.array([v[0] for v in vrt.sample(xy, indexes=1, masked=True)])
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float) * cfg.dem_z_scale_to_m


def _facade_patches(buildings, cfg):
    rows = []
    for row in buildings[["node_id", "height_m", "base_z_m", "geometry"]].itertuples(index=False):
        ordinal = 0
        column_ordinal = 0
        for polygon in _polygon_parts(row.geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                coords = np.asarray(ring.coords, dtype=float)[:, :2]
                for a, b in zip(coords[:-1], coords[1:]):
                    delta = b - a
                    edge_length = float(np.linalg.norm(delta))
                    if edge_length < cfg.min_edge_length_m:
                        continue
                    tangent = delta / edge_length
                    normal = np.array([tangent[1], -tangent[0]])
                    midpoint = (a + b) / 2
                    if polygon.covers(shapely.Point(*(midpoint + normal * 0.05))):
                        normal *= -1
                    n_horizontal = max(1, math.ceil(edge_length / cfg.max_patch_width_m))
                    # Regional quadrature is capped so a skyscraper does not
                    # create hundreds of layers on every facade column. Patch
                    # areas still span the complete finite building height.
                    n_vertical = min(
                        cfg.max_vertical_patches,
                        max(1, math.ceil(row.height_m / cfg.max_patch_height_m)),
                    )
                    width = edge_length / n_horizontal
                    patch_height = row.height_m / n_vertical
                    for hi in range(n_horizontal):
                        xy = a + tangent * width * (hi + 0.5)
                        if column_ordinal >= COLUMN_STRIDE_3D:
                            raise ValueError(f"Building {row.node_id} exceeds 3D column ID stride.")
                        column_id = row.node_id * COLUMN_STRIDE_3D + column_ordinal
                        for zi in range(n_vertical):
                            if ordinal >= PATCH_STRIDE_3D:
                                raise ValueError(f"Building {row.node_id} exceeds 3D patch ID stride.")
                            z_low = row.base_z_m + zi * patch_height
                            z_high = z_low + patch_height
                            rows.append((
                                row.node_id * PATCH_STRIDE_3D + ordinal, row.node_id,
                                column_id,
                                xy[0], xy[1], (z_low + z_high) / 2,
                                normal[0], normal[1], 0.0, width, patch_height,
                                width * patch_height, z_low, z_high,
                                math.floor(xy[0] / cfg.tile_size_m),
                                math.floor(xy[1] / cfg.tile_size_m),
                            ))
                            ordinal += 1
                        column_ordinal += 1
    if not rows:
        return _empty_patches()
    return pd.DataFrame(rows, columns=list(_empty_patches().columns))


def _contact_rows(patches, buildings, tile, cfg):
    if patches.empty or buildings.empty:
        return _empty_contacts(), np.empty(0, dtype=np.int64)
    points = shapely.points(patches.x_m, patches.y_m)
    probes = shapely.buffer(points, cfg.contact_tolerance_m)
    geometry = buildings.geometry.to_numpy()
    point_idx, polygon_idx = STRtree(geometry).query(probes, predicate="intersects")
    if not len(point_idx):
        return _empty_contacts(), np.empty(0, dtype=np.int64)
    b_id = buildings.node_id.to_numpy(np.int64)
    b_low = buildings.base_z_m.to_numpy(float)
    b_high = b_low + buildings.height_m.to_numpy(float)
    p_id = patches.patch_id.to_numpy(np.int64)
    p_bld = patches.building_id.to_numpy(np.int64)
    overlap = np.maximum(
        0.0,
        np.minimum(patches.z_high_m.to_numpy()[point_idx], b_high[polygon_idx])
        - np.maximum(patches.z_low_m.to_numpy()[point_idx], b_low[polygon_idx]),
    )
    keep = (p_bld[point_idx] != b_id[polygon_idx]) & (overlap > 0)
    point_idx, polygon_idx, overlap = point_idx[keep], polygon_idx[keep], overlap[keep]
    if not len(point_idx):
        return _empty_contacts(), np.empty(0, dtype=np.int64)
    contacts = pd.DataFrame({
        "local_patch": point_idx, "patch_id": p_id[point_idx],
        "building_id": p_bld[point_idx], "receiver_building": b_id[polygon_idx],
        "overlap_height_m": overlap,
    }).drop_duplicates(["patch_id", "receiver_building"])
    all_contact_ids = contacts.patch_id.unique().astype(np.int64)
    selected = set(buildings.loc[buildings.selected, "node_id"].astype(np.int64))
    owned_ids = set(patches.loc[
        (patches.home_ix == tile.ix) & (patches.home_iy == tile.iy), "patch_id"
    ].astype(np.int64))
    contacts = contacts[
        contacts.receiver_building.isin(selected) & contacts.patch_id.isin(owned_ids)
    ].copy()
    if contacts.empty:
        return _empty_contacts(), all_contact_ids
    patch_lookup = patches.set_index("patch_id")
    contacts["potential_area"] = (
        contacts.patch_id.map(patch_lookup.width_m) * contacts.overlap_height_m
    )
    total = contacts.groupby("patch_id").potential_area.transform("sum")
    area = contacts.patch_id.map(patch_lookup.area_m2)
    scale = np.minimum(1.0, area / total)
    contacts["contact_exchange_area_m2"] = contacts.potential_area * scale
    contacts["contact_fraction"] = contacts.contact_exchange_area_m2 / area
    return contacts[list(_empty_contacts().columns)].reset_index(drop=True), all_contact_ids


def _candidate_pairs(patches, cfg, progress_desc=None):
    """Pair facade columns in XY, then expand compatible vertical patches.

    A direct 3D radius query repeats the same horizontal search for every
    storey. Column-first pairing removes that redundancy without changing the
    subsequent 3D distance, angle, kernel, or ray tests.
    """
    xyz = patches[["x_m", "y_m", "z_m"]].to_numpy(float)
    normals = patches[["nx", "ny", "nz"]].to_numpy(float)
    columns = patches.drop_duplicates("column_id")[
        ["column_id", "building_id", "x_m", "y_m", "nx", "ny"]
    ].reset_index(drop=True)
    column_xy = columns[["x_m", "y_m"]].to_numpy(float)
    column_normals = columns[["nx", "ny"]].to_numpy(float)
    column_building = columns.building_id.to_numpy(np.int64)
    patch_by_column = {
        int(column_id): group.index.to_numpy(np.int64)
        for column_id, group in patches.groupby("column_id", sort=False)
    }
    tree = cKDTree(column_xy)
    cutoff = np.cos(np.deg2rad(cfg.max_angle_deg))
    out_p, out_q = [], []

    def flush():
        if not out_p:
            return None
        p, q = np.concatenate(out_p), np.concatenate(out_q)
        out_p.clear()
        out_q.clear()
        delta = xyz[q] - xyz[p]
        distance = np.linalg.norm(delta, axis=1)
        valid = (distance > 1e-9) & (distance <= cfg.max_distance_m)
        p, q, delta, distance = p[valid], q[valid], delta[valid], distance[valid]
        if not len(p):
            return None
        direction = delta / distance[:, None]
        cos_p = np.einsum("ij,ij->i", normals[p], direction)
        cos_q = np.einsum("ij,ij->i", normals[q], -direction)
        keep = (cos_p >= cutoff) & (cos_q >= cutoff)
        if not np.any(keep):
            return None
        return p[keep], q[keep], distance[keep], cos_p[keep], cos_q[keep]

    buffered = 0
    starts = range(0, len(columns), cfg.query_batch_size)
    if progress_desc:
        starts = tqdm(
            starts, total=math.ceil(len(columns) / cfg.query_batch_size),
            desc=progress_desc, unit="batch", leave=False, dynamic_ncols=True,
        )
    for start in starts:
        neighborhoods = tree.query_ball_point(
            column_xy[start:start + cfg.query_batch_size], cfg.max_distance_m, workers=1,
        )
        for offset, neighbors in enumerate(neighborhoods):
            c = start + offset
            nbr = np.asarray(neighbors, dtype=np.int64)
            nbr = nbr[(nbr > c) & (column_building[nbr] != column_building[c])]
            if not len(nbr):
                continue
            delta_xy = column_xy[nbr] - column_xy[c]
            distance_xy = np.linalg.norm(delta_xy, axis=1)
            direction_xy = delta_xy / np.maximum(distance_xy[:, None], 1e-12)
            cos_c = direction_xy @ column_normals[c]
            cos_n = np.einsum("ij,ij->i", column_normals[nbr], -direction_xy)
            nbr = nbr[(cos_c >= cutoff) & (cos_n >= cutoff)]
            p_idx = patch_by_column[int(columns.column_id.iloc[c])]
            for n in nbr:
                q_idx = patch_by_column[int(columns.column_id.iloc[n])]
                out_p.append(np.repeat(p_idx, len(q_idx)))
                out_q.append(np.tile(q_idx, len(p_idx)))
                buffered += len(p_idx) * len(q_idx)
                if buffered >= cfg.ray_batch_size:
                    result = flush()
                    buffered = 0
                    if result is not None:
                        yield result
    result = flush()
    if result is not None:
        yield result


def _append_quad(vertices, faces, a, b, c, d):
    base = len(vertices)
    vertices.extend([a, b, c, d])
    faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])


def _building_scene_mesh(buildings, local_origin):
    ox, oy, oz = local_origin
    vertices, faces = [], []
    for row in buildings[["geometry", "base_z_m", "height_m"]].itertuples(index=False):
        z0, z1 = row.base_z_m - oz, row.base_z_m + row.height_m - oz
        for polygon in _polygon_parts(row.geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                coords = np.asarray(ring.coords, dtype=float)[:, :2]
                for a, b in zip(coords[:-1], coords[1:]):
                    ax, ay, bx, by = a[0] - ox, a[1] - oy, b[0] - ox, b[1] - oy
                    _append_quad(vertices, faces, (ax, ay, z0), (bx, by, z0),
                                 (bx, by, z1), (ax, ay, z1))
            for tri in triangulate(polygon):
                if not polygon.covers(tri.representative_point()):
                    continue
                coords = np.asarray(tri.exterior.coords, dtype=float)[:3, :2]
                base = len(vertices)
                vertices.extend([(x - ox, y - oy, z1) for x, y in coords])
                faces.append((base, base + 1, base + 2))
    return vertices, faces


def _terrain_mesh(path, bounds, local_origin, cfg):
    if not path:
        return [], []
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.vrt import WarpedVRT

    ox, oy, oz = local_origin
    with rasterio.open(path) as src, WarpedVRT(
        src, crs=cfg.target_crs, resolution=cfg.terrain_resolution_m,
    ) as vrt:
        window = from_bounds(*bounds, transform=vrt.transform).round_offsets().round_lengths()
        values = vrt.read(1, window=window, masked=True).astype(float)
        transform = vrt.window_transform(window)
    z = values.filled(np.nan) * cfg.dem_z_scale_to_m
    height, width = z.shape
    if height < 2 or width < 2:
        return [], []
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)
    vertices = np.column_stack([(xs - ox).ravel(), (ys - oy).ravel(), (z - oz).ravel()])
    cell = np.arange((height - 1) * (width - 1)).reshape(height - 1, width - 1)
    row = cell // (width - 1)
    col = cell % (width - 1)
    a = row * width + col
    b, c, d = a + 1, a + width + 1, a + width
    f1 = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    f2 = np.column_stack([a.ravel(), c.ravel(), d.ravel()])
    faces = np.vstack([f1, f2]).astype(np.int32)
    valid = np.isfinite(vertices[:, 2])
    faces = faces[np.all(valid[faces], axis=1)]
    return vertices[valid].tolist(), _remap_faces_for_valid(faces, valid).tolist()


def _remap_faces_for_valid(faces, valid):
    remap = np.full(len(valid), -1, dtype=np.int64)
    remap[valid] = np.arange(np.count_nonzero(valid))
    return remap[faces].astype(np.int32)


def _scene_intersector(buildings, tile, cfg):
    halo = cfg.max_distance_m + cfg.contact_tolerance_m
    terrain_bounds = shapely.box(*tile.bounds).buffer(halo, cap_style="square").bounds
    min_z = float(buildings.base_z_m.min()) if len(buildings) else 0.0
    local_origin = np.array([tile.bounds[0], tile.bounds[1], min_z], dtype=float)
    vertices, faces = _building_scene_mesh(buildings, local_origin)
    terrain_v, terrain_f = _terrain_mesh(cfg.dem_path, terrain_bounds, local_origin, cfg)
    if terrain_v:
        offset = len(vertices)
        vertices.extend(terrain_v)
        faces.extend([tuple(np.asarray(face) + offset) for face in terrain_f])
    if not faces:
        return None, local_origin
    import point_cloud_utils as pcu

    intersector = pcu.RayMeshIntersector(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
    )
    return intersector, local_origin


def _visible_mask(p, q, patches, intersector, local_origin, cfg):
    if intersector is None:
        return np.ones(len(p), dtype=bool)
    xyz = patches[["x_m", "y_m", "z_m"]].to_numpy(float)
    normals = patches[["nx", "ny", "nz"]].to_numpy(float)
    origin = xyz[p] + cfg.ray_offset_m * normals[p] - local_origin
    target = xyz[q] + cfg.ray_offset_m * normals[q] - local_origin
    disp = target - origin
    distance = np.linalg.norm(disp, axis=1)
    direction = disp / np.maximum(distance[:, None], 1e-12)
    visible = np.empty(len(p), dtype=bool)
    for start in range(0, len(p), cfg.ray_batch_size):
        stop = min(start + cfg.ray_batch_size, len(p))
        face_id, _, hit_t = intersector.intersect_rays(
            np.ascontiguousarray(origin[start:stop], dtype=np.float32),
            np.ascontiguousarray(direction[start:stop], dtype=np.float32),
            ray_near=max(cfg.hit_tolerance_m, 1e-6),
            ray_far=float(np.max(distance[start:stop]) + cfg.hit_tolerance_m),
        )
        face_id = np.asarray(face_id).reshape(-1)
        hit_t = np.asarray(hit_t).reshape(-1)
        visible[start:stop] = (
            (face_id < 0) | (hit_t >= distance[start:stop] - cfg.hit_tolerance_m)
        )
    return visible


def _patch_exchange(patches, buildings, tile, allowed_tiles, cfg, show_progress=False):
    if patches.empty:
        return _empty_exchange()
    ids = patches.patch_id.to_numpy(np.int64)
    home_ix = patches.home_ix.to_numpy(np.int64)
    home_iy = patches.home_iy.to_numpy(np.int64)
    building = patches.building_id.to_numpy(np.int64)
    area = patches.area_m2.to_numpy(float)
    intersector, local_origin = _scene_intersector(buildings, tile, cfg)
    chunks = []
    progress_desc = (
        f"tile {tile.key}: {patches.column_id.nunique():,} facade columns"
        if show_progress else None
    )
    for p, q, distance, cos_p, cos_q in _candidate_pairs(
        patches, cfg, progress_desc=progress_desc,
    ):
        canonical = np.where(ids[p] < ids[q], p, q)
        owned = (home_ix[canonical] == tile.ix) & (home_iy[canonical] == tile.iy)
        included = np.fromiter(
            ((home_ix[x], home_iy[x]) in allowed_tiles for x in p), bool, len(p)
        ) & np.fromiter(
            ((home_ix[x], home_iy[x]) in allowed_tiles for x in q), bool, len(q)
        )
        keep = owned & included
        p, q = p[keep], q[keep]
        distance, cos_p, cos_q = distance[keep], cos_p[keep], cos_q[keep]
        if not len(p):
            continue
        visible = _visible_mask(p, q, patches, intersector, local_origin, cfg)
        p, q = p[visible], q[visible]
        distance, cos_p, cos_q = distance[visible], cos_p[visible], cos_q[visible]
        if not len(p):
            continue
        kernel = cos_p * cos_q / (np.pi * distance**2)
        raw = kernel * area[p] * area[q]
        prelim = np.minimum(raw, np.minimum(area[p], area[q]))
        chunks.append(pd.DataFrame({
            "patch_p": ids[p], "patch_q": ids[q],
            "building_p": building[p], "building_q": building[q],
            "distance_m": distance, "kernel_pq_m2_inv": kernel,
            "exchange_pq_raw_m2": raw, "exchange_pq_prelim_m2": prelim,
        }))
    return pd.concat(chunks, ignore_index=True) if chunks else _empty_exchange()


def _process_tile(buildings, source, tile, allowed_tiles, dirs, cfg, show_progress=False):
    data = _read_tile_buildings(buildings, source, tile, cfg)
    patches = _facade_patches(data[data.selected], cfg)
    owned = patches[(patches.home_ix == tile.ix) & (patches.home_iy == tile.iy)]
    contacts, _ = _contact_rows(patches, data, tile, cfg)
    exchange = _patch_exchange(
        patches, data, tile, allowed_tiles, cfg, show_progress=show_progress,
    )
    owned_ids = owned.building_id.drop_duplicates()
    lookup = data[data.node_id.isin(owned_ids)][["node_id", "height_m", "base_z_m"]].copy()
    lookup = lookup.rename(columns={"node_id": "building_id"})
    if cfg.source_id_column and cfg.source_id_column in data:
        lookup["source_building_id"] = data.loc[lookup.index, cfg.source_id_column].astype("string")
    else:
        lookup["source_building_id"] = lookup.building_id.astype("string")
    lookup = lookup.drop_duplicates("building_id")
    owned.to_parquet(dirs["patches"] / f"{tile.key}.parquet", index=False)
    contacts.to_parquet(dirs["contacts"] / f"{tile.key}.parquet", index=False)
    lookup.to_parquet(dirs["building_ids"] / f"{tile.key}.parquet", index=False)
    exchange.to_parquet(dirs["raw"] / f"{tile.key}.parquet", index=False)
    status = {
        "tile": tile.key, "buildings_loaded": len(data), "patches_owned": len(owned),
        "contacts": len(contacts), "raw_edges": len(exchange),
    }
    (dirs["done"] / f"{tile.key}.json").write_text(json.dumps(status))
    return status


def _reduce_partitions(out, dirs):
    import duckdb

    patch_exchange_path = out / "patch_exchange.parquet"
    building_edges_path = out / "building_edges.parquet"
    building_metrics_path = out / "building_metrics.parquet"
    building_ids_path = out / "building_ids.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW patches AS SELECT * FROM read_parquet('{dirs['patches']}/*.parquet')")
    con.execute(f"CREATE VIEW contacts AS SELECT * FROM read_parquet('{dirs['contacts']}/*.parquet')")
    con.execute(f"CREATE VIEW ids AS SELECT * FROM read_parquet('{dirs['building_ids']}/*.parquet')")
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{dirs['raw']}/*.parquet')")
    con.execute(f"""
        COPY (SELECT building_id, ANY_VALUE(source_building_id) source_building_id,
                     ANY_VALUE(height_m) height_m, ANY_VALUE(base_z_m) base_z_m
              FROM ids GROUP BY building_id)
        TO '{building_ids_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute("""
        CREATE TEMP VIEW contact_per_patch AS
        SELECT patch_id, LEAST(area_m2, COALESCE(SUM(contact_exchange_area_m2), 0)) contact_m2
        FROM patches LEFT JOIN contacts USING (patch_id) GROUP BY patch_id, area_m2
    """)
    con.execute("""
        CREATE TEMP VIEW incident AS
        SELECT patch_id, SUM(exchange_pq_prelim_m2) allocated_m2 FROM (
            SELECT patch_p patch_id, exchange_pq_prelim_m2 FROM raw
            UNION ALL SELECT patch_q, exchange_pq_prelim_m2 FROM raw
        ) GROUP BY patch_id
    """)
    con.execute("""
        CREATE TEMP VIEW patch_scale AS
        SELECT p.patch_id,
               LEAST(1.0, GREATEST(0.0, p.area_m2-c.contact_m2)/NULLIF(i.allocated_m2,0)) scale
        FROM patches p JOIN incident i USING (patch_id) JOIN contact_per_patch c USING (patch_id)
    """)
    con.execute(f"""
        COPY (SELECT r.*,
                     r.exchange_pq_prelim_m2 * LEAST(sp.scale,sq.scale) exchange_pq_m2
              FROM raw r JOIN patch_scale sp ON r.patch_p=sp.patch_id
                         JOIN patch_scale sq ON r.patch_q=sq.patch_id)
        TO '{patch_exchange_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"CREATE VIEW final_edges AS SELECT * FROM read_parquet('{patch_exchange_path}')")
    con.execute("""
        CREATE TEMP VIEW surfaces AS
        SELECT building_id, SUM(area_m2) facade_area_m2 FROM patches GROUP BY building_id
    """)
    con.execute("""
        CREATE TEMP VIEW patch_to_building AS WITH directed AS (
            SELECT e.patch_p source_patch,e.building_p source_building,e.building_q target_building,
                   e.exchange_pq_m2/p.area_m2 f FROM final_edges e JOIN patches p ON e.patch_p=p.patch_id
            UNION ALL
            SELECT e.patch_q,e.building_q,e.building_p,e.exchange_pq_m2/p.area_m2
            FROM final_edges e JOIN patches p ON e.patch_q=p.patch_id
            UNION ALL
            SELECT c.patch_id,c.building_id,c.receiver_building,c.contact_exchange_area_m2/p.area_m2
            FROM contacts c JOIN patches p USING (patch_id)
        ) SELECT source_patch,source_building,target_building,LEAST(1.0,SUM(f)) f_patch_to_building
          FROM directed GROUP BY 1,2,3
    """)
    con.execute("""
        CREATE TEMP VIEW pair_fmax AS
        SELECT LEAST(source_building,target_building) building_i,
               GREATEST(source_building,target_building) building_j,
               MAX(CASE WHEN source_building<target_building THEN f_patch_to_building ELSE 0 END) f_max_i_to_j,
               MAX(CASE WHEN source_building>target_building THEN f_patch_to_building ELSE 0 END) f_max_j_to_i
        FROM patch_to_building GROUP BY 1,2
    """)
    con.execute("""
        CREATE TEMP VIEW visible_building AS
        SELECT LEAST(building_p,building_q) building_i,GREATEST(building_p,building_q) building_j,
               SUM(exchange_pq_m2) exchange_visible_ij_m2,COUNT(*) visible_patch_pair_count
        FROM final_edges GROUP BY 1,2
    """)
    con.execute("""
        CREATE TEMP VIEW contact_building AS
        SELECT LEAST(building_id,receiver_building) building_i,
               GREATEST(building_id,receiver_building) building_j,
               SUM(CASE WHEN building_id<receiver_building THEN contact_exchange_area_m2 ELSE 0 END) contact_i_to_j_m2,
               SUM(CASE WHEN building_id>receiver_building THEN contact_exchange_area_m2 ELSE 0 END) contact_j_to_i_m2,
               COUNT(DISTINCT patch_id) contact_patch_count
        FROM contacts GROUP BY 1,2
    """)
    con.execute("""
        CREATE TEMP VIEW combined AS
        SELECT COALESCE(v.building_i,c.building_i) building_i,
               COALESCE(v.building_j,c.building_j) building_j,
               COALESCE(v.exchange_visible_ij_m2,0) exchange_visible_ij_m2,
               COALESCE(v.visible_patch_pair_count,0) visible_patch_pair_count,
               COALESCE(c.contact_i_to_j_m2,0) contact_i_to_j_m2,
               COALESCE(c.contact_j_to_i_m2,0) contact_j_to_i_m2,
               COALESCE(c.contact_patch_count,0) contact_patch_count
        FROM visible_building v FULL OUTER JOIN contact_building c USING(building_i,building_j)
    """)
    con.execute(f"""
        COPY (SELECT x.*,si.facade_area_m2 facade_area_i_m2,sj.facade_area_m2 facade_area_j_m2,
                     (exchange_visible_ij_m2+contact_i_to_j_m2)/si.facade_area_m2 vf_i_to_j,
                     (exchange_visible_ij_m2+contact_j_to_i_m2)/sj.facade_area_m2 vf_j_to_i,
                     f.f_max_i_to_j,f.f_max_j_to_i,contact_patch_count>0 has_contact
              FROM combined x JOIN surfaces si ON x.building_i=si.building_id
                              JOIN surfaces sj ON x.building_j=sj.building_id
                              JOIN pair_fmax f USING(building_i,building_j))
        TO '{building_edges_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"CREATE VIEW building_edges AS SELECT * FROM read_parquet('{building_edges_path}')")
    con.execute(f"""
        COPY (WITH directed AS (
            SELECT building_i building_id,building_j neighbor_id,vf_i_to_j vf,
                   f_max_i_to_j f_max,contact_i_to_j_m2>0 contact FROM building_edges
            UNION ALL SELECT building_j,building_i,vf_j_to_i,f_max_j_to_i,
                   contact_j_to_i_m2>0 FROM building_edges
        ), metrics AS (
            SELECT building_id,COUNT(DISTINCT neighbor_id) neighbor_count,SUM(vf) bvf_all_edges,
                   SUM(CASE WHEN NOT contact THEN vf ELSE 0 END) bvf_noncontact_edges,
                   MAX(f_max) bvf_max FROM directed GROUP BY building_id
        ) SELECT s.building_id,s.facade_area_m2,
                 s.facade_area_m2*COALESCE(m.bvf_all_edges,0) exchange_total_m2,
                 COALESCE(m.neighbor_count,0) neighbor_count,
                 COALESCE(m.bvf_all_edges,0) bvf_all_edges,
                 COALESCE(m.bvf_noncontact_edges,0) bvf_noncontact_edges,
                 COALESCE(m.bvf_all_edges,0)-COALESCE(m.bvf_noncontact_edges,0) bvf_contact_edges,
                 COALESCE(m.bvf_max,0) bvf_max
          FROM surfaces s LEFT JOIN metrics m USING(building_id))
        TO '{building_metrics_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    return BVF3DGraphResult(
        output_dir=out, patches_path=dirs["patches"], contacts_path=dirs["contacts"],
        building_ids_path=building_ids_path, patch_exchange_path=patch_exchange_path,
        building_edges_path=building_edges_path, building_metrics_path=building_metrics_path,
        manifest_path=out / "manifest.json",
    )
