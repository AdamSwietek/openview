"""Tiled, resumable construction of a 2D building coupling graph.

The hot path is deliberately mesh-free: facade line segments are paired with a
KD-tree, tested against footprint interiors with a Shapely STRtree, and reduced
to one row per building pair with DuckDB over Parquet partitions.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely import STRtree


PATCH_STRIDE = 100_000


def _empty_frame(schema: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in schema.items()})


def _empty_patches() -> pd.DataFrame:
    return _empty_frame({
        "patch_id": "int64", "building_id": "int64", "x_m": "float64",
        "y_m": "float64", "nx": "float64", "ny": "float64",
        "length_m": "float64", "home_ix": "int64", "home_iy": "int64",
    })


def _empty_contacts() -> pd.DataFrame:
    return _empty_frame({
        "patch_id": "int64", "building_id": "int64",
        "receiver_building": "int64", "contact_vf": "float64",
        "contact_exchange_length_m": "float64",
    })


def _empty_exchange() -> pd.DataFrame:
    return _empty_frame({
        "patch_p": "int64", "patch_q": "int64", "building_p": "int64",
        "building_q": "int64", "distance_m": "float64",
        "kernel_pq_m_inv": "float64", "exchange_pq_raw_m": "float64",
        "exchange_pq_prelim_m": "float64",
    })


@dataclass(frozen=True)
class BVFGraphConfig:
    """Parameters for the fast 2D line-segment approximation."""

    id_column: str = "OBJECTID"
    source_id_column: Optional[str] = "BLD_ID"
    layer: Optional[str] = None
    target_crs: str = "EPSG:26911"
    tile_size_m: float = 2_000.0
    max_distance_m: float = 152.4
    max_angle_deg: float = 80.0
    min_edge_length_m: float = 2.0
    min_crossing_m: float = 0.25
    contact_tolerance_m: float = 0.05
    query_batch_size: int = 500
    workers: int = 1
    resume: bool = True


@dataclass(frozen=True)
class BVFGraphResult:
    output_dir: Path
    patches_path: Path
    contacts_path: Path
    building_ids_path: Path
    patch_exchange_path: Path
    building_edges_path: Path
    building_metrics_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class _Tile:
    ix: int
    iy: int
    bounds: tuple[float, float, float, float]

    @property
    def key(self) -> str:
        return f"{self.ix}_{self.iy}"


def build_bvf_graph(
    buildings: str | Path | gpd.GeoDataFrame,
    output_dir: str | Path,
    *,
    config: Optional[BVFGraphConfig] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
    max_tiles: Optional[int] = None,
) -> BVFGraphResult:
    """Build a tiled 2D building graph and return paths to its products.

    Parameters
    ----------
    buildings
        LARIAC GDB/GPKG/other GDAL-readable path, or a GeoDataFrame.
    output_dir
        New or resumable run directory.
    config
        Input schema, geometry, tiling, and parallelism settings.
    bounds
        Optional processing bounds in ``config.target_crs``.
    max_tiles
        Optional development limit. Use with ``bounds`` for deterministic tests.
    """
    cfg = config or BVFGraphConfig()
    out = Path(output_dir)
    dirs = _output_dirs(out)
    out.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    source = _source_spec(buildings, cfg, bounds)
    tiles = list(_iter_tiles(source["target_bounds"], cfg.tile_size_m))
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
    if cfg.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(_process_tile, *job) for job in jobs]
            for future in as_completed(futures):
                future.result()
    else:
        for job in jobs:
            _process_tile(*job)

    result = _reduce_partitions(out, dirs, cfg)
    manifest = {
        "schema_version": "0.1",
        "model": "differential_line_2d",
        "input": str(buildings) if not isinstance(buildings, gpd.GeoDataFrame) else "GeoDataFrame",
        "config": asdict(cfg),
        "bounds": list(source["target_bounds"]),
        "tile_count": len(tiles),
        "outputs": {name: str(path) for name, path in asdict(result).items()},
    }
    result.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return result


def _output_dirs(out: Path) -> dict[str, Path]:
    return {
        "patches": out / "patches",
        "contacts": out / "contacts",
        "building_ids": out / "building_ids",
        "raw": out / "patch_exchange_raw",
        "done": out / "tiles_done",
    }


def _source_spec(buildings, cfg: BVFGraphConfig, bounds):
    if isinstance(buildings, gpd.GeoDataFrame):
        if buildings.crs is None:
            raise ValueError("Input buildings must have a CRS.")
        source_crs = str(buildings.crs)
        source_bounds = tuple(map(float, buildings.total_bounds))
    else:
        import pyogrio

        info = pyogrio.read_info(buildings, layer=cfg.layer)
        source_crs = info["crs"]
        source_bounds = tuple(map(float, info["total_bounds"]))
    transformer = Transformer.from_crs(source_crs, cfg.target_crs, always_xy=True)
    target_bounds = transformer.transform_bounds(*source_bounds, densify_pts=21)
    return {
        "source_crs": source_crs,
        "target_bounds": tuple(map(float, bounds or target_bounds)),
    }


def _iter_tiles(bounds, size) -> Iterable[_Tile]:
    minx, miny, maxx, maxy = bounds
    ix0, iy0 = math.floor(minx / size), math.floor(miny / size)
    ix1, iy1 = math.ceil(maxx / size), math.ceil(maxy / size)
    for iy in range(iy0, iy1):
        for ix in range(ix0, ix1):
            yield _Tile(ix, iy, (ix * size, iy * size, (ix + 1) * size, (iy + 1) * size))


def _read_tile_buildings(buildings, source, tile, cfg):
    halo = cfg.max_distance_m + cfg.contact_tolerance_m
    x0, y0, x1, y1 = tile.bounds
    target_bbox = (x0 - halo, y0 - halo, x1 + halo, y1 + halo)
    if isinstance(buildings, gpd.GeoDataFrame):
        data = buildings.to_crs(cfg.target_crs)
        data = data.cx[target_bbox[0]:target_bbox[2], target_bbox[1]:target_bbox[3]].copy()
    else:
        import pyogrio

        transformer = Transformer.from_crs(cfg.target_crs, source["source_crs"], always_xy=True)
        source_bbox = transformer.transform_bounds(*target_bbox, densify_pts=21)
        fid_id = cfg.id_column.upper() in {"FID", "OBJECTID"}
        columns = [] if fid_id else [cfg.id_column]
        if cfg.source_id_column and cfg.source_id_column != cfg.id_column:
            columns.append(cfg.source_id_column)
        data = pyogrio.read_dataframe(
            buildings, layer=cfg.layer, bbox=source_bbox, columns=columns,
            fid_as_index=fid_id,
        ).to_crs(cfg.target_crs)
    if data.empty:
        return data
    if cfg.id_column in data:
        raw_node_id = data[cfg.id_column]
    elif cfg.id_column.upper() in {"FID", "OBJECTID"} and data.index.name == "fid":
        raw_node_id = pd.Series(data.index, index=data.index)
    else:
        raise KeyError(f"Missing building ID column {cfg.id_column!r}.")
    node_id = pd.to_numeric(raw_node_id, errors="raise").astype(np.int64)
    if node_id.duplicated().any():
        raise ValueError(f"{cfg.id_column!r} must be unique per building feature.")
    data = data.assign(node_id=node_id)
    data.geometry = shapely.make_valid(data.geometry.to_numpy())
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    return data


def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [g for g in geometry.geoms if g.geom_type == "Polygon"]
    return []


def _facade_segments(buildings, cfg):
    rows = []
    for row in buildings[["node_id", "geometry"]].itertuples(index=False):
        ordinal = 0
        for polygon in _polygon_parts(row.geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                coords = np.asarray(ring.coords, dtype=float)
                for a, b in zip(coords[:-1], coords[1:]):
                    delta = b[:2] - a[:2]
                    length = float(np.linalg.norm(delta))
                    if length < cfg.min_edge_length_m:
                        continue
                    midpoint = (a[:2] + b[:2]) / 2
                    candidate = np.array([delta[1], -delta[0]]) / length
                    probe = shapely.Point(*(midpoint + candidate * 0.05))
                    if polygon.covers(probe):
                        candidate *= -1
                    if ordinal >= PATCH_STRIDE:
                        raise ValueError(f"Building {row.node_id} exceeds patch ID stride.")
                    rows.append((
                        row.node_id * PATCH_STRIDE + ordinal, row.node_id,
                        midpoint[0], midpoint[1], candidate[0], candidate[1], length,
                        math.floor(midpoint[0] / cfg.tile_size_m),
                        math.floor(midpoint[1] / cfg.tile_size_m),
                    ))
                    ordinal += 1
    if not rows:
        return _empty_patches()
    return pd.DataFrame(rows, columns=[
        "patch_id", "building_id", "x_m", "y_m", "nx", "ny", "length_m",
        "home_ix", "home_iy",
    ])


def _contact_rows(patches, buildings, tile, cfg):
    if patches.empty or buildings.empty:
        return _empty_contacts(), np.empty(0, dtype=np.int64)
    points = shapely.points(patches["x_m"], patches["y_m"])
    probes = shapely.buffer(points, cfg.contact_tolerance_m)
    geometry = buildings.geometry.to_numpy()
    building_ids = buildings["node_id"].to_numpy(dtype=np.int64)
    point_idx, polygon_idx = STRtree(geometry).query(probes, predicate="intersects")
    if not len(point_idx):
        return _empty_contacts(), np.empty(0, dtype=np.int64)
    contacts = pd.DataFrame({
        "local_patch": point_idx,
        "receiver_building": building_ids[polygon_idx],
    })
    contacts["patch_id"] = patches["patch_id"].to_numpy()[point_idx]
    contacts["building_id"] = patches["building_id"].to_numpy()[point_idx]
    contacts = contacts[contacts["building_id"] != contacts["receiver_building"]]
    contacts = contacts.drop_duplicates(["patch_id", "receiver_building"])
    all_contact_ids = contacts["patch_id"].unique().astype(np.int64)
    owned = (patches["home_ix"] == tile.ix) & (patches["home_iy"] == tile.iy)
    owned_ids = set(patches.loc[owned, "patch_id"].astype(np.int64))
    contacts = contacts[contacts["patch_id"].isin(owned_ids)].copy()
    if contacts.empty:
        return _empty_contacts(), all_contact_ids
    counts = contacts.groupby("patch_id")["receiver_building"].transform("size")
    lengths = patches.set_index("patch_id")["length_m"]
    contacts["contact_vf"] = 1.0 / counts
    contacts["contact_exchange_length_m"] = (
        contacts["patch_id"].map(lengths) * contacts["contact_vf"]
    )
    return contacts[list(_empty_contacts().columns)].reset_index(drop=True), all_contact_ids


def _candidate_pairs(patches, cfg):
    xy = patches[["x_m", "y_m"]].to_numpy(dtype=float)
    normals = patches[["nx", "ny"]].to_numpy(dtype=float)
    building = patches["building_id"].to_numpy(dtype=np.int64)
    tree = cKDTree(xy)
    cutoff = np.cos(np.deg2rad(cfg.max_angle_deg))
    for start in range(0, len(patches), cfg.query_batch_size):
        neighborhoods = tree.query_ball_point(
            xy[start:start + cfg.query_batch_size], cfg.max_distance_m, workers=1,
        )
        left, right = [], []
        for offset, neighbors in enumerate(neighborhoods):
            p = start + offset
            q = np.asarray(neighbors, dtype=np.int64)
            q = q[(q > p) & (building[q] != building[p])]
            if len(q):
                left.append(np.full(len(q), p, dtype=np.int64))
                right.append(q)
        if not left:
            continue
        p, q = np.concatenate(left), np.concatenate(right)
        delta = xy[q] - xy[p]
        distance = np.linalg.norm(delta, axis=1)
        keep = distance > 1e-9
        p, q, delta, distance = p[keep], q[keep], delta[keep], distance[keep]
        direction = delta / distance[:, None]
        cos_p = np.einsum("ij,ij->i", normals[p], direction)
        cos_q = np.einsum("ij,ij->i", normals[q], -direction)
        keep = (cos_p >= cutoff) & (cos_q >= cutoff)
        if np.any(keep):
            yield p[keep], q[keep], distance[keep], cos_p[keep], cos_q[keep]


def _visible_mask(p, q, patches, geometry, tree, min_crossing):
    xy = patches[["x_m", "y_m"]].to_numpy(dtype=float)
    lines = shapely.linestrings(np.stack([xy[p], xy[q]], axis=1))
    hits = tree.query(lines, predicate="intersects")
    blocked = np.zeros(len(lines), dtype=bool)
    if hits.size:
        line_idx, polygon_idx = hits
        hit_lines, hit_polygons = lines[line_idx], geometry[polygon_idx]
        crossing = shapely.intersection(hit_lines, hit_polygons)
        meaningful = (
            shapely.relate_pattern(hit_lines, hit_polygons, "T********")
            & (shapely.length(crossing) >= min_crossing)
        )
        np.logical_or.at(blocked, line_idx[meaningful], True)
    return ~blocked


def _patch_exchange(patches, buildings, tile, allowed_tiles, contact_ids, cfg):
    if patches.empty:
        return _empty_exchange()
    ids = patches["patch_id"].to_numpy(dtype=np.int64)
    home_ix = patches["home_ix"].to_numpy(dtype=np.int64)
    home_iy = patches["home_iy"].to_numpy(dtype=np.int64)
    building = patches["building_id"].to_numpy(dtype=np.int64)
    length = patches["length_m"].to_numpy(dtype=float)
    contact = np.isin(ids, contact_ids)
    geometry = buildings.geometry.to_numpy()
    footprint_tree = STRtree(geometry)
    chunks = []
    for p, q, distance, cos_p, cos_q in _candidate_pairs(patches, cfg):
        canonical = np.where(ids[p] < ids[q], p, q)
        owned = (home_ix[canonical] == tile.ix) & (home_iy[canonical] == tile.iy)
        included = np.fromiter(
            ((home_ix[x], home_iy[x]) in allowed_tiles for x in p), bool, len(p)
        ) & np.fromiter(
            ((home_ix[x], home_iy[x]) in allowed_tiles for x in q), bool, len(q)
        )
        keep = owned & included & ~(contact[p] | contact[q])
        p, q = p[keep], q[keep]
        distance, cos_p, cos_q = distance[keep], cos_p[keep], cos_q[keep]
        if not len(p):
            continue
        visible = _visible_mask(
            p, q, patches, geometry, footprint_tree, cfg.min_crossing_m,
        )
        p, q = p[visible], q[visible]
        distance, cos_p, cos_q = distance[visible], cos_p[visible], cos_q[visible]
        if not len(p):
            continue
        # Differential 2D line kernel: K has units 1/m, E has units m.
        kernel = cos_p * cos_q / (2.0 * distance)
        raw = kernel * length[p] * length[q]
        prelim = np.minimum(raw, np.minimum(length[p], length[q]))
        chunks.append(pd.DataFrame({
            "patch_p": ids[p], "patch_q": ids[q],
            "building_p": building[p], "building_q": building[q],
            "distance_m": distance, "kernel_pq_m_inv": kernel,
            "exchange_pq_raw_m": raw, "exchange_pq_prelim_m": prelim,
        }))
    return pd.concat(chunks, ignore_index=True) if chunks else _empty_exchange()


def _process_tile(buildings, source, tile, allowed_tiles, dirs, cfg):
    marker = dirs["done"] / f"{tile.key}.json"
    data = _read_tile_buildings(buildings, source, tile, cfg)
    patches = _facade_segments(data, cfg)
    owned = patches[(patches["home_ix"] == tile.ix) & (patches["home_iy"] == tile.iy)]
    contacts, contact_ids = _contact_rows(patches, data, tile, cfg)
    exchange = _patch_exchange(
        patches, data, tile, allowed_tiles, contact_ids, cfg,
    )
    owned_building_ids = owned["building_id"].drop_duplicates()
    lookup = data[data["node_id"].isin(owned_building_ids)][["node_id"]].copy()
    lookup = lookup.rename(columns={"node_id": "building_id"})
    if cfg.source_id_column and cfg.source_id_column in data:
        lookup["source_building_id"] = data.loc[lookup.index, cfg.source_id_column].astype("string")
    else:
        lookup["source_building_id"] = lookup["building_id"].astype("string")
    lookup = lookup.drop_duplicates("building_id")
    owned.to_parquet(dirs["patches"] / f"{tile.key}.parquet", index=False)
    contacts.to_parquet(dirs["contacts"] / f"{tile.key}.parquet", index=False)
    lookup.to_parquet(dirs["building_ids"] / f"{tile.key}.parquet", index=False)
    exchange.to_parquet(dirs["raw"] / f"{tile.key}.parquet", index=False)
    marker.write_text(json.dumps({
        "tile": tile.key, "buildings_loaded": len(data),
        "patches_owned": len(owned), "contacts": len(contacts),
        "raw_edges": len(exchange),
    }))


def _reduce_partitions(out, dirs, cfg):
    import duckdb

    patch_glob = str(dirs["patches"] / "*.parquet")
    contact_glob = str(dirs["contacts"] / "*.parquet")
    building_ids_glob = str(dirs["building_ids"] / "*.parquet")
    raw_glob = str(dirs["raw"] / "*.parquet")
    patch_exchange_path = out / "patch_exchange.parquet"
    building_edges_path = out / "building_edges.parquet"
    building_metrics_path = out / "building_metrics.parquet"
    building_ids_path = out / "building_ids.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE VIEW patches AS SELECT * FROM read_parquet('{patch_glob}')")
    con.execute(f"CREATE VIEW contacts AS SELECT * FROM read_parquet('{contact_glob}')")
    con.execute(f"CREATE VIEW building_ids AS SELECT * FROM read_parquet('{building_ids_glob}')")
    con.execute(f"CREATE VIEW raw_edges AS SELECT * FROM read_parquet('{raw_glob}')")
    con.execute(f"""
        COPY (
            SELECT building_id, ANY_VALUE(source_building_id) AS source_building_id
            FROM building_ids GROUP BY building_id
        ) TO '{building_ids_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute("""
        CREATE TEMP VIEW incident AS
        SELECT patch_id, SUM(exchange_pq_prelim_m) AS allocated_m
        FROM (
            SELECT patch_p AS patch_id, exchange_pq_prelim_m FROM raw_edges
            UNION ALL
            SELECT patch_q AS patch_id, exchange_pq_prelim_m FROM raw_edges
        ) GROUP BY patch_id
    """)
    con.execute("""
        CREATE TEMP VIEW patch_scale AS
        SELECT p.patch_id,
               LEAST(1.0, p.length_m / NULLIF(i.allocated_m, 0)) AS scale
        FROM patches p JOIN incident i USING (patch_id)
    """)
    con.execute(f"""
        COPY (
            SELECT r.*,
                   r.exchange_pq_prelim_m * LEAST(sp.scale, sq.scale) AS exchange_pq_m
            FROM raw_edges r
            JOIN patch_scale sp ON r.patch_p = sp.patch_id
            JOIN patch_scale sq ON r.patch_q = sq.patch_id
        ) TO '{patch_exchange_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"CREATE VIEW final_edges AS SELECT * FROM read_parquet('{patch_exchange_path}')")
    con.execute("""
        CREATE TEMP VIEW perimeters AS
        SELECT building_id, SUM(length_m) AS perimeter_m
        FROM patches GROUP BY building_id
    """)
    con.execute("""
        CREATE TEMP VIEW patch_to_building_visible AS
        WITH directed AS (
            SELECT e.patch_p AS source_patch, e.building_p AS source_building,
                   e.building_q AS target_building,
                   e.exchange_pq_m / p.length_m AS f_contribution
            FROM final_edges e JOIN patches p ON e.patch_p = p.patch_id
            UNION ALL
            SELECT e.patch_q, e.building_q, e.building_p,
                   e.exchange_pq_m / p.length_m
            FROM final_edges e JOIN patches p ON e.patch_q = p.patch_id
        )
        SELECT source_patch, source_building, target_building,
               LEAST(1.0, SUM(f_contribution)) AS f_patch_to_building
        FROM directed GROUP BY 1, 2, 3
    """)
    con.execute("""
        CREATE TEMP VIEW visible_fmax_building AS
        SELECT LEAST(source_building, target_building) AS building_i,
               GREATEST(source_building, target_building) AS building_j,
               MAX(CASE WHEN source_building < target_building
                        THEN f_patch_to_building ELSE 0 END) AS f_max_i_to_j_visible,
               MAX(CASE WHEN source_building > target_building
                        THEN f_patch_to_building ELSE 0 END) AS f_max_j_to_i_visible
        FROM patch_to_building_visible GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TEMP VIEW contact_fmax_building AS
        SELECT LEAST(building_id, receiver_building) AS building_i,
               GREATEST(building_id, receiver_building) AS building_j,
               MAX(CASE WHEN building_id < receiver_building THEN 1.0 ELSE 0 END)
                   AS f_max_i_to_j_contact,
               MAX(CASE WHEN building_id > receiver_building THEN 1.0 ELSE 0 END)
                   AS f_max_j_to_i_contact
        FROM contacts GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TEMP VIEW pair_fmax AS
        SELECT COALESCE(v.building_i, c.building_i) AS building_i,
               COALESCE(v.building_j, c.building_j) AS building_j,
               GREATEST(COALESCE(v.f_max_i_to_j_visible, 0),
                        COALESCE(c.f_max_i_to_j_contact, 0)) AS f_max_i_to_j,
               GREATEST(COALESCE(v.f_max_j_to_i_visible, 0),
                        COALESCE(c.f_max_j_to_i_contact, 0)) AS f_max_j_to_i
        FROM visible_fmax_building v
        FULL OUTER JOIN contact_fmax_building c USING (building_i, building_j)
    """)
    con.execute("""
        CREATE TEMP VIEW visible_building AS
        SELECT LEAST(building_p, building_q) AS building_i,
               GREATEST(building_p, building_q) AS building_j,
               SUM(exchange_pq_m) AS exchange_visible_ij_m,
               COUNT(*) AS visible_patch_pair_count
        FROM final_edges GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TEMP VIEW contact_building AS
        SELECT LEAST(building_id, receiver_building) AS building_i,
               GREATEST(building_id, receiver_building) AS building_j,
               SUM(CASE WHEN building_id < receiver_building
                        THEN contact_exchange_length_m ELSE 0 END) AS contact_i_to_j_m,
               SUM(CASE WHEN building_id > receiver_building
                        THEN contact_exchange_length_m ELSE 0 END) AS contact_j_to_i_m,
               COUNT(DISTINCT patch_id) AS contact_patch_count
        FROM contacts GROUP BY 1, 2
    """)
    con.execute("""
        CREATE TEMP VIEW combined AS
        SELECT COALESCE(v.building_i, c.building_i) AS building_i,
               COALESCE(v.building_j, c.building_j) AS building_j,
               COALESCE(v.exchange_visible_ij_m, 0) AS exchange_visible_ij_m,
               COALESCE(v.visible_patch_pair_count, 0) AS visible_patch_pair_count,
               COALESCE(c.contact_i_to_j_m, 0) AS contact_i_to_j_m,
               COALESCE(c.contact_j_to_i_m, 0) AS contact_j_to_i_m,
               COALESCE(c.contact_patch_count, 0) AS contact_patch_count
        FROM visible_building v FULL OUTER JOIN contact_building c USING (building_i, building_j)
    """)
    con.execute(f"""
        COPY (
            SELECT x.*, pi.perimeter_m AS perimeter_i_m, pj.perimeter_m AS perimeter_j_m,
                   (x.exchange_visible_ij_m + x.contact_i_to_j_m) / pi.perimeter_m AS vf_i_to_j,
                   (x.exchange_visible_ij_m + x.contact_j_to_i_m) / pj.perimeter_m AS vf_j_to_i,
                   f.f_max_i_to_j, f.f_max_j_to_i,
                   x.contact_patch_count > 0 AS has_contact
            FROM combined x
            JOIN perimeters pi ON x.building_i = pi.building_id
            JOIN perimeters pj ON x.building_j = pj.building_id
            JOIN pair_fmax f USING (building_i, building_j)
        ) TO '{building_edges_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"CREATE VIEW building_edges AS SELECT * FROM read_parquet('{building_edges_path}')")
    con.execute(f"""
        COPY (
            WITH directed AS (
                SELECT building_i AS building_id, building_j AS neighbor_id,
                       vf_i_to_j AS vf, f_max_i_to_j AS f_max,
                       contact_i_to_j_m > 0 AS contact
                FROM building_edges
                UNION ALL
                SELECT building_j, building_i, vf_j_to_i, f_max_j_to_i,
                       contact_j_to_i_m > 0
                FROM building_edges
            ), metrics AS (
                SELECT building_id, COUNT(DISTINCT neighbor_id) AS neighbor_count,
                       SUM(vf) AS bvf_all_edges,
                       SUM(CASE WHEN NOT contact THEN vf ELSE 0 END) AS bvf_noncontact_edges,
                       MAX(f_max) AS bvf_max
                FROM directed GROUP BY building_id
            )
            SELECT p.building_id, p.perimeter_m,
                   p.perimeter_m * COALESCE(m.bvf_all_edges, 0) AS exchange_total_m,
                   COALESCE(m.neighbor_count, 0) AS neighbor_count,
                   COALESCE(m.bvf_all_edges, 0) AS bvf_all_edges,
                   COALESCE(m.bvf_noncontact_edges, 0) AS bvf_noncontact_edges,
                   COALESCE(m.bvf_all_edges, 0) - COALESCE(m.bvf_noncontact_edges, 0)
                       AS bvf_contact_edges,
                   COALESCE(m.bvf_max, 0) AS bvf_max
            FROM perimeters p LEFT JOIN metrics m USING (building_id)
        ) TO '{building_metrics_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    return BVFGraphResult(
        output_dir=out,
        patches_path=dirs["patches"],
        contacts_path=dirs["contacts"],
        building_ids_path=building_ids_path,
        patch_exchange_path=patch_exchange_path,
        building_edges_path=building_edges_path,
        building_metrics_path=building_metrics_path,
        manifest_path=out / "manifest.json",
    )
