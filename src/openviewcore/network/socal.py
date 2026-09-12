"""Prepare Southern California WUI and bulk OSM inputs for the 2D graph."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from tqdm.auto import tqdm


WUI_FEATURE_SERVICE = (
    "https://services1.arcgis.com/pf6KDbd8NVL1IUHa/arcgis/rest/services/"
    "Wildland_Urban_Interface_vector/FeatureServer/1"
)
GEOFABRIK_SOCAL_GPKG_URL = (
    "https://download.geofabrik.de/north-america/us/california/"
    "socal-latest-free.gpkg.zip"
)
GEOFABRIK_CALIFORNIA_PBF_URL = (
    "https://download.geofabrik.de/north-america/us/california-latest.osm.pbf"
)
DEFAULT_SOCAL_COUNTIES = (
    "Imperial County", "Kern County", "Los Angeles County", "Orange County",
    "Riverside County", "San Bernardino County", "San Diego County",
    "San Luis Obispo County", "Santa Barbara County", "Ventura County",
)


@dataclass(frozen=True)
class SoCal2DSourceConfig:
    """Configuration for a reproducible OSM/WUI source snapshot."""

    target_crs: str = "EPSG:3310"
    wui_buffer_m: float = 1_609.344
    blocker_halo_m: float = 152.45
    wui_simplify_m: float = 25.0
    counties: tuple[str, ...] = DEFAULT_SOCAL_COUNTIES
    wui_class_column: Optional[str] = "WUI_DESC"
    wui_classes: Optional[tuple[str, ...]] = ("Interface",)
    osm_layer: Optional[str] = None
    osm_id_column: str = "osm_id"
    output_prefix: str = "socal_wui"


@dataclass(frozen=True)
class SoCal2DSourceResult:
    output_dir: Path
    wui_path: Path
    analysis_mask_path: Path
    source_mask_path: Path
    buildings_path: Path
    manifest_path: Path


def fetch_wui_service(
    output_path: str | Path,
    *,
    service_url: str = WUI_FEATURE_SERVICE,
    chunk_size: int = 200,
    timeout_s: float = 120.0,
) -> Path:
    """Download every WUI feature by object ID and cache a GeoPackage.

    Querying object IDs first avoids silent truncation at the service's maximum
    record count. The server layer combines Interface and Intermix geometries;
    it does not expose their original WUI class as an attribute.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    id_payload = _read_json(
        f"{service_url}/query?" + urllib.parse.urlencode({
            "where": "1=1", "returnIdsOnly": "true", "f": "json",
        }),
        timeout_s,
    )
    object_ids = sorted(map(int, id_payload.get("objectIds", [])))
    if not object_ids:
        raise RuntimeError(f"The WUI service returned no object IDs: {id_payload}")
    frames = []
    for start in tqdm(
        range(0, len(object_ids), chunk_size), desc="Downloading WUI",
        unit="batch", dynamic_ncols=True,
    ):
        ids = object_ids[start:start + chunk_size]
        query = urllib.parse.urlencode({
            "objectIds": ",".join(map(str, ids)), "outFields": "*",
            "returnGeometry": "true", "outSR": "4326", "f": "geojson",
        })
        payload = _read_json(f"{service_url}/query?{query}", timeout_s)
        frames.append(gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326"))
    wui = pd.concat(frames, ignore_index=True)
    wui = gpd.GeoDataFrame(wui, geometry="geometry", crs="EPSG:4326")
    if output.exists():
        output.unlink()
    wui.to_file(output, layer="wui", driver="GPKG")
    return output


def _read_json(url: str, timeout_s: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "openviewcore/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def download_geofabrik_socal(
    output_zip: str | Path,
    *,
    url: str = GEOFABRIK_SOCAL_GPKG_URL,
    overwrite: bool = False,
    timeout_s: float = 120.0,
) -> Path:
    """Stream the Southern California GeoPackage archive with a progress bar."""
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "openviewcore/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        with partial.open("wb") as stream, tqdm(
            total=total, desc="Downloading OSM GeoPackage", unit="B",
            unit_scale=True, unit_divisor=1024, dynamic_ncols=True,
        ) as progress:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                progress.update(len(block))
    partial.replace(output)
    return output


def download_geofabrik_pbf(
    output_pbf: str | Path,
    *,
    url: str = GEOFABRIK_CALIFORNIA_PBF_URL,
    overwrite: bool = False,
    timeout_s: float = 120.0,
) -> Path:
    """Stream a Geofabrik PBF and reject HTML/error-page responses."""
    output = Path(output_pbf)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output
    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "openviewcore/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content_type = response.headers.get_content_type().casefold()
        if content_type in {"text/html", "text/plain"}:
            raise RuntimeError(
                f"Expected an OSM PBF from {url}, but received {content_type} "
                f"from {response.geturl()}."
            )
        total = int(response.headers.get("Content-Length", 0)) or None
        with partial.open("wb") as stream, tqdm(
            total=total, desc="Downloading California OSM PBF", unit="B",
            unit_scale=True, unit_divisor=1024, dynamic_ncols=True,
        ) as progress:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                progress.update(len(block))
    partial.replace(output)
    return output


def convert_osm_pbf_buildings(
    osm_pbf: str | Path,
    output_gpkg: str | Path,
    *,
    overwrite: bool = False,
    ogr2ogr: str = "ogr2ogr",
) -> Path:
    """Convert only OSM building multipolygons from a PBF to indexed GPKG."""
    source, output = Path(osm_pbf), Path(output_gpkg)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        return output
    executable = shutil.which(ogr2ogr)
    if executable is None:
        raise RuntimeError(
            "ogr2ogr is required to convert the statewide OSM PBF. Install GDAL "
            "in the openview environment (for example, conda install -c conda-forge gdal)."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.gpkg")
    if partial.exists():
        partial.unlink()
    sql = (
        "SELECT CASE WHEN osm_id IS NOT NULL THEN 'r' || osm_id "
        "ELSE 'w' || osm_way_id END AS osm_id, geometry "
        "FROM multipolygons WHERE building IS NOT NULL"
    )
    subprocess.run([
        executable, "-f", "GPKG", str(partial), str(source),
        "-dialect", "SQLite", "-sql", sql,
        "-nln", "gis_osm_buildings_a_free_1", "-nlt", "PROMOTE_TO_MULTI",
        "-lco", "SPATIAL_INDEX=YES", "-progress",
    ], check=True)
    partial.replace(output)
    return output


def extract_geopackage_archive(
    archive: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Extract and return the single GeoPackage in a Geofabrik zip archive."""
    archive, output_dir = Path(archive), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive} is not a ZIP archive. It may be an HTML response saved "
            "from an unavailable Geofabrik product URL."
        )
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.lower().endswith(".gpkg")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one GeoPackage in {archive}, found {names}")
        destination = output_dir / Path(names[0]).name
        if destination.exists() and not overwrite:
            return destination
        bundle.extract(names[0], output_dir)
        extracted = output_dir / names[0]
        if extracted != destination:
            extracted.replace(destination)
    return destination


def build_wui_masks(
    wui: str | Path | gpd.GeoDataFrame,
    *,
    config: Optional[SoCal2DSourceConfig] = None,
    clip_bounds: Optional[tuple[float, float, float, float]] = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return selected WUI, one-mile node mask, and node-plus-blocker mask."""
    cfg = config or SoCal2DSourceConfig()
    frame = (
        _read_wui(wui, class_column=cfg.wui_class_column)
        if not isinstance(wui, gpd.GeoDataFrame) else wui.copy()
    )
    if frame.crs is None:
        raise ValueError("WUI features must have a CRS.")
    if cfg.wui_classes:
        class_column = _find_column(frame.columns, cfg.wui_class_column or "")
        if class_column is not None:
            wanted_classes = {str(value).strip().casefold() for value in cfg.wui_classes}
            frame = frame[
                frame[class_column].astype("string").str.strip().str.casefold().isin(wanted_classes)
            ].copy()
            if frame.empty:
                raise ValueError(
                    f"No WUI features matched {class_column}={cfg.wui_classes}."
                )
    name_column = _find_column(frame.columns, "NAMELSAD") or _find_column(frame.columns, "NAME")
    if name_column is not None and cfg.counties:
        wanted = {_canonical_county_name(name) for name in cfg.counties}
        available_names = frame[name_column].astype("string").map(_canonical_county_name)
        frame = frame[available_names.isin(wanted)].copy()
    selected = frame.to_crs(cfg.target_crs)
    if clip_bounds is not None:
        clip = shapely.box(*clip_bounds).buffer(
            cfg.wui_buffer_m + cfg.blocker_halo_m,
        )
        selected.geometry = shapely.intersection(selected.geometry.to_numpy(), clip)
        clipped = selected.geometry.to_numpy()
        selected = selected[
            ~shapely.is_missing(clipped) & ~shapely.is_empty(clipped)
        ].copy()
    if selected.empty:
        raise ValueError("No WUI features remain after the requested filters.")
    geometry = selected.geometry.simplify(cfg.wui_simplify_m, preserve_topology=True)
    analysis_geometry = shapely.union_all(shapely.buffer(
        geometry.to_numpy(), cfg.wui_buffer_m, quad_segs=4,
    ))
    source_geometry = shapely.buffer(
        analysis_geometry, cfg.blocker_halo_m, quad_segs=2,
    )
    analysis = gpd.GeoDataFrame({"mask": ["analysis"]}, geometry=[analysis_geometry], crs=cfg.target_crs)
    source = gpd.GeoDataFrame({"mask": ["source"]}, geometry=[source_geometry], crs=cfg.target_crs)
    return selected, analysis, source


def prepare_osm_buildings(
    osm_geopackage: str | Path,
    output_path: str | Path,
    *,
    source_mask: gpd.GeoDataFrame | gpd.GeoSeries,
    config: Optional[SoCal2DSourceConfig] = None,
) -> Path:
    """Clip Geofabrik buildings, normalize IDs, and write FlatGeobuf."""
    import pyogrio

    cfg = config or SoCal2DSourceConfig()
    source = Path(osm_geopackage)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = cfg.osm_layer or _detect_building_layer(source)
    info = pyogrio.read_info(source, layer=layer)
    layer_crs = info["crs"]
    mask = _union_mask(source_mask, cfg.target_crs)
    mask_source = gpd.GeoSeries([mask], crs=cfg.target_crs).to_crs(layer_crs).iloc[0]
    available = set(info["fields"])
    osm_id_column = _find_column(available, cfg.osm_id_column)
    if osm_id_column is None:
        raise KeyError(
            f"OSM building layer {layer!r} lacks {cfg.osm_id_column!r}; "
            f"available fields: {sorted(available)}"
        )
    optional = ["name", "type", "fclass", "code"]
    columns = [osm_id_column] + [c for c in optional if c in available]
    buildings = pyogrio.read_dataframe(
        source, layer=layer, mask=mask_source, columns=columns,
    ).to_crs(cfg.target_crs)
    buildings.geometry = shapely.make_valid(buildings.geometry.to_numpy())
    buildings = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    buildings = buildings[shapely.intersects(buildings.geometry.to_numpy(), mask)].copy()
    raw_id = buildings[osm_id_column].astype("string").str.strip()
    buildings = buildings[raw_id.notna() & raw_id.ne("")].copy()
    raw_id = buildings[osm_id_column].astype("string").str.strip()
    buildings["source_building_id"] = "osm:" + raw_id
    buildings = buildings.sort_values("source_building_id", kind="stable")
    duplicated = buildings["source_building_id"].duplicated(keep=False)
    if duplicated.any():
        examples = buildings.loc[duplicated, "source_building_id"].unique()[:5].tolist()
        raise ValueError(
            "OSM IDs are not unique within the building layer; refusing to discard "
            f"geometry. Examples: {examples}"
        )
    buildings = buildings.reset_index(drop=True)
    buildings["building_id"] = np.arange(1, len(buildings) + 1, dtype=np.int64)
    keep = ["building_id", "source_building_id"] + [
        c for c in optional if c in buildings
    ] + ["geometry"]
    buildings = buildings[keep]
    if output.exists():
        output.unlink()
    pyogrio.write_dataframe(buildings, output, driver="FlatGeobuf")
    return output


def prepare_socal_2d_source(
    osm_geopackage: str | Path,
    wui_path: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[SoCal2DSourceConfig] = None,
) -> SoCal2DSourceResult:
    """Create all local inputs required by ``build_bvf_graph``."""
    cfg = config or SoCal2DSourceConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    osm_bounds = _osm_bounds_in_target_crs(Path(osm_geopackage), cfg)
    selected, analysis, source = build_wui_masks(
        wui_path, config=cfg, clip_bounds=osm_bounds,
    )
    prefix = cfg.output_prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix.")
    selected_path = output / f"{prefix}.gpkg"
    analysis_path = output / f"{prefix}_analysis_mask.gpkg"
    source_path = output / f"{prefix}_source_mask.gpkg"
    buildings_path = output / f"{prefix}_osm_buildings.fgb"
    for path in (selected_path, analysis_path, source_path):
        if path.exists():
            path.unlink()
    selected.to_file(selected_path, layer="wui", driver="GPKG")
    analysis.to_file(analysis_path, layer="mask", driver="GPKG")
    source.to_file(source_path, layer="mask", driver="GPKG")
    prepare_osm_buildings(
        osm_geopackage, buildings_path, source_mask=source, config=cfg,
    )
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "0.1", "source_model": "geofabrik_osm_wui_2d",
        "config": asdict(cfg), "osm_geopackage": str(Path(osm_geopackage).resolve()),
        "wui_input": str(Path(wui_path).resolve()),
        "wui_class_column": cfg.wui_class_column,
        "wui_classes": cfg.wui_classes,
        "selected_wui_features": len(selected),
        "building_features": int(pyogrio_feature_count(buildings_path)),
    }, indent=2))
    return SoCal2DSourceResult(
        output_dir=output, wui_path=selected_path,
        analysis_mask_path=analysis_path, source_mask_path=source_path,
        buildings_path=buildings_path, manifest_path=manifest_path,
    )


def pyogrio_feature_count(path: str | Path) -> int:
    import pyogrio
    return int(pyogrio.read_info(path)["features"])


def _find_column(columns: Iterable[str], requested: str) -> Optional[str]:
    lookup = {str(column).casefold(): str(column) for column in columns}
    return lookup.get(requested.casefold())


def _canonical_county_name(value) -> str:
    name = str(value).strip().casefold()
    return name.removesuffix(" county").strip()


def _read_wui(
    path: str | Path, *, class_column: Optional[str] = None,
) -> gpd.GeoDataFrame:
    source = Path(path)
    if source.suffix.casefold() in {".parquet", ".geoparquet"}:
        import pyarrow.parquet as pq

        schema = pq.read_schema(source)
        metadata = schema.metadata or {}
        geo = json.loads(metadata.get(b"geo", b"{}").decode())
        geometry_column = geo.get("primary_column", "geometry")
        lookup = {name.casefold(): name for name in schema.names}
        requested = [geometry_column, "NAMELSAD", "NAME", class_column]
        columns = []
        for name in requested:
            actual = lookup.get(str(name).casefold()) if name else None
            if actual is not None and actual not in columns:
                columns.append(actual)
        return gpd.read_parquet(source, columns=columns)
    return gpd.read_file(source, layer="wui")


def _osm_bounds_in_target_crs(
    path: Path, config: SoCal2DSourceConfig,
) -> tuple[float, float, float, float]:
    import pyogrio

    layer = config.osm_layer or _detect_building_layer(path)
    info = pyogrio.read_info(path, layer=layer)
    transformer = Transformer.from_crs(info["crs"], config.target_crs, always_xy=True)
    return tuple(map(float, transformer.transform_bounds(
        *info["total_bounds"], densify_pts=21,
    )))


def _detect_building_layer(path: Path) -> str:
    import pyogrio
    layers = pyogrio.list_layers(path)
    candidates = [
        str(name) for name, geometry_type in layers
        if "building" in str(name).casefold()
        and "polygon" in str(geometry_type).casefold()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely identify the polygon building layer in {path}; "
            f"candidates={candidates}. Set osm_layer explicitly."
        )
    return candidates[0]


def _union_mask(mask, target_crs: str):
    if isinstance(mask, gpd.GeoDataFrame):
        frame = mask.to_crs(target_crs)
        return frame.geometry.union_all()
    if isinstance(mask, gpd.GeoSeries):
        return mask.to_crs(target_crs).union_all()
    raise TypeError("source_mask must be a GeoDataFrame or GeoSeries with a CRS.")
