"""End-to-end visibility pipeline.

Orchestrates: download -> preprocess -> mesh -> viewpoints -> simulate -> aggregate.

Two spatial extents:
- **aoi** (area of interest): buildings whose facades generate viewpoints.
- **visibility_region**: a larger envelope (aoi + buffer) used to download
  DEM, surrounding buildings, and land cover so that rays have geometry to
  hit beyond the AOI boundary.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import PipelineConfig, WGS84


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

import geopandas as gpd

def build_scene(
    aoi: gpd.GeoDataFrame,
    cfg,
    building_source: str = 'osm',
    canopy_source: str | None = None,
    canopy_height=None,
):
    """Build the full 3-D scene for an area of interest.

    Parameters
    ----------
    aoi : GeoDataFrame
        Area of interest in any CRS.
    cfg : PipelineConfig
        Pipeline configuration.
    building_source : str
        ``'osm'`` (default) or ``'lariac'``.
    canopy_source : str, optional
        GEE canopy-height product key passed to
        :func:`~openviewcore.download.gee.download_canopy_height`.
        One of ``'eth_10m'`` or ``'meta_1m'``.  When provided the canopy
        raster is downloaded automatically and added to the scene mesh.
    canopy_height : xr.DataArray, optional
        Pre-downloaded canopy-height raster (skips GEE download).  Takes
        precedence over *canopy_source* if both are supplied.
    """
    from openviewcore.pipeline import _buffer_aoi
    urb_region = _buffer_aoi(aoi, buffer_m=cfg.urban_buffer, target_crs=cfg.crs)
    vis_region = _buffer_aoi(aoi, buffer_m=cfg.vis_buffer,   target_crs=cfg.crs)

    # Centroid for solar calculations (WGS84)
    centroid_4326 = aoi.to_crs(cfg.crs).dissolve().centroid.to_crs(4326).iloc[0]
    lat, lon = centroid_4326.y, centroid_4326.x
    print(f'  Solar centroid: lat={lat:.4f}, lon={lon:.4f}')

    from openviewcore.download.dem import download_dem
    print('  [DEM] Downloading ...')
    local_dem = download_dem(urb_region, resolution=cfg.dem_resolution, crs=cfg.crs)
    dem       = download_dem(vis_region,  resolution=30,                 crs=cfg.crs)
    print(f'  [DEM] Local: {local_dem.shape}, Regional: {dem.shape}')

    # ── Canopy download (optional) ─────────────────────────────────────────
    if canopy_height is None and canopy_source is not None:
        from openviewcore.download.gee import download_canopy_height
        print(f'  [Canopy] Downloading from GEE ({canopy_source}) ...')
        canopy_height = download_canopy_height(
            urb_region, source=canopy_source, crs=cfg.crs,
            gee_project=cfg.gee_project,
        )
        print(f'  [Canopy] Downloaded: {canopy_height.shape}')

    from openviewcore.download.osm import download_buildings
    from openviewcore.preprocess.buildings import clean_buildings
    from openviewcore.preprocess.terrain import elevate_footprints
    from openviewcore.download.lariac import load_lariac

    if building_source == 'lariac':
        print('  [Buildings] Loading LARIAC ...')

        buildings_raw = load_lariac(urb_region, yr=2020)
        # buildings_raw = download_buildings(urb_region, crs=cfg.crs) # or OSM
        print(f'  [Buildings] {len(buildings_raw)} raw footprints')

        buildings = clean_buildings(
            buildings_raw.dropna(subset='HEIGHT'), #for local data
            crs=cfg.crs,
            height_col='height_m', # for local data
            oid='BLD_ID', # for local data
        )
    else:
        print('  [Buildings] Loading OSM ...')

        buildings_raw = download_buildings(urb_region, crs=cfg.crs) # or OSM
        print(f'  [Buildings] {len(buildings_raw)} raw footprints')

        buildings = clean_buildings(
            buildings_raw,
            crs=cfg.crs,
        )
    buildings = elevate_footprints(buildings, local_dem)
    print(f'  [Buildings] {len(buildings)} after cleaning')

    from openviewcore.mesh.terrain import meshify_terrain, merge_terrain_meshes

    print('  [Terrain] Meshing ...')
    inner_terrain = meshify_terrain(local_dem.fillna(0), downscale=cfg.dem_downscale, reduce=True)
    outer_terrain = meshify_terrain(dem.fillna(0),       downscale=cfg.dem_downscale, reduce=cfg.terrain_reduce)
    terrain_mesh  = merge_terrain_meshes(inner_terrain, outer_terrain)
    print(f'  [Terrain] Merged: {terrain_mesh["v"].shape[0]:,} verts, {terrain_mesh["f"].shape[0]:,} faces')

    from openviewcore.mesh.buildings import meshify_buildings
    from openviewcore.mesh.terrain import meshify_canopy
    from openviewcore.mesh.scene import build_scene

    print('  [Scene] Extruding buildings ...')
    building_meshes = meshify_buildings(buildings)

    # ── Optional canopy ───────────────────────────────────────────────────
    canopy_mesh = None
    if canopy_height is not None:
        print('  [Canopy] Meshing canopy surface ...')
        canopy_mesh = meshify_canopy(local_dem, canopy_height, downscale=cfg.canopy_downscale)
        print(f'  [Canopy] {canopy_mesh["v"].shape[0]:,} verts, {canopy_mesh["f"].shape[0]:,} faces')

    scene, ncells = build_scene(terrain_mesh, building_meshes, canopy_mesh=canopy_mesh)
    print(f'  [Scene] {scene["v"].shape[0]:,} verts, {scene["f"].shape[0]:,} faces  |  ncells={ncells}')

    local_scene, local_ncells = build_scene(inner_terrain, building_meshes, canopy_mesh=canopy_mesh)
    print(f'  [Local Scene] {local_scene["v"].shape[0]:,} verts, {local_scene["f"].shape[0]:,} faces')

    return {'scene':scene,
            'ncells':ncells,
            'local_scene':local_scene,
            'local_ncells':local_ncells,
            'buildings': buildings,
            'building_meshes': building_meshes,
            'inner_terrain': inner_terrain,
            'local_dem': local_dem,
            'dem': dem,
            'canopy_mesh': canopy_mesh,
            'canopy_height': canopy_height,
            }

def run_visibility_pipeline(
    aoi: gpd.GeoDataFrame,
    *,
    visibility_region: Optional[gpd.GeoDataFrame] = None,
    visibility_buffer: float = 500.0,
    cfg: Optional[PipelineConfig] = None,
) -> pd.DataFrame:
    """Run the full visibility simulation for an area of interest.

    Parameters
    ----------
    aoi : GeoDataFrame
        Area of interest polygon (single-row).  Viewpoints are generated
        only from buildings that intersect this region.
    visibility_region : GeoDataFrame, optional
        Larger polygon for DEM / building / land-cover downloads.  When
        *None*, an automatic buffer of *visibility_buffer* meters around
        *aoi* is used.
    visibility_buffer : float
        Buffer distance (meters) applied to *aoi* when
        *visibility_region* is not supplied.  Ignored otherwise.
    cfg : PipelineConfig, optional
        Override default settings.

    Returns
    -------
    pd.DataFrame
        Per-viewpoint visual share table.
    """
    if cfg is None:
        cfg = PipelineConfig()

    # --- Derive visibility region ------------------------------------------
    if visibility_region is None:
        visibility_region = _buffer_aoi(aoi, visibility_buffer, cfg.crs)

    # lazy imports so that module-level import is fast
    from .download.osm import download_buildings
    from .download.dem import download_dem
    from .download.landcover import download_landcover
    from .preprocess.buildings import clean_buildings
    from .preprocess.terrain import elevate_footprints
    from .mesh.terrain import meshify_terrain
    from .mesh.buildings import meshify_buildings
    from .mesh.scene import build_scene
    from .viewpoints.facade import generate_viewpoints
    from .simulate.rays import create_rays
    from .simulate.intersect import cast_rays
    from .aggregate.viewshed import classify_hits
    from .aggregate.landcover import sample_landcover_at_hits
    from .aggregate.summary import summarize_visibility

    # --- 1. Download (use the wider visibility region) ---------------------
    all_buildings_raw = download_buildings(visibility_region, crs=cfg.crs)
    dem = download_dem(visibility_region, resolution=cfg.dem_resolution,
                       crs=cfg.crs, cache_dir=cfg.cache_dir / "dem")
    landcover = download_landcover(visibility_region, crs=cfg.crs,
                                   landcover_path=cfg.landcover_path)

    # --- 2. Preprocess all buildings (full region) -------------------------
    all_buildings = clean_buildings(all_buildings_raw, crs=cfg.crs)
    all_buildings = elevate_footprints(all_buildings, dem)

    # --- 3. Mesh (full region — rays need surrounding context) -------------
    terrain_mesh = meshify_terrain(dem, downscale=cfg.dem_downscale,
                                   reduce=cfg.terrain_reduce)
    building_meshes = meshify_buildings(all_buildings)
    scene, ncells = build_scene(terrain_mesh, building_meshes)

    # --- 4. Viewpoints (AOI buildings only) --------------------------------
    aoi_proj = aoi.to_crs(cfg.crs) if aoi.crs != cfg.crs else aoi
    aoi_buildings = all_buildings[
        all_buildings.intersects(aoi_proj.union_all())
    ].copy()

    viewpoints = generate_viewpoints(
        aoi_buildings,
        floor_height=cfg.floor_height,
        ground_floor_h=cfg.ground_floor_h,
        facade_offset=cfg.facade_offset,
        min_segment_len=cfg.min_segment_len,
    )

    # --- 5. Simulate -------------------------------------------------------
    ray_origins, ray_dirs, vp_ids = create_rays(
        viewpoints,
        hfov=cfg.ray_hfov,
        resolution=cfg.ray_resolution,
    )
    fid, bc, t = cast_rays(ray_origins, ray_dirs, scene,
                           ray_near=cfg.ray_near,
                           chunk_size=cfg.chunk_size)

    # --- 6. Aggregate ------------------------------------------------------
    labels = classify_hits(fid, t, ncells,
                           bins=cfg.distance_bins,
                           bin_labels=cfg.distance_labels)
    labels = sample_landcover_at_hits(
        labels, fid, bc, scene, landcover, crs=cfg.crs,
    )
    results = summarize_visibility(labels, vp_ids, viewpoints)

    return results


# ---------------------------------------------------------------------------
# Chunked pipeline
# ---------------------------------------------------------------------------

def get_visual_exposure(
    viewpoints: np.ndarray,
    scene: dict,
    ncells: dict,
    local_scene: dict,
    lat: float,
    lon: float,
    *,
    vp_chunk_size: int = 500,
    cfg: Optional[PipelineConfig] = None,
    landcover=None,
    tz: str = "UTC",
    year: int = 2024,
    solar_freq: str = "1h",
    golden_freq: str = "5min",
    golden_hour_el: float = 6.0,
    dni=None,
    timestep_chunk: int = 200,
    n_workers: int = 0,
) -> pd.DataFrame:
    """Process viewpoints in chunks through landscape + solar pipelines.

    Builds both BVH intersectors once, then slices viewpoints into chunks of
    ``vp_chunk_size``, runs :func:`~openviewcore.simulate.compute_landscape_exposure`
    and :func:`~openviewcore.simulate.compute_solar_exposure` on each chunk,
    joins the two results on ``vp_uuid``, and concatenates across all chunks.

    Parameters
    ----------
    viewpoints : np.ndarray
        Structured array with ``vp_uuid``, ``osmid``, ``vp`` (XYZ), ``norm``.
    scene : dict
        Full scene mesh ``{'v', 'f'}`` (terrain + buildings) for landscape.
    ncells : dict
        Face counts ``{'terrain': int, 'buildings': int}`` for landscape.
    local_scene : dict
        Local scene mesh ``{'v', 'f'}`` used for solar shadow casting.
    lat, lon : float
        Site coordinates in WGS84 for pvlib solar positions.
    vp_chunk_size : int
        Number of viewpoints per chunk.
    cfg : PipelineConfig, optional
        Pipeline configuration.  A shallow copy is made internally so the
        caller's ``cfg.intersector`` is not overwritten.
    landcover : xr.DataArray, optional
        NLCD raster for land-cover refinement passed to landscape exposure.
    tz : str
        IANA timezone string (e.g. ``"America/Los_Angeles"``).
    year : int
        Calendar year for solar simulation.
    solar_freq : str
        Time resolution for the hourly solar loop.
    golden_freq : str
        Time resolution for the golden-hour loop.
    golden_hour_el : float
        Sun elevation threshold (degrees) splitting golden-hour from midday.
    dni : ndarray, optional
        Direct Normal Irradiance aligned to daylight timesteps.
    timestep_chunk : int
        Solar timesteps per worker call.
    n_workers : int
        Solar worker threads.  ``0`` = ``min(cpu_count, 4)``.

    Returns
    -------
    pd.DataFrame
        Landscape + solar columns merged for all viewpoints, indexed by
        ``vp_uuid``.  Columns are the union of both simulation outputs.
    """
    import contextlib
    import copy
    import os
    import point_cloud_utils as pcu
    from tqdm import tqdm
    from .simulate import compute_landscape_exposure, compute_solar_exposure

    if cfg is None:
        cfg = PipelineConfig()

    # Shallow copy so we can set cfg.intersector without touching the caller's object
    cfg = copy.copy(cfg)

    # ------------------------------------------------------------------
    # Build both BVH intersectors once
    # Redirect fd-level stderr to suppress OpenMP thread-affinity warnings
    # emitted by the native pcu library on macOS.
    # ------------------------------------------------------------------
    def _build_intersector(mesh: dict):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd   = os.dup(2)
        try:
            os.dup2(devnull_fd, 2)
            return pcu.RayMeshIntersector(
                np.ascontiguousarray(mesh["v"], dtype=np.float32),
                np.ascontiguousarray(mesh["f"], dtype=np.int32),
            )
        finally:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            os.close(devnull_fd)

    print("Building landscape intersector (full scene)...")
    landscape_ix = _build_intersector(scene)

    print("Building solar intersector (local scene)...")
    solar_ix = _build_intersector(local_scene)

    # ------------------------------------------------------------------
    # Chunk loop
    # ------------------------------------------------------------------
    n_vps    = len(viewpoints)
    n_chunks = (n_vps + vp_chunk_size - 1) // vp_chunk_size
    print(f"Processing {n_vps:,} viewpoints in {n_chunks} chunks of {vp_chunk_size}")

    chunk_results = []

    for start in tqdm(range(0, n_vps, vp_chunk_size), desc="VP chunks", unit="chunk", position=0, leave=True):
        end      = min(start + vp_chunk_size, n_vps)
        vp_chunk = viewpoints[start:end]

        # Landscape — attach the pre-built intersector to the cfg copy
        cfg.intersector = landscape_ix
        lulc_chunk = compute_landscape_exposure(
            vp_chunk, scene, ncells,
            cfg=cfg,
            landcover=landcover,
        )

        # Solar — intersector passed directly
        solar_chunk = compute_solar_exposure(
            vp_chunk, local_scene, lat, lon,
            tz=tz,
            year=year,
            solar_freq=solar_freq,
            golden_freq=golden_freq,
            golden_hour_el=golden_hour_el,
            dni=dni,
            timestep_chunk=timestep_chunk,
            n_workers=n_workers,
            intersector=solar_ix,
        )

        # Join on shared vp_uuid index
        chunk_results.append(lulc_chunk.join(solar_chunk, how="left"))

    return pd.concat(chunk_results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buffer_aoi(
    aoi: gpd.GeoDataFrame,
    buffer_m: float,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """Buffer the AOI in a projected CRS, return result in WGS 84."""
    projected = aoi.to_crs(target_crs) if aoi.crs != target_crs else aoi.copy()
    buffered = projected.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=buffered, crs=target_crs)
