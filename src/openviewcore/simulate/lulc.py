"""Unified visual landscape exposure pipeline.

Wraps ray creation, intersection, obstruction classification, optional
land-cover sampling, and per-viewpoint aggregation into a single call.

Typical usage
-------------
::

    from openviewcore.simulate import compute_viewshed

    vsh = compute_viewshed(viewpoints, scene, ncells, cfg=cfg)
    # or with land cover:
    vsh = compute_viewshed(viewpoints, scene, ncells, cfg=cfg, landcover=lc)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_RAY_HFOV,
    DEFAULT_RAY_RESOLUTION,
    DEFAULT_RAY_NEAR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PROJECT_CRS,
    DISTANCE_BINS,
    DISTANCE_LABELS,
    PipelineConfig,
)
from .rays import create_rays
from .intersect import cast_rays
from ..aggregate.viewshed import classify_hits
from ..aggregate.landcover import sample_landcover_at_hits
from ..aggregate.summary import summarize_visibility


def compute_landscape_exposure(
    viewpoints: np.ndarray,
    scene: dict,
    ncells: dict,
    *,
    cfg: Optional[PipelineConfig] = None,
    hfov: float = DEFAULT_RAY_HFOV,
    resolution: int = DEFAULT_RAY_RESOLUTION,
    ray_near: float = DEFAULT_RAY_NEAR,
    chunk_size: Optional[int] = None,
    bins: Sequence[float] = DISTANCE_BINS,
    bin_labels: Sequence[str] = DISTANCE_LABELS,
    landcover=None,
    crs: str = DEFAULT_PROJECT_CRS,
) -> pd.DataFrame:
    """Compute the visual share histogram (VSH) per facade viewpoint.

    Runs the full visual landscape exposure pipeline in one call:
    ``create_rays → cast_rays → classify_hits
    [→ sample_landcover_at_hits] → summarize_visibility``

    Parameters
    ----------
    viewpoints : np.ndarray
        Structured array with ``vp_uuid``, ``osmid``, ``vp`` (XYZ), and
        ``norm`` fields — output of
        :func:`~openviewcore.viewpoints.facade.generate_viewpoints`.
    scene : dict
        ``{'v': (N,3) float32, 'f': (M,3) int32}`` — combined scene mesh
        (terrain + buildings) from
        :func:`~openviewcore.mesh.scene.build_scene`.
    ncells : dict
        ``{'terrain': int, 'buildings': int}`` — face counts returned
        alongside *scene* by :func:`~openviewcore.mesh.scene.build_scene`.
    cfg : PipelineConfig, optional
        When supplied all ray/distance parameters are read from *cfg*;
        individual keyword overrides still take precedence.
    hfov : float
        Half field-of-view in degrees (ignored when *cfg* is given).
    resolution : int
        Grid cells per side of the view window (ignored when *cfg* given).
    ray_near : float
        Ignore intersections closer than this distance (m).
    chunk_size : int, optional
        Number of rays per intersection call.  ``None`` = all at once.
    bins : sequence of float
        Distance bin edges (meters) for the obstruction histogram.
    bin_labels : sequence of str
        Labels for each distance bin (length = ``len(bins) - 1``).
    landcover : xr.DataArray, optional
        NLCD (or equivalent) raster aligned to *crs*.  When provided,
        terrain ray hits are refined to land-cover class codes.
    crs : str
        CRS of the scene coordinates (used for land-cover reprojection).

    Returns
    -------
    pd.DataFrame
        One row per viewpoint.  Columns:

        - ``osmid``, ``X``, ``Y``, ``Z``, ``norm`` — viewpoint metadata
        - ``<obstruction>_<distance_band>`` — visual share proportions
          (sum to 1.0 per row), e.g. ``sky_Near``, ``buildings_Mid``.
    """
    # ------------------------------------------------------------------
    # Apply PipelineConfig defaults (keyword args still override)
    # ------------------------------------------------------------------
    if cfg is not None:
        hfov       = hfov       if hfov       != DEFAULT_RAY_HFOV        else cfg.ray_hfov
        resolution = resolution if resolution != DEFAULT_RAY_RESOLUTION   else cfg.ray_resolution
        ray_near   = ray_near   if ray_near   != DEFAULT_RAY_NEAR         else cfg.ray_near
        chunk_size = chunk_size if chunk_size is not None                  else cfg.chunk_size
        bins       = bins       if bins       is not DISTANCE_BINS        else cfg.distance_bins
        bin_labels = bin_labels if bin_labels is not DISTANCE_LABELS      else cfg.distance_labels
        crs        = crs        if crs        != DEFAULT_PROJECT_CRS      else cfg.crs

    # ------------------------------------------------------------------
    # 1. Ray creation
    # ------------------------------------------------------------------
    ray_origins, ray_dirs, vp_ids = create_rays(
        viewpoints, hfov=hfov, resolution=resolution,
    )

    # ------------------------------------------------------------------
    # 2. Ray–mesh intersection
    # ------------------------------------------------------------------
    fid, bc, t = cast_rays(
        ray_origins, ray_dirs, scene,
        ray_near=ray_near, chunk_size=chunk_size,
        intersector=cfg.intersector if cfg is not None else None,
    )

    # ------------------------------------------------------------------
    # 3. Obstruction + distance classification
    # ------------------------------------------------------------------
    labels = classify_hits(fid, t, ncells, bins=bins, bin_labels=bin_labels)

    # ------------------------------------------------------------------
    # 4. (Optional) refine terrain hits with land-cover class
    # ------------------------------------------------------------------
    if landcover is not None:
        labels = sample_landcover_at_hits(
            labels, fid, bc, scene, landcover, crs=crs,
        )

    # ------------------------------------------------------------------
    # 5. Aggregate to per-viewpoint visual share histogram
    # ------------------------------------------------------------------
    return summarize_visibility(labels, vp_ids, viewpoints)
