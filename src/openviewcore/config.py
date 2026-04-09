"""Runtime configuration for openviewcore.

All spatial operations use meters internally. Unit conversion happens only
at I/O boundaries (download and final output).  Paths are never hardcoded;
pass them at call‑sites or set the corresponding environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_PROC_BIND", "false")
os.environ.setdefault("OMP_PLACES", "threads")



# ---------------------------------------------------------------------------
# CRS
# ---------------------------------------------------------------------------
WGS84: str = "EPSG:4326"

DEFAULT_PROJECT_CRS: str = os.getenv("OPENVIEWCORE_CRS", "EPSG:6424")

# ---------------------------------------------------------------------------
# Building / viewpoint defaults (meters)
# ---------------------------------------------------------------------------
DEFAULT_FLOOR_HEIGHT: float = 3.0       # meters per storey
DEFAULT_GROUND_FLOOR_H: float = 2.0     # height of first viewpoint row
DEFAULT_FACADE_OFFSET: float = 0.5      # push viewpoint away from wall (m)
DEFAULT_MIN_SEGMENT_LEN: float = 1.0    # skip facade edges shorter than this

# ---------------------------------------------------------------------------
# Ray‑casting defaults
# ---------------------------------------------------------------------------
DEFAULT_RAY_HFOV: float = 60.0          # half field‑of‑view (degrees)
DEFAULT_RAY_RESOLUTION: int = 25        # grid cells per side
DEFAULT_RAY_NEAR: float = .5         # ignore hits closer than this (m)
DEFAULT_CHUNK_SIZE: int = 5000          # viewpoints per chunk

# ---------------------------------------------------------------------------
# Distance binning (meters)
# ---------------------------------------------------------------------------
DISTANCE_BINS: list[float] = [0, 100, 1_000, 50_000, float("inf")]
DISTANCE_LABELS: list[str] = ["Near", "Mid", "Far", "Inf"]

# ---------------------------------------------------------------------------
# Conversion factors
# ---------------------------------------------------------------------------
METER_TO_FEET: float = 3.28084

# ---------------------------------------------------------------------------
# Data paths (optional env‑var overrides, never hardcoded defaults)
# ---------------------------------------------------------------------------
LANDCOVER_PATH: Optional[Path] = (
    Path(os.environ["OPENVIEWCORE_LULC_PATH"]).expanduser()
    if "OPENVIEWCORE_LULC_PATH" in os.environ
    else None
)

CACHE_DIR: Path = Path(
    os.getenv("OPENVIEWCORE_CACHE_DIR", "cache")
).expanduser()


# ---------------------------------------------------------------------------
# Pipeline config (bundle all settings into one object)
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Immutable bag of settings passed through the pipeline."""

    crs: str = DEFAULT_PROJECT_CRS
    dem_resolution: int = 10
    floor_height: float = DEFAULT_FLOOR_HEIGHT
    ground_floor_h: float = DEFAULT_GROUND_FLOOR_H
    facade_offset: float = DEFAULT_FACADE_OFFSET
    min_segment_len: float = DEFAULT_MIN_SEGMENT_LEN
    ray_hfov: float = DEFAULT_RAY_HFOV
    ray_resolution: int = DEFAULT_RAY_RESOLUTION
    ray_near: float = DEFAULT_RAY_NEAR
    chunk_size: int = DEFAULT_CHUNK_SIZE
    distance_bins: list[float] = field(default_factory=lambda: list(DISTANCE_BINS))
    distance_labels: list[str] = field(default_factory=lambda: list(DISTANCE_LABELS))
    landcover_path: Optional[Path] = LANDCOVER_PATH
    cache_dir: Path = CACHE_DIR
    dem_downscale: int = 1
    canopy_downscale: int = 2
    terrain_reduce: bool = True
    vis_buffer: float = 1000.0
    urban_buffer: float = 1000.0
    intersector: Any = field(default=None, repr=False, compare=False)
    gee_project: Optional[str] = None   # Google Earth Engine project ID

    def build_intersector(self, scene: dict) -> None:
        """Build and cache the BVH ray-mesh intersector for *scene*.

        Call this once after constructing the scene mesh, then pass *cfg*
        to each simulation function so the BVH is not rebuilt on every call.
        """
        import numpy as np
        import point_cloud_utils as pcu

        v = np.ascontiguousarray(scene["v"], dtype=np.float32)
        f = np.ascontiguousarray(scene["f"], dtype=np.int32)
        self.intersector = pcu.RayMeshIntersector(v, f)
