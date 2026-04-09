"""Download USGS 3DEP LiDAR point clouds via PDAL EPT streaming."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pdal
import requests
from shapely.geometry import box

# Default cache location alongside this file — survives working-dir changes
_DEFAULT_CACHE = Path(__file__).parent / "usgs_3dep_cache.geojson"

# ASPRS noise/withheld classes stripped unconditionally
_NOISE_CLASSES = [7, 18]

# Classes needed for the full lidar pipeline (ground, veg low/med/high, building)
DEFAULT_CLASSES = [2, 3, 4, 5, 6]


class USGSLAZ:
    """Fetch, filter, and download USGS 3DEP LiDAR for an area of interest.

    Usage
    -----
    >>> fetcher = USGSLAZ(target_epsg=26911)
    >>> path = fetcher.fetch_aoi_laz(
    ...     aoi_wgs84_bounds=(-118.40, 34.00, -118.38, 34.02),
    ...     output_name="culver_city",
    ... )
    >>> from openviewcore.lidar import from_las
    >>> pts_down, cls_down, pts_orig, n_ret, ret_num, crs = from_las(path)
    """

    BOUNDARIES_URL = (
        "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master"
        "/boundaries/resources.geojson"
    )

    def __init__(
        self,
        target_epsg: int = 26911,
        cache_path: str | Path = _DEFAULT_CACHE,
    ):
        self.target_epsg = target_epsg
        self.cache_path  = Path(cache_path)
        self.catalog     = self._load_catalog()

    def _load_catalog(self) -> gpd.GeoDataFrame:
        """Load and cache the USGS EPT boundary catalog (projected to EPSG:3857)."""
        if not self.cache_path.exists():
            print("Fetching USGS 3DEP catalog...")
            r = requests.get(self.BOUNDARIES_URL, timeout=30)
            r.raise_for_status()
            self.cache_path.write_bytes(r.content)

        return gpd.read_file(self.cache_path).to_crs(epsg=3857)

    def get_intersecting_datasets(
        self, aoi_wgs84_bounds: tuple[float, float, float, float]
    ) -> tuple[list[str], object]:
        """Return EPT dataset names that intersect the AOI.

        Parameters
        ----------
        aoi_wgs84_bounds : (west, south, east, north) in WGS-84 degrees.

        Returns
        -------
        names : list[str]
        aoi_3857 : shapely geometry  (projected AOI used for the PDAL polygon filter)
        """
        aoi_3857 = (
            gpd.GeoSeries([box(*aoi_wgs84_bounds)], crs="EPSG:4326")
            .to_crs(epsg=3857)
            .iloc[0]
        )
        hits = self.catalog[self.catalog.intersects(aoi_3857)]
        return hits["name"].tolist(), aoi_3857

    def fetch_aoi_laz(
        self,
        aoi_wgs84_bounds: tuple[float, float, float, float],
        output_name: str,
        resolution: float = 1.0,
        class_filter: list[int] | None = None,
    ) -> str:
        """Build and execute a PDAL pipeline, writing a compressed LAZ file.

        Parameters
        ----------
        aoi_wgs84_bounds : (west, south, east, north) in WGS-84 degrees.
        output_name : str
            Output filename stem; ``.laz`` is appended automatically.
        resolution : float
            EPT streaming resolution in metres (lower = denser point cloud).
        class_filter : list[int], optional
            ASPRS classification codes to **keep**.  Defaults to
            ``[2, 3, 4, 5, 6]`` — ground, low/med/high vegetation, building.
            Noise classes 7 and 18 are always removed regardless of this setting.

        Returns
        -------
        str  Path to the written ``.laz`` file.
        """
        names, aoi_3857 = self.get_intersecting_datasets(aoi_wgs84_bounds)
        if not names:
            raise ValueError("No USGS 3DEP datasets found for this area.")

        keep: list[int] = class_filter if class_filter is not None else DEFAULT_CLASSES
        pipeline_nodes: list[dict] = []

        # 1. Readers — one per intersecting EPT dataset
        for name in names:
            pipeline_nodes.append({
                "type":       "readers.ept",
                "filename":   f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{name}/ept.json",
                "polygon":    aoi_3857.wkt,
                "resolution": resolution,
            })

        # 2. Merge when multiple datasets overlap the AOI
        if len(names) > 1:
            pipeline_nodes.append({"type": "filters.merge"})

        # 3. Strip noise/withheld classes and keep only the requested range.
        #    PDAL filters.range uses comma for AND; exclusions remove noise,
        #    then the inclusive range retains min(keep)..max(keep).
        #    This works correctly for the default [2,3,4,5,6] (contiguous).
        noise_expr = ",".join(f"Classification![{c}:{c}]" for c in _NOISE_CLASSES)
        keep_expr  = f"Classification[{min(keep)}:{max(keep)}]"
        pipeline_nodes.append({
            "type":   "filters.range",
            "limits": f"{noise_expr},{keep_expr}",
        })

        # 5. Reproject to metric UTM
        pipeline_nodes.append({
            "type":    "filters.reprojection",
            "out_srs": f"EPSG:{self.target_epsg}",
        })

        # 6. Writer — point format 6 retains return-number fields needed by
        #    classify_ground's multi-return signal; compression via laszip
        out_path = f"{output_name}.laz"
        pipeline_nodes.append({
            "type":          "writers.las",
            "filename":      out_path,
            "minor_version": 4,
            "dataformat_id": 6,
            "compression":   "laszip",
        })

        print(f"Streaming {len(names)} dataset(s), keeping classes {keep}...")
        pdal.Pipeline(json.dumps(pipeline_nodes)).execute()
        return out_path
