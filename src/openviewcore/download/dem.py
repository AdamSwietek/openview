"""Download a Digital Elevation Model from USGS 3DEP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import seamless_3dep as s3dep
import xarray as xr

from ..config import WGS84, DEFAULT_PROJECT_CRS, CACHE_DIR


def download_dem(
    gdf,
    *,
    resolution: int = 10,
    crs: str = DEFAULT_PROJECT_CRS,
    cache_dir: Optional[Path] = None,
) -> xr.DataArray:
    """Download a USGS 3DEP DEM clipped to the visibility region.

    Parameters
    ----------
    gdf : GeoDataFrame
        Visibility region polygon (reprojected to WGS 84 internally).
    resolution : int
        Desired DEM resolution in meters.
    crs : str
        Target projected CRS for the output raster.
    cache_dir : Path, optional
        Directory to cache downloaded tiles.  Falls back to
        ``OPENVIEWCORE_CACHE_DIR/dem`` or a temp directory.

    Returns
    -------
    xr.DataArray
        Elevation in **meters**, reprojected to *crs*.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR / "dem"

    data_dir = Path(cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # s3dep needs WGS 84 bbox
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)

    bbox = tuple(gdf.total_bounds)

    tiff_files = s3dep.get_map(
        map_type="DEM",
        bbox=bbox,
        save_dir=data_dir,
        res=resolution,
    )

    dem = s3dep.tiffs_to_da(tiff_files, bbox)
    # s3dep returns data in Web Mercator; assign then reproject
    dem = dem.rio.write_crs("EPSG:3857", inplace=False)
    dem = dem.rio.reproject(crs)

    return dem
