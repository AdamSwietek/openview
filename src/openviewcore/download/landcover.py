"""Download / load NLCD land cover raster."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from ..config import DEFAULT_PROJECT_CRS, LANDCOVER_PATH


def download_landcover(
    gdf: gpd.GeoDataFrame,
    *,
    crs: str = DEFAULT_PROJECT_CRS,
    landcover_path: Optional[Path] = None,
) -> Optional[xr.DataArray]:
    """Load and clip an NLCD land cover raster to the visibility region.

    If no path is provided and the ``OPENVIEWCORE_LULC_PATH`` env-var is
    unset, returns *None* (land cover sampling will be skipped later).

    Parameters
    ----------
    gdf : GeoDataFrame
        Visibility region polygon.
    crs : str
        CRS for the clipped output.
    landcover_path : Path, optional
        Path to the NLCD GeoTIFF.

    Returns
    -------
    xr.DataArray or None
    """
    fp = landcover_path or LANDCOVER_PATH
    if fp is None:
        return None

    fp = Path(fp)
    if not fp.exists():
        print('no file found')
        return None

    with rxr.open_rasterio(fp, masked=True) as da:
        raster_crs = da.rio.crs
        mask_gdf = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf
        clipped = da.rio.clip(mask_gdf.geometry, from_disk=True)
    if clipped.rio.crs != crs:
        clipped = clipped.rio.reproject(crs)
    return clipped
