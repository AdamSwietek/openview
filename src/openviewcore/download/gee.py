"""Download vegetation and surface data from Google Earth Engine.

Focused on canopy height rasters for appending vegetation geometry to the
scene mesh.  Other GEE assets (DSM, land cover) are available via the helper
primitives at the bottom of this module.

Requires optional dependencies::

    pip install earthengine-api geemap rioxarray

Authentication must be completed before use::

    import ee; ee.Authenticate(); ee.Initialize(project='your-project')
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import geopandas as gpd
import xarray as xr

from ..config import WGS84, DEFAULT_PROJECT_CRS


# ---------------------------------------------------------------------------
# Canopy height sources
# ---------------------------------------------------------------------------

CANOPY_SOURCES: dict[str, dict] = {
    "meta_1m": {
        "asset":      "projects/meta-forest-monitoring-okw37/assets/CanopyHeight",
        "type":       "ImageCollection",
        "band":       "height",
        "resolution": 1,
        "description": "Meta/WRI High-Resolution 1 m Global Canopy Height (2020)",
    },
    "eth_10m": {
        "asset":  "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1",
        "type":   "Image",
        "band":   "b1",
        "resolution": 10,
        "description": "ETH Global Sentinel-2 10 m Canopy Height (2020)",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_canopy_height(
    gdf: gpd.GeoDataFrame,
    *,
    source: str = "eth_10m",
    resolution: Optional[int] = None,
    crs: str = DEFAULT_PROJECT_CRS,
    gee_project: Optional[str] = None,
) -> xr.DataArray:
    """Download a canopy-height raster for *gdf* from Google Earth Engine.

    Parameters
    ----------
    gdf : GeoDataFrame
        Area of interest.  Any CRS is accepted; reprojected to WGS 84
        internally for the EE query.
    source : str
        One of ``"meta_1m"`` (1 m, global) or ``"eth_10m"`` (10 m, global).
        See :data:`CANOPY_SOURCES` for details.
    resolution : int, optional
        Export resolution in metres.  Defaults to the native resolution of
        the chosen source.
    crs : str
        Target projected CRS for the returned raster.

    Returns
    -------
    xr.DataArray
        Canopy height in **metres**, reprojected to *crs*.  NoData pixels
        are set to ``0`` (bare ground / no canopy).

    Examples
    --------
    >>> canopy = download_canopy_height(aoi.to_crs(cfg.crs), crs=cfg.crs)
    >>> canopy.plot()
    """
    import rioxarray  # noqa: F401  — activates .rio accessor

    _init_ee(project=gee_project) if gee_project else _init_ee()

    if source not in CANOPY_SOURCES:
        raise ValueError(
            f"Unknown source '{source}'. Choose from: {list(CANOPY_SOURCES)}"
        )

    src    = CANOPY_SOURCES[source]
    res    = resolution or src["resolution"]
    roi    = _gdf_to_ee_roi(gdf)
    image  = _fetch_ee_image(src["asset"], src["type"], roi).select(src["band"])

    with TemporaryDirectory() as tmp:
        tif = str(Path(tmp) / "canopy.tif")
        _export_to_tif(image, tif, scale=res, region=roi)
        da = xr.open_dataarray(tif, engine="rasterio").load()

    da = da.squeeze(drop=True)              # drop band dim (now always single-band)
    da = da.where(da < 255)               # 255 = uint8 NoData sentinel → NaN
    da = da.where(da >= 0, other=0)       # clip remaining negatives to 0
    da = da.fillna(0)                     # NaN → 0 (bare ground)
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.reproject(crs)
    da.attrs["source"]      = src["description"]
    da.attrs["resolution_m"] = res
    return da


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _init_ee(**kwargs) -> None:
    """Initialize Earth Engine"""
    import ee
    try:
        ee.data.getAssetRoots()
    except Exception:
        ee.Initialize(**kwargs)


def _gdf_to_ee_roi(gdf: gpd.GeoDataFrame):
    """Convert a GeoDataFrame to an ``ee.Geometry`` in WGS 84.

    Parameters
    ----------
    gdf : GeoDataFrame
        Area of interest in any CRS.

    Returns
    -------
    ee.Geometry.Polygon
    """
    import ee

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)

    geom   = gdf.dissolve().geometry.iloc[0]
    coords = [list(c) for c in geom.exterior.coords]
    return ee.Geometry.Polygon([coords])


def _fetch_ee_image(asset: str, asset_type: str, roi):
    """Return a clipped ``ee.Image`` from either an Image or ImageCollection.

    Parameters
    ----------
    asset : str
        Earth Engine asset ID.
    asset_type : str
        ``"Image"`` or ``"ImageCollection"``.
    roi : ee.Geometry
        Region to clip to.

    Returns
    -------
    ee.Image
    """
    import ee

    if asset_type == "ImageCollection":
        image = (
            ee.ImageCollection(asset)
            .filterBounds(roi)
            .sort("system:time_start")
            .first()
            .unmask()
        )
    elif asset_type == "Image":
        image = ee.Image(asset)
    else:
        raise ValueError(f"asset_type must be 'Image' or 'ImageCollection', got '{asset_type}'")

    return image.clip(roi)


def _export_to_tif(image, path: str, *, scale: int, region) -> None:
    """Download an ``ee.Image`` to a local GeoTIFF via geemap.

    Parameters
    ----------
    image : ee.Image
        Image to export.
    path : str
        Destination file path (must end in ``.tif``).
    scale : int
        Export resolution in metres.
    region : ee.Geometry
        Export bounds.
    """
    import geemap

    geemap.ee_export_image(
        image,
        filename=path,
        scale=scale,
        region=region,
        file_per_band=False,
        crs="EPSG:4326",
    )
