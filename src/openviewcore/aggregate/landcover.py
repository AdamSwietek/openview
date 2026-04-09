"""Sample land-cover raster at terrain ray-hit points.

Two modes
---------
* **Discrete (NLCD)**   — ``sample_landcover_at_hits``   → integer class code per hit
* **Continuous (embedding)** — ``sample_embedding_at_hits`` → mean-pooled (V, B) array
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import xarray as xr

from ..config import DEFAULT_PROJECT_CRS


import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import xarray as xr
from pyproj import Transformer

def sample_landcover_at_hits(
    labels: pd.DataFrame,
    fid: np.ndarray,
    bc: np.ndarray,
    scene: dict,
    landcover: xr.DataArray | None,
    *,
    crs: str = "EPSG:3857", # Example default
) -> pd.DataFrame:
    if landcover is None:
        return labels

    # Use underlying numpy array to avoid pandas indexing overhead
    obs_values = labels["obstruction"].values
    terrain_mask = (obs_values == "terrain")
    
    if not terrain_mask.any():
        return labels

    # 1. Vectorized interpolation (fast C++ via pcu)
    hit_pts = pcu.interpolate_barycentric_coords(
        scene["f"], fid[terrain_mask], bc[terrain_mask], scene["v"],
    )

    # 2. Sample LULC values as integers
    lulc_integers = _sample_raster(hit_pts, landcover, crs)

    # 3. Update the labels using Categorical codes
    # We convert the obstruction column to Categorical if it isn't already
    # This is MUCH faster than string replacement
    new_obs = labels["obstruction"].astype(object).values
    new_obs[terrain_mask] = lulc_integers
    
    labels["obstruction"] = pd.Categorical(new_obs)
    
    return labels

def _sample_raster(
    pts: np.ndarray,
    raster: xr.DataArray,
    src_crs: str,
) -> np.ndarray:
    """Zero-overhead sampling using raw index math."""
    x, y = pts[:, 0], pts[:, 1]
    raster_crs = str(raster.rio.crs)

    # 1. Vectorized Transform (keep this, pyproj is already C-speed)
    if raster_crs != src_crs:
        transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)
        x, y = transformer.transform(x, y)

    # 2. Extract Raster Metadata once
    # Affine transform: x_pixel = (x - x_min) / res
    x_min = float(raster.x.min())
    y_max = float(raster.y.max()) # Rasters usually start top-left
    res_x = float(raster.rio.transform()[0])
    res_y = abs(float(raster.rio.transform()[4]))

    # 3. Calculate Indices (The "Speed Demon" Step)
    # This replaces all the overhead of xarray.sel()
    cols = ((x - x_min) / res_x).astype(np.int32)
    rows = ((y_max - y) / res_y).astype(np.int32)

    # 4. Clip to ensure we don't index out of bounds
    cols = np.clip(cols, 0, raster.shape[-1] - 1)
    rows = np.clip(rows, 0, raster.shape[-2] - 1)

    # 5. Direct Fancy Indexing
    # Assuming raster is (band, y, x)
    return raster.values[0, rows, cols]


def _sample_embedding_raster(
    pts: np.ndarray,
    raster: xr.DataArray,
    src_crs: str,
) -> np.ndarray:
    """Sample all bands of a continuous raster at XY points.

    Returns
    -------
    ndarray (N, B) float32 — one row per hit point, one column per band.
    """
    x, y = pts[:, 0], pts[:, 1]
    raster_crs = str(raster.rio.crs)

    if raster_crs != src_crs:
        from pyproj import Transformer
        transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)
        x, y = transformer.transform(x, y)

    x_min = float(raster.x.min())
    y_max = float(raster.y.max())
    res_x = float(raster.rio.transform()[0])
    res_y = abs(float(raster.rio.transform()[4]))

    cols = np.clip(((x - x_min) / res_x).astype(np.int32), 0, raster.shape[-1] - 1)
    rows = np.clip(((y_max - y) / res_y).astype(np.int32), 0, raster.shape[-2] - 1)

    # raster.values is (B, H, W) → index all bands at once → (B, N) → (N, B)
    return raster.values[:, rows, cols].T.astype(np.float32)


def sample_embedding_at_hits(
    fid: np.ndarray,
    bc: np.ndarray,
    scene: dict,
    vp_ids: np.ndarray,
    embedding: xr.DataArray,
    *,
    t: np.ndarray | None = None,
    distance_weight: bool = True,
    crs: str = "EPSG:3857",
    band_names: list[str] | None = None,
) -> pd.DataFrame:
    """Distance-weighted mean-pool of a continuous raster over terrain ray hits.

    For each viewpoint, terrain-hit rays are sampled from the embedding raster
    and averaged weighted by ``1 / t`` (inverse hit distance) so that nearby
    surfaces contribute more than distant ones.  Pass ``distance_weight=False``
    to revert to uniform mean-pooling.

    Parameters
    ----------
    fid, bc : ndarray
        Face IDs and barycentric coords from ``cast_rays``.
    scene : dict  ``{'v', 'f'}``
    vp_ids : ndarray (N,) int
        Viewpoint index per ray — aligns with *fid* and *bc*.
    embedding : xr.DataArray  shape (B, H, W)
    t : ndarray (N,) float, optional
        Ray hit distances from ``cast_rays``.  Required when
        ``distance_weight=True`` (the default).
    distance_weight : bool
        Weight each hit by ``1 / t`` before averaging.  Default ``True``.
    crs : str
        CRS of the scene / ray hit coordinates.
    band_names : list[str], optional
        Output column names.

    Returns
    -------
    DataFrame  shape (V, B)
        Indexed by unique viewpoint ID.  NaN rows where no terrain hit exists.
    """
    terrain_mask = fid >= 0
    n_bands = embedding.shape[0]
    cols    = band_names or [f"emb_{i}" for i in range(n_bands)]

    if not terrain_mask.any():
        return pd.DataFrame(
            np.full((len(np.unique(vp_ids)), n_bands), np.nan, dtype=np.float32),
            columns=cols,
        )

    hit_pts = pcu.interpolate_barycentric_coords(
        scene["f"], fid[terrain_mask], bc[terrain_mask], scene["v"],
    )

    # (K, B)
    hit_emb = _sample_embedding_raster(hit_pts, embedding, crs)

    # inverse-distance weights: w = 1/t, clipped to avoid div-by-zero
    if distance_weight and t is not None:
        hit_t   = np.clip(t[terrain_mask], 1e-3, None)
        weights = (1.0 / hit_t).astype(np.float64)   # (K,)
    else:
        weights = np.ones(hit_emb.shape[0], dtype=np.float64)

    hit_vp_ids = vp_ids[terrain_mask]
    unique_vps = np.unique(vp_ids)
    vp_to_idx  = {v: i for i, v in enumerate(unique_vps)}
    idx        = np.array([vp_to_idx[v] for v in hit_vp_ids], dtype=np.int32)

    w_sums  = np.zeros((len(unique_vps), n_bands), dtype=np.float64)
    w_total = np.zeros(len(unique_vps),            dtype=np.float64)

    np.add.at(w_sums,  idx, hit_emb * weights[:, None])
    np.add.at(w_total, idx, weights)

    mean_emb = np.where(w_total[:, None] > 0, w_sums / w_total[:, None], np.nan)

    return pd.DataFrame(mean_emb.astype(np.float32), index=unique_vps, columns=cols)

# def _sample_raster(
#     pts: np.ndarray,
#     raster: xr.DataArray,
#     src_crs: str,
# ) -> np.ndarray:
#     """Fast nearest-neighbour sampling using pyproj and direct index lookup."""
#     x, y = pts[:, 0], pts[:, 1]
#     raster_crs = str(raster.rio.crs)

#     # 1. Faster Coordinate Transformation
#     if raster_crs != src_crs:
#         # always_xy=True ensures we don't accidentally swap Lat/Lon
#         transformer = Transformer.from_crs(src_crs, raster_crs, always_xy=True)
#         x, y = transformer.transform(x, y)

#     # 2. Vectorized Indexing
#     # Using sel(method="nearest") is okay, but ensure we use 'points' dimension correctly
#     # xarray.sel is generally fast, but direct indexing is faster if you pre-calculate
#     x_da = xr.DataArray(x, dims="z") # 'z' is just a placeholder dimension name
#     y_da = xr.DataArray(y, dims="z")
    
#     # Grab values and drop extra dimensions immediately
#     vals = raster.sel(x=x_da, y=y_da, method="nearest").values
    
#     # If the raster has a band dimension (e.g., shape (1, N)), squeeze it
#     if vals.ndim > 1:
#         vals = np.squeeze(vals)

#     return vals # Keep as int for now

# def sample_landcover_at_hits(
#     labels: pd.DataFrame,
#     fid: np.ndarray,
#     bc: np.ndarray,
#     scene: dict,
#     landcover: Optional[xr.DataArray],
#     *,
#     crs: str = DEFAULT_PROJECT_CRS,
# ) -> pd.DataFrame:
#     """Replace ``"terrain"`` labels with NLCD land-cover class names.

#     Parameters
#     ----------
#     labels : DataFrame
#         Output of :func:`~openviewcore.aggregate.viewshed.classify_hits`.
#     fid, bc : ndarray
#         Face IDs and barycentric coordinates from ray casting.
#     scene : dict
#         Combined ``{'v', 'f'}`` mesh.
#     landcover : xr.DataArray or None
#         NLCD raster.  If *None*, terrain hits are left as ``"terrain"``.
#     crs : str
#         CRS of the scene and ray coordinates.

#     Returns
#     -------
#     DataFrame
#         Same shape as *labels* with terrain obstruction refined to NLCD codes.
#     """
#     if landcover is None:
#         return labels

#     terrain_mask = labels["obstruction"].values == "terrain"
#     if not terrain_mask.any():
#         return labels

#     hit_pts = pcu.interpolate_barycentric_coords(
#         scene["f"], fid[terrain_mask], bc[terrain_mask], scene["v"],
#     )

#     lulc_values = _sample_raster(hit_pts, landcover, crs)

#     labels = labels.copy()
#     labels.loc[terrain_mask, "obstruction"] = lulc_values

#     return labels
# # print('testw')
# def _sample_raster(
#     pts: np.ndarray,
#     raster: xr.DataArray,
#     crs: str,
# ) -> np.ndarray:
#     """Nearest-neighbour sample of a raster at XY points."""
#     x, y = pts[:, 0], pts[:, 1]

    
#     raster_crs = raster.rio.crs

#     if str(raster_crs) != crs:
#         print('Reprojecting points to match raster CRS...')
#         gs = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=crs).to_crs(raster_crs)
#         x = gs.geometry.x.to_numpy()
#         y = gs.geometry.y.to_numpy()

#     x_da = xr.DataArray(x, dims="points")
#     y_da = xr.DataArray(y, dims="points")
#     vals = raster.sel(x=x_da, y=y_da, method="nearest").values

#     if vals.ndim > 1:
#         vals = vals[0]

#     return vals.astype(int).astype(str)
