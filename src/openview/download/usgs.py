"""Download and transform scene data (terrain, buildings, etc.)."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import seamless_3dep as s3dep
from pathlib import Path
import tempfile
import rioxarray as rxr
import rasterio
import os



def _get_map_with_timeout(
    map_type,
    bbox,
    data_dir,
    resolution,
    pixel_max,
    timeout,
    limit_per_host,
):
    if timeout is None:
        return s3dep.get_map(map_type, bbox, data_dir, resolution, pixel_max)

    original_download = s3dep.terry.download

    def _download_with_timeout(urls, file_paths, **kwargs):
        return original_download(
            urls,
            file_paths,
            timeout=timeout,
            limit_per_host=limit_per_host,
            **kwargs,
        )

    # s3dep.terry.download = _download_with_timeout
    try:
        return s3dep.get_map(map_type, bbox, data_dir, resolution, pixel_max)
    finally:
        s3dep.terry.download = original_download

def download_dem_from_gdf(
    gdf,
    resolution=30,
    crs="EPSG:4326",
    data_dir='cache/dem',
    feet = True
):
    """
    Downloads a DEM from the USGS National Map for a given bounding box. In/Out is WG84 CRS
    """
    if data_dir is None:
        data_dir = tempfile.mkdtemp(prefix="seamless_3dep_")
    
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    bbox = tuple(gdf.total_bounds)

    data_dir = Path(data_dir)

    tiff_files = s3dep.get_map(map_type = 'DEM',
                            bbox = bbox,
                            save_dir = data_dir,
                            res = resolution)
    
    dem = s3dep.tiffs_to_da(tiff_files, bbox) #source file is a 'engineeringCRS', need to reassign
    dem = dem.rio.write_crs("EPSG:3857", inplace=False) # 
    dem = dem.rio.reproject(crs)
    if feet:
        from src.openview.constants import METERtoFEET
        dem.data = dem.data * METERtoFEET
        
    return dem#, tiff_files


from ..constants import PATH_TO_LA_10DEM, PATH_TO_LA_30DEM, PROJECT_CRS


def load_dem_from_file(gdf, fp=PATH_TO_LA_10DEM, crs=PROJECT_CRS):
    """Load a local DEM, clip to geometry, and reproject."""
    with rxr.open_rasterio(fp) as da:
        if gdf.crs != da.rio.crs:
            gdf = gdf.to_crs(da.rio.crs)
        geometries = gdf.geometry if hasattr(gdf, "geometry") else gdf
        clipped = da.rio.clip(geometries, from_disk=True)

    if crs is not None and clipped.rio.crs != crs:
        clipped = clipped.rio.reproject(crs)

    return clipped


import requests
import os
import zipfile
from tqdm import tqdm

def download_nlcd(year: int, output_dir="nlcd_data"):
    """
    Downloads the 20xx Annual NLCD Land Cover raster (CONUS).
    """
    # 1. SETUP: Define URL and Paths
    # The Annual NLCD 2023 (Collection 1.1) Land Cover file URL
    # If this 404s, get the updated link from: https://www.mrlc.gov/data
    
    base_url = "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    filename = f"Annual_NLCD_LndCov_{year}_CU_C1V1.zip"

    url = os.path.join(base_url, filename)
    output_path = os.path.join(output_dir, filename)
    
    # 2. DOWNLOAD: Stream the file with a progress bar
    print(f"Starting download: {filename}")
    print(f"Source: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for broken links (404/500)
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # 8KB chunks

        with open(output_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        print("\nDownload complete!")

    except requests.exceptions.HTTPError as err:
        print(f"\nError: Could not download file. {err}")
        print("Tip: The URL might have changed. Go to https://www.mrlc.gov/data to copy the latest 'Direct Download' link.")
        return

    # 3. EXTRACT: Unzip the raster
    print("Extracting files...")
    try:
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted to: {output_dir}")
        
        # Cleanup: Option to delete the zip file to save space
        # os.remove(output_path) 
        
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip archive.")
