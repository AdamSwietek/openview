"""Set in stone."""
# pylint: disable=invalid-name
KiB = 1024**1
MiB = 1024**2
GiB = 1024**3

SECOND: int = 1
MINUTE: int = 60
HOUR: int = 60 * 60
DAY: int = 24 * 60 * 60
MONTH: int = 30 * 24 * 60 * 60

METERtoFEET = 3.28084

WGS84_EPSG: str = "EPSG:4326"
PROJECT_CRS: str = "EPSG:6424"

#sample bounding boxes for tests
BBOX = dict(
    DEFAULT = (-118.4135, 34.0040, -118.3790, 34.0300),  # Carson Park, Culver City
    SMALL = (-118.39798, 34.01570, -118.39453, 34.01830),
    TINY = (-118.396505, 34.016825, -118.396005, 34.017175),
    LARGE = (-118.48515, 34.10441,-118.21564,33.98977), # WestLA to DTLA 
)


#paths
from pathlib import Path
RAW_DATA_DIR = '/Users/adamswietek/Documents/PostDoc/data/raw_la'

PATH_TO_LULC = '/Users/adamswietek/Documents/PostDoc/data/raw_national/Annual_NLCD_LndCov_2023_CU_C1V0.tif'

PATH_TO_LA_10DEM = '/Users/adamswietek/Documents/PostDoc/data/raw_la/dem/USGS_1_3_Arc_Second_LosAngelesCounty.tif'
PATH_TO_LA_30DEM = '/Users/adamswietek/Documents/PostDoc/data/raw_la/dem/USGS_1_Arc_Second_LosAngelesCounty.tif'