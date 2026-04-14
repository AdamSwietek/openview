"""Data download: OSM buildings, USGS DEM, NLCD land cover, USGS 3DEP LiDAR."""


def __getattr__(name):
    if name == "download_buildings":
        from .osm import download_buildings
        return download_buildings
    if name == "download_dem":
        from .dem import download_dem
        return download_dem
    if name == "download_landcover":
        from .landcover import download_landcover
        return download_landcover
    if name == "USGSLAZ":
        from .lidar import USGSLAZ
        return USGSLAZ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["download_buildings", "download_dem", "download_landcover", "USGSLAZ"]
