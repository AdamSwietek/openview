"""Preprocess buildings and terrain."""

from .buildings import clean_buildings
from .terrain import elevate_footprints

__all__ = ["clean_buildings", "elevate_footprints"]
