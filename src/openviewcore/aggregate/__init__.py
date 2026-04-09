"""Post-process ray hits into visibility metrics."""

from .viewshed import classify_hits
from .landcover import sample_landcover_at_hits
from .summary import summarize_visibility

__all__ = ["classify_hits", "sample_landcover_at_hits", "summarize_visibility"]
