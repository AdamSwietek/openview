"""openviewcore — visibility analysis from building facades.

Public API
----------
>>> from openviewcore import run_visibility_pipeline, PipelineConfig
>>> results = run_visibility_pipeline(aoi=my_polygon)
"""

from .config import PipelineConfig
from .pipeline import run_visibility_pipeline, get_visual_exposure

__all__ = [
    "PipelineConfig",
    "run_visibility_pipeline",
    "get_visual_exposure",
]
