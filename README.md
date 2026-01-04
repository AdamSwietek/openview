# OpenView

OpenView is a visibility and view simulation toolkit for urban terrain and buildings.
It supports data download, mesh construction, viewpoint generation, and ray-based
visibility analysis.

## Features
- Download building footprints and terrain data
- Meshify rasters and building footprints
- Generate viewpoints on facades
- Ray/mesh intersection and view-factor summaries

## Install (editable)
```bash
pip install -e .
```

## Example
```python
from openview.simulate.view import run_windowview_sim_chunked
```

## Notes
- Large scenes require chunking and tiling for memory and runtime.
- See `scripts/` and `notebooks/` for pipelines and experiments.
