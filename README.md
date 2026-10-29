# openview

**Street- and facade-level visual exposure simulation for urban environments.**

`openview` (`openviewcore`) is a Python library that computes what a person sees from a street or building facade — combining 3D scene geometry, hemisphere ray casting, solar simulation, and satellite embedding sampling into a single, reproducible pipeline. It is designed for urban researchers who need physically grounded visual exposure metrics at the scale of a neighbourhood or city.

---

## Why openview

Existing tools address parts of this problem in isolation: fisheye-image methods (Treepedia, Green View Index) measure vegetation visibility but depend on street-view imagery availability and lack 3D solar geometry; solar tools (ladybug, pvlib alone) require pre-built geometry and produce no landscape composition metrics; viewshed tools (GRASS r.viewshed, WhiteboxTools) operate on rasters without sub-metre building geometry or spectral information. `openview` integrates all three into one pipeline that works from LiDAR or open-data building footprints and scales to city-wide analysis.

---

## Key capabilities

- **Viewpoint generation** — street-network viewpoints (OSM) or building facade viewpoints (floor-by-floor), elevation-corrected against terrain
- **Ray casting** — forward view cone or equal-area hemisphere rays; BVH-accelerated intersection via `point_cloud_utils`
- **Solar exposure** — annual direct sun hours, golden-hour access, and shadow fraction using `pvlib`; parallelised shadow casting with a facing-mask optimisation
- **Landscape visual share** — per-viewpoint proportions of sky, terrain, and buildings by distance band (Near / Mid / Far)
- **Satellite embedding exposure** — distance-weighted mean-pool of Google Satellite Embedding (64 bands, 10 m) over terrain ray hits — the spectral signature of what a viewpoint *sees*
- **Optional city mesh** — auto-built from LiDAR (Poisson + terrain) or OSM footprints + DEM; or supply your own triangle mesh

---

## Installation

```bash
pip install openviewcore
```

With optional extras:

```bash
pip install "openviewcore[gee]"   # Google Earth Engine satellite embedding download
pip install "openviewcore[viz]"   # pyvista + contextily for 3D and map plots
pip install "openviewcore[all]"   # everything including dev/test dependencies
```

Requires Python ≥ 3.9.

---

## Quickstart

```python
import geopandas as gpd
from openviewcore.config import PipelineConfig
from openviewcore.pipeline import build_scene, get_visual_exposure
from openviewcore.viewpoints.street import download_streets, generate_street_viewpoints

# 1. Define an area of interest (any polygon GeoDataFrame in WGS84)
aoi = gpd.read_file("my_neighbourhood.geojson")

# 2. Build the 3D scene (downloads OSM buildings + DEM automatically)
cfg = PipelineConfig(crs="EPSG:32611", gee_project="my-gcp-project")
result = build_scene(aoi, cfg, building_source="osm")

# 3. Generate street-level viewpoints every 10 m
streets = download_streets(aoi)
viewpoints = generate_street_viewpoints(
    streets,
    spacing=10.0,
    terrain_mesh=result["inner_terrain"],
)

# 4. Run landscape + solar simulation
exposure = get_visual_exposure(
    viewpoints,
    result["scene"], result["ncells"],
    result["local_scene"],
    lat=34.05, lon=-118.25,
    tz="America/Los_Angeles",
    year=2024,
)

print(exposure.head())
# vp_uuid  sky_Near  buildings_Near  terrain_Mid  annual_solar_hours  ...
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [WIKI_OPENVIEWCORE.md](WIKI_OPENVIEWCORE.md) | Full methods reference — viewpoints, rays, solar, aggregation |
| [docs/tutorials/](docs/tutorials/) | Step-by-step notebooks |
| [WIKI_CAUSAL.md](WIKI_CAUSAL.md) | Causal study design for LA County interventions |

### Tutorials

| Notebook | Topic |
|----------|-------|
| [00_quickstart.ipynb](docs/tutorials/00_quickstart.ipynb) | Synthetic scene — no downloads, runs in < 2 min |
| [01_viewpoints.ipynb](docs/tutorials/01_viewpoints.ipynb) | Facade vs street viewpoints, spacing, elevation |
| [02_ray_casting.ipynb](docs/tutorials/02_ray_casting.ipynb) | Forward cone vs hemisphere, scene mesh options |
| [03_solar.ipynb](docs/tutorials/03_solar.ipynb) | Solar exposure, golden-hour, orientation |
| [04_landscape.ipynb](docs/tutorials/04_landscape.ipynb) | Visual share, LULC refinement, distance bands |
| [05_embedding.ipynb](docs/tutorials/05_embedding.ipynb) | Satellite embedding exposure vs at-viewpoint |
| [06_la_county.ipynb](docs/tutorials/06_la_county.ipynb) | Full application — LA County building simulation |

---

## Project structure

```
openview/
├── src/openviewcore/          # installable package
│   ├── viewpoints/            # VP generation (facade, street)
│   ├── simulate/              # ray generation, solar, landscape
│   ├── aggregate/             # hit classification, raster sampling, summary
│   ├── mesh/                  # optional scene mesh construction
│   ├── download/              # DEM, OSM, GEE, LARIAC data access
│   ├── preprocess/            # building cleaning, terrain elevation
│   ├── config.py              # PipelineConfig dataclass
│   └── pipeline.py            # high-level orchestration
├── docs/tutorials/            # step-by-step Jupyter notebooks
├── tests/                     # pytest test suite
├── notebooks/                 # research notebooks (UHI pilot study)
├── paper.md                   # JOSS submission paper
├── WIKI_OPENVIEWCORE.md       # methods reference
└── pyproject.toml
```


---

## License

MIT License. See [LICENSE](LICENSE).
