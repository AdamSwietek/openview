"""Tests for viewpoint generation — street and facade modes.

All tests use synthetic geometry (no network downloads required).
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Polygon

from openviewcore.viewpoints.facade import VP_DTYPE, generate_viewpoints
from openviewcore.viewpoints.street import generate_street_viewpoints


# ---------------------------------------------------------------------------
# Helpers — synthetic geometry
# ---------------------------------------------------------------------------

def _flat_terrain_mesh():
    """A simple 2×2 flat quad mesh at Z=0, centred near the origin."""
    v = np.array([
        [-100, -100, 0],
        [ 100, -100, 0],
        [ 100,  100, 0],
        [-100,  100, 0],
    ], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return {"v": v, "f": f}


def _one_building_gdf(crs="EPSG:32611"):
    """A single 10×10 m building footprint, height=9 m."""
    footprint = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame(
        {
            "osmid":     [1],
            "height":    [9.0],
            "elevation": [0.0],
            "old_geom":  [footprint],
        },
        geometry=[footprint],
        crs=crs,
    )
    return gdf


def _street_gdf(crs="EPSG:32611"):
    """A single 50 m street segment running east along Y=5."""
    line = LineString([(0, 5), (50, 5)])
    gdf = gpd.GeoDataFrame(
        {"osmid": [42], "length": [50.0]},
        geometry=[line],
        crs=crs,
    )
    return gdf


# ---------------------------------------------------------------------------
# VP_DTYPE
# ---------------------------------------------------------------------------

def test_vp_dtype_fields():
    fields = VP_DTYPE.names
    assert "vp_uuid"     in fields
    assert "osmid"       in fields
    assert "vp"          in fields
    assert "norm"        in fields
    assert "segment_len" in fields


# ---------------------------------------------------------------------------
# Facade viewpoints
# ---------------------------------------------------------------------------

class TestFacadeViewpoints:

    def test_returns_structured_array(self):
        vps = generate_viewpoints(_one_building_gdf())
        assert isinstance(vps, np.ndarray)
        assert vps.dtype == VP_DTYPE

    def test_nonzero_viewpoints(self):
        vps = generate_viewpoints(_one_building_gdf())
        assert len(vps) > 0

    def test_vp_uuid_unique(self):
        vps = generate_viewpoints(_one_building_gdf())
        assert len(np.unique(vps["vp_uuid"])) == len(vps)

    def test_floor_count(self):
        """A 9 m building with 3 m floors and ground_floor_h=1.5 → 3 floors."""
        vps = generate_viewpoints(
            _one_building_gdf(),
            floor_height=3.0,
            ground_floor_h=1.5,
        )
        # 4 wall segments, 2 viewpoints per segment (two directions) not used here
        # facade.py places one VP per segment per floor — 4 segments × 3 floors = 12
        assert len(vps) == 12

    def test_norms_are_unit_vectors(self):
        vps = generate_viewpoints(_one_building_gdf())
        norms = vps["norm"]
        magnitudes = np.linalg.norm(norms, axis=1)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-5)

    def test_osmid_matches(self):
        vps = generate_viewpoints(_one_building_gdf())
        assert (vps["osmid"] == 1).all()

    def test_terrain_filter_removes_subterrain(self):
        terrain = _flat_terrain_mesh()
        # All viewpoints of a ground-level building should survive terrain filter
        vps_with = generate_viewpoints(_one_building_gdf(), terrain_mesh=terrain)
        vps_without = generate_viewpoints(_one_building_gdf())
        assert len(vps_with) <= len(vps_without)
        assert len(vps_with) > 0


# ---------------------------------------------------------------------------
# Street viewpoints
# ---------------------------------------------------------------------------

class TestStreetViewpoints:

    def test_returns_structured_array(self):
        vps = generate_street_viewpoints(_street_gdf())
        assert isinstance(vps, np.ndarray)
        assert vps.dtype == VP_DTYPE

    def test_two_viewpoints_per_sample(self):
        """Each sample point produces exactly two viewpoints (opposing norms)."""
        vps = generate_street_viewpoints(_street_gdf(), spacing=None)
        # One segment → one midpoint → two opposing viewpoints
        assert len(vps) == 2

    def test_opposing_norms(self):
        """The two viewpoints at each sample share the same XY but have opposite norms."""
        vps = generate_street_viewpoints(_street_gdf(), spacing=None)
        assert len(vps) == 2
        np.testing.assert_allclose(vps["norm"][0], -vps["norm"][1], atol=1e-5)

    def test_spacing_produces_more_viewpoints(self):
        """spacing=10 on a 50 m segment → 5 samples × 2 = 10 viewpoints."""
        vps = generate_street_viewpoints(_street_gdf(), spacing=10.0)
        assert len(vps) == 10

    def test_vp_uuid_unique(self):
        vps = generate_street_viewpoints(_street_gdf(), spacing=10.0)
        assert len(np.unique(vps["vp_uuid"])) == len(vps)

    def test_eye_height_applied(self):
        """Default eye height (1.83 m) should be the Z coordinate on flat terrain."""
        vps = generate_street_viewpoints(_street_gdf(), eye_height=1.83)
        np.testing.assert_allclose(vps["vp"][:, 2], 1.83, atol=1e-5)

    def test_osmid_attached(self):
        vps = generate_street_viewpoints(_street_gdf())
        assert (vps["osmid"] == 42).all()
