"""Tests for ray generation and BVH intersection.

Uses a synthetic flat-box scene (terrain quad + one box building).
No file I/O or network access.
"""

import numpy as np
import pytest

from openviewcore.viewpoints.facade import VP_DTYPE
from openviewcore.simulate.rays import create_rays, create_hemisphere_rays
from openviewcore.simulate.intersect import cast_rays


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_viewpoints(n=3, eye_height=1.83):
    """Create *n* viewpoints facing east (+X) on flat terrain."""
    arr = np.zeros(n, dtype=VP_DTYPE)
    arr["vp_uuid"]     = np.arange(n, dtype=np.int64)
    arr["osmid"]       = 0
    arr["vp"][:, 0]    = np.linspace(0, 10, n)   # X
    arr["vp"][:, 1]    = 0.0                      # Y
    arr["vp"][:, 2]    = eye_height               # Z
    arr["norm"][:, 0]  = 1.0                      # facing +X (east)
    arr["norm"][:, 2]  = 0.0
    arr["segment_len"] = 5.0
    return arr


def _box_scene():
    """A flat terrain quad + a solid box building 20 m to the east.

    Terrain: Z=0 plane covering [-50, 50]² in XY.
    Building: a solid rectangular prism at X=[15,25], Y=[-5,5], Z=[0,10].
    """
    # Terrain — two triangles
    tv = np.array([
        [-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0],
    ], dtype=np.float32)
    tf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # Box building — 6 faces × 2 triangles = 12 faces
    bv = np.array([
        [15, -5, 0], [25, -5, 0], [25,  5, 0], [15,  5, 0],  # bottom
        [15, -5,10], [25, -5,10], [25,  5,10], [15,  5,10],  # top
    ], dtype=np.float32)
    bf = np.array([
        # west face  (X=15)
        [0, 3, 7], [0, 7, 4],
        # east face  (X=25)
        [1, 5, 6], [1, 6, 2],
        # south face (Y=-5)
        [0, 4, 5], [0, 5, 1],
        # north face (Y=5)
        [3, 2, 6], [3, 6, 7],
        # bottom
        [0, 1, 2], [0, 2, 3],
        # top
        [4, 7, 6], [4, 6, 5],
    ], dtype=np.int32) + len(tv)   # offset to match combined vertex array

    v = np.concatenate([tv, bv], axis=0)
    f = np.concatenate([tf, bf], axis=0)
    ncells = {"terrain": len(tf), "buildings": len(bf)}
    return {"v": v, "f": f}, ncells


# ---------------------------------------------------------------------------
# create_rays
# ---------------------------------------------------------------------------

class TestCreateRays:

    def test_output_shapes(self):
        vps = _make_viewpoints(4)
        res = 5
        origins, dirs, ids = create_rays(vps, hfov=45.0, resolution=res)
        n_total = len(vps) * res * res
        assert origins.shape == (n_total, 3)
        assert dirs.shape    == (n_total, 3)
        assert ids.shape     == (n_total,)

    def test_ray_directions_unit_length(self):
        vps = _make_viewpoints(2)
        _, dirs, _ = create_rays(vps, hfov=45.0, resolution=5)
        mags = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-5)

    def test_vp_ids_repeat_correctly(self):
        vps = _make_viewpoints(3)
        res = 4
        _, _, ids = create_rays(vps, resolution=res)
        # Each vp_uuid should appear res² times
        for uid in vps["vp_uuid"]:
            assert (ids == uid).sum() == res * res

    def test_origins_match_viewpoint_positions(self):
        vps = _make_viewpoints(2)
        origins, _, ids = create_rays(vps, resolution=3)
        for vp in vps:
            mask = ids == vp["vp_uuid"]
            # All rays for this viewpoint must start at the viewpoint position
            expected = np.broadcast_to(vp["vp"], origins[mask].shape)
            np.testing.assert_allclose(origins[mask], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# create_hemisphere_rays
# ---------------------------------------------------------------------------

class TestHemisphereRays:

    def test_output_shapes(self):
        vps = _make_viewpoints(2)
        n_az, n_ze = 12, 6
        origins, dirs, ids = create_hemisphere_rays(vps, n_azimuth=n_az, n_zenith=n_ze)
        n_total = len(vps) * n_az * n_ze
        assert origins.shape == (n_total, 3)
        assert dirs.shape    == (n_total, 3)

    def test_all_rays_point_upward(self):
        """Hemisphere rays for an upward-looking viewpoint must have Z > 0."""
        # Use viewpoints with norm=[0,0,1] (looking straight up); hemisphere
        # is rotated into the viewpoint frame so norm direction matters.
        vps = _make_viewpoints(3)
        vps["norm"] = [0.0, 0.0, 1.0]   # override to look straight up
        _, dirs, _ = create_hemisphere_rays(vps)
        assert (dirs[:, 2] > -1e-5).all()

    def test_unit_length(self):
        vps = _make_viewpoints(2)
        _, dirs, _ = create_hemisphere_rays(vps)
        mags = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(mags, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# cast_rays (BVH intersection)
# ---------------------------------------------------------------------------

class TestCastRays:

    def test_output_shapes(self):
        vps = _make_viewpoints(2)
        scene, _ = _box_scene()
        origins, dirs, _ = create_rays(vps, hfov=45.0, resolution=5)
        fid, bc, t = cast_rays(origins, dirs, scene)
        n = len(origins)
        assert fid.shape == (n,)
        assert bc.shape  == (n, 3)
        assert t.shape   == (n,)

    def test_hit_returns_nonnegative_t(self):
        """All mesh hits must have t > 0."""
        vps = _make_viewpoints(2)
        scene, _ = _box_scene()
        origins, dirs, _ = create_rays(vps, hfov=45.0, resolution=10)
        fid, _, t = cast_rays(origins, dirs, scene)
        hit_mask = fid >= 0
        assert (t[hit_mask] > 0).all()

    def test_miss_returns_minus_one(self):
        """Rays pointing straight up from below the terrain must miss."""
        scene, _ = _box_scene()
        origins = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        dirs    = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        fid, _, _ = cast_rays(origins, dirs, scene, ray_near=0.1)
        # fid may come back as 0-d or 1-d depending on pcu version
        assert int(np.ravel(fid)[0]) == -1

    def test_east_facing_rays_hit_building(self):
        """A ray from origin facing east should hit the box building."""
        scene, _ = _box_scene()
        origins = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
        dirs    = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        fid, _, t = cast_rays(origins, dirs, scene, ray_near=0.1)
        assert int(np.ravel(fid)[0]) >= 0                              # hit
        assert float(np.ravel(t)[0]) == pytest.approx(15.0, abs=0.5)  # west face at X=15

    def test_chunked_matches_single_pass(self):
        """cast_rays with chunk_size should return the same result as without."""
        vps = _make_viewpoints(3)
        scene, _ = _box_scene()
        origins, dirs, _ = create_rays(vps, hfov=45.0, resolution=5)

        fid_full, bc_full, t_full = cast_rays(origins, dirs, scene)
        fid_ch,   bc_ch,   t_ch   = cast_rays(origins, dirs, scene, chunk_size=20)

        np.testing.assert_array_equal(fid_full, fid_ch)
        np.testing.assert_allclose(t_full, t_ch, atol=1e-4)
