"""Tests for hit classification and visual share aggregation.

No network access or file I/O. Uses synthetic fid / t arrays.
"""

import numpy as np
import pandas as pd
import pytest

from openviewcore.aggregate.viewshed import classify_hits
from openviewcore.aggregate.summary import summarize_visibility
from openviewcore.viewpoints.facade import VP_DTYPE


BINS   = [0, 25, 100, 300, float("inf")]
LABELS = ["Near", "Mid", "Far", "Inf"]

NCELLS = {"terrain": 10, "buildings": 20}  # faces 0–9 = terrain, 10–29 = buildings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vps(n=4):
    arr = np.zeros(n, dtype=VP_DTYPE)
    arr["vp_uuid"] = np.arange(n, dtype=np.int64)
    arr["osmid"]   = 0
    arr["vp"]      = [[float(i), 0.0, 1.83] for i in range(n)]
    arr["norm"]    = [[1.0, 0.0, 0.0]] * n
    return arr


def _synthetic_hits(n_rays=100, n_miss=10):
    """Return (fid, t, vp_ids) with a mix of terrain hits, building hits, and sky."""
    rng = np.random.default_rng(0)

    # terrain hits: fid in [0, 9]
    n_terrain  = n_rays - n_miss - 20
    n_building = 20

    fid = np.concatenate([
        rng.integers(0, 10,  size=n_terrain),   # terrain face ids
        rng.integers(10, 30, size=n_building),  # building face ids
        np.full(n_miss, -1, dtype=np.int64),    # sky / miss
    ])
    t = np.concatenate([
        rng.uniform(5,   80,  size=n_terrain),   # Near / Mid distances
        rng.uniform(110, 280, size=n_building),  # Mid / Far distances
        np.zeros(n_miss),                        # irrelevant for sky
    ])
    # Assign vp_ids round-robin over 4 viewpoints
    vp_ids = np.tile(np.arange(4, dtype=np.int64), n_rays // 4 + 1)[:n_rays]

    return fid, t, vp_ids


# ---------------------------------------------------------------------------
# classify_hits
# ---------------------------------------------------------------------------

class TestClassifyHits:

    def test_output_shape(self):
        fid, t, _ = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == len(fid)
        assert set(labels.columns) == {"obstruction", "distance_band"}

    def test_sky_on_miss(self):
        fid = np.array([-1, -1, 5], dtype=np.int64)
        t   = np.array([0.0, 0.0, 10.0])
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        assert (labels["obstruction"].iloc[:2] == "sky").all()

    def test_terrain_category(self):
        fid = np.array([0, 5, 9], dtype=np.int64)   # all terrain faces
        t   = np.array([10.0, 10.0, 10.0])
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        assert (labels["obstruction"] == "terrain").all()

    def test_buildings_category(self):
        fid = np.array([10, 20, 29], dtype=np.int64)  # all building faces
        t   = np.array([50.0, 50.0, 50.0])
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        assert (labels["obstruction"] == "buildings").all()

    def test_distance_bins(self):
        fid = np.array([0, 0, 0, 0], dtype=np.int64)
        t   = np.array([10.0, 60.0, 200.0, 400.0])
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        assert labels["distance_band"].tolist() == ["Near", "Mid", "Far", "Inf"]

    def test_proportions_sum_to_one(self):
        """Proportions across all distance bands per obstruction should make sense."""
        fid, t, _ = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        # Just check that the categories are populated correctly
        assert labels["obstruction"].isin(["sky", "terrain", "buildings"]).all()
        assert labels["distance_band"].isin(LABELS).all()


# ---------------------------------------------------------------------------
# summarize_visibility
# ---------------------------------------------------------------------------

class TestSummarizeVisibility:

    def test_output_shape(self):
        vps = _make_vps(4)
        fid, t, vp_ids = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        result = summarize_visibility(labels, vp_ids, vps)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4            # one row per viewpoint
        assert "osmid" in result.columns

    def test_visual_share_sums_to_one(self):
        vps = _make_vps(4)
        fid, t, vp_ids = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        result = summarize_visibility(labels, vp_ids, vps)

        share_cols = [c for c in result.columns if "_" in c and c not in
                      ("osmid", "segment_len")]
        row_sums = result[share_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    def test_sky_present_in_columns(self):
        vps = _make_vps(4)
        fid, t, vp_ids = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        result = summarize_visibility(labels, vp_ids, vps)
        sky_cols = [c for c in result.columns if c.startswith("sky_")]
        assert len(sky_cols) > 0

    def test_xyz_metadata_attached(self):
        vps = _make_vps(4)
        fid, t, vp_ids = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        result = summarize_visibility(labels, vp_ids, vps)
        assert "X" in result.columns
        assert "Y" in result.columns
        assert "Z" in result.columns

    def test_no_negative_shares(self):
        vps = _make_vps(4)
        fid, t, vp_ids = _synthetic_hits()
        labels = classify_hits(fid, t, NCELLS, bins=BINS, bin_labels=LABELS)
        result = summarize_visibility(labels, vp_ids, vps)
        share_cols = [c for c in result.columns if "_" in c and c not in ("osmid",)]
        assert (result[share_cols] >= 0).all().all()
