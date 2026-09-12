"""Small end-to-end checks for the tiled 2D building graph."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, box

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from openviewcore.network import BVFGraphConfig, build_bvf_graph


@pytest.mark.parametrize("visibility_backend", ["strtree", "bvh"])
def test_two_tile_graph_has_visible_and_contact_edges(tmp_path, visibility_backend):
    buildings = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3],
            "BLD_ID": ["A", "B", "C"],
            "geometry": [box(12, 4, 18, 12), box(22, 4, 28, 12), box(28, 4, 34, 12)],
        },
        crs="EPSG:32611",
    )
    config = BVFGraphConfig(
        id_column="OBJECTID",
        source_id_column="BLD_ID",
        target_crs="EPSG:32611",
        tile_size_m=20,
        max_distance_m=20,
        min_edge_length_m=1,
        workers=1,
        resume=False,
        visibility_backend=visibility_backend,
    )

    result = build_bvf_graph(
        buildings, tmp_path / "graph", config=config, bounds=(0, 0, 40, 20),
    )

    edges = pd.read_parquet(result.building_edges_path).set_index(["building_i", "building_j"])
    metrics = pd.read_parquet(result.building_metrics_path)
    ids = pd.read_parquet(result.building_ids_path).set_index("building_id")

    assert (1, 2) in edges.index
    assert not bool(edges.loc[(1, 2), "has_contact"])
    assert (2, 3) in edges.index
    assert bool(edges.loc[(2, 3), "has_contact"])
    assert edges.loc[(2, 3), "vf_i_to_j"] == pytest.approx(8 / 28)
    assert edges.loc[(2, 3), "vf_j_to_i"] == pytest.approx(8 / 28)
    assert edges.loc[(2, 3), "f_max_i_to_j"] == pytest.approx(1.0)
    assert edges.loc[(2, 3), "f_max_j_to_i"] == pytest.approx(1.0)
    assert edges["f_max_i_to_j"].between(0, 1).all()
    assert edges["f_max_j_to_i"].between(0, 1).all()
    assert (edges["f_max_i_to_j"] >= edges["vf_i_to_j"]).all()
    assert (edges["f_max_j_to_i"] >= edges["vf_j_to_i"]).all()
    assert ids.loc[1, "source_building_id"] == "A"
    assert np.allclose(
        metrics["exchange_total_m"],
        metrics["perimeter_m"] * metrics["bvf_all_edges"],
    )
    assert metrics["bvf_all_edges"].between(0, 1 + 1e-12).all()
    assert metrics["bvf_max"].between(0, 1).all()
    assert metrics.set_index("building_id").loc[2, "bvf_max"] == pytest.approx(1.0)
    directed_max = pd.concat([
        edges.reset_index()[["building_i", "f_max_i_to_j"]].rename(
            columns={"building_i": "building_id", "f_max_i_to_j": "f_max"}
        ),
        edges.reset_index()[["building_j", "f_max_j_to_i"]].rename(
            columns={"building_j": "building_id", "f_max_j_to_i": "f_max"}
        ),
    ]).groupby("building_id")["f_max"].max()
    observed_max = metrics.set_index("building_id").loc[directed_max.index, "bvf_max"]
    assert np.allclose(observed_max, directed_max)
    assert (
        metrics["bvf_all_edges"]
        - metrics["bvf_noncontact_edges"]
        - metrics["bvf_contact_edges"]
    ).abs().max() < 1e-12


@pytest.mark.parametrize("visibility_backend", ["strtree", "bvh"])
def test_selection_mask_limits_nodes_but_retains_external_blockers(
    tmp_path, visibility_backend,
):
    a = box(2, 4, 8, 12)
    blocker = box(10, 2, 14, 14)
    c = box(16, 4, 22, 12)
    buildings = gpd.GeoDataFrame(
        {"OBJECTID": [1, 2, 3], "BLD_ID": ["A", "blocker", "C"]},
        geometry=[a, blocker, c], crs="EPSG:32611",
    )
    selection = MultiPolygon([a.buffer(0.1), c.buffer(0.1)])
    config = BVFGraphConfig(
        id_column="OBJECTID", source_id_column="BLD_ID",
        target_crs="EPSG:32611", tile_size_m=20, max_distance_m=30,
        min_edge_length_m=1, workers=1, resume=False,
        visibility_backend=visibility_backend,
    )

    result = build_bvf_graph(
        buildings, tmp_path / "selected_graph", config=config,
        selection_mask=selection,
    )

    ids = pd.read_parquet(result.building_ids_path)
    edges = pd.read_parquet(result.building_edges_path)
    assert set(ids["building_id"]) == {1, 3}
    assert edges.empty


def test_bvh_visibility_is_stable_at_large_projected_coordinates(tmp_path):
    footprints = [
        translate(box(2, 4, 8, 12), xoff=384_000, yoff=3_768_000),
        translate(box(10, 2, 14, 14), xoff=384_000, yoff=3_768_000),
        translate(box(16, 4, 22, 12), xoff=384_000, yoff=3_768_000),
    ]
    buildings = gpd.GeoDataFrame(
        {"OBJECTID": [1, 2, 3], "BLD_ID": ["A", "blocker", "C"]},
        geometry=footprints, crs="EPSG:26911",
    )
    selection = MultiPolygon([
        footprints[0].buffer(0.1), footprints[2].buffer(0.1),
    ])
    config = BVFGraphConfig(
        id_column="OBJECTID", source_id_column="BLD_ID",
        target_crs="EPSG:26911", tile_size_m=2_000,
        max_distance_m=30, min_edge_length_m=1, workers=1,
        resume=False, visibility_backend="bvh",
    )

    result = build_bvf_graph(
        buildings, tmp_path / "large_coordinates", config=config,
        selection_mask=selection,
    )

    assert pd.read_parquet(result.building_edges_path).empty
