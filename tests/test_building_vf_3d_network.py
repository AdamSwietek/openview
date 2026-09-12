import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from openviewcore.network import BVF3DGraphConfig, build_bvf_3d_graph


def _config(**overrides):
    values = dict(
        id_column="OBJECTID", source_id_column="BLD_ID",
        target_crs="EPSG:26911", tile_size_m=100.0, max_distance_m=30.0,
        height_column="HEIGHT", height_scale_to_m=1.0,
        elevation_column="ELEV", elevation_scale_to_m=1.0,
        max_patch_width_m=10.0, max_patch_height_m=3.0,
        min_edge_length_m=0.5, workers=1, resume=False,
    )
    values.update(overrides)
    return BVF3DGraphConfig(**values)


def test_height_aware_rays_see_over_shorter_building(tmp_path):
    buildings = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3], "BLD_ID": ["a", "blocker", "c"],
            "HEIGHT": [9.0, 3.0, 9.0], "ELEV": [0.0, 0.0, 0.0],
        },
        geometry=[box(10, 10, 12, 12), box(15, 10, 17, 12), box(20, 10, 22, 12)],
        crs="EPSG:26911",
    )
    result = build_bvf_3d_graph(buildings, tmp_path / "height", config=_config())
    exchange = pd.read_parquet(result.patch_exchange_path)
    patches = pd.concat(
        [pd.read_parquet(path) for path in result.patches_path.glob("*.parquet")],
        ignore_index=True,
    ).set_index("patch_id")
    ac = exchange[
        ((exchange.building_p == 1) & (exchange.building_q == 3))
        | ((exchange.building_p == 3) & (exchange.building_q == 1))
    ]
    assert len(ac) > 0
    z_p = ac.patch_p.map(patches.z_m)
    z_q = ac.patch_q.map(patches.z_m)
    assert np.any((z_p > 3.0) & (z_q > 3.0))
    assert not np.any((z_p < 3.0) & (z_q < 3.0))


def test_3d_graph_has_area_outputs_and_contact_coupling(tmp_path):
    buildings = gpd.GeoDataFrame(
        {
            "OBJECTID": [10, 11], "BLD_ID": ["left", "right"],
            "HEIGHT": [6.0, 3.0], "ELEV": [0.0, 0.0],
        },
        geometry=[box(10, 10, 12, 12), box(12, 10, 14, 12)],
        crs="EPSG:26911",
    )
    result = build_bvf_3d_graph(buildings, tmp_path / "contact", config=_config())
    edges = pd.read_parquet(result.building_edges_path)
    metrics = pd.read_parquet(result.building_metrics_path)
    assert len(edges) == 1
    assert bool(edges.has_contact.iloc[0])
    assert edges.contact_i_to_j_m2.iloc[0] > 0
    assert {"facade_area_m2", "exchange_total_m2", "bvf_max"} <= set(metrics.columns)
    assert np.all(metrics.bvf_all_edges >= 0)


def test_selection_mask_keeps_external_building_as_blocker_only(tmp_path):
    buildings = gpd.GeoDataFrame(
        {
            "OBJECTID": [1, 2, 3], "BLD_ID": ["a", "blocker", "c"],
            "HEIGHT": [6.0, 20.0, 6.0], "ELEV": [0.0, 0.0, 0.0],
        },
        geometry=[box(10, 10, 12, 12), box(15, 10, 17, 12), box(20, 10, 22, 12)],
        crs="EPSG:26911",
    )
    selection = box(9, 9, 13, 13).union(box(19, 9, 23, 13))
    result = build_bvf_3d_graph(
        buildings, tmp_path / "selection", config=_config(), selection_mask=selection,
    )
    ids = pd.read_parquet(result.building_ids_path)
    exchange = pd.read_parquet(result.patch_exchange_path)
    assert set(ids.building_id) == {1, 3}
    assert 2 not in set(exchange.building_p) | set(exchange.building_q)
    assert exchange.empty
