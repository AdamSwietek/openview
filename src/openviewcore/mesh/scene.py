"""Combine terrain and building meshes into a single scene."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def build_scene(
    terrain_mesh: dict,
    building_meshes: list[dict],
    *,
    building_ids: Optional[Sequence] = None,
    canopy_mesh: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Join terrain + buildings (+ optional canopy) into one combined mesh.

    Parameters
    ----------
    terrain_mesh : dict
        ``{'v': (N,3), 'f': (M,3)}`` terrain mesh.
    building_meshes : list[dict]
        Per-building ``{'v', 'f'}`` meshes.
    building_ids : sequence, optional
        ID per building mesh.  Defaults to ``0, 1, 2, ...``
    canopy_mesh : dict, optional
        Canopy-top mesh ``{'v', 'f'}`` from
        :func:`~openviewcore.mesh.terrain.meshify_canopy`.  When provided,
        faces are appended after buildings and tagged ``"canopy"`` in
        *ncells* so ray hits are classified correctly.

    Returns
    -------
    scene : dict
        Combined ``{'v': (N,3), 'f': (M,3)}`` mesh.
    ncells : dict
        Maps ``"terrain"``, ``"buildings"`` (and ``"canopy"`` when present)
        to their face counts in the combined mesh.
    """
    all_meshes = [terrain_mesh] + building_meshes
    if canopy_mesh is not None:
        all_meshes.append(canopy_mesh)

    combined = _join_meshes(all_meshes)

    n_terrain   = terrain_mesh["f"].shape[0]
    n_buildings = sum(m["f"].shape[0] for m in building_meshes)
    ncells = {
        "terrain":   n_terrain,
        "buildings": n_buildings,
    }
    if canopy_mesh is not None:
        ncells["canopy"] = canopy_mesh["f"].shape[0]

    return combined, ncells


def _join_meshes(mesh_list: list[dict]) -> dict:
    """Concatenate meshes with proper face-index offsetting."""
    vert_list = [m["v"] for m in mesh_list]
    face_list = [m["f"] for m in mesh_list]

    offsets = np.cumsum([0] + [v.shape[0] for v in vert_list[:-1]])

    vertices = np.vstack(vert_list)
    faces = np.vstack([f + off for f, off in zip(face_list, offsets)])

    return {
        "v": np.ascontiguousarray(vertices, dtype=np.float32),
        "f": np.ascontiguousarray(faces, dtype=np.int32),
    }
