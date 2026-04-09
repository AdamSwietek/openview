"""Classify ray hits into obstruction categories (sky / terrain / building)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config import DISTANCE_BINS, DISTANCE_LABELS


# def classify_hits(
#     fid: np.ndarray,
#     t: np.ndarray,
#     ncells: dict,
#     *,
#     bins: Sequence[float] = DISTANCE_BINS,
#     bin_labels: Sequence[str] = DISTANCE_LABELS,
# ) -> pd.DataFrame:
#     """Label each ray hit with an obstruction type and distance band.

#     Parameters
#     ----------
#     fid : ndarray (N,) int
#         Face index per ray (-1 = miss / sky).
#     t : ndarray (N,) float
#         Hit distance per ray.
#     ncells : dict
#         ``{"terrain": int, "buildings": int}`` — face counts from
#         :func:`~openviewcore.mesh.scene.build_scene`.
#     bins : list[float]
#         Distance bin edges (meters).
#     bin_labels : list[str]
#         Labels for each distance bin.

#     Returns
#     -------
#     DataFrame
#         Columns: ``obstruction`` (str), ``distance_band`` (str).
#     """
#     obstruction = _bin_obstruction(fid, ncells)
#     distance_band = pd.cut(t, bins=bins, labels=bin_labels)

#     return pd.DataFrame({
#         "obstruction": obstruction,
#         "distance_band": distance_band.astype(str),
#     })


# def _bin_obstruction(fid: np.ndarray, ncells: dict) -> np.ndarray:
#     """Map face ids to mesh category using cumulative face counts."""
#     cumulative = -1
#     cut_bins = [-1]
#     labels = []
#     for name, count in ncells.items():
#         cumulative += int(count) + 1
#         cut_bins.append(cumulative)
#         labels.append(name)

#     result = pd.cut(fid, bins=cut_bins, labels=labels, include_lowest=True)

#     out = result.astype(str)
#     out[fid < 0] = "sky"
#     return out

def classify_hits(
    fid: np.ndarray,
    t: np.ndarray,
    ncells: dict,
    *,
    bins: Sequence[float] = DISTANCE_BINS,
    bin_labels: Sequence[str] = DISTANCE_LABELS,
) -> pd.DataFrame:
    
    # 1. Use NumPy digitize for distance binning (Stay as integers)
    # digitize is significantly faster than pd.cut for raw arrays
    dist_indices = np.digitize(t, bins) - 1
    # Clip to ensure indices stay within label range
    dist_indices = np.clip(dist_indices, 0, len(bin_labels) - 1)

    # 2. Map obstruction categories using NumPy logic
    # We calculate the thresholds once
    thresholds = np.cumsum([0] + [int(count) for count in ncells.values()])
    names = list(ncells.keys())
    
    # Initialize with 'sky' (we'll use a special value like -1 or len(names))
    obs_indices = np.full(fid.shape, -1, dtype=np.int8)
    
    # Efficiently mask ranges
    # Since ncells is usually just 2-3 items, this loop is negligible
    for i, (lower, upper) in enumerate(zip(thresholds[:-1], thresholds[1:])):
        mask = (fid >= lower) & (fid < upper)
        obs_indices[mask] = i
    
    # 3. Create DataFrame using Categoricals (DO NOT USE STRINGS)
    # Categoricals store data as integers + a small lookup table
    return pd.DataFrame({
        "obstruction": pd.Categorical.from_codes(
            obs_indices + 1, categories=["sky"] + names
        ),
        "distance_band": pd.Categorical.from_codes(
            dist_indices, categories=bin_labels
        ),
    })