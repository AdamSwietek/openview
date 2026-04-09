"""Summarise ray-cast results into per-viewpoint visual share tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def summarize_visibility(
    labels: pd.DataFrame,
    vp_ids: np.ndarray,
    viewpoints: np.ndarray,
) -> pd.DataFrame:
    # 1. Ensure categories are integers
    # We assume 'labels' columns are already Categorical from previous steps
    obs_codes  = labels["obstruction"].cat.codes.values.copy()
    dist_codes = labels["distance_band"].cat.codes.values.copy()

    obs_names  = list(labels["obstruction"].cat.categories)
    dist_names = list(labels["distance_band"].cat.categories)

    # Remap -1 codes (NaN / outside-raster hits) to a dedicated "unsampled"
    # category so they are preserved rather than causing bincount to fail.
    if (obs_codes == -1).any():
        obs_codes[obs_codes == -1] = len(obs_names)
        obs_names = obs_names + ["unsampled"]

    if (dist_codes == -1).any():
        dist_codes[dist_codes == -1] = len(dist_names)
        dist_names = dist_names + ["unsampled"]

    # Map vp_ids to 0 -> N range
    # Using np.unique with return_inverse is the fastest way to get local IDs
    unique_vpids, vpid_codes = np.unique(vp_ids, return_inverse=True)

    n_vps  = len(unique_vpids)
    n_obs  = len(obs_names)
    n_dist = len(dist_names)

    # 2. Linearize the indices
    # We treat the (VP, OBS, DIST) triplet as a single address in a 1D array
    # index = vpid * (total_possible_combinations) + obs * (dist_count) + dist
    flat_indices = (
        vpid_codes.astype(np.int64) * (n_obs * n_dist) +
        obs_codes.astype(np.int64) * n_dist +
        dist_codes.astype(np.int64)
    )

    # 3. Use np.bincount (The fastest way to aggregate in NumPy)
    counts = np.bincount(flat_indices, minlength=n_vps * n_obs * n_dist)
    
    # 4. Reshape and Normalize
    # Reshape to (Viewpoints, Combined_Labels)
    counts_matrix = counts.reshape(n_vps, n_obs * n_dist).astype(np.float32)
    
    # Normalize by row (visual share proportion)
    row_sums = counts_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for viewpoints with no hits
    counts_matrix /= np.where(row_sums == 0, 1, row_sums)

    # 5. Build Column Names
    col_names = [f"{o}_{d}" for o in obs_names for d in dist_names]

    # 6. Assemble Final DataFrame
    result_df = pd.DataFrame(
        counts_matrix,
        index=unique_vpids,   # keep as int64 to match solar index
        columns=col_names
    )

    # Attach Metadata (Join with viewpoints)
    meta_df = pd.DataFrame({
        "osmid": viewpoints["osmid"],
        "X": viewpoints["vp"][:, 0],
        "Y": viewpoints["vp"][:, 1],
        "Z": viewpoints["vp"][:, 2],
        "norm": [v for v in viewpoints["norm"]]
    }, index=viewpoints["vp_uuid"])  # keep as int64

    out = meta_df.join(result_df, how="left").fillna(0.0)
    out.index.name = "vp_uuid"
    return out

# def summarize_visibility(
#     labels: pd.DataFrame,
#     vp_ids: np.ndarray,
#     viewpoints: np.ndarray,
# ) -> pd.DataFrame:
#     """Ultra-fast aggregation using categorical pivoting."""
    
#     # 1. Prepare Categorical Data (Avoid String Concatenation)
#     # If not already categorical, converting now is still faster than + strings
#     obs = labels["obstruction"].astype("category")
#     dist = labels["distance_band"].astype("category")
    
#     # 2. Use a Cross-tabulation approach
#     # We use pd.crosstab because it is optimized for categorical data
#     # This replaces the groupby/value_counts/unstack flow
#     counts = pd.crosstab(
#         index=vp_ids, 
#         columns=[obs, dist], 
#         dropna=False, 
#         normalize='index'
#     )
    
#     # Flatten the multi-index columns from (obs, dist) to "obs_dist"
#     counts.columns = [f"{o}_{d}" for o, d in counts.columns]

#     # 3. Build Metadata DataFrame directly from NumPy
#     # Keep everything in NumPy until the very last moment
#     meta_df = pd.DataFrame({
#         "osmid": viewpoints["osmid"],
#         "X": viewpoints["vp"][:, 0],
#         "Y": viewpoints["vp"][:, 1],
#         "Z": viewpoints["vp"][:, 2],
#     }, index=viewpoints["vp_uuid"].astype(str)) # Ensure index type matches counts.index

#     # Handle the 'norm' vector without converting to a slow Python list
#     # We store it as an object array of the original vectors
#     meta_df["norm"] = [v for v in viewpoints["norm"]]

#     # 4. Final Join
#     # counts.index will be string if vp_ids was string, or int if vp_ids was int
#     # Ensure consistency before joining
#     counts.index = counts.index.astype(str)
    
#     return meta_df.join(counts, how="left").fillna(0.0)

# def summarize_visibility(
#     labels: pd.DataFrame,
#     vp_ids: np.ndarray,
#     viewpoints: np.ndarray,
# ) -> pd.DataFrame:
    
#     # 1. Create the Ray-level DataFrame
#     combined_label = labels["obstruction"] + labels["distance_band"]
#     df = pd.DataFrame({"vpid": vp_ids, "label": combined_label})

#     # 2. Aggregate (normalize proportions)
#     # unstack() makes labels the columns, vpid the index
#     result = df.groupby("vpid")["label"].value_counts(normalize=True).unstack().fillna(0)

#     # 3. Create a Metadata DataFrame from the viewpoints array
#     # This ensures every VP property stays tied to its specific UUID
#     meta_df = pd.DataFrame({
#         "vpid": viewpoints["vp_uuid"], # Assuming 'vp_uuid' is the field name
#         "osmid": viewpoints["osmid"],
#         "X": viewpoints["vp"][:, 0],
#         "Y": viewpoints["vp"][:, 1],
#         "Z": viewpoints["vp"][:, 2],
#         "norm": list(viewpoints["norm"]) # list() handles the 3D vector nicely in a cell
#     }).set_index("vpid")

#     # 4. Join the results (Left join keeps all viewpoints even if they saw nothing)
#     final_output = meta_df.join(result, how="left").fillna(0)

#     return final_output

# def summarize_visibility(
#     labels: pd.DataFrame,
#     vp_ids: np.ndarray,
#     viewpoints: np.ndarray,
# ) -> pd.DataFrame:
#     """Aggregate per-ray labels into lulc view factor per viewpoint.

#     Parameters
#     ----------
#     labels : DataFrame
#         Columns ``obstruction`` and ``distance_band`` from earlier stages.
#     vp_ids : ndarray (N,)
#         Viewpoint UUID per ray (same length as *labels*).
#     viewpoints : np.ndarray
#         Structured viewpoint array with ``osmid``, ``vp``, ``norm`` fields.

#     Returns
#     -------
#     DataFrame
#         One row per viewpoint.  Columns are ``<obstruction><distance_band>``
#         proportions, plus ``osmid``, ``X``, ``Y``, ``Z``, ``norm``.
#     """
#     combined_label = labels["obstruction"] + labels["distance_band"]
#     df = pd.DataFrame({"vpid": vp_ids, "label": combined_label})

#     result = df.groupby("vpid")["label"].value_counts(normalize=True).unstack()
#     result = result.fillna(0)

#     result["osmid"] = viewpoints["osmid"].tolist()
#     result["X"] = viewpoints["vp"][:, 0].tolist()
#     result["Y"] = viewpoints["vp"][:, 1].tolist()
#     result["Z"] = viewpoints["vp"][:, 2].tolist()
#     result["norm"] = viewpoints["norm"].tolist()

#     return result
