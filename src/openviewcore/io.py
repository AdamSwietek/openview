"""Save / load meshes, viewpoints, and results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mesh (vertices + faces) as compressed .npz
# ---------------------------------------------------------------------------

def save_mesh(path: str | Path, vf: dict) -> None:
    """Save a mesh dict to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, v=vf["v"], f=vf["f"])


def load_mesh(path: str | Path) -> dict:
    """Load a mesh dict from a ``.npz`` file."""
    data = np.load(path)
    return {"v": data["v"], "f": data["f"]}


# ---------------------------------------------------------------------------
# Viewpoints (structured array)
# ---------------------------------------------------------------------------

def save_viewpoints(path: str | Path, vp: np.ndarray) -> None:
    """Save a structured viewpoint array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vp)


def load_viewpoints(path: str | Path) -> np.ndarray:
    """Load a structured viewpoint array."""
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Results (DataFrame)
# ---------------------------------------------------------------------------

def save_results(path: str | Path, df: pd.DataFrame) -> None:
    """Save visibility results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(path: str | Path) -> pd.DataFrame:
    """Load visibility results from CSV."""
    return pd.read_csv(path)
