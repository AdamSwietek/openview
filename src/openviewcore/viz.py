"""Interactive 3-D visualisation of scenes, viewpoints, and results."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def show_scene(
    terrain_mesh: dict,
    building_meshes: list[dict],
    viewpoints: Optional[np.ndarray] = None,
    *,
    results: Optional = None,
    results_col: str = "sky_total",
    title: str = "openviewcore — 3D Scene",
    window_size: tuple[int, int] = (1200, 800),
) -> None:
    """Launch an interactive PyVista window showing the 3-D scene.

    Parameters
    ----------
    terrain_mesh : dict
        ``{'v': (N,3), 'f': (M,3)}`` terrain mesh.
    building_meshes : list[dict]
        Per-building ``{'v', 'f'}`` meshes.
    viewpoints : np.ndarray, optional
        Structured viewpoint array (with ``vp`` and ``norm`` fields).
        Shown as a point cloud when provided.
    results : DataFrame, optional
        Per-viewpoint results.  If provided together with *viewpoints*,
        viewpoints are coloured by *results_col*.
    results_col : str
        Column in *results* to colour viewpoints by.
    title : str
        Window title.
    window_size : tuple
        ``(width, height)`` in pixels.
    """
    import pyvista as pv

    # Force native OS window
    pv.global_theme.jupyter_backend = None

    # --- Terrain ---
    terrain_pv = _vf_to_polydata(terrain_mesh)
    terrain_pv["elevation"] = terrain_mesh["v"][:, 2]

    # --- Buildings (combined) ---
    bldg_pv = _combine_building_meshes(building_meshes)

    # --- Plotter ---
    pl = pv.Plotter(
        notebook=False, window_size=window_size, title=title,
    )
    pl.add_mesh(
        terrain_pv, color = 'grey',#scalars="elevation", color="grey",
        opacity=0.8, #show_scalar_bar=False,
    )
    pl.add_mesh(bldg_pv, color="yellow", opacity=1, label="Buildings")

    # --- Viewpoints ---
    if viewpoints is not None:
        vp_pts = np.vstack(viewpoints["vp"])
        vp_cloud = pv.PolyData(vp_pts)
        vp_cloud["normals"] = np.vstack(viewpoints["norm"])

        if results is not None and results_col in results.columns:
            vp_cloud[results_col] = results[results_col].values
            pl.add_points(
                vp_cloud, scalars=results_col, cmap="viridis",
                point_size=5, label="Viewpoints",
            )
            # pl.add_arrows(
            #     cent=vp_cloud.points,          # The starting positions (XYZ)
            #     direction=vp_cloud["normals"], # The direction vectors
            #     mag=0.5,                       # Length of the arrows
            #     color="black",                 # Base color (overridden if scalars are used)
            #     label="Viewpoint normals",
            #     scalars=vp_cloud[results_col],           # Data to color the arrows by
            #     cmap="viridis",                # Colormap for the results
            # )
        else:
            pl.add_points(
                vp_cloud, color="red", point_size=5, label="Viewpoints",
            )

    
    pl.add_legend()
    pl.camera_position = "iso"
    pl.show()


def show_building(
    building_idx: int,
    buildings,
    building_meshes: list[dict],
    viewpoints: np.ndarray,
    *,
    title: str | None = None,
    window_size: tuple[int, int] = (1000, 700),
) -> None:
    """Inspect a single building: show its mesh and associated viewpoints.

    Parameters
    ----------
    building_idx : int
        Row position in *buildings* (0-based integer iloc index).
    buildings : GeoDataFrame
        Preprocessed buildings with ``osmid`` column.
    building_meshes : list[dict]
        Per-building mesh list — same order as *buildings*.
    viewpoints : np.ndarray
        Full structured viewpoint array.  Filtered to this building's osmid.
    title : str, optional
        Window title.  Defaults to ``"Building <osmid>"``.
    window_size : tuple
        ``(width, height)`` in pixels.
    """
    import pyvista as pv

    pv.global_theme.jupyter_backend = None

    print(buildings.shape == len(building_meshes))
    row = buildings.iloc[building_idx]
    osmid = row.osmid
    mesh = building_meshes[building_idx]

    # Viewpoints belonging to this building
    mask = viewpoints["osmid"] == osmid
    vps = viewpoints[mask]

    title = title or f"Building osmid={osmid}  |  {mask.sum()} viewpoints"

    # Build PyVista objects
    bldg_pv = _vf_to_polydata(mesh)
    bldg_pv["z"] = mesh["v"][:, 2]

    pl = pv.Plotter(notebook=False, window_size=window_size, title=title)
    pl.add_mesh(bldg_pv, scalars="z", cmap="Oranges",
                show_scalar_bar=False, label="Building")

    if len(vps):
        vp_pts = np.vstack(vps["vp"])
        vp_norms = np.vstack(vps["norm"])

        vp_cloud = pv.PolyData(vp_pts)
        vp_cloud["floor"] = vp_pts[:, 2]   # colour by height (floor level)

        pl.add_points(vp_cloud, scalars="floor", cmap="cool",
                      point_size=10, label="Viewpoints", render_points_as_spheres=True)

        # Arrows showing facade normals
        pl.add_arrows(vp_pts, vp_norms, mag=0.8,
                      color="black", label="Normals")

    pl.add_legend()
    pl.camera_position = "iso"
    pl.show()


# ---------------------------------------------------------------------------
# Solar time-of-day map
# ---------------------------------------------------------------------------

def plot_solar_tod(
    buildings,
    building_solar: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (24, 6),
    title: str = "Facade Solar Exposure — Time of Day",
) -> None:
    """Plot buildings coloured by sunrise, midday, afternoon hours and TOD category.

    Parameters
    ----------
    buildings : GeoDataFrame
        Preprocessed buildings with an ``osmid`` column and projected geometry.
    building_solar : DataFrame
        Output of :func:`~openviewcore.simulate.solar.aggregate_solar_to_buildings`.
        Must be indexed by ``osmid`` and contain ``sunrise_solar_hours``,
        ``midday_solar_hours``, ``afternoon_solar_hours``, and ``tod_category``.
    figsize : tuple
        Overall figure size ``(width, height)`` in inches.
    title : str
        Figure suptitle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Join solar stats onto buildings
    gdf = buildings.set_index("osmid").join(building_solar, how="left")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, y=1.01)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # -- Panel 1: Golden morning hours (el 0–6°, eastern sky) --
    gdf.plot(
        column="golden_morning_hours",
        ax=ax1,
        cmap="Blues",
        missing_kwds={"color": "#dddddd"},
        legend=True,
        legend_kwds={"label": "hrs/yr", "shrink": 0.7},
    )
    ax1.set_title("Golden Morning (el 0–6°, east)", fontsize=10)
    ax1.set_axis_off()

    # -- Panel 2: Midday solar hours (el >= 6°) --
    gdf.plot(
        column="midday_solar_hours",
        ax=ax2,
        cmap="YlOrRd",
        missing_kwds={"color": "#dddddd"},
        legend=True,
        legend_kwds={"label": "hrs/yr", "shrink": 0.7},
    )
    ax2.set_title("Midday (el ≥ 6°)", fontsize=10)
    ax2.set_axis_off()

    # -- Panel 3: Golden afternoon hours (el 0–6°, western sky) --
    gdf.plot(
        column="golden_afternoon_hours",
        ax=ax3,
        cmap="Oranges",
        missing_kwds={"color": "#dddddd"},
        legend=True,
        legend_kwds={"label": "hrs/yr", "shrink": 0.7},
    )
    ax3.set_title("Golden Afternoon (el 0–6°, west)", fontsize=10)
    ax3.set_axis_off()

    # -- Panel 4: TOD category --
    _TOD_COLORS = {
        "Sunrise":  "#4e79a7",   # blue
        "Sunset":   "#e15759",   # red
        "Both":     "#59a14f",   # green
        "Neither":  "#d3d3d3",   # light grey
    }
    ax = ax4
    for category, color in _TOD_COLORS.items():
        subset = gdf[gdf["tod_category"] == category]
        if not subset.empty:
            subset.plot(ax=ax, color=color, linewidth=0.3, edgecolor="white")

    # Grey out buildings with no solar data
    gdf[gdf["tod_category"].isna()].plot(ax=ax, color="#dddddd", linewidth=0.3)

    legend_patches = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in _TOD_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8)
    ax.set_title("TOD Category", fontsize=10)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vf_to_polydata(vf: dict):
    """Convert a ``{'v', 'f'}`` dict to a PyVista PolyData."""
    import pyvista as pv

    faces = np.hstack([
        np.full((vf["f"].shape[0], 1), 3, dtype=np.int32),
        vf["f"],
    ]).ravel()
    return pv.PolyData(vf["v"], faces)


def _combine_building_meshes(meshes: list[dict]):
    """Stack building meshes into a single PolyData."""
    import pyvista as pv

    verts = [m["v"] for m in meshes]
    faces = [m["f"] for m in meshes]
    offsets = np.cumsum([0] + [v.shape[0] for v in verts[:-1]])

    all_v = np.vstack(verts)
    all_f = np.vstack([f + off for f, off in zip(faces, offsets)])

    pv_faces = np.hstack([
        np.full((all_f.shape[0], 1), 3, dtype=np.int32),
        all_f,
    ]).ravel()
    return pv.PolyData(all_v, pv_faces)
