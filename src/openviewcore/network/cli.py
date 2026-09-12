"""Command-line entry point for regional 2D and 3D BVF graph construction."""

from __future__ import annotations

import argparse

from .building_vf import BVFGraphConfig, build_bvf_graph
from .building_vf_3d import BVF3DGraphConfig, build_bvf_3d_graph
from .socal import (
    DEFAULT_SOCAL_COUNTIES,
    SoCal2DSourceConfig,
    fetch_wui_service,
    prepare_socal_2d_source,
)


def _add_selection_arguments(parser):
    parser.add_argument("--selection", help="vector layer selecting graph nodes")
    parser.add_argument("--selection-layer")
    parser.add_argument("--selection-field", help="attribute field used to filter selection features")
    parser.add_argument("--selection-value", help="exact string value retained from --selection-field")
    parser.add_argument("--selection-buffer", type=float, default=0.0)
    parser.add_argument("--selection-simplify", type=float, default=0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openview-bvf")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build a tiled 2D building graph")
    build.add_argument("--buildings", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--layer")
    build.add_argument("--id-column", default="OBJECTID")
    build.add_argument("--source-id-column", default="BLD_ID")
    build.add_argument("--crs", default="EPSG:26911")
    build.add_argument("--tile-size", type=float, default=2_000.0)
    build.add_argument("--max-distance", type=float, default=152.4)
    build.add_argument("--max-angle", type=float, default=80.0)
    build.add_argument(
        "--visibility-backend", choices=("strtree", "bvh"), default="strtree",
        help="footprint visibility engine (default: strtree)",
    )
    build.add_argument("--ray-offset", type=float, default=0.01)
    build.add_argument("--hit-tolerance", type=float, default=0.005)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--bounds", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    build.add_argument("--max-tiles", type=int)
    build.add_argument("--no-resume", action="store_true")
    _add_selection_arguments(build)
    build3d = sub.add_parser("build3d", help="build a tiled height-aware 3D building graph")
    build3d.add_argument("--buildings", required=True)
    build3d.add_argument("--output", required=True)
    build3d.add_argument("--layer")
    build3d.add_argument("--id-column", default="OBJECTID")
    build3d.add_argument("--source-id-column", default="BLD_ID")
    build3d.add_argument("--crs", default="EPSG:26911")
    build3d.add_argument("--tile-size", type=float, default=2_000.0)
    build3d.add_argument("--max-distance", type=float, default=152.4)
    build3d.add_argument("--max-angle", type=float, default=60.0)
    build3d.add_argument("--height-column", default="HEIGHT")
    build3d.add_argument("--height-units", choices=("m", "ft"), default="ft")
    build3d.add_argument("--elevation-column", default="ELEV")
    build3d.add_argument("--elevation-units", choices=("m", "ft"), default="ft")
    build3d.add_argument("--default-height", type=float, default=3.0)
    build3d.add_argument("--patch-width", type=float, default=20.0)
    build3d.add_argument("--patch-height", type=float, default=3.0)
    build3d.add_argument("--max-vertical-patches", type=int, default=3)
    build3d.add_argument("--dem")
    build3d.add_argument("--terrain-resolution", type=float, default=30.0)
    build3d.add_argument("--no-dem-elevation", action="store_true")
    _add_selection_arguments(build3d)
    build3d.add_argument("--workers", type=int, default=1)
    build3d.add_argument("--bounds", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    build3d.add_argument("--max-tiles", type=int)
    build3d.add_argument("--no-resume", action="store_true")
    fetch_wui = sub.add_parser("fetch-wui", help="cache the complete ArcGIS WUI layer")
    fetch_wui.add_argument("--output", required=True)
    prepare = sub.add_parser(
        "prepare-socal",
        help="clip bulk Geofabrik OSM buildings to the Southern California WUI study area",
    )
    prepare.add_argument("--osm-gpkg", required=True)
    prepare.add_argument("--wui", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--osm-layer")
    prepare.add_argument("--crs", default="EPSG:3310")
    prepare.add_argument("--wui-buffer", type=float, default=1_609.344)
    prepare.add_argument("--blocker-halo", type=float, default=152.45)
    prepare.add_argument("--wui-simplify", type=float, default=25.0)
    prepare.add_argument("--wui-class", nargs="+", default=["Interface"])
    prepare.add_argument("--counties", nargs="+", default=list(DEFAULT_SOCAL_COUNTIES))
    return parser


def _load_selection(args):
    if not args.selection:
        return None
    import geopandas as gpd

    selection_gdf = gpd.read_file(args.selection, layer=args.selection_layer)
    if bool(args.selection_field) != bool(args.selection_value):
        raise ValueError("--selection-field and --selection-value must be supplied together")
    if args.selection_field:
        if args.selection_field not in selection_gdf:
            raise KeyError(f"Selection field not found: {args.selection_field!r}")
        selection_gdf = selection_gdf[
            selection_gdf[args.selection_field].astype(str).str.casefold()
            == args.selection_value.casefold()
        ]
        if selection_gdf.empty:
            raise ValueError("The selection attribute filter retained no features.")
    selection_gdf = selection_gdf.to_crs(args.crs)
    geometry = selection_gdf.geometry
    if args.selection_simplify:
        geometry = geometry.simplify(args.selection_simplify, preserve_topology=True)
    if args.selection_buffer:
        geometry = geometry.buffer(args.selection_buffer, resolution=4)
    return geometry.union_all()


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch-wui":
        print(fetch_wui_service(args.output))
        return 0
    if args.command == "prepare-socal":
        cfg = SoCal2DSourceConfig(
            target_crs=args.crs, wui_buffer_m=args.wui_buffer,
            blocker_halo_m=args.blocker_halo, wui_simplify_m=args.wui_simplify,
            counties=tuple(args.counties), osm_layer=args.osm_layer,
            wui_classes=tuple(args.wui_class) if args.wui_class else None,
        )
        result = prepare_socal_2d_source(
            args.osm_gpkg, args.wui, args.output, config=cfg,
        )
        print(result.manifest_path)
        return 0
    selection = _load_selection(args)
    if args.command == "build3d":
        unit_scale = {"m": 1.0, "ft": 0.3048}
        cfg = BVF3DGraphConfig(
            id_column=args.id_column, source_id_column=args.source_id_column,
            layer=args.layer, target_crs=args.crs, tile_size_m=args.tile_size,
            max_distance_m=args.max_distance, max_angle_deg=args.max_angle,
            height_column=args.height_column,
            height_scale_to_m=unit_scale[args.height_units],
            elevation_column=args.elevation_column or None,
            elevation_scale_to_m=unit_scale[args.elevation_units],
            default_height_m=args.default_height,
            max_patch_width_m=args.patch_width,
            max_patch_height_m=args.patch_height,
            max_vertical_patches=args.max_vertical_patches,
            dem_path=args.dem, terrain_resolution_m=args.terrain_resolution,
            use_dem_for_building_elevation=not args.no_dem_elevation,
            workers=args.workers, resume=not args.no_resume,
        )
        result = build_bvf_3d_graph(
            args.buildings, args.output, config=cfg,
            bounds=tuple(args.bounds) if args.bounds else None,
            max_tiles=args.max_tiles, selection_mask=selection,
        )
        print(result.manifest_path)
        return 0
    cfg = BVFGraphConfig(
        id_column=args.id_column,
        source_id_column=args.source_id_column,
        layer=args.layer,
        target_crs=args.crs,
        tile_size_m=args.tile_size,
        max_distance_m=args.max_distance,
        max_angle_deg=args.max_angle,
        visibility_backend=args.visibility_backend,
        ray_offset_m=args.ray_offset,
        hit_tolerance_m=args.hit_tolerance,
        workers=args.workers,
        resume=not args.no_resume,
    )
    result = build_bvf_graph(
        args.buildings, args.output, config=cfg,
        bounds=tuple(args.bounds) if args.bounds else None,
        max_tiles=args.max_tiles,
        selection_mask=selection,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
