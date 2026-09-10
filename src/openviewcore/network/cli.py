"""Command-line entry point for regional BVF graph construction."""

from __future__ import annotations

import argparse

from .building_vf import BVFGraphConfig, build_bvf_graph


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
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--bounds", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    build.add_argument("--max-tiles", type=int)
    build.add_argument("--no-resume", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    cfg = BVFGraphConfig(
        id_column=args.id_column,
        source_id_column=args.source_id_column,
        layer=args.layer,
        target_crs=args.crs,
        tile_size_m=args.tile_size,
        max_distance_m=args.max_distance,
        max_angle_deg=args.max_angle,
        workers=args.workers,
        resume=not args.no_resume,
    )
    result = build_bvf_graph(
        args.buildings, args.output, config=cfg,
        bounds=tuple(args.bounds) if args.bounds else None,
        max_tiles=args.max_tiles,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
