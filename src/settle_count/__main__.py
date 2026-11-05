"""Command line interface for settle-count.

Run from the repository root without installing::

    PYTHONPATH=src python -m settle_count coverage examples/synthetic/day_35.png

or, after ``pip install .``, simply ``python -m settle_count``.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from skimage.io import imread

from .counting import count_settled
from .coverage import analyze_coverage
from .timeseries import coverage_timeseries, write_coverage_csv


def _background(values: list[float] | None) -> np.ndarray | None:
    """Turn an optional RGB triple into a float colour in [0, 1]."""
    if values is None:
        return None
    rgb = np.asarray(values, dtype=np.float64)
    if rgb.size != 3:
        raise SystemExit("--background needs exactly three values")
    if float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the segmentation options shared by every subcommand."""
    parser.add_argument(
        "--background",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        help="plate colour; values above 1 are read as 0-255",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing of the colour-distance score, in pixels (default 2.0)",
    )
    parser.add_argument(
        "--min-delta-e",
        type=float,
        default=None,
        help="fixed segmentation threshold in delta-E units (default: adaptive)",
    )


def _expand(patterns: list[str]) -> list[Path]:
    """Expand shell-style patterns, since native arguments are not globbed on Windows."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise SystemExit(f"no files match {pattern!r}")
        paths.extend(Path(match) for match in matches)
    return paths


def _save_heatmap(heat: np.ndarray, path: Path) -> None:
    """Write a heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    image = axis.imshow(heat, cmap="inferno", vmin=0.0, vmax=1.0)
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, fraction=0.046, label="relative fouling signal")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_labels(labels: np.ndarray, path: Path) -> None:
    """Write a coloured label image PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    axis.imshow(labels, cmap="nipy_spectral", interpolation="nearest")
    axis.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _command_coverage(args: argparse.Namespace) -> int:
    """Print the covered fraction of one image."""
    image = imread(args.image)
    result = analyze_coverage(
        image,
        background=_background(args.background),
        smooth_sigma=args.smooth_sigma,
        min_delta_e=args.min_delta_e,
    )
    print(f"image          {args.image}")
    print(f"coverage       {result.fraction:.3f}  ({100.0 * result.fraction:.1f} %)")
    print(f"threshold      {result.threshold:.1f} delta-E")
    print(f"background RGB {tuple(round(float(channel) * 255.0) for channel in result.background_rgb)}")
    if args.heatmap:
        _save_heatmap(result.heat, Path(args.heatmap))
        print(f"heatmap        {args.heatmap}")
    return 0


def _command_count(args: argparse.Namespace) -> int:
    """Print the individual count of one image."""
    image = imread(args.image)
    result = count_settled(
        image,
        background=_background(args.background),
        smooth_sigma=args.smooth_sigma,
        min_delta_e=args.min_delta_e,
        min_area=args.min_area,
    )
    print(f"image          {args.image}")
    print(f"individuals    {result.count}")
    if result.count:
        areas = result.areas
        print(f"area median    {float(np.median(areas)):.0f} px^2")
        print(f"area range     {int(areas.min())} - {int(areas.max())} px^2")
        print(f"area total     {int(areas.sum())} px^2")
    if args.labels:
        _save_labels(result.labels, Path(args.labels))
        print(f"labels         {args.labels}")
    return 0


def _command_timeseries(args: argparse.Namespace) -> int:
    """Print coverage and count against time for a set of images."""
    files = _expand(args.images)
    series = coverage_timeseries(
        files,
        days=args.days,
        count=not args.no_count,
        background=_background(args.background),
        smooth_sigma=args.smooth_sigma,
        min_delta_e=args.min_delta_e,
    )
    print(f"{'day':>8}  {'coverage':>9}  {'count':>6}")
    for index, day in enumerate(series["day"]):
        count = series["count"][index] if "count" in series else ""
        print(f"{day:>8.1f}  {series['coverage'][index]:>9.3f}  {count:>6}")
    if args.csv:
        write_coverage_csv(series, Path(args.csv))
        print(f"csv            {args.csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m settle_count``."""
    parser = argparse.ArgumentParser(
        prog="settle-count",
        description=(
            "Quantify marine biofouling on settlement plates: coverage, heatmaps, "
            "clump-splitting counts and coverage-time curves."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser("coverage", help="covered fraction and heatmap")
    coverage.add_argument("image", help="photograph of a settlement plate")
    coverage.add_argument("--heatmap", help="write the heatmap to this PNG path")
    _add_common_arguments(coverage)
    coverage.set_defaults(handler=_command_coverage)

    count = subparsers.add_parser("count", help="individual count, clumps split by watershed")
    count.add_argument("image", help="photograph of a settlement plate")
    count.add_argument("--labels", help="write the label image to this PNG path")
    count.add_argument("--min-area", type=int, default=None, help="minimum object area in pixels")
    _add_common_arguments(count)
    count.set_defaults(handler=_command_count)

    timeseries = subparsers.add_parser("timeseries", help="coverage and count against time")
    timeseries.add_argument("images", nargs="+", help="image paths or shell-style patterns")
    timeseries.add_argument("--days", nargs="+", type=float, default=None, help="explicit time points")
    timeseries.add_argument("--csv", help="write the time series to this CSV path")
    timeseries.add_argument("--no-count", action="store_true", help="skip counting")
    _add_common_arguments(timeseries)
    timeseries.set_defaults(handler=_command_timeseries)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
