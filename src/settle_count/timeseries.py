"""Coverage and count against time, from a series of plate photographs.

The module is deliberately pandas-free: the time series is a plain dict of
equal-length column lists, which is what ``DataFrame`` conversion or a CSV
writer needs anyway.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from skimage.io import imread

from .counting import count_settled
from .coverage import coverage_map

__all__ = ["coverage_timeseries", "infer_day", "write_coverage_csv"]

_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


def infer_day(path: str | Path, *, fallback: float | None = None) -> float:
    """Extract a time point from a filename such as ``day_07.png``.

    The last number in the filename stem is used, so ``siteA_day_21.tif``
    yields ``21``. Files without any number fall back to ``fallback`` when it
    is given, and to ``nan`` otherwise.

    Args:
        path: Image path.
        fallback: Time point to use when the filename holds no number.

    Returns:
        The inferred time point as a float.
    """
    numbers = _NUMBER.findall(Path(path).stem)
    if numbers:
        return float(numbers[-1])
    return float(fallback) if fallback is not None else float("nan")


def coverage_timeseries(
    paths: Sequence[str | Path] | Mapping[float, str | Path],
    *,
    days: Sequence[float] | None = None,
    count: bool = True,
    **kwargs: object,
) -> dict[str, list]:
    """Measure coverage (and optionally count) on a series of photographs.

    Args:
        paths: Image paths, or a mapping of time point to image path. Plain
            sequences are ordered by time point; the time point comes from
            ``days`` when given, otherwise from the filename via
            :func:`infer_day`, otherwise from the sequence index.
        days: Explicit time points, one per entry of ``paths`` when ``paths``
            is a sequence.
        count: Also count individuals on every frame.
        **kwargs: Forwarded to :func:`settle_count.coverage.coverage_map` and
            :func:`settle_count.counting.count_settled`, e.g. ``roi``,
            ``background`` or ``min_delta_e``.

    Returns:
        Column-oriented dict with keys ``day``, ``coverage``, ``path`` and
        (when ``count`` is true) ``count``, sorted by time point.

    Raises:
        ValueError: If ``days`` does not match the number of paths.
    """
    if isinstance(paths, Mapping):
        pairs = sorted((float(day), Path(path)) for day, path in paths.items())
    else:
        files = [Path(path) for path in paths]
        if days is not None and len(days) != len(files):
            raise ValueError(f"got {len(days)} days for {len(files)} paths")
        if days is None:
            day_values = [infer_day(path, fallback=float(index)) for index, path in enumerate(files)]
        else:
            day_values = [float(day) for day in days]
        pairs = sorted(zip(day_values, files), key=lambda pair: pair[0])

    series: dict[str, list] = {"day": [], "coverage": [], "path": []}
    if count:
        series["count"] = []
    for day, path in pairs:
        image = imread(path)
        fraction, _ = coverage_map(image, **kwargs)
        series["day"].append(float(day))
        series["coverage"].append(float(fraction))
        series["path"].append(str(path))
        if count:
            series["count"].append(int(count_settled(image, **kwargs).count))
    return series


def write_coverage_csv(series: Mapping[str, Sequence[object]], path: str | Path) -> Path:
    """Write a time series produced by :func:`coverage_timeseries` to CSV.

    Args:
        series: Column-oriented dict, as returned by
            :func:`coverage_timeseries`.
        path: Destination CSV path.

    Returns:
        The destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(series)
    rows = zip(*(series[column] for column in columns))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return destination
