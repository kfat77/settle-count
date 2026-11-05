"""Quantitative image analysis for marine biofouling settlement plates.

Three measurements, one small package:

* :func:`~settle_count.coverage.coverage_map` - covered fraction and a
  per-pixel heatmap of fouling signal.
* :func:`~settle_count.counting.count_settled` - individual count, with
  touching individuals separated by a seeded watershed.
* :func:`~settle_count.timeseries.coverage_timeseries` - coverage and count
  against time, from a set of photographs of the same plate.

Everything runs on plain RGB photographs; no trained model, no camera
calibration, no lab-specific data.
"""

from __future__ import annotations

from .counting import CountResult, count_settled
from .coverage import (
    CoverageResult,
    analyze_coverage,
    coverage_map,
    estimate_background,
    fouling_mask,
    fouling_score,
)
from .demo import DEMO_DAYS, DemoFrame, make_demo_frame, make_demo_series, write_demo_images
from .timeseries import coverage_timeseries, write_coverage_csv

__version__ = "0.1.0"

__all__ = [
    "CoverageResult",
    "CountResult",
    "DEMO_DAYS",
    "DemoFrame",
    "__version__",
    "analyze_coverage",
    "count_settled",
    "coverage_map",
    "coverage_timeseries",
    "estimate_background",
    "fouling_mask",
    "fouling_score",
    "make_demo_frame",
    "make_demo_series",
    "write_coverage_csv",
    "write_demo_images",
]
