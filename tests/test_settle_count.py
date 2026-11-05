"""Tests for settle-count.

Run from the repository root::

    python -m unittest discover -s tests -v

The tests need no extra dependency: synthetic frames come from
:mod:`settle_count.demo`, and every check is either a smoke test or an
accuracy check against the planted ground truth of synthetic data.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from settle_count import (  # noqa: E402  (import after sys.path setup)
    analyze_coverage,
    count_settled,
    coverage_map,
    coverage_timeseries,
    make_demo_frame,
    make_demo_series,
    write_coverage_csv,
    write_demo_images,
)
from settle_count.coverage import as_float_rgb  # noqa: E402


def _disc_image(centres, radius, *, size, seed=0):
    """Render dark discs on a pale plate, with sensor noise."""
    rng = np.random.default_rng(seed)
    image = np.empty((size, size, 3), dtype=np.float64)
    image[:] = (0.855, 0.810, 0.720)
    rows, columns = np.mgrid[0:size, 0:size]
    for x, y in centres:
        disc = (rows - y) ** 2 + (columns - x) ** 2 <= radius * radius
        image[disc] = (0.330, 0.300, 0.205)
    image += rng.normal(0.0, 0.010, image.shape)
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


class CoverageTests(unittest.TestCase):
    def test_heatmap_is_normalised(self):
        frame = make_demo_frame(28, size=256)
        fraction, heat = coverage_map(frame.image)
        self.assertEqual(heat.shape, frame.image.shape[:2])
        self.assertGreaterEqual(float(heat.min()), 0.0)
        self.assertLessEqual(float(heat.max()), 1.0)
        self.assertGreater(fraction, 0.0)
        self.assertLessEqual(fraction, 1.0)

    def test_clean_plate_has_no_coverage(self):
        frame = make_demo_frame(0, size=256)
        fraction, _ = coverage_map(frame.image)
        self.assertLess(fraction, 0.01)
        self.assertEqual(count_settled(frame.image).count, 0)

    def test_as_float_rgb_accepts_grayscale_and_uint8(self):
        grayscale = np.full((8, 8), 128, dtype=np.uint8)
        rgb = as_float_rgb(grayscale)
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertAlmostEqual(float(rgb.mean()), 128.0 / 255.0, places=6)
        with self.assertRaises(ValueError):
            as_float_rgb(np.zeros((8, 8, 5)))


class AccuracyTests(unittest.TestCase):
    """Accuracy against planted ground truth of synthetic frames."""

    @classmethod
    def setUpClass(cls):
        cls.frame = make_demo_frame(35, size=384)
        cls.result = analyze_coverage(cls.frame.image)
        cls.counted = count_settled(cls.frame.image)

    def test_coverage_matches_planted_fraction(self):
        self.assertLess(abs(self.result.fraction - self.frame.coverage), 0.08)

    def test_count_matches_planted_count(self):
        self.assertLess(abs(self.counted.count - self.frame.count), 4)

    def test_labels_and_areas_are_consistent(self):
        self.assertEqual(int(self.counted.labels.max()), self.counted.count)
        self.assertEqual(int(self.counted.labels.min()), 0)
        np.testing.assert_array_equal(
            np.bincount(self.counted.labels.ravel())[1:], self.counted.areas
        )


class CountingTests(unittest.TestCase):
    def test_splits_three_touching_discs(self):
        image = _disc_image([(60, 110), (95, 110), (130, 110)], 22, size=190)
        result = count_settled(image)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.areas.size, 3)

    def test_blank_image_counts_zero(self):
        image = _disc_image([], 0, size=64)
        result = count_settled(image)
        self.assertEqual(result.count, 0)
        self.assertFalse(result.labels.any())


class TimeSeriesTests(unittest.TestCase):
    def test_days_inferred_from_filenames(self):
        frames = make_demo_series(days=[7, 21], size=192)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_demo_images(Path(directory), frames)
            series = coverage_timeseries(paths, count=False)
        self.assertEqual(series["day"], [7.0, 21.0])
        self.assertEqual(len(series["path"]), 2)
        for value in series["coverage"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_explicit_mapping_and_csv_roundtrip(self):
        frames = make_demo_series(days=[3, 6], size=192)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_demo_images(Path(directory), frames)
            series = coverage_timeseries({3: paths[0], 6: paths[1]}, count=True)
            destination = write_coverage_csv(series, Path(directory) / "curve.csv")
            text = destination.read_text(encoding="utf-8").splitlines()
        self.assertEqual(series["day"], [3.0, 6.0])
        self.assertEqual(len(series["count"]), 2)
        self.assertEqual(text[0], "day,coverage,path,count")
        self.assertEqual(len(text), 3)


if __name__ == "__main__":
    unittest.main()
