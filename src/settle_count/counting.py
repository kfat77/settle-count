"""Counting settled individuals, with touching individuals split by watershed.

The pipeline is the textbook one for clumped round objects:

1. segment fouled pixels with :func:`settle_count.coverage.analyze_coverage`;
2. compute the Euclidean distance transform of the binary mask and smooth it
   with a small Gaussian, which removes pixel-scale ridges without erasing the
   shallow saddles between two touching individuals;
3. find one seed per individual as an h-maximum of that distance transform;
4. run a seeded watershed on the negated distance transform, so every pixel of
   a clump is assigned to the individual whose seed basin it falls in.

h-maxima are used instead of peak detection because the only parameter is a
depth in pixels, derived from the image itself (a fraction of the typical
object radius), rather than an absolute intensity threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .coverage import analyze_coverage

__all__ = ["CountResult", "count_settled", "label_areas"]

_SEED_DEPTH_FRACTION = 0.08
_DISTANCE_SMOOTH_SIGMA = 1.0
_MIN_OBJECT_PIXELS = 4
_ARTIFACT_AREA_FRACTION = 0.15
_NEIGHBOUR_STRUCTURE = ndi.generate_binary_structure(2, 1)


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` pixels.

    Implemented with SciPy rather than :func:`skimage.morphology.remove_small_objects`
    so the tool does not depend on a keyword argument that scikit-image 0.26
    deprecated.
    """
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_size)
    keep[0] = False
    return keep[labels]


@dataclass(frozen=True)
class CountResult:
    """Output of one count.

    Attributes:
        labels: Integer label image, ``0`` for background, ``1..count`` for
            individuals.
        areas: Pixel area of each labelled individual, in label order.
    """

    labels: np.ndarray
    areas: np.ndarray

    @property
    def count(self) -> int:
        """Number of individuals kept after area filtering."""
        return int(self.areas.size)


def label_areas(labels: np.ndarray) -> np.ndarray:
    """Return the pixel area of every non-zero label.

    Args:
        labels: Integer label image.

    Returns:
        Array of pixel counts ordered by label value (label 1 first).
    """
    return np.bincount(np.asarray(labels, dtype=np.int64).ravel())[1:].astype(np.int64)


def count_settled(
    img: np.ndarray,
    *,
    background: np.ndarray | None = None,
    roi: np.ndarray | None = None,
    smooth_sigma: float = 2.0,
    min_delta_e: float | None = None,
    min_area: int | None = None,
    seed_depth: float | None = None,
) -> CountResult:
    """Count settled individuals in a photograph of a settlement plate.

    Args:
        img: Image array accepted by
            :func:`settle_count.coverage.as_float_rgb`.
        background: Optional explicit plate colour as float RGB in [0, 1].
        roi: Optional boolean region to analyse.
        smooth_sigma: Gaussian smoothing sigma applied to the colour-distance
            score before thresholding, in pixels.
        min_delta_e: Optional absolute segmentation threshold in delta-E
            units. Leave as ``None`` to derive it from the image.
        min_area: Minimum object area in pixels. ``None`` derives it as 15% of
            the median object area, which removes watershed slivers without
            deleting genuinely small settlers.
        seed_depth: h-maxima depth used to place seeds, in pixels. ``None``
            derives it as 8% of the 95th percentile of the smoothed distance
            transform, i.e. about 8% of the typical object radius.

    Returns:
        :class:`CountResult` holding the label image and per-individual areas.
    """
    coverage = analyze_coverage(
        img,
        background=background,
        roi=roi,
        smooth_sigma=smooth_sigma,
        min_delta_e=min_delta_e,
    )
    empty = CountResult(
        labels=np.zeros(coverage.mask.shape, dtype=np.int32),
        areas=np.zeros(0, dtype=np.int64),
    )
    mask = np.asarray(coverage.mask, dtype=bool)
    if not mask.any():
        return empty

    binary = ndi.binary_fill_holes(mask)
    binary = ndi.binary_closing(binary, structure=_NEIGHBOUR_STRUCTURE)
    binary = _remove_small_objects(binary, _MIN_OBJECT_PIXELS)
    if not binary.any():
        return empty

    distance = ndi.distance_transform_edt(binary)
    if _DISTANCE_SMOOTH_SIGMA > 0.0:
        distance = ndi.gaussian_filter(distance, _DISTANCE_SMOOTH_SIGMA)
    if seed_depth is None:
        depth = max(1.0, _SEED_DEPTH_FRACTION * float(np.percentile(distance[binary], 95.0)))
    else:
        depth = float(seed_depth)
    if depth <= 0.0:
        markers, marker_count = ndi.label(binary)
    else:
        markers, marker_count = ndi.label(h_maxima(distance, depth))
    if marker_count == 0:
        return empty

    segmented = watershed(-distance, markers, mask=binary).astype(np.int32)
    areas = label_areas(segmented)
    threshold_area = (
        float(min_area)
        if min_area is not None
        else max(float(_MIN_OBJECT_PIXELS), _ARTIFACT_AREA_FRACTION * float(np.median(areas)))
    )
    keep = np.flatnonzero(areas >= threshold_area) + 1
    if keep.size == 0:
        return empty

    remap = np.zeros(areas.size + 1, dtype=np.int32)
    remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return CountResult(labels=remap[segmented], areas=areas[keep - 1].astype(np.int64))
