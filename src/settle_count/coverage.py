"""Coverage estimation for marine fouling settlement plates.

The estimator is deliberately simple and auditable: the plate surface is
modelled by a single background colour, every pixel is scored by its CIELAB
colour distance (delta-E 1976) from that background, and the score map is
thresholded to obtain the fouled area.

No trained model and no camera calibration are involved, so the same code runs
on any RGB photograph of a settlement plate, provided the plate surface fills
most of the cropped frame and fouling is chromatically distinct from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu

__all__ = [
    "CoverageResult",
    "analyze_coverage",
    "as_float_rgb",
    "coverage_map",
    "estimate_background",
    "fouling_mask",
    "fouling_score",
]

_MAD_TO_SIGMA = 1.4826
_NOISE_SIGMAS = 3.0
_MIN_FOULING_DELTA_E = 3.0
"""Absolute floor for the adaptive threshold, in delta-E units.

Camera noise and mild illumination gradients stay well below 3 delta-E for a
uniformly lit plate, while settled organisms and biofilm are tens of delta-E
away. The floor is what keeps a clean plate at zero coverage instead of
reporting the tail of the noise distribution.
"""

_EPS = 1e-9


@dataclass(frozen=True)
class CoverageResult:
    """Output of one coverage estimate.

    Attributes:
        fraction: Covered fraction of the analysed region, in [0, 1].
        heat: Per-pixel relative fouling signal, normalised to [0, 1] by the
            99th percentile of the analysed region.
        mask: Boolean fouled-pixel mask at the selected threshold.
        threshold: Threshold in CIELAB distance units (delta-E 1976).
        background_rgb: Background colour the score was computed against,
            as float RGB in [0, 1].
    """

    fraction: float
    heat: np.ndarray
    mask: np.ndarray
    threshold: float
    background_rgb: np.ndarray


def as_float_rgb(img: np.ndarray) -> np.ndarray:
    """Return an ``(H, W, 3)`` float RGB image in [0, 1].

    Args:
        img: Grayscale ``(H, W)``, RGB ``(H, W, 3)`` or RGBA ``(H, W, 4)``
            array. Integer dtypes are scaled by their dtype maximum; float
            input is clipped to [0, 1].

    Returns:
        Float64 RGB array of shape ``(H, W, 3)``.

    Raises:
        ValueError: If the array is not 2-D, or is 3-D with a channel count
            other than 3 or 4.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected an (H, W, 3) or (H, W, 4) image, got shape {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def _region_mask(shape: tuple[int, int], roi: np.ndarray | None) -> np.ndarray:
    """Return the boolean analysis region, defaulting to the whole image."""
    if roi is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(roi, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"roi shape {mask.shape} does not match image shape {shape}")
    if not mask.any():
        raise ValueError("roi selects no pixels")
    return mask


def estimate_background(
    img: np.ndarray,
    *,
    roi: np.ndarray | None = None,
    levels: int = 8,
) -> np.ndarray:
    """Estimate the dominant plate colour of an image.

    Colours are quantised into ``levels`` bins per channel and the most
    populated bin is taken as the plate background; the returned colour is the
    per-channel median of the pixels in that bin, which is robust to sensor
    noise and to a scattered minority of fouled pixels.

    Args:
        img: Image array accepted by :func:`as_float_rgb`.
        roi: Optional boolean region to estimate from.
        levels: Histogram bins per channel (default 8, i.e. 32 grey levels).

    Returns:
        Float RGB background colour, shape ``(3,)``.

    Raises:
        ValueError: If ``roi`` does not match the image shape or selects no
            pixels.
    """
    rgb = as_float_rgb(img)
    region = _region_mask(rgb.shape[:2], roi)
    pixels = rgb[region]
    quantised = np.clip((pixels * levels).astype(np.int64), 0, levels - 1)
    keys = (quantised[:, 0] * levels + quantised[:, 1]) * levels + quantised[:, 2]
    counts = np.bincount(keys, minlength=levels**3)
    dominant = int(np.argmax(counts))
    return np.median(pixels[keys == dominant], axis=0)


def fouling_score(img: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    """CIELAB distance of every pixel from ``background_rgb``.

    Colour distances are computed in the perceptually uniform CIELAB space, so
    a single threshold is meaningful for dark barnacles and pale tube worms
    alike, and is only weakly affected by brightness changes of the lamp.

    Args:
        img: Image array accepted by :func:`as_float_rgb`.
        background_rgb: Reference colour as float RGB in [0, 1].

    Returns:
        Float64 score map, same height and width as ``img``.

    Raises:
        ValueError: If ``background_rgb`` does not hold three values.
    """
    rgb = as_float_rgb(img)
    lab = rgb2lab(rgb)
    background = np.asarray(background_rgb, dtype=np.float64).ravel()
    if background.size != 3:
        raise ValueError(f"background_rgb must hold 3 values, got {background.size}")
    background = np.clip(background, 0.0, 1.0).reshape(1, 1, 3)
    background_lab = rgb2lab(background)[0, 0]
    return np.linalg.norm(lab - background_lab, axis=-1)


def analyze_coverage(
    img: np.ndarray,
    *,
    background: np.ndarray | None = None,
    roi: np.ndarray | None = None,
    smooth_sigma: float = 2.0,
    min_delta_e: float | None = None,
) -> CoverageResult:
    """Estimate coverage and return the intermediate maps.

    The score map is smoothed, then thresholded at the largest of: the Otsu
    threshold of the analysed region, a robust noise floor (median + 3 sigma of
    the MAD-based noise estimate) and a 3 delta-E absolute floor. The two floors
    keep an unfouled plate at zero coverage instead of splitting sensor noise in
    half; Otsu dominates as soon as genuine fouling is present.

    Args:
        img: Image array accepted by :func:`as_float_rgb`.
        background: Optional explicit plate colour as float RGB in [0, 1].
            Estimate it with :func:`estimate_background` when the plate covers
            less than about half of the frame.
        roi: Optional boolean region to analyse, e.g. the plate disc when the
            photograph includes the surrounding tray.
        smooth_sigma: Gaussian smoothing sigma of the score map, in pixels.
            Use 0 to threshold the raw pixel score.
        min_delta_e: Optional absolute threshold in delta-E units, overriding
            the adaptive threshold entirely.

    Returns:
        :class:`CoverageResult` with fraction, heatmap, mask and threshold.

    Raises:
        ValueError: If ``background`` or ``roi`` is malformed.
    """
    rgb = as_float_rgb(img)
    region = _region_mask(rgb.shape[:2], roi)
    if background is None:
        bg = estimate_background(rgb, roi=region)
    else:
        bg = np.asarray(background, dtype=np.float64).ravel()
        if bg.size != 3:
            raise ValueError(f"background must hold 3 values, got {bg.size}")
        bg = np.clip(bg, 0.0, 1.0)

    score = fouling_score(rgb, bg)
    smooth = ndi.gaussian_filter(score, smooth_sigma) if smooth_sigma > 0 else score
    values = smooth[region]
    if values.size == 0:
        raise ValueError("roi selects no pixels")

    if float(values.max()) <= _EPS:
        heat = np.zeros_like(smooth)
        mask = np.zeros_like(region)
        threshold = float("inf")
    else:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        noise_floor = median + _NOISE_SIGMAS * _MAD_TO_SIGMA * mad
        otsu = float(threshold_otsu(values))
        threshold = (
            float(min_delta_e)
            if min_delta_e is not None
            else max(otsu, noise_floor, _MIN_FOULING_DELTA_E)
        )
        mask = (smooth > threshold) & region
        unit = max(float(np.percentile(values, 99.0)), _EPS)
        heat = np.clip(smooth / unit, 0.0, 1.0)
    heat = np.where(region, heat, 0.0)
    fraction = float(mask.sum()) / float(region.sum())
    return CoverageResult(
        fraction=fraction,
        heat=heat,
        mask=mask,
        threshold=threshold,
        background_rgb=np.asarray(bg, dtype=np.float64),
    )


def coverage_map(img: np.ndarray, **kwargs: object) -> tuple[float, np.ndarray]:
    """Estimate the covered fraction and a relative fouling heatmap.

    This is the two-value convenience wrapper around
    :func:`analyze_coverage`; pass its keyword arguments through to select a
    background colour, a region of interest or a fixed threshold.

    Args:
        img: Image array accepted by :func:`as_float_rgb`.
        **kwargs: Forwarded to :func:`analyze_coverage`.

    Returns:
        ``(fraction, heat)``: covered fraction in [0, 1] and a float heatmap
        in [0, 1] of the same height and width as ``img``.
    """
    result = analyze_coverage(img, **kwargs)
    return result.fraction, result.heat


def fouling_mask(img: np.ndarray, **kwargs: object) -> np.ndarray:
    """Return the boolean fouled-pixel mask.

    Args:
        img: Image array accepted by :func:`as_float_rgb`.
        **kwargs: Forwarded to :func:`analyze_coverage`.

    Returns:
        Boolean mask of the same height and width as ``img``.
    """
    return analyze_coverage(img, **kwargs).mask
