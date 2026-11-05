"""Synthetic settlement-plate photographs, for demos and tests only.

**None of the images produced here are real experimental data.** They are
rendered from a small generative model: individuals arrive at random times
during a 49-day window, grow to a mature radius, and settle in clumps of one to
four members. Clumps are assigned one per cell of a small grid and their members
sit in a compact chain, so distinct clumps stay separate while members of a
clump overlap. The model is only realistic enough to exercise the analysis
pipeline and to give it a known answer to compare against.

Every frame therefore carries its planted coverage and planted count, so the
demo can report measurement error instead of asking the reader to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imsave

__all__ = [
    "DEMO_DAYS",
    "DEFAULT_SEED",
    "DemoFrame",
    "make_demo_frame",
    "make_demo_series",
    "write_demo_images",
]

DEMO_DAYS: tuple[int, ...] = (0, 7, 14, 21, 28, 35, 42, 49)
DEFAULT_SEED = 20221105
"""Seed chosen only to make the demo deterministic."""

PLATE_RGB = np.array([0.845, 0.800, 0.715])
"""Synthetic plate colour: a pale, warm grey."""

FOUL_RGB = np.array(
    [
        [0.34, 0.30, 0.20],
        [0.43, 0.38, 0.25],
        [0.29, 0.33, 0.24],
        [0.46, 0.36, 0.22],
        [0.24, 0.26, 0.20],
    ]
)
"""Synthetic fouling palette: olive, brown and grey-green settlers."""

MATURE_RADIUS = 17.5
"""Mature radius of a synthetic individual, in pixels."""
MAX_INDIVIDUALS = 40
GROWTH_DAYS = 6.0
EDGE_SOFTNESS = 0.6
NOISE_SIGMA = 0.008
_VIGNETTE = 0.004
_TEXTURE_AMPLITUDE = 0.006
_CLUSTER_SIZES = (1, 2, 3, 4)
_CLUSTER_WEIGHTS = (0.12, 0.28, 0.30, 0.30)
_CLUSTER_SPACING = 1.45
"""Centre-to-centre distance inside a clump, as a fraction of the mature radius.

At 1.45 radii, neighbours overlap by about a quarter of their diameter: enough
that a connected-component count undercounts, but shallow enough that the
distance-transform saddle between two individuals survives.
"""
_GRID_SIDE = 4
"""Clumps are assigned one per cell of a :data:`_GRID_SIDE` square grid."""
_MARGIN_FRACTION = 0.04
_JITTER_FRACTION = 0.02


@dataclass(frozen=True)
class DemoFrame:
    """One synthetic frame with its ground truth.

    Attributes:
        day: Time point in days.
        image: RGB uint8 image.
        coverage: Planted covered fraction in [0, 1].
        count: Planted number of individuals.
        mask: Planted boolean union of every individual.
    """

    day: float
    image: np.ndarray
    coverage: float
    count: int
    mask: np.ndarray


@dataclass(frozen=True)
class _Individual:
    x: float
    y: float
    radius: float
    arrival: float


def _plan_individuals(size: int, seed: int) -> list[_Individual]:
    """Place individuals in clumps on a jittered grid of cells.

    One clump is assigned to an unused grid cell, so distinct clumps stay
    separated while the members of a clump form a compact chain. Chain members
    share a mature radius but arrive on different days.
    """
    rng = np.random.default_rng(seed)
    margin = _MARGIN_FRACTION * size
    cell = (size - 2.0 * margin) / _GRID_SIDE
    jitter = _JITTER_FRACTION * cell
    individuals: list[_Individual] = []
    for cell_index in rng.permutation(_GRID_SIDE * _GRID_SIDE):
        if len(individuals) >= MAX_INDIVIDUALS:
            break
        row, column = divmod(int(cell_index), _GRID_SIDE)
        cluster_x = margin + (column + 0.5) * cell + rng.uniform(-jitter, jitter)
        cluster_y = margin + (row + 0.5) * cell + rng.uniform(-jitter, jitter)
        base = float(np.clip(rng.normal(MATURE_RADIUS, 1.5), 14.0, 19.0))
        wanted = int(rng.choice(_CLUSTER_SIZES, p=_CLUSTER_WEIGHTS))
        cluster_size = min(wanted, MAX_INDIVIDUALS - len(individuals))
        direction = rng.uniform(0.0, 2.0 * np.pi)
        step = _CLUSTER_SPACING * base
        for member in range(cluster_size):
            along = (member - 0.5 * (cluster_size - 1)) * step
            radius = float(
                np.clip(rng.normal(base, 0.04 * base), 0.92 * base, 1.08 * base)
            )
            individuals.append(
                _Individual(
                    x=float(cluster_x + along * np.cos(direction)),
                    y=float(cluster_y + along * np.sin(direction)),
                    radius=radius,
                    arrival=float(rng.uniform(1.0, DEMO_DAYS[-1] - 9.0)),
                )
            )
    return individuals


def _radius(individual: _Individual, day: float) -> float:
    """Radius of one individual at ``day``; 0 before it has settled."""
    age = day - individual.arrival
    if age < 0.0:
        return 0.0
    growth = 0.5 + 0.5 * min(age / GROWTH_DAYS, 1.0)
    return individual.radius * growth


def _render(
    individuals: list[_Individual],
    day: float,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one frame; returns the uint8 image and the planted union mask."""
    rows, columns = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    label = np.zeros((size, size), dtype=np.int32)
    for index, individual in enumerate(individuals, start=1):
        radius = _radius(individual, day)
        if radius <= 0.0:
            continue
        disc = (rows - individual.y) ** 2 + (columns - individual.x) ** 2 <= radius * radius
        if not disc.any():
            continue
        mask |= disc
        label[disc] = index

    palette = FOUL_RGB[rng.integers(0, len(FOUL_RGB), size=len(individuals) + 1)]
    palette = palette * rng.normal(1.0, 0.05, size=palette.shape)
    fouling = palette[label]
    alpha = ndi.gaussian_filter(mask.astype(np.float64), EDGE_SOFTNESS)

    image = PLATE_RGB[None, None, :] * (1.0 - alpha[..., None]) + fouling * alpha[..., None]

    grid = np.linspace(-1.0, 1.0, size)
    radial = grid[None, :] ** 2 + grid[:, None] ** 2
    image *= (1.0 - _VIGNETTE * radial)[..., None]

    texture = ndi.gaussian_filter(rng.normal(0.0, 1.0, (size, size)), 24)
    texture *= _TEXTURE_AMPLITUDE / max(float(np.abs(texture).max()), 1e-6)
    image += texture[..., None]
    image += rng.normal(0.0, NOISE_SIGMA, image.shape)
    np.clip(image, 0.0, 1.0, out=image)
    return np.round(image * 255.0).astype(np.uint8), mask


def make_demo_frame(day: float, *, size: int = 512, seed: int = DEFAULT_SEED) -> DemoFrame:
    """Render one synthetic frame with its planted coverage and count.

    Args:
        day: Time point in days.
        size: Square image side in pixels.
        seed: Random seed; the same seed always yields the same settlement
            layout, so frames of one series are nested (individuals persist).

    Returns:
        A :class:`DemoFrame`.
    """
    individuals = _plan_individuals(size, seed)
    rng = np.random.default_rng(seed + 10007 * int(round(day)) + 1)
    image, mask = _render(individuals, float(day), size, rng)
    return DemoFrame(
        day=float(day),
        image=image,
        coverage=float(mask.mean()),
        count=sum(1 for individual in individuals if _radius(individual, day) > 0.0),
        mask=mask,
    )


def make_demo_series(
    days: tuple[int, ...] | list[int] = DEMO_DAYS,
    *,
    size: int = 512,
    seed: int = DEFAULT_SEED,
) -> list[DemoFrame]:
    """Render a synthetic time series of settlement plates.

    Args:
        days: Time points in days.
        size: Square image side in pixels.
        seed: Random seed, shared by every frame of the series.

    Returns:
        Frames ordered by day.
    """
    return [make_demo_frame(day, size=size, seed=seed) for day in sorted(days)]


def write_demo_images(directory: str | Path, frames: list[DemoFrame]) -> list[Path]:
    """Write demo frames as ``day_NN.png`` files.

    Args:
        directory: Destination directory, created if missing.
        frames: Frames from :func:`make_demo_series`.

    Returns:
        Written paths, in frame order.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        path = destination / f"day_{int(frame.day):02d}.png"
        imsave(path, frame.image, check_contrast=False)
        paths.append(path)
    return paths
