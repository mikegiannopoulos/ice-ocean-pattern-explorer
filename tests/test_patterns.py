import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioblp.patterns import (
    channelness_metric,
    compute_hessian,
    compute_orientation,
    extract_channels,
    hessian_eigenvalues,
    skeletonize_mask,
    smooth_field,
)


def _synthetic_ridge_field(size: int = 64) -> np.ndarray:
    x = np.linspace(0.0, 4.0 * np.pi, size)
    y = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    return np.sin(xx) + 0.1 * np.cos(2.0 * np.pi * yy)


def test_synthetic_sinusoidal_ridge_metric_identifies_structure() -> None:
    field = _synthetic_ridge_field()
    smoothed = smooth_field(field, sigma=1.0)
    dxx, dyy, dxy = compute_hessian(smoothed)
    lambda1, lambda2 = hessian_eigenvalues(dxx, dyy, dxy)
    metric = channelness_metric(lambda1, lambda2)

    assert metric.shape == field.shape
    assert metric.max() > 0.0
    assert metric.max() > metric.mean()

    mask = extract_channels(metric, threshold=0.5 * float(metric.max()))
    assert 0 < np.count_nonzero(mask) < mask.size


def test_gaussian_smoothing_reduces_variance() -> None:
    field = _synthetic_ridge_field()
    smoothed = smooth_field(field, sigma=1.5)

    assert np.var(smoothed) < np.var(field)


def test_hessian_eigenvalues_are_finite() -> None:
    field = smooth_field(_synthetic_ridge_field(), sigma=1.0)
    dxx, dyy, dxy = compute_hessian(field)
    lambda1, lambda2 = hessian_eigenvalues(dxx, dyy, dxy)

    assert np.isfinite(lambda1).all()
    assert np.isfinite(lambda2).all()


def test_skeletonization_returns_thinner_structure_than_mask() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:22, 8:24] = True

    skeleton = skeletonize_mask(mask)

    assert skeleton.dtype == np.bool_
    assert np.count_nonzero(skeleton) < np.count_nonzero(mask)


def test_orientation_values_stay_within_half_circle() -> None:
    angles = compute_orientation(_synthetic_ridge_field())

    assert angles.shape == (64, 64)
    assert np.all(angles >= 0.0)
    assert np.all(angles < 180.0)


def test_pipeline_is_deterministic() -> None:
    field = _synthetic_ridge_field()

    smoothed_a = smooth_field(field, sigma=1.0)
    smoothed_b = smooth_field(field, sigma=1.0)
    dxx_a, dyy_a, dxy_a = compute_hessian(smoothed_a)
    dxx_b, dyy_b, dxy_b = compute_hessian(smoothed_b)
    lambda1_a, lambda2_a = hessian_eigenvalues(dxx_a, dyy_a, dxy_a)
    lambda1_b, lambda2_b = hessian_eigenvalues(dxx_b, dyy_b, dxy_b)
    metric_a = channelness_metric(lambda1_a, lambda2_a)
    metric_b = channelness_metric(lambda1_b, lambda2_b)
    orient_a = compute_orientation(field)
    orient_b = compute_orientation(field)

    assert np.array_equal(metric_a, metric_b)
    assert np.array_equal(orient_a, orient_b)
