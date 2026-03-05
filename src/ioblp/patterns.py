"""Synthetic-array pattern detection utilities for basal channel analysis.

These functions operate on 2D scalar fields that represent idealized ice-draft
surfaces. The core signal is local curvature: elongated troughs or ridges
produce anisotropic Hessian signatures that can be converted into a simple
channelness metric.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize


def _validate_2d_array(array: np.ndarray, name: str) -> np.ndarray:
    """Return an array view after enforcing a 2D input contract."""
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return values


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to suppress short-wavelength draft noise."""
    values = _validate_2d_array(field, "field")
    return gaussian_filter(values, sigma=float(sigma))


def compute_hessian(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute second-derivative curvature terms for a 2D draft field."""
    values = _validate_2d_array(field, "field")

    grad_y, grad_x = np.gradient(values)
    dxx = np.gradient(grad_x, axis=1)
    dyy = np.gradient(grad_y, axis=0)
    dxy_x = np.gradient(grad_x, axis=0)
    dxy_y = np.gradient(grad_y, axis=1)
    dxy = 0.5 * (dxy_x + dxy_y)
    return dxx, dyy, dxy


def hessian_eigenvalues(
    dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hessian eigenvalues sorted by absolute magnitude.

    The eigenvalues describe principal curvatures. Large negative values
    indicate locally concave-down features consistent with narrow troughs.
    """
    hxx = _validate_2d_array(dxx, "dxx")
    hyy = _validate_2d_array(dyy, "dyy")
    hxy = _validate_2d_array(dxy, "dxy")

    if hxx.shape != hyy.shape or hxx.shape != hxy.shape:
        raise ValueError("Hessian components must share the same shape")

    trace = hxx + hyy
    determinant_term = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
    eig_a = 0.5 * (trace - determinant_term)
    eig_b = 0.5 * (trace + determinant_term)

    swap = np.abs(eig_a) > np.abs(eig_b)
    lambda1 = np.where(swap, eig_b, eig_a)
    lambda2 = np.where(swap, eig_a, eig_b)
    return lambda1, lambda2


def channelness_metric(lambda1: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
    """Convert Hessian curvature into a simple trough-strength metric.

    The metric is the magnitude of the strongest negative principal curvature.
    Regions with weak or positive curvature map to zero.
    """
    eig1 = _validate_2d_array(lambda1, "lambda1")
    eig2 = _validate_2d_array(lambda2, "lambda2")

    if eig1.shape != eig2.shape:
        raise ValueError("Eigenvalue fields must share the same shape")

    metric = np.maximum(0.0, np.maximum(-eig1, -eig2))
    return np.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0)


def extract_channels(metric: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold the channelness field into a binary candidate mask."""
    values = _validate_2d_array(metric, "metric")
    mask = values >= float(threshold)
    if mask.shape != values.shape:
        raise ValueError("Thresholding changed array shape")
    return mask


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary channel mask to one-pixel-wide centerlines."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return skeletonize(values).astype(bool)


def compute_orientation(field: np.ndarray) -> np.ndarray:
    """Estimate local channel axis orientation in degrees within [0, 180).

    The gradient gives the local normal direction; rotating by 90 degrees
    approximates the along-channel orientation.
    """
    values = _validate_2d_array(field, "field")
    grad_y, grad_x = np.gradient(values)
    angles = (np.degrees(np.arctan2(grad_y, grad_x)) + 90.0) % 180.0
    return np.nan_to_num(angles, nan=0.0) % 180.0
