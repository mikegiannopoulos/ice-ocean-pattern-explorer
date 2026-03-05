"""Reusable BedMachine detection pipeline utilities for CLI and app frontends."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from skimage.measure import label

from ioblp.data import get_project_root, load_bedmachine_default
from ioblp.patterns import (
    channelness_metric,
    compute_hessian,
    compute_orientation,
    extract_channels,
    hessian_eigenvalues,
    skeletonize_mask,
    smooth_field,
)
from ioblp.preprocess import compute_ice_draft, subset_xy

ROI_PRESETS = {
    "ross": {"xmin": -500000.0, "xmax": 500000.0, "ymin": -1200000.0, "ymax": -400000.0},
    "filchner_ronne": {
        "xmin": -1600000.0,
        "xmax": -300000.0,
        "ymin": 300000.0,
        "ymax": 1600000.0,
    },
    "pine_island": {
        "xmin": -1900000.0,
        "xmax": -1200000.0,
        "ymin": -900000.0,
        "ymax": 200000.0,
    },
    "thwaites": {
        "xmin": -1850000.0,
        "xmax": -1000000.0,
        "ymin": -1200000.0,
        "ymax": 0.0,
    },
}
AUTOFIT_ROIS = {"pine_island", "thwaites"}
TUNING_PRESETS = {
    "conservative": {"threshold_quantile": 0.99, "sigma_small": 4.0, "min_length": 30},
    "balanced": {"threshold_quantile": 0.975, "sigma_small": 2.0, "min_length": 30},
    "sensitive": {"threshold_quantile": 0.95, "sigma_small": 2.0, "min_length": 10},
}
RUN_INDEX_COLUMNS = [
    "timestamp",
    "roi_name",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "tuning",
    "sigma_small",
    "sigma_large",
    "threshold_quantile",
    "edge_buffer",
    "min_length",
    "floating_fraction",
    "threshold",
    "detected_pixels",
    "skeleton_pixels",
    "num_components",
    "median_length_pixels",
    "mean_length_pixels",
    "outdir",
]


def format_bound(value: float) -> str:
    """Format coordinate bounds for human-readable output and folder names."""
    return f"{value:g}"


def format_bounds(bounds: dict[str, float]) -> str:
    """Return a compact bounds string."""
    return (
        f"xmin={format_bound(bounds['xmin'])}, "
        f"xmax={format_bound(bounds['xmax'])}, "
        f"ymin={format_bound(bounds['ymin'])}, "
        f"ymax={format_bound(bounds['ymax'])}"
    )


def resolve_tuning(
    tuning: str,
    *,
    sigma_small: float | None,
    threshold_quantile: float | None,
    min_length: int | None,
) -> dict[str, float | int | str]:
    """Resolve effective detection parameters from tuning plus explicit overrides."""
    if tuning not in TUNING_PRESETS:
        names = ", ".join(sorted(TUNING_PRESETS))
        raise ValueError(f"Unknown tuning preset '{tuning}'. Available presets: {names}")

    preset = TUNING_PRESETS[tuning]
    return {
        "tuning": tuning,
        "sigma_small": float(sigma_small if sigma_small is not None else preset["sigma_small"]),
        "threshold_quantile": float(
            threshold_quantile
            if threshold_quantile is not None
            else preset["threshold_quantile"]
        ),
        "min_length": int(min_length if min_length is not None else preset["min_length"]),
    }


def autofit_to_floating(ds_region: xr.Dataset, pad_pixels: int = 50) -> dict[str, float]:
    """Fit a candidate ROI to the largest contiguous floating component."""
    if "mask" not in ds_region.data_vars:
        available = ", ".join(sorted(ds_region.data_vars))
        raise ValueError(f"Dataset does not contain 'mask'. Available variables: {available}")

    mask_da = ds_region["mask"]
    if "y" not in mask_da.dims or "x" not in mask_da.dims:
        raise ValueError("Expected 'mask' to have 'y' and 'x' dimensions")

    floating = np.asarray((mask_da == 3).transpose("y", "x"), dtype=bool)
    labels = label(floating, connectivity=2)
    if labels.max() == 0:
        raise ValueError("No floating-ice pixels found in candidate ROI")

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_id = int(np.argmax(counts))
    locations = np.argwhere(labels == largest_id)
    y_min_idx, x_min_idx = locations.min(axis=0)
    y_max_idx, x_max_idx = locations.max(axis=0)

    ny, nx = floating.shape
    pad = max(0, int(pad_pixels))
    y0 = max(0, int(y_min_idx) - pad)
    y1 = min(ny - 1, int(y_max_idx) + pad)
    x0 = max(0, int(x_min_idx) - pad)
    x1 = min(nx - 1, int(x_max_idx) + pad)

    x_vals = np.asarray(ds_region["x"], dtype=float)
    y_vals = np.asarray(ds_region["y"], dtype=float)
    return {
        "xmin": float(min(x_vals[x0], x_vals[x1])),
        "xmax": float(max(x_vals[x0], x_vals[x1])),
        "ymin": float(min(y_vals[y0], y_vals[y1])),
        "ymax": float(max(y_vals[y0], y_vals[y1])),
    }


def resolve_bounds(
    ds: xr.Dataset,
    *,
    roi: str | None,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Resolve final bounds from preset and explicit overrides.

    Returns:
      final bounds, and optional candidate bounds used before auto-fit.
    """
    if roi is None:
        missing = [
            name
            for name, value in (("xmin", xmin), ("xmax", xmax), ("ymin", ymin), ("ymax", ymax))
            if value is None
        ]
        if missing:
            missing_list = ", ".join(f"--{name}" for name in missing)
            raise ValueError(
                "Bounds are required when --roi is not provided. "
                f"Missing: {missing_list}"
            )
        return (
            {
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
            },
            None,
        )

    if roi not in ROI_PRESETS:
        names = ", ".join(sorted(ROI_PRESETS))
        raise ValueError(f"Unknown ROI preset '{roi}'. Available presets: {names}")

    candidate_bounds = ROI_PRESETS[roi].copy()
    overrides = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    for name, value in overrides.items():
        if value is not None:
            candidate_bounds[name] = float(value)

    manual_override = any(value is not None for value in overrides.values())
    if roi in AUTOFIT_ROIS and not manual_override:
        candidate_region = subset_xy(
            ds,
            candidate_bounds["xmin"],
            candidate_bounds["xmax"],
            candidate_bounds["ymin"],
            candidate_bounds["ymax"],
        )
        fitted_bounds = autofit_to_floating(candidate_region)
        return fitted_bounds, candidate_bounds

    return candidate_bounds, None


def resolve_run_outdir(
    *,
    bounds: dict[str, float],
    outdir: str | Path | None = None,
    project_root: Path | None = None,
    expected_paths: list[Path] | None = None,
) -> tuple[Path, Path]:
    """Resolve run-specific output directory and create figures directory."""
    root = project_root if project_root is not None else get_project_root()
    if outdir:
        resolved = Path(outdir).expanduser()
        if not resolved.is_absolute():
            resolved = root / resolved
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"run_{stamp}_"
            f"xmin{format_bound(bounds['xmin'])}_"
            f"xmax{format_bound(bounds['xmax'])}_"
            f"ymin{format_bound(bounds['ymin'])}_"
            f"ymax{format_bound(bounds['ymax'])}"
        )
        resolved = root / "outputs" / run_name

    outdir_path = resolved.resolve()
    if outdir_path == root:
        raise ValueError("Refusing to write outputs directly to the project root.")

    figures_dir = outdir_path / "figures"
    expected = (
        expected_paths
        if expected_paths is not None
        else [
            outdir_path / "channels_mask.nc",
            outdir_path / "skeleton.nc",
            outdir_path / "diagnostics.json",
            outdir_path / "run_config.json",
            figures_dir / "fig01_region_draft.png",
            figures_dir / "fig02_channels_overlay.png",
            figures_dir / "fig03_orientation_hist.png",
            figures_dir / "fig04_channel_lengths.png",
        ]
    )
    if outdir_path.exists():
        existing = [path for path in expected if path.exists()]
        if existing:
            paths = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Output overwrite prevention: existing files detected in outdir: {paths}"
            )

    figures_dir.mkdir(parents=True, exist_ok=True)
    return outdir_path, figures_dir


def fill_nans_nearest(values: np.ndarray) -> np.ndarray:
    """Fill NaN locations with nearest valid neighbors."""
    nan_mask = np.isnan(values)
    if nan_mask.all():
        raise ValueError("Draft field contains only NaNs in the selected subset.")
    if not nan_mask.any():
        return values

    nearest_indices = distance_transform_edt(
        nan_mask,
        return_distances=False,
        return_indices=True,
    )
    return values[tuple(nearest_indices)]


def component_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components and return labels and component lengths."""
    labels = label(mask, connectivity=2)
    lengths = np.bincount(labels.ravel())[1:].astype(float)
    lengths = lengths[lengths > 0]
    return labels, lengths


def save_mask_netcdf(path: Path, name: str, mask: np.ndarray, template: xr.DataArray) -> None:
    """Write a 2D boolean mask to NetCDF using template coords."""
    coords = {dim: template.coords[dim] for dim in template.dims if dim in template.coords}
    array = xr.DataArray(mask.astype(bool), dims=template.dims, coords=coords, name=name)
    xr.Dataset({name: array}).to_netcdf(path)


def run_detection_core(
    ds_region: xr.Dataset,
    *,
    sigma_small: float,
    sigma_large: float,
    threshold_quantile: float,
    edge_buffer: int,
    min_length: int,
    return_masks: bool = False,
) -> dict[str, Any]:
    """Run the detection core without output-side effects."""
    if "mask" not in ds_region.data_vars:
        available = ", ".join(sorted(ds_region.data_vars))
        raise ValueError(f"Dataset does not contain 'mask'. Available variables: {available}")

    draft = compute_ice_draft(ds_region)
    floating_mask = np.asarray(ds_region["mask"] == 3, dtype=bool)
    floating_interior = binary_erosion(floating_mask, iterations=edge_buffer)
    if not floating_interior.any():
        raise ValueError(
            "Floating interior is empty. Reduce edge_buffer or change the region bounds."
        )

    draft_values = np.asarray(draft, dtype=float)
    draft_filled = fill_nans_nearest(draft_values)
    background = gaussian_filter(draft_filled, sigma=sigma_large)
    draft_detrended = draft_filled - background

    smoothed = smooth_field(draft_detrended, sigma=sigma_small)
    dxx, dyy, dxy = compute_hessian(smoothed)
    lambda1, lambda2 = hessian_eigenvalues(dxx, dyy, dxy)
    metric = channelness_metric(lambda1, lambda2)

    positive_metric = metric[(metric > 0) & floating_interior]
    threshold = (
        float(np.quantile(positive_metric, threshold_quantile))
        if positive_metric.size
        else 0.0
    )

    detected = extract_channels(metric, threshold=threshold) & floating_interior
    skeleton = skeletonize_mask(detected)

    labels_before, lengths_before = component_lengths(skeleton)
    keep_component_ids = np.where(lengths_before >= float(min_length))[0] + 1
    skeleton_filtered = np.isin(labels_before, keep_component_ids)

    detected_labels = label(detected, connectivity=2)
    keep_detected_ids = np.unique(detected_labels[skeleton_filtered])
    keep_detected_ids = keep_detected_ids[keep_detected_ids > 0]
    detected_filtered = np.isin(detected_labels, keep_detected_ids)

    _, lengths_pixels = component_lengths(skeleton_filtered)
    num_components = int(lengths_pixels.size)
    diagnostics: dict[str, Any] = {
        "floating_fraction": float(floating_mask.mean()),
        "threshold": threshold,
        "detected_pixels": int(detected_filtered.sum()),
        "skeleton_pixels": int(skeleton_filtered.sum()),
        "num_components": num_components,
        "median_length_pixels": float(np.median(lengths_pixels)) if num_components else 0.0,
        "mean_length_pixels": float(np.mean(lengths_pixels)) if num_components else 0.0,
    }

    if return_masks:
        orientation = compute_orientation(smoothed)
        diagnostics.update(
            {
                "channels_mask": detected_filtered,
                "skeleton": skeleton_filtered,
                "draft_values": draft_values,
                "floating_mask": floating_mask,
                "floating_interior": floating_interior,
                "skeleton_orientations": orientation[skeleton_filtered],
                "lengths_pixels": lengths_pixels,
                "draft_template": draft,
            }
        )

    return diagnostics


def save_detection_figures(
    *,
    figures_dir: Path,
    draft_values: np.ndarray,
    floating_mask: np.ndarray,
    floating_interior: np.ndarray,
    skeleton_mask: np.ndarray,
    skeleton_orientations: np.ndarray,
    lengths_pixels: np.ndarray,
) -> dict[str, Path]:
    """Save standard detection figures and return their paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig01_path = figures_dir / "fig01_region_draft.png"
    fig02_path = figures_dir / "fig02_channels_overlay.png"
    fig03_path = figures_dir / "fig03_orientation_hist.png"
    fig04_path = figures_dir / "fig04_channel_lengths.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(draft_values, origin="lower", cmap="viridis")
    ax.contour(floating_mask.astype(int), levels=[0.5], colors="white", linewidths=0.8)
    ax.contour(floating_interior.astype(int), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title("Fig01: Region Draft with Floating Ice Mask")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, label="Ice draft")
    fig.tight_layout()
    fig.savefig(fig01_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(draft_values, origin="lower", cmap="gray")
    overlay = np.ma.masked_where(~skeleton_mask, skeleton_mask)
    ax.imshow(overlay, origin="lower", cmap="autumn", alpha=0.9)
    ax.set_title("Fig02: Detected Channels on Draft")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(fig02_path, dpi=800)
    plt.close(fig)

    plt.figure()
    if skeleton_orientations.size:
        plt.hist(skeleton_orientations, bins=36)
    plt.xlabel("Orientation (degrees)")
    plt.ylabel("Frequency")
    plt.title("Fig03: Skeleton Orientation Distribution")
    plt.savefig(fig03_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    if lengths_pixels.size:
        plt.hist(lengths_pixels, bins=40)
    plt.xlabel("Channel length (pixels)")
    plt.ylabel("Count")
    plt.title("Fig04: Channel Length Distribution")
    plt.savefig(fig04_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "fig01": fig01_path,
        "fig02": fig02_path,
        "fig03": fig03_path,
        "fig04": fig04_path,
    }


def append_runs_index(project_root: Path, row: dict[str, Any]) -> Path:
    """Append a row to outputs/runs_index.csv, creating the file if missing."""
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    index_path = outputs_dir / "runs_index.csv"
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_INDEX_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return index_path


def run_detection(
    bounds: dict[str, float],
    params: dict[str, Any],
    *,
    roi_name: str = "custom",
    outdir: str | Path | None = None,
    dataset: xr.Dataset | None = None,
    project_root: Path | None = None,
    save_figures: bool = True,
    save_masks: bool = True,
    channels_filename: str = "channels_mask.nc",
    skeleton_filename: str = "skeleton.nc",
    append_index: bool = True,
) -> tuple[dict[str, Path | str], dict[str, float]]:
    """Execute one detection run and persist standard outputs.

    Returns:
      outputs dictionary with key file paths and diagnostics dictionary.
    """
    root = project_root if project_root is not None else get_project_root()
    outdir_path, figures_dir = resolve_run_outdir(
        bounds=bounds,
        outdir=outdir,
        project_root=root,
    )

    ds = dataset if dataset is not None else load_bedmachine_default()
    region = subset_xy(ds, bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"])
    core = run_detection_core(
        region,
        sigma_small=float(params["sigma_small"]),
        sigma_large=float(params["sigma_large"]),
        threshold_quantile=float(params["threshold_quantile"]),
        edge_buffer=int(params["edge_buffer"]),
        min_length=int(params["min_length"]),
        return_masks=True,
    )
    diagnostics = {
        "floating_fraction": float(core["floating_fraction"]),
        "threshold": float(core["threshold"]),
        "detected_pixels": int(core["detected_pixels"]),
        "skeleton_pixels": int(core["skeleton_pixels"]),
        "num_components": int(core["num_components"]),
        "median_length_pixels": float(core["median_length_pixels"]),
        "mean_length_pixels": float(core["mean_length_pixels"]),
    }

    run_config = {
        "roi_name": roi_name if roi_name else "custom",
        "xmin": float(bounds["xmin"]),
        "xmax": float(bounds["xmax"]),
        "ymin": float(bounds["ymin"]),
        "ymax": float(bounds["ymax"]),
        "tuning": str(params.get("tuning", "balanced")),
        "sigma_small": float(params["sigma_small"]),
        "sigma_large": float(params["sigma_large"]),
        "threshold_quantile": float(params["threshold_quantile"]),
        "edge_buffer": int(params["edge_buffer"]),
        "min_length": int(params["min_length"]),
        "outdir": str(outdir_path),
        "mode": str(params.get("mode", "single")),
    }

    diagnostics_path = outdir_path / "diagnostics.json"
    run_config_path = outdir_path / "run_config.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True))

    outputs: dict[str, Path | str] = {
        "outdir": str(outdir_path),
        "figures_dir": str(figures_dir),
        "diagnostics_json": diagnostics_path,
        "run_config_json": run_config_path,
    }

    if save_masks:
        save_mask_netcdf(
            outdir_path / channels_filename,
            Path(channels_filename).stem,
            np.asarray(core["channels_mask"], dtype=bool),
            core["draft_template"],
        )
        save_mask_netcdf(
            outdir_path / skeleton_filename,
            Path(skeleton_filename).stem,
            np.asarray(core["skeleton"], dtype=bool),
            core["draft_template"],
        )
        outputs["channels_mask_nc"] = outdir_path / channels_filename
        outputs["skeleton_nc"] = outdir_path / skeleton_filename

    if save_figures:
        figure_paths = save_detection_figures(
            figures_dir=figures_dir,
            draft_values=np.asarray(core["draft_values"], dtype=float),
            floating_mask=np.asarray(core["floating_mask"], dtype=bool),
            floating_interior=np.asarray(core["floating_interior"], dtype=bool),
            skeleton_mask=np.asarray(core["skeleton"], dtype=bool),
            skeleton_orientations=np.asarray(core["skeleton_orientations"], dtype=float),
            lengths_pixels=np.asarray(core["lengths_pixels"], dtype=float),
        )
        outputs.update(figure_paths)

    if append_index:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "roi_name": roi_name if roi_name else "custom",
            "xmin": float(bounds["xmin"]),
            "xmax": float(bounds["xmax"]),
            "ymin": float(bounds["ymin"]),
            "ymax": float(bounds["ymax"]),
            "tuning": str(params.get("tuning", "balanced")),
            "sigma_small": float(params["sigma_small"]),
            "sigma_large": float(params["sigma_large"]),
            "threshold_quantile": float(params["threshold_quantile"]),
            "edge_buffer": int(params["edge_buffer"]),
            "min_length": int(params["min_length"]),
            "floating_fraction": diagnostics["floating_fraction"],
            "threshold": diagnostics["threshold"],
            "detected_pixels": diagnostics["detected_pixels"],
            "skeleton_pixels": diagnostics["skeleton_pixels"],
            "num_components": diagnostics["num_components"],
            "median_length_pixels": diagnostics["median_length_pixels"],
            "mean_length_pixels": diagnostics["mean_length_pixels"],
            "outdir": str(outdir_path),
        }
        outputs["runs_index_csv"] = append_runs_index(root, row)

    return outputs, diagnostics
