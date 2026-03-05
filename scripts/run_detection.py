"""Run the BedMachine channel detection pipeline from the command line."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ioblp.data import load_bedmachine_default
from ioblp.pipeline import (
    AUTOFIT_ROIS,
    ROI_PRESETS,
    TUNING_PRESETS,
    autofit_to_floating,
    format_bounds,
    resolve_bounds,
    resolve_run_outdir,
    resolve_tuning,
    run_detection,
    run_detection_core,
    save_mask_netcdf,
)
from ioblp.preprocess import subset_xy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run channel detection on a BedMachine Antarctica subset."
    )
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--roi", type=str, default=None)
    parser.add_argument("--list-rois", action="store_true")
    parser.add_argument("--check-rois", action="store_true")
    parser.add_argument(
        "--tuning",
        type=str,
        default="balanced",
        choices=sorted(TUNING_PRESETS),
    )
    parser.add_argument("--sigma-small", type=float, default=None)
    parser.add_argument("--sigma-large", type=float, default=20.0)
    parser.add_argument("--threshold-quantile", type=float, default=None)
    parser.add_argument("--edge-buffer", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-quantiles", type=str, default="0.95,0.975,0.99,0.995")
    parser.add_argument("--sweep-min-lengths", type=str, default="10,30")
    parser.add_argument("--sweep-sigma-smalls", type=str, default="2,4,6")
    parser.add_argument("--sweep-save-best", action="store_true")
    return parser.parse_args()


def parse_float_grid(values: str, arg_name: str) -> list[float]:
    """Parse comma-separated floats from a CLI argument."""
    parts = [part.strip() for part in values.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{arg_name} must contain at least one numeric value")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be a comma-separated list of floats") from exc


def parse_int_grid(values: str, arg_name: str) -> list[int]:
    """Parse comma-separated integers from a CLI argument."""
    parts = [part.strip() for part in values.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{arg_name} must contain at least one integer value")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be a comma-separated list of integers") from exc


def print_roi_presets() -> None:
    """Print available ROI presets."""
    for name in sorted(ROI_PRESETS):
        print(f"{name}: {format_bounds(ROI_PRESETS[name])}")


def check_roi_fractions(ds: xr.Dataset) -> None:
    """Print floating fractions for ROI presets."""
    print("ROI preset floating-ice fractions (mask == 3):")
    for name in sorted(ROI_PRESETS):
        candidate_bounds = ROI_PRESETS[name]
        candidate_region = subset_xy(
            ds,
            candidate_bounds["xmin"],
            candidate_bounds["xmax"],
            candidate_bounds["ymin"],
            candidate_bounds["ymax"],
        )
        if name in AUTOFIT_ROIS:
            fitted_bounds = autofit_to_floating(candidate_region)
            fitted_region = subset_xy(
                ds,
                fitted_bounds["xmin"],
                fitted_bounds["xmax"],
                fitted_bounds["ymin"],
                fitted_bounds["ymax"],
            )
            floating_fraction = float(np.asarray(fitted_region["mask"] == 3, dtype=bool).mean())
            print(f"{name}:")
            print(f"  candidate: {format_bounds(candidate_bounds)}")
            print(f"  fitted:    {format_bounds(fitted_bounds)}")
            print(f"  floating_fraction(fitted): {floating_fraction:.4f}")
            continue

        floating_fraction = float(np.asarray(candidate_region["mask"] == 3, dtype=bool).mean())
        print(f"{name}:")
        print(f"  candidate: {format_bounds(candidate_bounds)}")
        print(f"  floating_fraction(candidate): {floating_fraction:.4f}")


def plot_sweep_summary(csv_path: Path, figures_dir: Path) -> None:
    """Generate sweep summary plots from sweep_results.csv."""
    if not csv_path.exists():
        print(f"No sweep results found at {csv_path}; skipping sweep plots.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "roi_name": row["roi_name"],
                    "threshold_quantile": float(row["threshold_quantile"]),
                    "sigma_small": float(row["sigma_small"]),
                    "min_length": int(float(row["min_length"])),
                    "num_components": int(float(row["num_components"])),
                    "median_length_pixels": float(row["median_length_pixels"]),
                }
            )
    if not rows:
        print(f"No rows in {csv_path}; skipping sweep plots.")
        return

    grouped: dict[tuple[float, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (float(row["sigma_small"]), int(row["min_length"]))
        grouped.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (sigma_small, min_length), group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda item: float(item["threshold_quantile"]))
        xs = [float(item["threshold_quantile"]) for item in sorted_rows]
        ys = [int(item["num_components"]) for item in sorted_rows]
        ax.plot(xs, ys, marker="o", label=f"sigma={sigma_small:g}, minlen={min_length}")
    ax.set_xlabel("threshold_quantile")
    ax.set_ylabel("num_components")
    ax.set_title("Sweep: Components vs Threshold Quantile")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_sweep_components.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (sigma_small, min_length), group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda item: float(item["threshold_quantile"]))
        xs = [float(item["threshold_quantile"]) for item in sorted_rows]
        ys = [float(item["median_length_pixels"]) for item in sorted_rows]
        ax.plot(xs, ys, marker="o", label=f"sigma={sigma_small:g}, minlen={min_length}")
    ax.set_xlabel("threshold_quantile")
    ax.set_ylabel("median_length_pixels")
    ax.set_title("Sweep: Median Length vs Threshold Quantile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_sweep_median_length.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]
    fig, ax = plt.subplots(figsize=(8, 5))
    quantiles = sorted({float(row["threshold_quantile"]) for row in rows})
    for index, quantile in enumerate(quantiles):
        subset = [row for row in rows if float(row["threshold_quantile"]) == quantile]
        xs = [int(row["num_components"]) for row in subset]
        ys = [float(row["median_length_pixels"]) for row in subset]
        ax.scatter(xs, ys, marker=markers[index % len(markers)], label=f"q={quantile:g}")

    best_path = csv_path.parent / "best_params.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_x = float(best["num_components"])
        best_y = float(best["median_length_pixels"])
        ax.scatter([best_x], [best_y], marker="x", color="black", s=80, linewidths=1.5)
        ax.annotate("best", (best_x, best_y), textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("num_components")
    ax.set_ylabel("median_length_pixels")
    ax.set_title("Sweep Tradeoff: Components vs Median Length")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_sweep_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Execute detection workflow."""
    args = parse_args()
    if args.list_rois:
        print_roi_presets()
        return

    ds = load_bedmachine_default()
    if args.check_rois:
        check_roi_fractions(ds)
        return

    try:
        bounds, candidate_bounds = resolve_bounds(
            ds,
            roi=args.roi,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        tuning = resolve_tuning(
            args.tuning,
            sigma_small=args.sigma_small,
            threshold_quantile=args.threshold_quantile,
            min_length=args.min_length,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if candidate_bounds is not None:
        print(f"Auto-fit ROI '{args.roi}':")
        print(f"- candidate: {format_bounds(candidate_bounds)}")
        print(f"- fitted:    {format_bounds(bounds)}")

    roi_name = args.roi if args.roi else "custom"
    params = {
        "tuning": tuning["tuning"],
        "sigma_small": float(tuning["sigma_small"]),
        "sigma_large": float(args.sigma_large),
        "threshold_quantile": float(tuning["threshold_quantile"]),
        "edge_buffer": int(args.edge_buffer),
        "min_length": int(tuning["min_length"]),
    }

    if args.sweep:
        outdir, figures_dir = resolve_run_outdir(
            bounds=bounds,
            outdir=args.outdir,
            project_root=PROJECT_ROOT,
            expected_paths=[],
        )
        expected = [
            outdir / "sweep_results.csv",
            outdir / "run_config.json",
            outdir / "best_params.json",
            outdir / "channels_mask_best.nc",
            outdir / "skeleton_best.nc",
            figures_dir / "fig_sweep_components.png",
            figures_dir / "fig_sweep_median_length.png",
            figures_dir / "fig_sweep_tradeoff.png",
        ]
        if outdir.exists() and any(path.exists() for path in expected):
            existing = ", ".join(str(path) for path in expected if path.exists())
            raise SystemExit(
                f"Output overwrite prevention: existing files detected in outdir: {existing}"
            )

        try:
            quantiles = parse_float_grid(args.sweep_quantiles, "--sweep-quantiles")
            min_lengths = parse_int_grid(args.sweep_min_lengths, "--sweep-min-lengths")
            sigma_smalls = parse_float_grid(args.sweep_sigma_smalls, "--sweep-sigma-smalls")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        region = subset_xy(ds, bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"])
        total = len(quantiles) * len(min_lengths) * len(sigma_smalls)
        rows: list[dict[str, object]] = []
        i = 0
        for threshold_quantile in quantiles:
            for sigma_small in sigma_smalls:
                for min_length in min_lengths:
                    i += 1
                    metrics = run_detection_core(
                        region,
                        sigma_small=sigma_small,
                        sigma_large=params["sigma_large"],
                        threshold_quantile=threshold_quantile,
                        edge_buffer=params["edge_buffer"],
                        min_length=min_length,
                        return_masks=False,
                    )
                    row = {
                        "roi_name": roi_name,
                        "xmin": bounds["xmin"],
                        "xmax": bounds["xmax"],
                        "ymin": bounds["ymin"],
                        "ymax": bounds["ymax"],
                        "sigma_small": sigma_small,
                        "sigma_large": params["sigma_large"],
                        "threshold_quantile": threshold_quantile,
                        "edge_buffer": params["edge_buffer"],
                        "min_length": min_length,
                        "floating_fraction": metrics["floating_fraction"],
                        "threshold": metrics["threshold"],
                        "detected_pixels": metrics["detected_pixels"],
                        "skeleton_pixels": metrics["skeleton_pixels"],
                        "num_components": metrics["num_components"],
                        "median_length_pixels": metrics["median_length_pixels"],
                        "mean_length_pixels": metrics["mean_length_pixels"],
                    }
                    rows.append(row)
                    print(
                        f"Sweep {i}/{total}: q={threshold_quantile}, "
                        f"sigma_small={sigma_small}, min_length={min_length} -> "
                        f"components={row['num_components']}, "
                        f"median_len={row['median_length_pixels']}"
                    )

        sweep_csv_path = outdir / "sweep_results.csv"
        fieldnames = [
            "roi_name",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
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
        ]
        with sweep_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        run_config = {
            "mode": "sweep",
            "roi_name": roi_name,
            "xmin": bounds["xmin"],
            "xmax": bounds["xmax"],
            "ymin": bounds["ymin"],
            "ymax": bounds["ymax"],
            "tuning": tuning["tuning"],
            "sigma_small": tuning["sigma_small"],
            "sigma_large": params["sigma_large"],
            "threshold_quantile": tuning["threshold_quantile"],
            "edge_buffer": params["edge_buffer"],
            "min_length": tuning["min_length"],
            "sweep_quantiles": quantiles,
            "sweep_sigma_smalls": sigma_smalls,
            "sweep_min_lengths": min_lengths,
            "sweep_save_best": args.sweep_save_best,
            "outdir": str(outdir),
        }
        (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

        print(f"Wrote sweep_results.csv to {sweep_csv_path}")
        if args.sweep_save_best and rows:
            best_row = max(
                rows,
                key=lambda row: (
                    float(row["median_length_pixels"]),
                    int(row["num_components"]),
                    int(row["detected_pixels"]),
                ),
            )
            best_result = run_detection_core(
                region,
                sigma_small=float(best_row["sigma_small"]),
                sigma_large=float(best_row["sigma_large"]),
                threshold_quantile=float(best_row["threshold_quantile"]),
                edge_buffer=int(best_row["edge_buffer"]),
                min_length=int(best_row["min_length"]),
                return_masks=True,
            )
            save_mask_netcdf(
                outdir / "channels_mask_best.nc",
                "channels_mask_best",
                np.asarray(best_result["channels_mask"], dtype=bool),
                best_result["draft_template"],
            )
            save_mask_netcdf(
                outdir / "skeleton_best.nc",
                "skeleton_best",
                np.asarray(best_result["skeleton"], dtype=bool),
                best_result["draft_template"],
            )
            best_params = {
                "roi_name": roi_name,
                "xmin": bounds["xmin"],
                "xmax": bounds["xmax"],
                "ymin": bounds["ymin"],
                "ymax": bounds["ymax"],
                "tuning": tuning["tuning"],
                "sigma_small": float(best_row["sigma_small"]),
                "sigma_large": float(best_row["sigma_large"]),
                "threshold_quantile": float(best_row["threshold_quantile"]),
                "edge_buffer": int(best_row["edge_buffer"]),
                "min_length": int(best_row["min_length"]),
                "floating_fraction": float(best_result["floating_fraction"]),
                "threshold": float(best_result["threshold"]),
                "detected_pixels": int(best_result["detected_pixels"]),
                "skeleton_pixels": int(best_result["skeleton_pixels"]),
                "num_components": int(best_result["num_components"]),
                "median_length_pixels": float(best_result["median_length_pixels"]),
                "mean_length_pixels": float(best_result["mean_length_pixels"]),
            }
            (outdir / "best_params.json").write_text(
                json.dumps(best_params, indent=2, sort_keys=True)
            )
            print(
                "Saved best combo outputs: "
                f"sigma_small={best_params['sigma_small']}, "
                f"threshold_quantile={best_params['threshold_quantile']}, "
                f"min_length={best_params['min_length']}"
            )

        plot_sweep_summary(sweep_csv_path, figures_dir)
        print(f"Saved sweep plots to {figures_dir}")
        return

    outputs, diagnostics = run_detection(
        bounds,
        params,
        roi_name=roi_name,
        outdir=args.outdir,
        dataset=ds,
        project_root=PROJECT_ROOT,
        save_figures=True,
        save_masks=True,
        append_index=True,
    )
    if diagnostics["floating_fraction"] < 0.10:
        print(
            "WARNING: ROI contains little floating ice "
            f"(floating_fraction={diagnostics['floating_fraction']:.4f}). "
            "Detections may be empty."
        )

    print("Saved outputs:")
    for key in ("channels_mask_nc", "skeleton_nc", "diagnostics_json", "run_config_json"):
        if key in outputs:
            print(f"- {outputs[key]}")
    for key in ("fig01", "fig02", "fig03", "fig04"):
        if key in outputs:
            print(f"- {outputs[key]}")
    if "runs_index_csv" in outputs:
        print(f"- {outputs['runs_index_csv']}")

    print("\nRun summary:")
    print(f"- outdir: {outputs['outdir']}")
    print(f"- tuning: {tuning['tuning']}")
    print(f"- sigma_small: {params['sigma_small']}")
    print(f"- sigma_large: {params['sigma_large']}")
    print(f"- threshold_quantile: {params['threshold_quantile']}")
    print(f"- min_length: {params['min_length']}")
    print(f"- floating_fraction: {diagnostics['floating_fraction']}")
    print(f"- threshold: {diagnostics['threshold']}")
    print(f"- detected_pixels: {diagnostics['detected_pixels']}")
    print(f"- skeleton_pixels: {diagnostics['skeleton_pixels']}")
    print(f"- num_components: {diagnostics['num_components']}")
    print(f"- median_length_pixels: {diagnostics['median_length_pixels']}")


if __name__ == "__main__":
    main()
