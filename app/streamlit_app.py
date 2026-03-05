"""Streamlit UI for running the IOBLP channel detection pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    from ioblp.data import load_bedmachine_default
    from ioblp.pipeline import ROI_PRESETS, resolve_bounds, resolve_tuning, run_detection
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from ioblp.data import load_bedmachine_default
    from ioblp.pipeline import ROI_PRESETS, resolve_bounds, resolve_tuning, run_detection
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def load_dataset():
    """Load BedMachine once for app sessions."""
    return load_bedmachine_default()


def load_dataset_safe():
    """Load dataset and return (dataset, error_message)."""
    try:
        return load_dataset(), None
    except FileNotFoundError as exc:
        return None, (
            "BedMachine file not found. Expected path: "
            "`data/raw/bedmachine_antarctica.nc`. "
            f"Details: {exc}"
        )
    except Exception as exc:  # pragma: no cover - UI surface
        return None, f"Failed to load BedMachine dataset: {exc}"


def main() -> None:
    """Render and execute the Streamlit demo app."""
    st.set_page_config(page_title="IOBLP Detection Demo", layout="wide")
    st.title("IOBLP Channel Detection Demo")
    st.caption("Runs the same detection pipeline as the CLI and writes outputs per run.")

    dataset, dataset_error = load_dataset_safe()
    if dataset_error:
        st.error(dataset_error)
        st.stop()

    mode = st.radio("ROI mode", ["Preset", "Custom"], horizontal=True)
    roi_name = None
    override_preset_bounds = False
    xmin = xmax = ymin = ymax = None

    if mode == "Preset":
        roi_name = st.selectbox("ROI preset", sorted(ROI_PRESETS))
        preset = ROI_PRESETS[roi_name]
        override_preset_bounds = st.checkbox("Override preset bounds", value=False)
        if override_preset_bounds:
            c1, c2, c3, c4 = st.columns(4)
            xmin = c1.number_input("xmin", value=float(preset["xmin"]), format="%0.1f")
            xmax = c2.number_input("xmax", value=float(preset["xmax"]), format="%0.1f")
            ymin = c3.number_input("ymin", value=float(preset["ymin"]), format="%0.1f")
            ymax = c4.number_input("ymax", value=float(preset["ymax"]), format="%0.1f")
        else:
            st.code(
                f"xmin={preset['xmin']}, xmax={preset['xmax']}, "
                f"ymin={preset['ymin']}, ymax={preset['ymax']}"
            )
    else:
        c1, c2, c3, c4 = st.columns(4)
        xmin = c1.number_input("xmin", value=-1600000.0, format="%0.1f")
        xmax = c2.number_input("xmax", value=-1200000.0, format="%0.1f")
        ymin = c3.number_input("ymin", value=-500000.0, format="%0.1f")
        ymax = c4.number_input("ymax", value=-100000.0, format="%0.1f")

    tuning = st.selectbox("Tuning preset", ["balanced", "conservative", "sensitive"], index=0)
    advanced = st.checkbox("Advanced overrides", value=False)

    sigma_small_override = None
    threshold_override = None
    min_length_override = None
    if advanced:
        d1, d2, d3 = st.columns(3)
        sigma_small_override = d1.number_input(
            "sigma_small override", value=2.0, min_value=0.1, format="%0.3f"
        )
        threshold_override = d2.number_input(
            "threshold_quantile override",
            value=0.975,
            min_value=0.0,
            max_value=1.0,
            format="%0.4f",
        )
        min_length_override = d3.number_input(
            "min_length override", value=30, min_value=1, step=1
        )

    e1, e2 = st.columns(2)
    sigma_large = e1.number_input("sigma_large", value=20.0, min_value=0.1, format="%0.3f")
    edge_buffer = e2.number_input("edge_buffer", value=5, min_value=0, step=1)

    outdir_input = st.text_input(
        "Output directory (optional, relative to project root or absolute)",
        value="",
    ).strip()

    run_clicked = st.button("Run detection", type="primary")
    if not run_clicked:
        return

    try:
        bounds, candidate_bounds = resolve_bounds(
            dataset,
            roi=roi_name,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        tuning_params = resolve_tuning(
            tuning,
            sigma_small=float(sigma_small_override) if sigma_small_override is not None else None,
            threshold_quantile=(
                float(threshold_override) if threshold_override is not None else None
            ),
            min_length=int(min_length_override) if min_length_override is not None else None,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    params = {
        "tuning": tuning_params["tuning"],
        "sigma_small": float(tuning_params["sigma_small"]),
        "sigma_large": float(sigma_large),
        "threshold_quantile": float(tuning_params["threshold_quantile"]),
        "edge_buffer": int(edge_buffer),
        "min_length": int(tuning_params["min_length"]),
    }

    with st.spinner("Running detection..."):
        try:
            outputs, diagnostics = run_detection(
                bounds,
                params,
                roi_name=roi_name if roi_name else "custom",
                outdir=outdir_input or None,
                dataset=dataset,
                project_root=PROJECT_ROOT,
                save_figures=True,
                save_masks=True,
                append_index=True,
            )
        except Exception as exc:  # pragma: no cover - UI surface
            st.error(f"Detection failed: {exc}")
            return

    if candidate_bounds is not None and not override_preset_bounds:
        st.info("Auto-fit applied for this ROI preset before running detection.")

    st.success(f"Run complete: {outputs['outdir']}")
    if diagnostics["floating_fraction"] < 0.10:
        st.warning(
            "ROI contains little floating ice; detections may be empty "
            f"(floating_fraction={diagnostics['floating_fraction']:.4f})."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Floating fraction", f"{diagnostics['floating_fraction']:.4f}")
    m2.metric("Detected pixels", f"{int(diagnostics['detected_pixels'])}")
    m3.metric("Components", f"{int(diagnostics['num_components'])}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Skeleton pixels", f"{int(diagnostics['skeleton_pixels'])}")
    m5.metric("Threshold", f"{diagnostics['threshold']:.6f}")
    m6.metric("Median length (px)", f"{diagnostics['median_length_pixels']:.2f}")

    st.subheader("Generated Figures")
    figure_keys = [
        ("fig01", "Fig01: Region Draft with Floating Ice Mask"),
        ("fig02", "Fig02: Detected Channels on Draft"),
        ("fig03", "Fig03: Skeleton Orientation Distribution"),
        ("fig04", "Fig04: Channel Length Distribution"),
    ]
    for key, caption in figure_keys:
        if key in outputs:
            st.image(str(outputs[key]), caption=caption, use_container_width=True)

    st.subheader("Saved Files")
    for key in (
        "channels_mask_nc",
        "skeleton_nc",
        "diagnostics_json",
        "run_config_json",
        "runs_index_csv",
    ):
        if key in outputs:
            st.code(str(outputs[key]))


if __name__ == "__main__":
    main()
