# Ice–Ocean Boundary Layer Pattern Explorer (IOBL-PX)

## Overview

IOBL-PX provides a reproducible BedMachine-based basal channel detection workflow with a command-line runner and an optional Streamlit interface, using the same underlying pipeline logic for both.

## Data Requirement

BedMachine data is not included in this repository. Place the NetCDF file at:

`data/raw/bedmachine_antarctica.nc`

If your downloaded file has a different name, rename it to `bedmachine_antarctica.nc`.

## Install

Base install:

```bash
pip install -e .
```

App extra:

```bash
pip install -e ".[app]"
```

Test extra:

```bash
pip install -e ".[test]"
```

## CLI Quickstart

List and check ROI presets:

```bash
python scripts/run_detection.py --list-rois
python scripts/run_detection.py --check-rois
```

Run a standard ROI/tuning detection:

```bash
python scripts/run_detection.py --roi ross --tuning balanced
```

Run a sweep:

```bash
python scripts/run_detection.py --roi pine_island --sweep
```

## Outputs

Each run writes to a run-specific output directory under `outputs/` by default, including:

- `channels_mask.nc` and `skeleton.nc` (single-run mode)
- `diagnostics.json`
- `run_config.json`
- `figures/fig01_region_draft.png`
- `figures/fig02_channels_overlay.png`
- `figures/fig03_orientation_hist.png`
- `figures/fig04_channel_lengths.png`

Sweep mode additionally writes `sweep_results.csv` and sweep summary figures in `figures/`.

## Streamlit (Local)

```bash
streamlit run app/streamlit_app.py
```

## Notes on Parameters

- `--tuning conservative` uses `threshold_quantile=0.99`, `sigma_small=4`, `min_length=30`.
- `--tuning balanced` uses `threshold_quantile=0.975`, `sigma_small=2`, `min_length=30`.
- `--tuning sensitive` uses `threshold_quantile=0.95`, `sigma_small=2`, `min_length=10`.
- Explicit CLI overrides (`--threshold-quantile`, `--sigma-small`, `--min-length`) take precedence over tuning defaults.
