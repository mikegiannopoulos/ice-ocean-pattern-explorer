"""Data loading and handling utilities for IOBL-PX."""

from pathlib import Path

import xarray as xr


def get_project_root() -> Path:
    """Return the project root directory containing ``pyproject.toml``."""
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("Could not locate project root containing pyproject.toml")


def get_data_dirs() -> dict[str, Path]:
    """Return project data directories, creating them if needed."""
    data_dir = get_project_root() / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    for directory in (data_dir, raw_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
    }


def load_netcdf(path: Path) -> xr.Dataset:
    """Load a NetCDF dataset from disk.

    BedMachine Antarctica is commonly distributed as NetCDF4/HDF5, so this
    loader prefers the ``netcdf4`` backend when the ``netCDF4`` package is
    available. If that backend cannot be imported, xarray falls back to its
    default engine selection.
    """
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    open_kwargs: dict[str, str] = {}
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        open_kwargs = {}
    else:
        open_kwargs["engine"] = "netcdf4"

    dataset = xr.open_dataset(path, **open_kwargs)
    try:
        return dataset.load()
    finally:
        dataset.close()


def load_from_raw(filename: str) -> xr.Dataset:
    """Load a NetCDF file from the project's raw data directory."""
    raw_dir = get_data_dirs()["raw"]
    return load_netcdf(raw_dir / filename)


def load_bedmachine_default() -> xr.Dataset:
    """Load the expected default BedMachine Antarctica file from ``data/raw``."""
    return load_from_raw("bedmachine_antarctica.nc")
