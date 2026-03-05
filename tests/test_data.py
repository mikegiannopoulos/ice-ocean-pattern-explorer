import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioblp.data import get_data_dirs, get_project_root, load_from_raw, load_netcdf


def test_get_project_root_contains_pyproject() -> None:
    project_root = get_project_root()
    assert (project_root / "pyproject.toml").exists()


def test_get_data_dirs_creates_expected_directories() -> None:
    data_dirs = get_data_dirs()

    assert data_dirs["data"].is_dir()
    assert data_dirs["raw"].is_dir()
    assert data_dirs["processed"].is_dir()


def test_load_from_raw_reads_synthetic_netcdf() -> None:
    data_dirs = get_data_dirs()
    test_file = data_dirs["raw"] / "test.nc"
    dataset = xr.Dataset(
        {
            "values": (("x", "y"), np.arange(25).reshape(5, 5)),
        }
    )

    try:
        dataset.to_netcdf(test_file)
        loaded = load_from_raw("test.nc")
        assert loaded["values"].shape == (5, 5)
    finally:
        dataset.close()
        if test_file.exists():
            test_file.unlink()


def test_load_netcdf_raises_for_missing_file() -> None:
    missing_file = Path("data/raw/does-not-exist.nc")

    with pytest.raises(FileNotFoundError):
        load_netcdf(missing_file)
