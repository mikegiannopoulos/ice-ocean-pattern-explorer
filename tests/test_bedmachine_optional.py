import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioblp.data import get_data_dirs, load_bedmachine_default
from ioblp.preprocess import compute_ice_draft, get_floating_mask


def test_bedmachine_optional_integration() -> None:
    dataset_path = get_data_dirs()["raw"] / "bedmachine_antarctica.nc"
    if not dataset_path.exists():
        pytest.skip("Optional BedMachine dataset not present at data/raw/bedmachine_antarctica.nc")

    ds = load_bedmachine_default()
    draft = compute_ice_draft(ds)
    floating_mask = get_floating_mask(ds)

    assert draft.ndim == 2
    assert floating_mask.ndim == 2
    assert floating_mask.dtype == bool
