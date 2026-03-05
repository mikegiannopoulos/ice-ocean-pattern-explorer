"""Preprocessing helpers for BedMachine-style x/y gridded datasets."""

import xarray as xr


def _coordinate_slice(coord: xr.DataArray, lower: float, upper: float) -> slice:
    """Build a slice that respects the coordinate sort direction."""
    start = min(lower, upper)
    stop = max(lower, upper)
    if coord.size > 1 and float(coord[0]) > float(coord[-1]):
        return slice(stop, start)
    return slice(start, stop)


def subset_xy(
    ds: xr.Dataset, x_min: float, x_max: float, y_min: float, y_max: float
) -> xr.Dataset:
    """Subset a dataset over x/y coordinates using inclusive bounds."""
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("Dataset must contain x and y coordinates")

    x_slice = _coordinate_slice(ds["x"], x_min, x_max)
    y_slice = _coordinate_slice(ds["y"], y_min, y_max)
    return ds.sel(x=x_slice, y=y_slice)


def get_floating_mask(ds: xr.Dataset) -> xr.DataArray:
    """Return a boolean floating-ice mask from BedMachine-style mask values.

    BedMachine commonly uses ``mask == 3`` for floating ice. This helper
    assumes that convention when a ``mask`` variable is present and returns a
    boolean DataArray with the same grid. If ``mask`` is missing, a clear error
    is raised because no robust floating/grounded inference is attempted here.
    """
    if "mask" not in ds.data_vars:
        available = ", ".join(sorted(ds.data_vars))
        raise ValueError(f"Dataset does not contain 'mask'. Available variables: {available}")

    return (ds["mask"] == 3).astype(bool)


def compute_ice_draft(ds: xr.Dataset) -> xr.DataArray:
    """Return ice draft from a direct field or a simple surface-thickness relation.

    Preferred variable names:
    - ``draft``: returned directly
    - otherwise ``surface`` and ``thickness``: draft = surface - thickness
    """
    if "draft" in ds.data_vars:
        return ds["draft"]

    if "surface" in ds.data_vars and "thickness" in ds.data_vars:
        return ds["surface"] - ds["thickness"]

    available = ", ".join(sorted(ds.data_vars))
    raise ValueError(
        "Could not compute ice draft. Expected 'draft' or both 'surface' and "
        f"'thickness'. Available variables: {available}"
    )
