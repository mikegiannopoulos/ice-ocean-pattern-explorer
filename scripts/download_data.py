"""Manual BedMachine placement instructions and local file verification."""

from pathlib import Path
import sys


EXPECTED_FILENAME = "bedmachine_antarctica.nc"


def main() -> None:
    """Print manual download instructions and verify the expected file path."""
    project_root = Path(__file__).resolve().parents[1]
    target_path = project_root / "data" / "raw" / EXPECTED_FILENAME

    print("Manual download required: BedMachine Antarctica is not fetched automatically.")
    print("Source: obtain the NetCDF dataset from the official BedMachine Antarctica distribution.")
    print(f"Expected filename: {EXPECTED_FILENAME}")
    print(f"Expected destination: {target_path.relative_to(project_root)}")
    print("If your downloaded file has a different name, rename it before running the notebook.")

    if target_path.exists():
        print(f"Verified dataset path exists: {target_path.relative_to(project_root)}")
        return

    print(
        f"Missing dataset file: {target_path.relative_to(project_root)}",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
