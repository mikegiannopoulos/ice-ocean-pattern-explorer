"""Command-line interface for IOBL-PX.

This module will expose entry points for running data processing and analysis
via command-line commands. Currently only a scaffold and placeholder.
"""

import argparse
from typing import Any


def main(args: Any = None) -> None:
    """Placeholder main function for CLI."""
    parser = argparse.ArgumentParser(
        description="Ice–Ocean Boundary Layer Pattern Explorer CLI"
    )
    # future arguments would be added here
    parser.parse_args(args)


if __name__ == "__main__":
    main()
