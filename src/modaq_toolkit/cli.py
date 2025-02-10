"""Command line interface for MODAQ toolkit."""

import argparse
import sys
from pathlib import Path

from .parser import process_mcap_files


def main(args: list[str] | None = None) -> int:
    """Extract and Transform Raw MODAQ data into a standardized parquet files"""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Extract and Transform Raw MODAQ data into a standardized parquet files"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing MCAP files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./data/"),
        help="Directory for output (default: ./data)",
    )

    parsed_args = parser.parse_args(args)
    process_mcap_files(parsed_args.input_dir, parsed_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
