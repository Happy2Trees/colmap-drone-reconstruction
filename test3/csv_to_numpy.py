#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np


def _clean_field(name: str) -> str:
    """Normalize CSV header field: strip whitespace, drop BOM, uppercase."""
    if name is None:
        return ""
    return name.replace("\ufeff", "").strip().upper()


def read_xyz_from_csv(csv_path: Path, columns: List[str]) -> np.ndarray:
    """Read selected XYZ columns from a CSV into an [N,3] array.

    Args:
        csv_path: Path to the CSV file.
        columns: List of 3 column names to extract (case-insensitive).

    Returns:
        np.ndarray of shape [N, 3] with dtype float64 (can be cast by caller).
    """
    wanted = [_clean_field(c) for c in columns]
    rows: List[List[float]] = []

    # Use utf-8-sig to gracefully handle BOM if present
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        # Build a mapping from cleaned header name -> original header
        header_map = {_clean_field(h): h for h in reader.fieldnames}

        missing = [c for c in wanted if c not in header_map]
        if missing:
            available = ", ".join(sorted(header_map.keys()))
            raise KeyError(
                f"Missing required columns {missing}. Available: {available}"
            )

        src_cols = [header_map[c] for c in wanted]

        for idx, row in enumerate(reader):
            try:
                x = float(row[src_cols[0]])
                y = float(row[src_cols[1]])
                z = float(row[src_cols[2]])
            except (KeyError, TypeError, ValueError):
                # Skip malformed lines quietly
                continue
            rows.append([x, y, z])

    if not rows:
        raise ValueError(f"No valid numeric rows parsed from {csv_path}")

    return np.asarray(rows, dtype=np.float64)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "measurement.csv"
    default_output = repo_root / "data" / "measurement_xyz.npy"

    parser = argparse.ArgumentParser(
        description="Convert measurement.csv to an [N,3] NumPy array saved as .npy",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input CSV (default: data/measurement.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output .npy path (default: data/measurement_xyz.npy)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Output array dtype",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default="POINTX,POINTY,POINTZ",
        help="Comma-separated column names to extract (case-insensitive)",
    )

    args = parser.parse_args()

    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    if len(columns) != 3:
        raise ValueError("--columns must specify exactly 3 columns, e.g., POINTX,POINTY,POINTZ")

    arr = read_xyz_from_csv(args.input, columns)
    arr = arr.astype(np.float32 if args.dtype == "float32" else np.float64, copy=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, arr)

    print(f"Saved: {args.output} | shape={arr.shape} | dtype={arr.dtype}")


if __name__ == "__main__":
    main()

