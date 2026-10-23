"""Create thresholds.json from the saved study results.

Run from the project root with:

    python experiments/make_thresholds.py

The degradation study gives P1 and H1. When the interval study has also been
run, its results give P0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from yieldlite.thresholds import (
    coverage_check,
    derive_thresholds,
    paired_gain,
    write_thresholds,
)


def _read_intervals(
    folder: Path,
) -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    """Load the interval-study summary and settings, if they were saved."""
    summary_path = folder / "interval_summary.csv"
    if not summary_path.exists():
        return None, None

    summary = pd.read_csv(summary_path)
    summary["h"] = pd.to_numeric(summary["h"])

    settings = json.loads(
        (folder / "interval_settings.json").read_text(encoding="utf-8")
    )
    return summary, settings


def main() -> None:
    """Read the study results, derive thresholds, and save the JSON file."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--results",
        default="experiments/results",
        help="Folder containing the degradation results and settings.",
    )
    parser.add_argument(
        "--out",
        default="src/yieldlite/data/thresholds.json",
        help="Location where the threshold file should be written.",
    )
    args = parser.parse_args()

    results_folder = Path(args.results)

    results = pd.read_csv(
        results_folder / "degradation_results.csv"
    )
    results["h"] = pd.to_numeric(results["h"])

    run_info = json.loads(
        (results_folder / "settings.json").read_text(
            encoding="utf-8"
        )
    )

    intervals, interval_info = _read_intervals(results_folder)

    thresholds = derive_thresholds(
        results,
        run_info=run_info,
        intervals=intervals,
        interval_info=interval_info,
    )
    write_thresholds(thresholds, args.out)

    # Print the evidence first so the saved thresholds can be checked easily.
    gains = paired_gain(results)
    print(
        gains.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    if intervals is not None:
        print()
        print(
            coverage_check(intervals).to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print()
    for name, entry in thresholds["thresholds"].items():
        print(f"{name}: {entry['status']} {entry['value']}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
