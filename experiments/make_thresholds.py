"""Create thresholds.json from the saved degradation-study results.

Run from the project root with:

    python experiments/make_thresholds.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from yieldlite.thresholds import (
    derive_thresholds,
    paired_gain,
    write_thresholds,
)


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

    thresholds = derive_thresholds(
        results,
        run_info=run_info,
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

    print()
    for name, entry in thresholds["thresholds"].items():
        print(f"{name}: {entry['status']} {entry['value']}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
