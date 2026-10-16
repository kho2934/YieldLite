"""Run the degradation study on the development panels.

Example
-------
python experiments/degradation_study.py --p-form oxide --k-form oxide

The script removes the locked 2023-2024 holdout before fitting anything. That
holdout is not evaluated here. Detailed results, summaries, and settings are
saved separately so the study can be reviewed and reproduced later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout
from yieldlite.study import run_degradation, summarize_degradation


def _int_list(text: str) -> list[int]:
    """Read a comma-separated list of integers."""
    return [
        int(part)
        for part in text.split(",")
    ]


def _cap_list(text: str) -> list[int | None]:
    """Read household caps, using 'all' to mean no cap."""
    return [
        None
        if part.strip().lower() == "all"
        else int(part)
        for part in text.split(",")
    ]


def main() -> None:
    """Load the data, run the study, and save the results."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--data",
        default="data/processed/dataset_final.csv",
        help="path to the cleaned survey table",
    )
    parser.add_argument(
        "--p-form",
        required=True,
        choices=["elemental", "oxide"],
        help="form used to record phosphorus",
    )
    parser.add_argument(
        "--k-form",
        required=True,
        choices=["elemental", "oxide"],
        help="form used to record potassium",
    )
    parser.add_argument(
        "--p-values",
        type=_int_list,
        default=[3, 4, 5, 6, 7],
        help="comma-separated numbers of training panels",
    )
    parser.add_argument(
        "--h-values",
        type=_cap_list,
        default=[5, 10, 20, None],
        help="comma-separated household caps; use 'all' for no cap",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=20,
        help="maximum panel draws for each study setting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed used to reproduce the study",
    )
    parser.add_argument(
        "--out",
        default="experiments/results",
        help="directory where the study outputs will be saved",
    )
    args = parser.parse_args()

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form=args.p_form,
        k_form=args.k_form,
    )

    development, _ = lock_holdout(frame)

    print(
        f"development rows: {len(development)}, "
        f"panels: {development['panel_id'].nunique()}"
    )

    results = run_degradation(
        development,
        p_values=args.p_values,
        h_values=args.h_values,
        n_draws=args.draws,
        seed=args.seed,
    )
    summary = summarize_degradation(results)

    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        out_dir / "degradation_results.csv",
        index=False,
    )
    summary.to_csv(
        out_dir / "degradation_summary.csv",
        index=False,
    )

    settings = {
        "p_form": args.p_form,
        "k_form": args.k_form,
        "p_values": args.p_values,
        "h_values": [
            "all" if h is None else h
            for h in args.h_values
        ],
        "draws": args.draws,
        "seed": args.seed,
        "note": (
            "P and K units are provisional until confirmed by IRRI."
        ),
    }

    (out_dir / "settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
