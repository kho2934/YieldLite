"""Run the prediction-interval study on the development panels.

Run from the project root with:

    python experiments/interval_study.py

The locked 2023–2024 holdout is removed before any model is fitted and is
never evaluated by this script. Detailed results, summarized results, and the
study settings are saved separately so figures and thresholds can be created
without rerunning the full study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yieldlite.interval_study import (
    run_interval_study,
    summarize_intervals,
)
from yieldlite.io.final_dataset import (
    load_final_dataset,
)
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout


def _int_list(text: str) -> list[int]:
    """Read a comma-separated list of integers."""
    return [
        int(part)
        for part in text.split(",")
    ]


def _float_list(text: str) -> list[float]:
    """Read a comma-separated list of decimal values."""
    return [
        float(part)
        for part in text.split(",")
    ]


def _cap_list(
    text: str,
) -> list[int | None]:
    """Read household caps, using ``all`` to mean no cap."""
    return [
        (
            None
            if part.strip().lower() == "all"
            else int(part)
        )
        for part in text.split(",")
    ]


def main() -> None:
    """Load the data, run the study, and save its results."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--data",
        default="data/processed/dataset_final.csv",
        help="path to the cleaned survey dataset",
    )
    parser.add_argument(
        "--p-form",
        default="oxide",
        choices=[
            "elemental",
            "oxide",
        ],
        help=(
            "form used for phosphorus "
            "(IRRI confirmed oxide)"
        ),
    )
    parser.add_argument(
        "--k-form",
        default="oxide",
        choices=[
            "elemental",
            "oxide",
        ],
        help=(
            "form used for potassium "
            "(IRRI confirmed oxide)"
        ),
    )
    parser.add_argument(
        "--p-values",
        type=_int_list,
        default=[
            3,
            4,
            5,
            6,
            7,
        ],
        help=(
            "comma-separated numbers of "
            "training panels"
        ),
    )
    parser.add_argument(
        "--h-values",
        type=_cap_list,
        default=[
            5,
            10,
            20,
            None,
        ],
        help=(
            "comma-separated households per cell; "
            "use 'all' for no cap"
        ),
    )
    parser.add_argument(
        "--levels",
        type=_float_list,
        default=[
            0.5,
            0.8,
        ],
        help=(
            "comma-separated prediction-interval "
            "levels to evaluate"
        ),
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=10,
        help="maximum random panel draws per setting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--out",
        default="experiments/results",
        help="folder where study results will be saved",
    )
    args = parser.parse_args()

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form=args.p_form,
        k_form=args.k_form,
    )

    # Keep the final holdout completely separate from this study.
    development, _ = lock_holdout(frame)

    print(
        f"development rows: {len(development)}, "
        f"panels: {development['panel_id'].nunique()}"
    )

    results = run_interval_study(
        development,
        p_values=args.p_values,
        h_values=args.h_values,
        levels=args.levels,
        n_draws=args.draws,
        seed=args.seed,
        progress=print,
    )
    summary = summarize_intervals(results)

    output_folder = Path(args.out)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_folder / "interval_results.csv",
        index=False,
    )
    summary.to_csv(
        output_folder / "interval_summary.csv",
        index=False,
    )

    # Save enough information to reproduce and explain this study run.
    settings = {
        "p_form": args.p_form,
        "k_form": args.k_form,
        "p_values": args.p_values,
        "h_values": [
            "all"
            if cap is None
            else cap
            for cap in args.h_values
        ],
        "levels": args.levels,
        "draws": args.draws,
        "seed": args.seed,
        "note": (
            "N, P2O5 and K2O are nutrient mass, "
            "confirmed by IRRI on 2026-10-22."
        ),
    }

    settings_path = (
        output_folder
        / "interval_settings.json"
    )
    settings_path.write_text(
        json.dumps(
            settings,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Show the highest requested level in the terminal summary.
    level = max(args.levels)
    shown = (
        summary[
            summary["level"] == level
        ]
        .sort_values(
            [
                "model",
                "h",
                "p",
            ]
        )
        .copy()
    )

    shown["h"] = shown["h"].map(
        lambda cap: (
            "all"
            if cap == float("inf")
            else f"{cap:.0f}"
        )
    )

    interval_columns = [
        "model",
        "h",
        "p",
        "supported",
        "coverage",
        "coverage_se",
        "width",
        "ref_width",
        "informative",
    ]

    print(f"\nlevel {level:.0%}")
    print(
        shown.loc[
            :,
            interval_columns,
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    # Historical ranges use the same training rows for every model. Showing
    # one model is therefore enough and avoids printing duplicate rows.
    first_model = shown["model"].iloc[0]
    history_summary = shown[
        shown["model"] == first_model
    ]

    history_columns = [
        "h",
        "p",
        "assessable",
        "history_coverage",
        "history_coverage_se",
        "ref_width",
    ]

    print(
        "\nusual historical yield range, "
        f"{level:.0%} "
        "(same for every model)"
    )
    print(
        history_summary.loc[
            :,
            history_columns,
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print(
        f"\nwrote results to {output_folder}"
    )


if __name__ == "__main__":
    main()
