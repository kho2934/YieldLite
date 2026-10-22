"""Run the interval study on the development panels.

Example
-------
python experiments/interval_study.py

The locked 2023-2024 holdout is removed before anything is fitted and is
never evaluated here. Results are written as CSV files so that the figures
and thresholds can be produced from them separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yieldlite.interval_study import run_interval_study, summarize_intervals
from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",")]


def _cap_list(text: str) -> list[int | None]:
    return [
        None if part.strip().lower() == "all" else int(part) for part in text.split(",")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        default="data/processed/dataset_final.csv",
        help="path to the cleaned survey table",
    )
    parser.add_argument(
        "--p-form",
        default="oxide",
        choices=["elemental", "oxide"],
        help="how phosphorus was recorded (IRRI confirmed oxide)",
    )
    parser.add_argument(
        "--k-form",
        default="oxide",
        choices=["elemental", "oxide"],
        help="how potassium was recorded (IRRI confirmed oxide)",
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
        help="comma-separated households per cell; use 'all' for no cap",
    )
    parser.add_argument(
        "--levels",
        type=_float_list,
        default=[0.5, 0.8],
        help="comma-separated interval levels to score",
    )
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="experiments/results")
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "interval_results.csv", index=False)
    summary.to_csv(out_dir / "interval_summary.csv", index=False)
    settings = {
        "p_form": args.p_form,
        "k_form": args.k_form,
        "p_values": args.p_values,
        "h_values": ["all" if h is None else h for h in args.h_values],
        "levels": args.levels,
        "draws": args.draws,
        "seed": args.seed,
        "note": "N, P2O5 and K2O are nutrient mass, confirmed by IRRI on 2026-10-22.",
    }
    (out_dir / "interval_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    level = max(args.levels)
    shown = summary[summary["level"] == level].sort_values(["model", "h", "p"])
    shown = shown.assign(
        h=shown["h"].map(lambda h: "all" if h == float("inf") else f"{h:.0f}")
    )
    columns = [
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
        shown.loc[:, columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
