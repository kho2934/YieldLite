"""Run the final test on the locked 2023-2024 panels.

Example
-------
python experiments/run_forward_test.py

The model learns from the 2020-2022 development panels and is then evaluated
on the locked panels. These panels are used only once so that they remain a
fair test of performance on new seasons.

After the test finishes, the script saves a record in ``forward_done.json``.
It will stop any later attempt to run the test again unless ``--rerun`` is
used. That option is only meant for recovering from an interrupted run. If
the model is changed after looking at the results, the holdout can no longer
be treated as an unseen final test.

The results folder will contain:

* ``forward_households.csv``: one row per household and interval level;
* ``forward_summary.csv``: results for each panel and interval level;
* ``forward_done.json``: a record showing when the test was completed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yieldlite
from yieldlite import YieldModel
from yieldlite.forward_test import run_forward_test, summarize_forward
from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout

RECORD_NAME = "forward_done.json"


def _float_list(text: str) -> list[float]:
    """Read comma-separated interval levels from the command line."""
    return [float(part.strip()) for part in text.split(",")]


def check_not_run_before(folder: Path, rerun: bool) -> None:
    """Stop an accidental second use of the locked panels."""
    record_path = folder / RECORD_NAME

    if record_path.is_file() and not rerun:
        previous_run = json.loads(
            record_path.read_text(encoding="utf-8")
        )
        run_time = previous_run.get("run_at", "an earlier date")

        raise SystemExit(
            f"The forward test was already completed on {run_time}. "
            "The locked panels should only be used once. Use --rerun only "
            "if the previous run failed before it could finish."
        )


def main() -> None:
    """Train on the development data and evaluate the locked panels."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument(
        "--data",
        default="data/processed/dataset_final.csv",
        help="path to the cleaned survey dataset",
    )
    parser.add_argument(
        "--out",
        default="experiments/results",
        help="folder where the results will be saved",
    )
    parser.add_argument(
        "--levels",
        type=_float_list,
        default=[0.5, 0.8],
        help="comma-separated interval levels, such as 0.5,0.8",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="repeat the test only after an interrupted run",
    )
    args = parser.parse_args()

    results_folder = Path(args.out)
    check_not_run_before(results_folder, args.rerun)

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form="oxide",
        k_form="oxide",
    )
    development, holdout = lock_holdout(frame)

    print(
        f"Development data: {len(development)} rows across "
        f"{development['panel_id'].nunique()} panels"
    )
    print(
        f"Locked holdout:   {len(holdout)} rows across "
        f"{holdout['panel_id'].nunique()} panels "
        f"({', '.join(sorted(holdout['panel_id'].unique()))})"
    )

    model = YieldModel().fit(development)

    print("\nModel selected from the development data:")
    for explanation in model.explain("en"):
        print(f" - {explanation}")

    households = run_forward_test(
        development,
        holdout,
        levels=args.levels,
    )
    summary = summarize_forward(households)

    results_folder.mkdir(parents=True, exist_ok=True)

    households.to_csv(
        results_folder / "forward_households.csv",
        index=False,
    )
    summary.to_csv(
        results_folder / "forward_summary.csv",
        index=False,
    )

    # Keep a record so the locked data is not used again by accident.
    record = {
        "run_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "yieldlite_version": yieldlite.__version__,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "holdout_panels": sorted(
            holdout["panel_id"].unique().tolist()
        ),
        "rung": model.selection_.rung,
        "levels": args.levels,
    }
    (results_folder / RECORD_NAME).write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    columns = [
        "panel_id",
        "n",
        "n_refused",
        "n_invalid",
        "mae",
        "mae_m0",
        "skill",
        "bias",
        "coverage",
        "width",
        "history_coverage",
    ]

    for level, level_rows in summary.groupby("level"):
        print(f"\nResults for the {level:.0%} interval:")
        print(
            level_rows.loc[:, columns]
            .round(3)
            .to_string(index=False)
        )

    print(f"\nResults saved in {results_folder}")


if __name__ == "__main__":
    main()
