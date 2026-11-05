"""Refit the final model using all 12 available panels.

Example
-------
python experiments/fit_final.py

The one-time forward test is already complete, so the former holdout panels
can now be added back to the development data. This script fits YieldModel
using the full dataset and reports which model rung the selection rule chooses,
along with its explanation in English and Vietnamese.

This is a final refit, not another evaluation, because no unseen panels remain.
A short record of the fit is saved as ``final_fit.json``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yieldlite
from yieldlite import YieldModel
from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame


def main() -> None:
    """Fit YieldModel on the full dataset and record its final choice."""
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
        help="folder where the fit record will be saved",
    )
    args = parser.parse_args()

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form="oxide",
        k_form="oxide",
    )
    panels = sorted(frame["panel_id"].unique().tolist())

    print(
        f"Fitting the final model on {len(frame)} rows "
        f"from {len(panels)} panels"
    )

    model = YieldModel().fit(frame)

    for language in ("en", "vi"):
        print(f"\nModel explanation ({language}):")
        for explanation in model.explain(language):
            print(f" - {explanation}")

    fit_record = {
        "fitted_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "yieldlite_version": yieldlite.__version__,
        "rows": len(frame),
        "panels": panels,
        "rung": model.selection_.rung,
    }

    results_folder = Path(args.out)
    results_folder.mkdir(parents=True, exist_ok=True)

    record_path = results_folder / "final_fit.json"
    record_path.write_text(
        json.dumps(fit_record, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinal fit record saved to {record_path}")


if __name__ == "__main__":
    main()
