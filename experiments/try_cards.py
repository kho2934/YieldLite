"""Draw prediction cards from the fitted model and save them as images.

Fits YieldModel on the development panels (the locked 2023-2024 panels stay
out of sight), then draws four cards for one typical household, each in
English and Vietnamese:

* ok: an ordinary prediction.
* flagged: the same household with a variety the model has never seen.
* refused_width: the same household with the interval-width rule switched on.
* refused_district: a district that is not in the training data.

The images go to a folder outside the repository by default, so nothing from
this script can be committed by accident.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from yieldlite import YieldModel, plot_prediction
from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout


def typical_household(development: pd.DataFrame) -> dict:
    """Return the most common district and season with a typical household."""
    district, season = (
        development.groupby(["district_code", "season"])
        .size()
        .idxmax()
    )
    cell = development[
        (development["district_code"] == district)
        & (development["season"] == season)
    ]

    return {
        "district": district,
        "season": season,
        "variety": cell["variety_raw"].mode().iloc[0],
        "npk_total": float(cell["npk_total_kg_ha"].median()),
    }


def draw_cards(development: pd.DataFrame, out_dir: Path, level: float) -> list[Path]:
    """Draw the four example cards in both languages and return the paths."""
    household = typical_household(development)
    default_model = YieldModel().fit(development)
    strict_model = YieldModel(refuse_if_uninformative=True).fit(development)

    cards = {
        "ok": default_model.predict(level=level, **household),
        "flagged": default_model.predict(
            level=level, **{**household, "variety": "UNSEEN_VARIETY"}
        ),
        "refused_width": strict_model.predict(level=level, **household),
        "refused_district": default_model.predict(
            level=level, **{**household, "district": "XX"}
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for name, result in cards.items():
        for lang in ("en", "vi"):
            path = out_dir / f"card_{name}_{lang}.png"
            plot_prediction(result, lang).savefig(path, dpi=150)
            paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/processed/dataset_final.csv")
    parser.add_argument("--level", type=float, default=0.8)
    parser.add_argument(
        "--out",
        default=str(Path(tempfile.gettempdir()) / "yieldlite_cards"),
    )
    args = parser.parse_args()

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form="oxide",
        k_form="oxide",
    )
    development, _ = lock_holdout(frame)

    for path in draw_cards(development, Path(args.out), args.level):
        print(path)


if __name__ == "__main__":
    main()
