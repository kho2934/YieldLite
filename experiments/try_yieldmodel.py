"""Fit YieldModel on the development panels and look at what it returns.

Reads the cleaned survey table, keeps the locked 2023-2024 panels out of
sight, fits the model, and predicts one typical household for every district
and season: its most common variety at the cell's median fertilizer amount.
Nothing is saved; the point is to read the output.
"""

from __future__ import annotations

import argparse

import pandas as pd

from yieldlite import YieldModel
from yieldlite.io.final_dataset import load_final_dataset
from yieldlite.prepare import prepare_frame
from yieldlite.split import lock_holdout


def typical_households(development: pd.DataFrame) -> pd.DataFrame:
    """Return one typical household for every district and season."""
    rows = []

    for (district, season), cell in development.groupby(["district_code", "season"]):
        rows.append(
            {
                "district": district,
                "season": season,
                "variety": cell["variety_raw"].mode().iloc[0],
                "npk_total": float(cell["npk_total_kg_ha"].median()),
                "n_rows": len(cell),
            }
        )

    return pd.DataFrame(rows)


def predict_all(model: YieldModel, households: pd.DataFrame, level: float) -> pd.DataFrame:
    """Predict every typical household and tabulate the results."""
    rows = []

    for row in households.itertuples(index=False):
        result = model.predict(
            district=row.district,
            season=row.season,
            variety=row.variety,
            npk_total=row.npk_total,
            level=level,
        )

        history_width = None
        if result.history is not None:
            history_width = result.history.upper - result.history.lower

        interval_width = None
        if result.interval is not None:
            interval_width = result.interval.upper - result.interval.lower

        rows.append(
            {
                "district": row.district,
                "season": row.season,
                "n_rows": row.n_rows,
                "status": result.status,
                "estimate": result.estimate,
                "interval_width": interval_width,
                "history_width": history_width,
                "reasons": ",".join(reason.code for reason in result.reasons),
            }
        )

    return pd.DataFrame(rows)


def show(development: pd.DataFrame, level: float) -> None:
    """Print the model's decisions, then the table for both width rules."""
    households = typical_households(development)

    for refuse in (True, False):
        model = YieldModel(refuse_if_uninformative=refuse).fit(development)

        print()
        print("=" * 72)
        print(f"refuse_if_uninformative={refuse}, level={level:.0%}")
        print("=" * 72)

        if refuse:
            print("\nWhy this model (English):")
            for line in model.explain("en"):
                print(" -", line)

            print("\nVì sao chọn mô hình này (Tiếng Việt):")
            for line in model.explain("vi"):
                print(" -", line)

        table = predict_all(model, households, level)
        print()
        print(table.round(2).to_string(index=False))

        shares = table["status"].value_counts(normalize=True).round(2).to_dict()
        print("\nshare of typical households by status:", shares)

        if refuse:
            first = table.index[table["status"] == "refused"]
            example = households.iloc[first[0] if len(first) else 0]
        else:
            first = table.index[table["status"] != "refused"]
            example = households.iloc[first[0] if len(first) else 0]

        result = model.predict(
            district=example.district,
            season=example.season,
            variety=example.variety,
            npk_total=example.npk_total,
            level=level,
        )
        print(f"\nexample: {example.district} {example.season} -> {result.status}")
        for lang in ("en", "vi"):
            for line in result.summary(lang):
                print(f"  [{lang}] {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/processed/dataset_final.csv")
    parser.add_argument("--level", type=float, default=0.8)
    args = parser.parse_args()

    frame = prepare_frame(
        load_final_dataset(args.data),
        p_form="oxide",
        k_form="oxide",
    )
    development, _ = lock_holdout(frame)
    print(
        f"development rows: {len(development)}, "
        f"panels: {development['panel_id'].nunique()}"
    )

    show(development, args.level)


if __name__ == "__main__":
    main()
