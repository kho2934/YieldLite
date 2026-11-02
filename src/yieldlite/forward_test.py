"""Evaluate the finished model on the locked holdout panels.

The model is fitted using only the development panels. Each holdout household
is then passed through ``YieldModel.predict`` in the same way a real user
would request a prediction. Nothing is adjusted after seeing these results,
so this test represents performance on genuinely unseen seasons.

``run_forward_test`` keeps one row for every household and interval level,
including refusals and invalid inputs. ``summarize_forward`` turns those rows
into panel-level results and gives every panel equal weight in the final
average.

M0 is evaluated on the same households, allowing the selected model to be
compared fairly with the basic season-and-district average.

This module only performs the evaluation. The calling script is responsible
for making sure the locked holdout is not used more than once.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from yieldlite.ladder import build_rung
from yieldlite.yieldmodel import YieldModel

_REQUIRED_COLUMNS = (
    "panel_id",
    "season",
    "district_code",
    "variety_raw",
    "variety_flag",
    "npk_total_kg_ha",
    "flagged",
    "yield_t_ha",
)

# These flags identify entries that contain a usable variety name.
_NAMED_VARIETY_FLAGS = {"known", "other"}

HOUSEHOLD_COLUMNS = (
    "panel_id",
    "district",
    "season",
    "level",
    "status",
    "reasons",
    "observed",
    "estimate",
    "lower",
    "upper",
    "history_lower",
    "history_upper",
    "m0_estimate",
)


def _check_frame(df: pd.DataFrame, name: str) -> None:
    """Check that one side of the split has the columns we need."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = [
        column for column in _REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if df.empty:
        raise ValueError(f"{name} cannot be empty")


def _check_levels(levels: Iterable[float]) -> list[float]:
    """Turn the requested interval levels into a validated list."""
    checked = [float(level) for level in levels]

    if not checked:
        raise ValueError("levels cannot be empty")

    if any(
        not np.isfinite(level) or not 0 < level < 1
        for level in checked
    ):
        raise ValueError(
            "every level must be a finite number strictly between 0 and 1"
        )

    return checked


def run_forward_test(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    levels: Iterable[float] = (0.5, 0.8),
    min_variety_count: int = 30,
    refuse_if_uninformative: bool = False,
) -> pd.DataFrame:
    """Fit on the development panels and predict every holdout household.

    Parameters
    ----------
    development : pandas.DataFrame
        Panels available for training. The model learns only from these rows.
    holdout : pandas.DataFrame
        Locked panels that remain unseen until this evaluation.
    levels : iterable of float, default (0.5, 0.8)
        Prediction-interval levels to request. If the training data cannot
        support a level, that prediction is kept as a refusal.
    min_variety_count : int, default 30
        Minimum number of training rows needed for a variety to keep its own
        category. Passed to both YieldModel and M0.
    refuse_if_uninformative : bool, default False
        Whether YieldModel should refuse when its interval is no narrower
        than the usual historical range.

    Returns
    -------
    pandas.DataFrame
        One row per holdout household and interval level. Refused or invalid
        predictions remain in the table, but their estimate and interval
        columns are missing. A row is marked ``invalid`` when its input
        values cannot be used, such as when fertilizer information is
        missing.
    """
    _check_frame(development, "development")
    _check_frame(holdout, "holdout")
    levels = _check_levels(levels)

    overlapping_panels = set(development["panel_id"]) & set(
        holdout["panel_id"]
    )
    if overlapping_panels:
        raise ValueError(
            "development and holdout share panels: "
            f"{sorted(overlapping_panels)}"
        )

    model = YieldModel(
        min_variety_count=min_variety_count,
        refuse_if_uninformative=refuse_if_uninformative,
    ).fit(development)

    # Score M0 on exactly the same holdout rows used for YieldModel.
    baseline = build_rung(
        "m0",
        min_variety_count=min_variety_count,
    )
    baseline.fit(development, development["yield_t_ha"])
    m0_predictions = baseline.predict(holdout).to_numpy(dtype=float)

    records = []

    for position, household in enumerate(
        holdout.itertuples(index=False)
    ):
        has_named_variety = (
            household.variety_flag in _NAMED_VARIETY_FLAGS
        )
        variety = household.variety_raw if has_named_variety else None

        for level in levels:
            record = {
                "panel_id": household.panel_id,
                "district": household.district_code,
                "season": household.season,
                "level": level,
                "status": "invalid",
                "reasons": "",
                "observed": float(household.yield_t_ha),
                "estimate": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "history_lower": np.nan,
                "history_upper": np.nan,
                "m0_estimate": m0_predictions[position],
            }

            try:
                result = model.predict(
                    district=household.district_code,
                    season=household.season,
                    variety=variety,
                    npk_total=household.npk_total_kg_ha,
                    level=level,
                )
            except ValueError:
                # Keep the household in the results even when its input
                # cannot be used. This makes the invalid-input rate visible.
                records.append(record)
                continue

            record["status"] = result.status
            record["reasons"] = ",".join(
                reason.code for reason in result.reasons
            )

            if result.interval is not None:
                record["estimate"] = result.estimate
                record["lower"] = result.interval.lower
                record["upper"] = result.interval.upper

            if result.history is not None:
                record["history_lower"] = result.history.lower
                record["history_upper"] = result.history.upper

            records.append(record)

    return pd.DataFrame(
        records,
        columns=list(HOUSEHOLD_COLUMNS),
    )


def _score_group(group: pd.DataFrame) -> pd.Series:
    """Calculate the results for one panel at one interval level."""
    scored = group.dropna(subset=["estimate"])
    n_scored = len(scored)

    metrics = {
        "n": len(group),
        "n_refused": int((group["status"] == "refused").sum()),
        "n_invalid": int((group["status"] == "invalid").sum()),
        "n_scored": n_scored,
        "refusal_rate": float(
            (group["status"] == "refused").mean()
        ),
        "mae": np.nan,
        "mae_m0": np.nan,
        "skill": np.nan,
        "bias": np.nan,
        "coverage": np.nan,
        "width": np.nan,
        "history_coverage": np.nan,
    }

    if n_scored:
        absolute_error = (
            scored["estimate"] - scored["observed"]
        ).abs()
        m0_absolute_error = (
            scored["m0_estimate"] - scored["observed"]
        ).abs()

        metrics["mae"] = float(absolute_error.mean())
        metrics["mae_m0"] = float(m0_absolute_error.mean())

        # Skill is undefined when M0 makes no error because there is no
        # baseline error available for the selected model to improve upon.
        if metrics["mae_m0"] > 0:
            metrics["skill"] = (
                1 - metrics["mae"] / metrics["mae_m0"]
            )

        metrics["bias"] = float(
            (scored["estimate"] - scored["observed"]).mean()
        )
        metrics["coverage"] = float(
            (
                (scored["observed"] >= scored["lower"])
                & (scored["observed"] <= scored["upper"])
            ).mean()
        )
        metrics["width"] = float(
            (scored["upper"] - scored["lower"]).mean()
        )

    with_history = group.dropna(
        subset=["history_lower", "history_upper"]
    )

    if not with_history.empty:
        metrics["history_coverage"] = float(
            (
                (
                    with_history["observed"]
                    >= with_history["history_lower"]
                )
                & (
                    with_history["observed"]
                    <= with_history["history_upper"]
                )
            ).mean()
        )

    return pd.Series(metrics)


def summarize_forward(households: pd.DataFrame) -> pd.DataFrame:
    """Summarize the forward test by panel and interval level.

    Panels are summarized separately because season-year panels, rather than
    individual households, are the independent units in this dataset. The
    final ``mean of panels`` row gives every panel equal weight, even when
    some panels contain more households than others.

    Parameters
    ----------
    households : pandas.DataFrame
        Household-level results returned by :func:`run_forward_test`.

    Returns
    -------
    pandas.DataFrame
        One row for each panel and interval level, followed by an equally
        weighted panel average. The table reports:

        * the number of total, refused, invalid and scored households;
        * refusal rate, MAE and bias;
        * M0 MAE and skill relative to M0;
        * prediction-interval coverage and width;
        * coverage of the usual historical yield range.
    """
    if not isinstance(households, pd.DataFrame):
        raise TypeError("households must be a pandas DataFrame")

    missing = [
        column
        for column in HOUSEHOLD_COLUMNS
        if column not in households.columns
    ]
    if missing:
        raise ValueError(f"households is missing columns: {missing}")

    if households.empty:
        raise ValueError("households cannot be empty")

    panel_rows = []

    for (level, panel_id), group in households.groupby(
        ["level", "panel_id"],
        sort=True,
    ):
        panel_rows.append(
            {
                "level": level,
                "panel_id": panel_id,
                **_score_group(group).to_dict(),
            }
        )

    per_panel = pd.DataFrame(panel_rows)

    score_columns = [
        "refusal_rate",
        "mae",
        "mae_m0",
        "skill",
        "bias",
        "coverage",
        "width",
        "history_coverage",
    ]
    count_columns = [
        "n",
        "n_refused",
        "n_invalid",
        "n_scored",
    ]

    # Average performance across panels instead of across households. This
    # prevents a large panel from having more influence than a small one.
    averages = per_panel.groupby(
        "level",
        as_index=False,
    )[score_columns].mean()

    totals = per_panel.groupby(
        "level",
        as_index=False,
    )[count_columns].sum()

    averages = totals.merge(averages, on="level")
    averages["panel_id"] = "mean of panels"

    table = pd.concat(
        [per_panel, averages],
        ignore_index=True,
    )
    table["_is_mean"] = table["panel_id"] == "mean of panels"

    return (
        table.sort_values(["level", "_is_mean", "panel_id"])
        .drop(columns="_is_mean")
        .astype({column: int for column in count_columns})
        .reset_index(drop=True)
    )
