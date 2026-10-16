"""Measure how model accuracy changes when less training data is available.

The study holds out one development panel at a time and trains the models on
smaller samples of the remaining data. It changes two things:

- ``p``: the number of season-year panels used for training
- ``h``: the maximum number of household rows kept in each district-season cell

Every model is evaluated on a panel it did not see during training. This module
only collects the results. Model-selection thresholds will be chosen later,
after the results have been reviewed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from yieldlite.degradation import cap_households_per_cell, sample_panel_subsets
from yieldlite.features import VarietyPooler
from yieldlite.models import SeasonDistrictTable, build_m1

TARGET = "yield_t_ha"

RESULT_COLUMNS = (
    "held_out",
    "p",
    "draw",
    "h",
    "model",
    "n_train",
    "n_test",
    "mae",
    "bias",
)

_REQUIRED_COLUMNS = (
    "panel_id",
    "season",
    "district_code",
    "variety_raw",
    "variety_flag",
    "npk_total_kg_ha",
    TARGET,
)


def _estimators() -> dict[str, object]:
    """Return the model templates used in the study."""
    return {
        "m0": SeasonDistrictTable(),
        "m1": build_m1(),
    }


def _check_development_rows(dev: pd.DataFrame) -> pd.Series:
    """Check the development data and return one season label per panel."""
    if not isinstance(dev, pd.DataFrame):
        raise TypeError("dev must be a pandas DataFrame")

    if dev.empty:
        raise ValueError("dev cannot be empty")

    if not dev.columns.is_unique:
        raise ValueError("dev must have unique column names")

    if not dev.index.is_unique:
        raise ValueError("dev must use a unique row index")

    missing = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in dev.columns
    ]
    if missing:
        raise ValueError(f"dev is missing columns: {missing}")

    if dev.loc[:, list(_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required columns cannot contain missing values")

    seasons_per_panel = dev.groupby("panel_id")["season"].nunique()
    if (seasons_per_panel > 1).any():
        raise ValueError("each panel must belong to exactly one season")

    return dev.groupby("panel_id")["season"].first()


def run_degradation(
    dev: pd.DataFrame,
    p_values: Iterable[int],
    h_values: Iterable[int | None],
    n_draws: int = 20,
    seed: int = 0,
    models: Sequence[str] = ("m0", "m1"),
    min_variety_count: int = 30,
) -> pd.DataFrame:
    """Run the degradation study on the development panels.

    Parameters
    ----------
    dev : pandas.DataFrame
        Development data only. The locked final holdout must not be included.
        The table needs ``panel_id``, ``season``, ``district_code``,
        ``variety_raw``, ``variety_flag``, ``npk_total_kg_ha``, and
        ``yield_t_ha``.
    p_values : iterable of int
        Numbers of training panels to test. Each value must leave at least one
        panel out and retain at least one panel from every season.
    h_values : iterable of int or None
        Maximum household rows kept in each district-season cell. ``None``
        keeps every available row.
    n_draws : int, default 20
        Maximum number of panel samples for each held-out panel and value of
        ``p``. If fewer valid samples exist, all of them are used.
    seed : int, default 0
        Random seed used to make the study reproducible.
    models : sequence of {"m0", "m1"}, default ("m0", "m1")
        Models included in the study.
    min_variety_count : int, default 30
        Minimum number of training rows required for a variety to keep its own
        category instead of being pooled.

    Returns
    -------
    pandas.DataFrame
        One result for each held-out panel, panel draw, household cap, and
        model. An ``h`` value of infinity means no household cap was applied.
        Bias is calculated as predicted yield minus observed yield.
    """
    panel_season = _check_development_rows(dev)

    estimators = _estimators()
    unknown_models = [
        name
        for name in models
        if name not in estimators
    ]
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")

    p_list = [int(p) for p in p_values]
    h_list = list(h_values)
    if not p_list or not h_list:
        raise ValueError("p_values and h_values cannot be empty")

    n_seasons = int(panel_season.nunique())
    n_panels = len(panel_season)

    bad_p = [
        p
        for p in p_list
        if not n_seasons <= p <= n_panels - 1
    ]
    if bad_p:
        raise ValueError(
            f"p values must be between {n_seasons} (one per season) and "
            f"{n_panels - 1} (all panels except the held-out panel), "
            f"got {bad_p}"
        )

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for held_out in sorted(panel_season.index):
        pool_season = panel_season.drop(held_out)
        test = dev.loc[dev["panel_id"] == held_out]

        for p in p_list:
            subsets = sample_panel_subsets(
                pool_season,
                p,
                n_draws,
                rng,
            )

            for draw, subset in enumerate(subsets):
                subset_rows = dev.loc[dev["panel_id"].isin(subset)]

                for h in h_list:
                    train = cap_households_per_cell(
                        subset_rows,
                        h,
                        rng,
                    )

                    # Variety support is learned from the current training
                    # sample only, so the held-out panel cannot influence it.
                    pooler = VarietyPooler(min_variety_count).fit(
                        train["variety_raw"],
                        train["variety_flag"],
                    )

                    x_train = train.assign(
                        variety_pooled=pooler.transform(
                            train["variety_raw"],
                            train["variety_flag"],
                        )
                    )
                    x_test = test.assign(
                        variety_pooled=pooler.transform(
                            test["variety_raw"],
                            test["variety_flag"],
                        )
                    )

                    for name in models:
                        estimator = clone(estimators[name])
                        estimator.fit(x_train, train[TARGET])

                        predicted = np.asarray(
                            estimator.predict(x_test),
                            dtype=float,
                        )
                        observed = test[TARGET].to_numpy(dtype=float)
                        errors = predicted - observed

                        records.append(
                            {
                                "held_out": held_out,
                                "p": p,
                                "draw": draw,
                                "h": np.inf if h is None else float(h),
                                "model": name,
                                "n_train": len(train),
                                "n_test": len(test),
                                "mae": float(np.abs(errors).mean()),
                                "bias": float(errors.mean()),
                            }
                        )

    return pd.DataFrame.from_records(
        records,
        columns=list(RESULT_COLUMNS),
    )


def summarize_degradation(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize results across draws and held-out panels.

    Results are first averaged across repeated draws for each held-out panel.
    They are then averaged across panels so that every panel has equal weight,
    regardless of how many household rows it contains.

    ``skill_vs_m0`` is calculated as ``1 - MAE / MAE_M0`` for each panel
    before being averaged. It is undefined for M0 itself and whenever the M0
    error is zero.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from :func:`run_degradation`.

    Returns
    -------
    pandas.DataFrame
        One row for each model, value of ``p``, and value of ``h``. The output
        includes mean MAE, MAE spread across panels, mean bias, skill over M0,
        and the number of held-out panels.
    """
    missing = [
        column
        for column in RESULT_COLUMNS
        if column not in results.columns
    ]
    if missing:
        raise ValueError(f"results is missing columns: {missing}")

    if results.empty:
        raise ValueError("results cannot be empty")

    keys = ["held_out", "p", "h"]

    # Average repeated draws before combining different held-out panels.
    per_panel = results.groupby(
        ["model", *keys],
        as_index=False,
    )[["mae", "bias"]].mean()

    baseline = per_panel.loc[
        per_panel["model"] == "m0",
        [*keys, "mae"],
    ].rename(columns={"mae": "mae_m0"})

    per_panel = per_panel.merge(
        baseline,
        on=keys,
        how="left",
    )

    usable_baseline = (
        per_panel["mae_m0"].notna()
        & (per_panel["mae_m0"] > 0)
    )
    skill = (
        1.0
        - per_panel["mae"]
        / per_panel["mae_m0"].where(usable_baseline)
    )
    per_panel["skill_vs_m0"] = skill.where(
        per_panel["model"] != "m0"
    )

    summary = per_panel.groupby(
        ["model", "p", "h"],
        as_index=False,
    ).agg(
        mae=("mae", "mean"),
        mae_sd=("mae", "std"),
        bias=("bias", "mean"),
        skill_vs_m0=("skill_vs_m0", "mean"),
        n_panels=("held_out", "nunique"),
    )

    return (
        summary.sort_values(["model", "p", "h"])
        .reset_index(drop=True)
    )
