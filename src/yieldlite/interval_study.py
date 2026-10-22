"""Study how prediction intervals behave as the training data gets smaller.

This is the uncertainty version of the model degradation study. One
development panel is held out at a time, while each model is trained and
calibrated on progressively smaller samples of the remaining panels. The
resulting intervals are then evaluated on the households in the held-out
panel.

The study records three main measurements:

* ``coverage``: how often the observed yield falls inside the interval;
* ``width``: the average size of the interval;
* ``informative``: the share of households whose interval is narrower than
  the usual yield range for their district and season.

The last measurement matters because an interval that is as wide as the
historical yield range does not tell a farmer much more than the past data
already showed.

With ``p`` training panels, the highest interval level that can remain finite
is ``p / (p + 1)``. See ``yieldlite.conformal`` for the explanation. When a
requested level is higher than this limit, the study records it as unsupported
instead of trying to score an invalid interval.

This module only measures interval behavior. The decision about how these
results should determine the ``P0`` threshold is made separately after the
study has been reviewed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from yieldlite.conformal import (
    LevelUnsupportedError,
    PanelCVPlus,
    interval_coverage,
)
from yieldlite.degradation import (
    cap_households_per_cell,
    sample_panel_subsets,
)
from yieldlite.history import typical_range
from yieldlite.ladder import RUNGS, build_rung
from yieldlite.study import TARGET, _check_development_rows

INTERVAL_COLUMNS = (
    "held_out",
    "p",
    "draw",
    "h",
    "model",
    "level",
    "supported",
    "n_train",
    "n_test",
    "coverage",
    "width",
    "ref_width",
    "informative",
)


def _check_levels(levels: Iterable[float]) -> list[float]:
    """Check the requested interval levels and return them as floats."""
    level_list = [float(level) for level in levels]

    if not level_list:
        raise ValueError("levels cannot be empty")

    if any(
        not 0 < level < 1
        for level in level_list
    ):
        raise ValueError(
            "every interval level must be strictly between 0 and 1"
        )

    if len(set(level_list)) != len(level_list):
        raise ValueError("interval levels cannot be repeated")

    return level_list


def _score_interval(
    intervals: PanelCVPlus,
    test: pd.DataFrame,
    observed: np.ndarray,
    level: float,
    usual: np.ndarray,
) -> dict[str, object]:
    """Score one interval level on the held-out households.

    If the fitted model does not have enough panels to support the requested
    level, the result is marked as unsupported and the four interval
    measurements are returned as ``NaN``.
    """
    try:
        lower, upper = intervals.predict_interval(
            test,
            level,
        )
    except LevelUnsupportedError:
        return {
            "supported": False,
            "coverage": np.nan,
            "width": np.nan,
            "ref_width": np.nan,
            "informative": np.nan,
        }

    widths = upper - lower

    return {
        "supported": True,
        "coverage": interval_coverage(
            observed,
            lower,
            upper,
        ),
        "width": float(np.mean(widths)),
        "ref_width": float(np.mean(usual)),
        "informative": float(
            np.mean(widths < usual)
        ),
    }


def run_interval_study(
    dev: pd.DataFrame,
    p_values: Iterable[int],
    h_values: Iterable[int | None],
    levels: Iterable[float] = (0.5, 0.8),
    n_draws: int = 10,
    seed: int = 0,
    models: Sequence[str] = RUNGS,
    min_variety_count: int = 30,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Evaluate prediction intervals while reducing the training data.

    For each development panel, this function holds that panel out and builds
    interval models from subsets of the remaining panels. It varies both the
    number of training panels and the maximum number of households retained
    in each district-season cell.

    Parameters
    ----------
    dev : pandas.DataFrame
        Development data only. The locked final holdout must not be included.
        The table needs ``panel_id``, ``season``, ``district_code``,
        ``variety_raw``, ``variety_flag``, ``npk_total_kg_ha``, and
        ``yield_t_ha``.
    p_values : iterable of int
        Numbers of training panels to test. Each value must be at least the
        number of seasons and no larger than the number of available panels
        minus one.
    h_values : iterable of int or None
        Maximum number of households to keep in each district-season cell.
        ``None`` keeps every available household.
    levels : iterable of float, default (0.5, 0.8)
        Nominal interval levels to evaluate.
    n_draws : int, default 10
        Maximum number of randomly sampled panel sets for each held-out panel
        and value of ``p``. Fewer draws are used when fewer valid panel sets
        exist.
    seed : int, default 0
        Random seed used to make the study reproducible.
    models : sequence of str, default all registered rungs
        Names of the models to evaluate. Each name must appear in
        ``yieldlite.ladder.RUNGS``.
    min_variety_count : int, default 30
        Minimum number of training rows required for a variety to keep its
        own category.
    progress : callable, optional
        Function that receives a short update after each held-out panel has
        been completed. Passing ``print`` displays progress during the study.

    Returns
    -------
    pandas.DataFrame
        One row for every held-out panel, panel count, random draw, household
        cap, model, and interval level. The returned columns follow
        ``INTERVAL_COLUMNS``.

        ``h`` is stored as infinity when no household cap is used.
        ``ref_width`` is the average historical yield range used as the
        comparison point. ``informative`` is the share of held-out households
        whose prediction interval is narrower than that range.

        When a panel count cannot support the requested interval level,
        ``supported`` is false and the four interval scores are ``NaN``.
    """
    panel_season = _check_development_rows(dev)

    unknown_models = [
        name
        for name in models
        if name not in RUNGS
    ]
    if unknown_models:
        raise ValueError(
            f"unknown models: {unknown_models}"
        )

    level_list = _check_levels(levels)

    p_list = [int(p) for p in p_values]
    h_list = list(h_values)

    if not p_list or not h_list:
        raise ValueError(
            "p_values and h_values cannot be empty"
        )

    n_seasons = int(panel_season.nunique())
    n_panels = len(panel_season)

    invalid_panel_counts = [
        p
        for p in p_list
        if not n_seasons <= p <= n_panels - 1
    ]
    if invalid_panel_counts:
        raise ValueError(
            f"p values must be between {n_seasons} "
            "(at least one panel per season) and "
            f"{n_panels - 1} "
            "(all available panels except the held-out panel); "
            f"received {invalid_panel_counts}"
        )

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    held_out_panels = sorted(panel_season.index)

    for completed, held_out in enumerate(
        held_out_panels,
        start=1,
    ):
        remaining_panel_seasons = panel_season.drop(
            held_out
        )
        test = dev.loc[
            dev["panel_id"] == held_out
        ]
        observed = test[TARGET].to_numpy(
            dtype=float
        )

        for p in p_list:
            subsets = sample_panel_subsets(
                remaining_panel_seasons,
                p,
                n_draws,
                rng,
            )

            for draw, subset in enumerate(subsets):
                subset_rows = dev.loc[
                    dev["panel_id"].isin(subset)
                ]

                for h in h_list:
                    train = cap_households_per_cell(
                        subset_rows,
                        h,
                        rng,
                    )

                    # Calculate the historical comparison range separately
                    # for every interval level.
                    usual_ranges = {
                        level: typical_range(
                            train,
                            test,
                            level,
                        )
                        for level in level_list
                    }

                    for model_name in models:
                        pipeline = build_rung(
                            model_name,
                            min_variety_count,
                        )

                        intervals = PanelCVPlus(
                            pipeline
                        ).fit(
                            train,
                            train[TARGET],
                            train["panel_id"],
                        )

                        for level in level_list:
                            scores = _score_interval(
                                intervals,
                                test,
                                observed,
                                level,
                                usual_ranges[level],
                            )

                            records.append(
                                {
                                    "held_out": held_out,
                                    "p": p,
                                    "draw": draw,
                                    "h": (
                                        np.inf
                                        if h is None
                                        else float(h)
                                    ),
                                    "model": model_name,
                                    "level": level,
                                    "n_train": len(train),
                                    "n_test": len(test),
                                    **scores,
                                }
                            )

        if progress is not None:
            progress(
                "finished held-out panel "
                f"{held_out} "
                f"({completed} of {len(held_out_panels)})"
            )

    return pd.DataFrame.from_records(
        records,
        columns=list(INTERVAL_COLUMNS),
    )


def summarize_intervals(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize interval performance across draws and held-out panels.

    Results are first averaged across random draws within each held-out panel.
    They are then averaged across panels. This gives every panel equal weight,
    because panels—not individual households—are the independent units used
    to judge generalization.

    ``coverage_se`` is the standard error of mean coverage across the held-out
    panels.

    Parameters
    ----------
    results : pandas.DataFrame
        Results returned by :func:`run_interval_study`.

    Returns
    -------
    pandas.DataFrame
        One row for each model, interval level, panel count, and household
        cap. The summary includes:

        * ``supported``: true only when every draw was supported;
        * ``coverage``: mean coverage across held-out panels;
        * ``coverage_sd``: variation in coverage across panels;
        * ``coverage_se``: standard error of mean panel coverage;
        * ``width``: mean prediction-interval width;
        * ``ref_width``: mean historical comparison width;
        * ``informative``: share of intervals narrower than the historical
          comparison range;
        * ``n_panels``: number of held-out panels represented.

        Scores remain ``NaN`` when the requested level was unsupported.
    """
    missing = [
        column
        for column in INTERVAL_COLUMNS
        if column not in results.columns
    ]
    if missing:
        raise ValueError(
            f"results is missing columns: {missing}"
        )

    if results.empty:
        raise ValueError("results cannot be empty")

    keys = [
        "model",
        "level",
        "p",
        "h",
    ]

    # Average repeated draws within each held-out panel first.
    per_panel = results.groupby(
        [*keys, "held_out"],
        as_index=False,
    ).agg(
        supported=("supported", "all"),
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        ref_width=("ref_width", "mean"),
        informative=("informative", "mean"),
    )

    # Then average across panels so that a large panel does not receive more
    # influence simply because it contains more households.
    summary = per_panel.groupby(
        keys,
        as_index=False,
    ).agg(
        supported=("supported", "all"),
        coverage=("coverage", "mean"),
        coverage_sd=("coverage", "std"),
        width=("width", "mean"),
        ref_width=("ref_width", "mean"),
        informative=("informative", "mean"),
        n_panels=("held_out", "nunique"),
    )

    summary["coverage_se"] = (
        summary["coverage_sd"]
        / np.sqrt(summary["n_panels"])
    )

    columns = [
        *keys,
        "supported",
        "coverage",
        "coverage_sd",
        "coverage_se",
        "width",
        "ref_width",
        "informative",
        "n_panels",
    ]

    return (
        summary.loc[:, columns]
        .sort_values(keys)
        .reset_index(drop=True)
    )
