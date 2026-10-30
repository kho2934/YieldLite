"""Measure how prediction intervals change as training data becomes smaller.

This study mirrors the model degradation study, but focuses on uncertainty.
It holds out one development panel at a time, trains each model on smaller
samples of the remaining panels, and evaluates the resulting intervals on the
households in the unseen panel.

The study records five main measurements:

* ``coverage``: how often the observed yield falls inside the prediction
  interval;
* ``width``: the average width of that interval;
* ``informative``: among households with a historical comparison, the share
  whose prediction interval is narrower than the usual yield range;
* ``history_coverage``: how often the observed yield falls inside the usual
  historical range itself;
* ``assessable``: the share of households whose district-season cell has
  enough training rows to calculate a historical range.

The historical range gives the prediction interval a practical comparison.
If both are equally wide, the model may not be telling the farmer much more
than past yields already showed. However, that comparison is meaningful only
when the historical range itself covers new observations reasonably well.
A range that misses too many new-season yields may be overconfident.

Historical ranges are calculated from the same training rows used to fit the
interval model. A district-season cell with too few rows is left unassessable
instead of borrowing a range from unrelated farms. Those households are not
included in ``informative``, ``history_coverage``, or ``ref_width``.

With ``p`` training panels, the largest interval level that can remain finite
is ``p / (p + 1)``. See ``yieldlite.conformal`` for the reasoning. A requested
level above that limit is recorded as unsupported rather than scored as if it
were valid.

This module measures interval behavior only. The separate threshold process
uses these results to decide the value of ``P0``.
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
from yieldlite.history import typical_interval
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
    "assessable",
    "history_coverage",
)


def _check_levels(
    levels: Iterable[float],
) -> list[float]:
    """Validate the requested interval levels and return them as floats."""
    level_list = [
        float(level)
        for level in levels
    ]

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
        raise ValueError(
            "interval levels cannot be repeated"
        )

    return level_list


def _assessable_history(
    usual_lower: np.ndarray,
    usual_upper: np.ndarray,
) -> np.ndarray:
    """Return which rows have two finite historical bounds."""
    return (
        np.isfinite(usual_lower)
        & np.isfinite(usual_upper)
    )


def _score_history(
    observed: np.ndarray,
    usual_lower: np.ndarray,
    usual_upper: np.ndarray,
) -> dict[str, float]:
    """Evaluate the usual yield range on the held-out households.

    Only households with two finite historical bounds are included.
    ``assessable`` records the share of held-out households that meet this
    requirement.

    If no household can be assessed, ``history_coverage`` and ``ref_width``
    are returned as ``NaN``.
    """
    assessable = _assessable_history(
        usual_lower,
        usual_upper,
    )

    if not assessable.any():
        return {
            "assessable": 0.0,
            "history_coverage": np.nan,
            "ref_width": np.nan,
        }

    lower = usual_lower[assessable]
    upper = usual_upper[assessable]

    return {
        "assessable": float(
            np.mean(assessable)
        ),
        "history_coverage": interval_coverage(
            observed[assessable],
            lower,
            upper,
        ),
        "ref_width": float(
            np.mean(upper - lower)
        ),
    }


def _score_interval(
    intervals: PanelCVPlus,
    test: pd.DataFrame,
    observed: np.ndarray,
    level: float,
    usual_lower: np.ndarray,
    usual_upper: np.ndarray,
) -> dict[str, object]:
    """Evaluate one prediction-interval level on a held-out panel.

    If the fitted panel count cannot support the requested level, the result
    is marked as unsupported and its coverage, width, and informativeness are
    returned as ``NaN``.

    Informativeness is measured only for households with an assessable
    historical range. It is also ``NaN`` when none of the held-out households
    has one.
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
            "informative": np.nan,
        }

    widths = np.asarray(
        upper - lower,
        dtype=float,
    )
    assessable = _assessable_history(
        usual_lower,
        usual_upper,
    )

    informative = np.nan
    if assessable.any():
        historical_widths = (
            usual_upper[assessable]
            - usual_lower[assessable]
        )
        informative = float(
            np.mean(
                widths[assessable]
                < historical_widths
            )
        )

    return {
        "supported": True,
        "coverage": interval_coverage(
            observed,
            lower,
            upper,
        ),
        "width": float(
            np.mean(widths)
        ),
        "informative": informative,
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
    """Evaluate prediction intervals as the training data is reduced.

    Each development panel is held out in turn. Interval models are trained
    on sampled subsets of the remaining panels while varying both the number
    of panels and the number of households kept in each district-season cell.

    Parameters
    ----------
    dev : pandas.DataFrame
        Development data only. The locked final holdout must not be included.
        Required columns include ``panel_id``, ``season``, ``district_code``,
        ``variety_raw``, ``variety_flag``, ``npk_total_kg_ha``, and
        ``yield_t_ha``.
    p_values : iterable of int
        Numbers of training panels to evaluate. Each value must be at least
        the number of seasons and no larger than the total number of
        development panels minus one.
    h_values : iterable of int or None
        Maximum households retained in each district-season cell. ``None``
        keeps every available household.
    levels : iterable of float, default (0.5, 0.8)
        Nominal prediction-interval levels to evaluate.
    n_draws : int, default 10
        Maximum number of sampled panel sets for each held-out panel and value
        of ``p``. Fewer draws are used when fewer valid sets exist.
    seed : int, default 0
        Random seed that makes the complete study reproducible.
    models : sequence of str, default all registered rungs
        Model names listed in ``yieldlite.ladder.RUNGS``.
    min_variety_count : int, default 30
        Minimum number of training rows required for a variety to remain its
        own category.
    progress : callable, optional
        Function that receives a short update after each held-out panel is
        completed. Passing ``print`` displays progress while the study runs.

    Returns
    -------
    pandas.DataFrame
        One row for every held-out panel, panel count, random draw, household
        cap, model, and interval level. Columns follow ``INTERVAL_COLUMNS``.

        ``h`` is stored as infinity when no household cap is applied.

        The usual yield interval contains the central historical share
        matching the requested prediction level. It is calculated from the
        training yields in the same district and season. A cell needs at
        least ten training rows, as defined by ``typical_interval``.

        ``assessable`` is the share of held-out households with a historical
        range. ``history_coverage`` is the share of those households whose
        observed yield falls inside that range, and ``ref_width`` is its
        average width. ``informative`` is the share whose prediction interval
        is narrower than the historical range.

        If no historical range can be assessed, ``history_coverage``,
        ``ref_width``, and ``informative`` are ``NaN``.

        When the training panels cannot support a requested prediction level,
        ``supported`` is false and the prediction-interval measurements are
        ``NaN``. The separate historical measurements remain available.
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

    p_list = [
        int(p)
        for p in p_values
    ]
    h_list = list(h_values)

    if not p_list or not h_list:
        raise ValueError(
            "p_values and h_values cannot be empty"
        )

    n_seasons = int(
        panel_season.nunique()
    )
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

    held_out_panels = sorted(
        panel_season.index
    )

    for completed, held_out in enumerate(
        held_out_panels,
        start=1,
    ):
        remaining_panel_seasons = (
            panel_season.drop(held_out)
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

                    # The historical ranges use only the current training
                    # rows, so they do not see the held-out panel. They are
                    # also independent of the model and need to be calculated
                    # only once for each requested level.
                    usual_ranges = {
                        level: typical_interval(
                            train,
                            test,
                            level,
                        )
                        for level in level_list
                    }

                    history_scores = {
                        level: _score_history(
                            observed,
                            *usual_ranges[level],
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
                            interval_scores = _score_interval(
                                intervals,
                                test,
                                observed,
                                level,
                                *usual_ranges[level],
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
                                    **history_scores[level],
                                    **interval_scores,
                                }
                            )

        if progress is not None:
            progress(
                "finished held-out panel "
                f"{held_out} "
                f"({completed} of "
                f"{len(held_out_panels)})"
            )

    return pd.DataFrame.from_records(
        records,
        columns=list(INTERVAL_COLUMNS),
    )


def summarize_intervals(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize interval performance across draws and held-out panels.

    Random draws are averaged within each held-out panel first. The panel
    results are then averaged together. This gives every panel equal weight,
    because panels—not individual household rows—are the independent units
    used to judge generalization.

    ``coverage_se`` is the standard error of prediction-interval coverage
    across held-out panels. ``history_coverage_se`` applies the same
    calculation to the historical range, using only panels where that range
    could be assessed.

    Parameters
    ----------
    results : pandas.DataFrame
        Results returned by :func:`run_interval_study`.

    Returns
    -------
    pandas.DataFrame
        One row for every model, interval level, panel count, and household
        cap. The summary includes:

        * ``supported``: true only when every draw was supported;
        * ``coverage``: average prediction-interval coverage;
        * ``coverage_sd``: variation in coverage across held-out panels;
        * ``coverage_se``: standard error of average panel coverage;
        * ``width``: average prediction-interval width;
        * ``ref_width``: average historical-range width;
        * ``informative``: share of intervals narrower than the historical
          range among assessable households;
        * ``assessable``: share of households with a historical range;
        * ``history_coverage``: average historical-range coverage;
        * ``history_coverage_se``: standard error of historical coverage;
        * ``n_panels``: number of held-out panels represented.

        Prediction-interval scores remain ``NaN`` when the requested level is
        unsupported. Historical-range scores remain ``NaN`` when no
        household in a panel can be assessed.
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
        raise ValueError(
            "results cannot be empty"
        )

    keys = [
        "model",
        "level",
        "p",
        "h",
    ]

    # Average repeated draws for each held-out panel before comparing panels.
    per_panel = results.groupby(
        [*keys, "held_out"],
        as_index=False,
    ).agg(
        supported=("supported", "all"),
        coverage=("coverage", "mean"),
        width=("width", "mean"),
        ref_width=("ref_width", "mean"),
        informative=("informative", "mean"),
        assessable=("assessable", "mean"),
        history_coverage=(
            "history_coverage",
            "mean",
        ),
    )

    # Give each held-out panel equal weight, even when panel sizes differ.
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
        assessable=("assessable", "mean"),
        history_coverage=(
            "history_coverage",
            "mean",
        ),
        history_coverage_sd=(
            "history_coverage",
            "std",
        ),
        history_panels=(
            "history_coverage",
            "count",
        ),
        n_panels=("held_out", "nunique"),
    )

    summary["coverage_se"] = (
        summary["coverage_sd"]
        / np.sqrt(summary["n_panels"])
    )
    summary["history_coverage_se"] = (
        summary["history_coverage_sd"]
        / np.sqrt(
            summary["history_panels"]
        )
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
        "assessable",
        "history_coverage",
        "history_coverage_se",
        "n_panels",
    ]

    return (
        summary.loc[:, columns]
        .sort_values(keys)
        .reset_index(drop=True)
    )
