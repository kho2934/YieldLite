"""Turn study results into thresholds for choosing a model.

A model earns a place on the ladder only when it improves on the simpler
model by more than one standard error across the held-out panels. That
improvement must also continue at every larger data setting tested. This
prevents one unusually good setting from creating an unreliable threshold.

The ``P0`` coverage threshold comes from the interval study. An interval is
considered reliable at a panel count only when that level is mathematically
supported and its observed coverage is not clearly below the requested level.

When a study never reaches a dependable crossover, the result is recorded as
``not_reached``. The package does not estimate a threshold that the evidence
did not actually reach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

CRITERION = (
    "The mean paired reduction in MAE across held-out panels must be larger "
    "than its standard error and remain so at every larger tested setting."
)

COVERAGE_CRITERION = (
    "At a panel count, an interval is trusted only when its level is "
    "supported and its mean coverage across held-out panels is at least the "
    "level minus one standard error, for every model and household setting "
    "tested. This must also hold at every larger tested panel count."
)

_RESULT_COLUMNS = (
    "held_out",
    "p",
    "h",
    "model",
    "mae",
)

_INTERVAL_COLUMNS = (
    "model",
    "level",
    "p",
    "h",
    "supported",
    "coverage",
    "coverage_se",
)


def paired_gain(
    results: pd.DataFrame,
    better: str = "m1",
    baseline: str = "m0",
) -> pd.DataFrame:
    """Measure how much one model improves on a simpler baseline.

    Repeated draws are averaged before the comparison. For each combination
    of panel count and household cap, the two models are compared separately
    on every held-out panel. This keeps the comparison paired because both
    models are judged on the same panels.

    Parameters
    ----------
    results : pandas.DataFrame
        Results produced by ``run_degradation``.
    better, baseline : str
        Names of the candidate model and the simpler comparison model.

    Returns
    -------
    pandas.DataFrame
        One row for each combination of ``p`` and ``h``. The output includes
        the mean improvement in MAE, its standard error, the number of panels,
        and whether that setting passes the selection rule.

        A positive ``mean_gain`` means that ``better`` had a lower MAE than
        ``baseline``. A setting passes only when the gain is larger than its
        standard error.
    """
    missing = [
        column
        for column in _RESULT_COLUMNS
        if column not in results.columns
    ]
    if missing:
        raise ValueError(
            f"results is missing columns: {missing}"
        )

    models = set(results["model"])
    if better not in models or baseline not in models:
        raise ValueError(
            f"results must contain both '{baseline}' and '{better}'"
        )

    # Average repeated draws first so that each held-out panel contributes
    # one paired model comparison at each data setting.
    per_panel = results.groupby(
        ["model", "held_out", "p", "h"],
        as_index=False,
    )["mae"].mean()

    wide = per_panel.pivot_table(
        index=["held_out", "p", "h"],
        columns="model",
        values="mae",
    ).reset_index()

    # A positive value means the candidate model reduced MAE.
    wide["gain"] = (
        wide[baseline] - wide[better]
    )

    grouped = wide.groupby(
        ["p", "h"]
    )["gain"]

    gains = grouped.agg(
        mean_gain="mean",
        sd="std",
        n_panels="count",
    )
    gains["se"] = (
        gains["sd"]
        / np.sqrt(gains["n_panels"])
    )
    gains["passes"] = (
        gains["mean_gain"] > gains["se"]
    ).fillna(False)

    return (
        gains.drop(columns="sd")
        .reset_index()
        .sort_values(["p", "h"])
    )


def coverage_check(
    summary: pd.DataFrame,
    level: float = 0.8,
) -> pd.DataFrame:
    """Check whether intervals are reliable at each panel count.

    A single model and household cap make up one setting. That setting passes
    when the requested interval level is supported and its average coverage
    is at least the requested level minus one standard error.

    A panel count passes only when every tested setting passes. This
    conservative rule prevents the package from accepting a panel count that
    works for one model or sample size but fails for another.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output from ``summarize_intervals`` or the saved
        ``interval_summary.csv`` file.
    level : float, default 0.8
        Interval level to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row for each panel count. The result includes the total number of
        settings, the number that passed, the lowest observed mean coverage,
        and whether the panel count passed overall.

        ``lowest_coverage`` is ``NaN`` when none of the settings are
        supported.
    """
    missing = [
        column
        for column in _INTERVAL_COLUMNS
        if column not in summary.columns
    ]
    if missing:
        raise ValueError(
            f"summary is missing columns: {missing}"
        )

    rows = summary[
        np.isclose(summary["level"], level)
    ]
    if rows.empty:
        raise ValueError(
            f"summary has no rows for level {level}"
        )

    # Missing coverage or standard-error values fail naturally because the
    # comparison below evaluates to false.
    clears_level = (
        rows["coverage"]
        >= level - rows["coverage_se"]
    )

    rows = rows.assign(
        passes=(
            rows["supported"].astype(bool)
            & clears_level
        )
    )

    by_panels = rows.groupby("p").agg(
        n_settings=("passes", "size"),
        n_passing=("passes", "sum"),
        lowest_coverage=("coverage", "min"),
    )
    by_panels["passes"] = (
        by_panels["n_passing"]
        == by_panels["n_settings"]
    )

    return (
        by_panels.reset_index()
        .sort_values("p")
    )


def _first_stable(
    values: list[float],
    passes: list[bool],
) -> float | None:
    """Find the first value that passes along with every larger value."""
    for index in range(len(values)):
        if all(passes[index:]):
            return values[index]

    return None


def _label(
    value: float | None,
) -> int | str | None:
    """Use the readable label ``"all"`` when there is no household cap."""
    if value is None:
        return None

    if np.isinf(value):
        return "all"

    return int(value)


def _entry(
    value: float | None,
    evaluated: bool = True,
    grid_minimum: float | None = None,
) -> dict[str, object]:
    """Create a threshold entry with its status and supporting context.

    A reached threshold also records whether it occurred at the smallest
    value included in the study. If it did, the study found only a floor:
    the real crossover may be below the range that was tested.
    """
    if not evaluated:
        return {
            "status": "not_evaluated",
            "value": None,
        }

    if value is None:
        return {
            "status": "not_reached",
            "value": None,
        }

    return {
        "status": "reached",
        "value": _label(value),
        "at_grid_minimum": bool(
            value == grid_minimum
        ),
    }


def derive_thresholds(
    results: pd.DataFrame,
    run_info: dict[str, object] | None = None,
    intervals: pd.DataFrame | None = None,
    interval_level: float = 0.8,
    interval_info: dict[str, object] | None = None,
) -> dict[str, object]:
    """Derive the model and interval thresholds from the completed studies.

    ``P1`` is the smallest number of panels where M1 reliably improves on M0
    when all available households are used.

    ``H1`` is the smallest household count per cell that works at every tested
    panel count from ``P1`` upward. Together, ``P1`` and ``H1`` define a safe
    region in which every tested setting passed. A failed setting can never
    sit inside that region.

    ``P0`` is the smallest panel count where intervals at ``interval_level``
    pass the coverage rule at that count and every larger tested count. The
    interval-study summary is required to calculate it. When no interval
    results are provided, ``P0`` is marked as ``not_evaluated``.

    Thresholds for model rungs that have not been built are also marked as
    ``not_evaluated``.

    Parameters
    ----------
    results : pandas.DataFrame
        Results produced by ``run_degradation``.
    run_info : dict, optional
        Settings and other information from the degradation study. These are
        stored with the output so the thresholds can be traced back to the
        run that produced them.
    intervals : pandas.DataFrame, optional
        Summary produced by ``summarize_intervals``. This is required to
        derive ``P0``.
    interval_level : float, default 0.8
        Interval level used to derive ``P0``.
    interval_info : dict, optional
        Settings and other information from the interval study.

    Returns
    -------
    dict
        A JSON-ready result containing the thresholds, supporting evidence,
        selection rules, and information about both study runs.
    """
    gains = paired_gain(results)

    # P1 uses only the settings with no household cap.
    uncapped = gains[
        np.isinf(gains["h"])
    ].sort_values("p")

    p1 = None
    if not uncapped.empty:
        p1 = _first_stable(
            uncapped["p"].tolist(),
            uncapped["passes"].tolist(),
        )

    # H1 must remain sufficient at every panel count from P1 upward. If P1
    # was not evaluated, the check uses every panel count in the study.
    h1 = None
    if uncapped.empty or p1 is not None:
        first_panel_count = (
            gains["p"].min()
            if p1 is None
            else p1
        )

        region = gains[
            gains["p"] >= first_panel_count
        ]

        passes_by_h = (
            region.groupby("h")["passes"]
            .all()
            .sort_index()
        )

        h1 = _first_stable(
            passes_by_h.index.tolist(),
            passes_by_h.tolist(),
        )

    # Keep the full supporting table instead of storing only the final P1 and
    # H1 values.
    evidence = [
        {
            "p": int(row.p),
            "h": _label(row.h),
            "mean_gain": round(
                float(row.mean_gain),
                4,
            ),
            "se": round(
                float(row.se),
                4,
            ),
            "n_panels": int(row.n_panels),
            "passes": bool(row.passes),
        }
        for row in gains.itertuples()
    ]

    # P0 can be calculated only when interval-study results are available.
    p0 = None
    p0_grid_minimum = None
    coverage_evidence = []

    if intervals is not None:
        by_panels = coverage_check(
            intervals,
            interval_level,
        )

        p0 = _first_stable(
            by_panels["p"].tolist(),
            by_panels["passes"].tolist(),
        )
        p0_grid_minimum = by_panels["p"].min()

        coverage_evidence = [
            {
                "p": int(row.p),
                "n_settings": int(
                    row.n_settings
                ),
                "n_passing": int(
                    row.n_passing
                ),
                "lowest_coverage": (
                    None
                    if np.isnan(
                        row.lowest_coverage
                    )
                    else round(
                        float(row.lowest_coverage),
                        4,
                    )
                ),
                "passes": bool(row.passes),
            }
            for row in by_panels.itertuples()
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "criterion": CRITERION,
        "n_development_panels": int(
            results["held_out"].nunique()
        ),
        "thresholds": {
            "P0": _entry(
                p0,
                evaluated=intervals is not None,
                grid_minimum=p0_grid_minimum,
            ),
            "P1": _entry(
                p1,
                evaluated=not uncapped.empty,
                grid_minimum=gains["p"].min(),
            ),
            "H1": _entry(
                h1,
                grid_minimum=gains["h"].min(),
            ),
            "W2": _entry(
                None,
                evaluated=False,
            ),
            "N3": _entry(
                None,
                evaluated=False,
            ),
            "P3": _entry(
                None,
                evaluated=False,
            ),
        },
        "evidence": evidence,
        "coverage": {
            "criterion": COVERAGE_CRITERION,
            "level": (
                interval_level
                if intervals is not None
                else None
            ),
            "evidence": coverage_evidence,
            "run": interval_info or {},
        },
        "run": run_info or {},
    }


def write_thresholds(
    thresholds: dict[str, object],
    path: str | Path,
) -> None:
    """Save the thresholds as readable JSON, creating the folder if needed."""
    target = Path(path)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_text(
        json.dumps(
            thresholds,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
