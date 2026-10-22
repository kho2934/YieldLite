"""Convert the degradation-study results into model-selection thresholds.

A model rung earns a threshold only when it improves on the simpler model by
more than one standard error across the held-out panels. The improvement must
also remain reliable at every larger data setting that was tested.

If the study never reaches a reliable crossover, the threshold is recorded as
``not_reached`` instead of estimating or guessing a value.
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

_RESULT_COLUMNS = ("held_out", "p", "h", "model", "mae")


def paired_gain(
    results: pd.DataFrame,
    better: str = "m1",
    baseline: str = "m0",
) -> pd.DataFrame:
    """Compare one model with a simpler baseline using paired MAE differences.

    Repeated draws are averaged first. The function then compares the two
    models separately on every held-out panel for each combination of ``p``
    and ``h``. Pairing the results this way keeps the comparison fair because
    both models are evaluated on the same panels.

    Parameters
    ----------
    results : pandas.DataFrame
        Results returned by ``run_degradation``.
    better, baseline : str
        Names of the two models being compared.

    Returns
    -------
    pandas.DataFrame
        One row for each ``p`` and ``h`` setting. It includes the average MAE
        improvement, its standard error, the number of held-out panels, and
        whether the setting passes the selection rule.

        A positive ``mean_gain`` means that ``better`` has a lower MAE than
        ``baseline``. A setting passes when ``mean_gain`` is larger than its
        standard error.
    """
    missing = [column for column in _RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"results is missing columns: {missing}")

    models = set(results["model"])
    if better not in models or baseline not in models:
        raise ValueError(
            f"results must contain both {baseline!r} and {better!r}"
        )

    # Average repeated draws before comparing the models within each panel.
    per_panel = results.groupby(
        ["model", "held_out", "p", "h"],
        as_index=False,
    )["mae"].mean()

    wide = per_panel.pivot_table(
        index=["held_out", "p", "h"],
        columns="model",
        values="mae",
    ).reset_index()

    # Positive gain means the new model reduced MAE.
    wide["gain"] = wide[baseline] - wide[better]

    grouped = wide.groupby(["p", "h"])["gain"]
    gains = grouped.agg(mean_gain="mean", sd="std", n_panels="count")
    gains["se"] = gains["sd"] / np.sqrt(gains["n_panels"])
    gains["passes"] = (gains["mean_gain"] > gains["se"]).fillna(False)

    return (
        gains.drop(columns="sd")
        .reset_index()
        .sort_values(["p", "h"])
    )


def _first_stable(values: list[float], passes: list[bool]) -> float | None:
    """Find the first value where the rule passes and keeps passing afterward."""
    for index in range(len(values)):
        if all(passes[index:]):
            return values[index]
    return None


def _label(value: float | None) -> int | str | None:
    """Turn an unlimited household setting into the readable label ``"all"``."""
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
    """Create one threshold entry with a clear status and value.

    A reached threshold also says whether it sits at the smallest value the
    study tested. In that case the study never saw the rung fall short, so
    the true crossover could lie below the tested range.
    """
    if not evaluated:
        return {"status": "not_evaluated", "value": None}
    if value is None:
        return {"status": "not_reached", "value": None}
    return {
        "status": "reached",
        "value": _label(value),
        "at_grid_minimum": bool(value == grid_minimum),
    }


def derive_thresholds(
    results: pd.DataFrame,
    run_info: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build model-selection thresholds from the degradation-study results.

    ``P1`` is the smallest number of panels where M1 reliably beats M0 when
    all available households are used. ``H1`` is the smallest number of
    households per cell that is sufficient at every tested panel count from
    ``P1`` upward. Together they describe a region in which every tested
    setting passes, so no tested setting that failed lies inside it.

    Thresholds for models that have not been built yet are marked as
    ``not_evaluated``. The coverage threshold ``P0`` is handled the same way
    because the current degradation study does not evaluate it.

    Parameters
    ----------
    results : pandas.DataFrame
        Results returned by ``run_degradation``.
    run_info : dict, optional
        Information about the study settings, saved with the thresholds so
        the result can be traced back to the run that produced it.

    Returns
    -------
    dict
        A JSON-ready dictionary containing the thresholds, the evidence used
        to derive them, the selection rule, and the run information.
    """
    gains = paired_gain(results)

    # P1 is based only on runs that used all available households.
    uncapped = gains[np.isinf(gains["h"])].sort_values("p")
    p1 = None
    if not uncapped.empty:
        p1 = _first_stable(
            uncapped["p"].tolist(),
            uncapped["passes"].tolist(),
        )

    # H1 must hold at every tested panel count from P1 upward. Without P1
    # there is no region, and an unevaluated P1 falls back to every tested
    # panel count.
    h1 = None
    if uncapped.empty or p1 is not None:
        first_panel_count = gains["p"].min() if p1 is None else p1
        region = gains[gains["p"] >= first_panel_count]
        passes_by_h = region.groupby("h")["passes"].all().sort_index()
        h1 = _first_stable(
            passes_by_h.index.tolist(),
            passes_by_h.tolist(),
        )

    # Keep the supporting numbers in the output instead of saving only the
    # final thresholds.
    evidence = [
        {
            "p": int(row.p),
            "h": _label(row.h),
            "mean_gain": round(float(row.mean_gain), 4),
            "se": round(float(row.se), 4),
            "n_panels": int(row.n_panels),
            "passes": bool(row.passes),
        }
        for row in gains.itertuples()
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "criterion": CRITERION,
        "n_development_panels": int(results["held_out"].nunique()),
        "thresholds": {
            "P0": _entry(None, evaluated=False),
            "P1": _entry(
                p1,
                evaluated=not uncapped.empty,
                grid_minimum=gains["p"].min(),
            ),
            "H1": _entry(h1, grid_minimum=gains["h"].min()),
            "W2": _entry(None, evaluated=False),
            "N3": _entry(None, evaluated=False),
            "P3": _entry(None, evaluated=False),
        },
        "evidence": evidence,
        "run": run_info or {},
    }


def write_thresholds(
    thresholds: dict[str, object],
    path: str | Path,
) -> None:
    """Save the thresholds as readable JSON and create the folder if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(thresholds, indent=2) + "\n",
        encoding="utf-8",
    )
