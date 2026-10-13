"""Panel-aware validation for the yieldlite model ladder.

Each development row belongs to one season-year panel. Validation holds out
one complete panel at a time, trains on the remaining panels, and predicts
the held-out households.

Keeping a panel together prevents the model from seeing part of a season's
shared conditions during training and then being evaluated on other rows
from that same season. The degradation study and grouped uncertainty methods
can reuse this same fold structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone


_PREDICTION_COLUMNS = (
    "panel_id",
    "y_true",
    "y_pred",
)


def leave_one_panel_out(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    panels: pd.Series,
) -> pd.DataFrame:
    """Create out-of-panel predictions for every row.

    The function holds out one panel, clones and fits the estimator on all
    remaining panels, and predicts the held-out rows. It repeats this process
    until every row has been predicted exactly once.

    Cloning prevents information learned in one fold from carrying into the
    next fold. It also leaves the estimator passed by the caller unfitted.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        An unfitted estimator with ``fit`` and ``predict`` methods.
    X : pandas.DataFrame
        Model features.
    y : pandas.Series
        Observed yield values.
    panels : pandas.Series
        One season-year panel ID for each row.

    Returns
    -------
    pandas.DataFrame
        One out-of-panel prediction per input row. The result contains
        ``panel_id``, ``y_true``, and ``y_pred`` and keeps the index from
        ``X``.

    Raises
    ------
    TypeError
        If ``X`` is not a pandas DataFrame.
    ValueError
        If the inputs are not aligned, contain invalid values, or include
        fewer than two panels.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not (len(X) == len(y) == len(panels)):
        raise ValueError(
            "X, y, and panels must have the same length"
        )

    y = pd.Series(y)
    panels = pd.Series(panels)

    if not (
        X.index.equals(y.index)
        and X.index.equals(panels.index)
    ):
        raise ValueError(
            "X, y, and panels must share the same index"
        )

    if not X.index.is_unique:
        raise ValueError(
            "X, y, and panels must use a unique row index"
        )

    if panels.isna().any():
        raise ValueError("panel IDs cannot be missing")

    numeric_y = pd.to_numeric(y, errors="coerce")
    if not np.isfinite(
        numeric_y.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "y must contain only finite numeric values"
        )

    unique_panels = panels.unique()
    if len(unique_panels) < 2:
        raise ValueError(
            "leave-one-panel-out needs at least 2 "
            f"distinct panels, got {len(unique_panels)}"
        )

    y_pred = pd.Series(
        np.nan,
        index=X.index,
        dtype=float,
        name="y_pred",
    )

    for held_out in unique_panels:
        test_mask = panels == held_out
        train_mask = ~test_mask

        model = clone(estimator)
        model.fit(
            X.loc[train_mask],
            numeric_y.loc[train_mask],
        )

        fold_predictions = np.asarray(
            model.predict(X.loc[test_mask]),
            dtype=float,
        )

        if fold_predictions.ndim != 1:
            raise ValueError(
                "predict() must return a one-dimensional result"
            )

        expected_rows = int(test_mask.sum())
        if len(fold_predictions) != expected_rows:
            raise ValueError(
                "predict() returned the wrong number of values: "
                f"expected {expected_rows}, "
                f"got {len(fold_predictions)}"
            )

        if not np.isfinite(fold_predictions).all():
            raise ValueError(
                f"model returned a non-finite prediction "
                f"for panel {held_out!r}"
            )

        y_pred.loc[test_mask] = fold_predictions

    if y_pred.isna().any():
        raise RuntimeError(
            "some rows did not receive an out-of-panel prediction"
        )

    return pd.DataFrame(
        {
            "panel_id": panels,
            "y_true": numeric_y.astype(float),
            "y_pred": y_pred,
        },
        index=X.index,
    )


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Check and copy a prediction table before calculating metrics."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = [
        column
        for column in _PREDICTION_COLUMNS
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    if predictions.empty:
        raise ValueError(f"{name} cannot be empty")

    if not predictions.index.is_unique:
        raise ValueError(
            f"{name} must use a unique row index"
        )

    checked = predictions.loc[
        :,
        list(_PREDICTION_COLUMNS),
    ].copy()

    if checked["panel_id"].isna().any():
        raise ValueError(
            f"{name} contains a missing panel ID"
        )

    for column in ("y_true", "y_pred"):
        checked[column] = pd.to_numeric(
            checked[column],
            errors="coerce",
        )

        if not np.isfinite(
            checked[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name}.{column} must contain only "
                "finite numeric values"
            )

    return checked


def panel_metrics(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate MAE and signed bias for each held-out panel.

    Bias is calculated as prediction minus truth. A positive value means the
    model tends to overpredict yield, while a negative value means it tends
    to underpredict.

    When baseline predictions are supplied, the result also includes:

    ``skill_over_baseline = 1 - candidate MAE / baseline MAE``

    Positive skill means the candidate model performed better. Negative skill
    means it performed worse. If the baseline MAE is zero, relative skill is
    undefined and is reported as ``NaN``.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output from ``leave_one_panel_out`` for the model being evaluated.
    baseline_predictions : pandas.DataFrame, optional
        Predictions for the simpler model used as the comparison baseline.
        They must describe the same rows, panels, and true target values.

    Returns
    -------
    pandas.DataFrame
        One row per panel with ``panel_id``, ``n``, ``mae``, ``bias``, and,
        when requested, ``skill_over_baseline``.
    """
    checked = _validate_prediction_frame(
        predictions,
        "predictions",
    )

    error = checked["y_pred"] - checked["y_true"]

    summary = (
        checked.assign(
            _abs_error=error.abs(),
            _error=error,
        )
        .groupby("panel_id", observed=True)
        .agg(
            n=("y_true", "size"),
            mae=("_abs_error", "mean"),
            bias=("_error", "mean"),
        )
        .reset_index()
    )

    if baseline_predictions is None:
        return summary

    baseline = _validate_prediction_frame(
        baseline_predictions,
        "baseline_predictions",
    )

    if not checked.index.equals(baseline.index):
        raise ValueError(
            "predictions and baseline_predictions must "
            "use the same row index"
        )

    if not checked["panel_id"].equals(baseline["panel_id"]):
        raise ValueError(
            "predictions and baseline_predictions must "
            "use the same panel ID for every row"
        )

    if not np.array_equal(
        checked["y_true"].to_numpy(dtype=float),
        baseline["y_true"].to_numpy(dtype=float),
    ):
        raise ValueError(
            "predictions and baseline_predictions must "
            "contain the same true target values"
        )

    baseline_error = (
        baseline["y_pred"] - baseline["y_true"]
    )
    baseline_mae = (
        baseline.assign(
            _abs_error=baseline_error.abs()
        )
        .groupby("panel_id", observed=True)["_abs_error"]
        .mean()
        .rename("_baseline_mae")
    )

    summary = summary.merge(
        baseline_mae,
        on="panel_id",
        how="left",
        validate="one_to_one",
    )

    summary["skill_over_baseline"] = np.nan
    valid_baseline = summary["_baseline_mae"] > 0

    summary.loc[
        valid_baseline,
        "skill_over_baseline",
    ] = (
        1.0
        - summary.loc[valid_baseline, "mae"]
        / summary.loc[valid_baseline, "_baseline_mae"]
    )

    return summary.drop(columns="_baseline_mae")
