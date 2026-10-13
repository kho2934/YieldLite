"""Tests for yieldlite's panel-aware validation tools."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from yieldlite.models import SeasonDistrictTable, build_m1
from yieldlite.validate import (
    leave_one_panel_out,
    panel_metrics,
)


def _small_panel_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create three small panels with two households in each."""
    X = pd.DataFrame(
        {
            "season": [
                "DX",
                "DX",
                "HT",
                "HT",
                "TD",
                "TD",
            ],
            "district_code": [
                "OM",
                "CD",
                "OM",
                "CD",
                "OM",
                "CD",
            ],
        }
    )
    y = pd.Series(
        [6.0, 7.0, 5.0, 5.5, 6.5, 6.0]
    )
    panels = pd.Series(
        ["A", "A", "B", "B", "C", "C"]
    )

    return X, y, panels


def test_leave_one_panel_out_covers_every_row_once():
    X, y, panels = _small_panel_data()

    predictions = leave_one_panel_out(
        SeasonDistrictTable(),
        X,
        y,
        panels,
    )

    assert len(predictions) == len(X)
    assert (
        predictions["panel_id"].tolist()
        == panels.tolist()
    )
    assert predictions["y_true"].tolist() == y.tolist()
    assert predictions["y_pred"].notna().all()


def test_leave_one_panel_out_never_trains_on_test_panel():
    X, y, panels = _small_panel_data()
    training_indexes = []

    class RecordingEstimator(
        RegressorMixin,
        BaseEstimator,
    ):
        """Record the rows received during each fit call."""

        def fit(self, X, y):
            training_indexes.append(set(X.index))
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    leave_one_panel_out(
        RecordingEstimator(),
        X,
        y,
        panels,
    )

    for training_index, held_out_panel in zip(
        training_indexes,
        panels.unique(),
        strict=True,
    ):
        held_out_index = set(
            X.index[panels == held_out_panel]
        )
        assert training_index.isdisjoint(held_out_index)


def test_leave_one_panel_out_does_not_mutate_estimator():
    X, y, panels = _small_panel_data()
    model = SeasonDistrictTable()

    leave_one_panel_out(model, X, y, panels)

    assert not hasattr(model, "group_means_")


def test_leave_one_panel_out_rejects_mismatched_lengths():
    X, y, panels = _small_panel_data()

    with pytest.raises(ValueError):
        leave_one_panel_out(
            SeasonDistrictTable(),
            X,
            y.iloc[:-1],
            panels,
        )


def test_leave_one_panel_out_rejects_mismatched_index():
    X, y, panels = _small_panel_data()
    y.index = y.index + 10

    with pytest.raises(ValueError):
        leave_one_panel_out(
            SeasonDistrictTable(),
            X,
            y,
            panels,
        )


def test_leave_one_panel_out_rejects_duplicate_index():
    X, y, panels = _small_panel_data()

    duplicate_index = pd.Index([0, 0, 1, 2, 3, 4])
    X.index = duplicate_index
    y.index = duplicate_index
    panels.index = duplicate_index

    with pytest.raises(ValueError):
        leave_one_panel_out(
            SeasonDistrictTable(),
            X,
            y,
            panels,
        )


def test_leave_one_panel_out_rejects_missing_panel_id():
    X, y, panels = _small_panel_data()
    panels.iloc[0] = None

    with pytest.raises(ValueError):
        leave_one_panel_out(
            SeasonDistrictTable(),
            X,
            y,
            panels,
        )


def test_leave_one_panel_out_needs_two_panels():
    X, y, _ = _small_panel_data()
    one_panel = pd.Series(["A"] * len(X))

    with pytest.raises(ValueError):
        leave_one_panel_out(
            SeasonDistrictTable(),
            X,
            y,
            one_panel,
        )


def test_leave_one_panel_out_rejects_nonfinite_prediction():
    X, y, panels = _small_panel_data()

    class NanEstimator(
        RegressorMixin,
        BaseEstimator,
    ):
        """Return NaN to confirm invalid predictions are rejected."""

        def fit(self, X, y):
            self.is_fitted_ = True
            return self

        def predict(self, X):
            return np.full(len(X), np.nan)

    with pytest.raises(ValueError):
        leave_one_panel_out(
            NanEstimator(),
            X,
            y,
            panels,
        )


def test_leave_one_panel_out_works_with_m1_array_output():
    rng = np.random.default_rng(0)
    districts = ["OM", "CD", "TN"]

    rows = []
    targets = []
    panel_ids = []

    for panel_number in range(4):
        for row_number in range(10):
            district = districts[
                row_number % len(districts)
            ]
            npk = rng.uniform(80, 220)

            rows.append(
                {
                    "district_code": district,
                    "season": "DX",
                    "variety_pooled": "pooled",
                    "npk_total_kg_ha": npk,
                }
            )
            targets.append(
                6.0
                + 0.01 * npk
                + rng.normal(0, 0.1)
            )
            panel_ids.append(f"panel_{panel_number}")

    X = pd.DataFrame(rows)
    y = pd.Series(targets)
    panels = pd.Series(panel_ids)

    # M1 returns a NumPy array, while M0 returns a pandas Series.
    # The validation function should handle either format.
    predictions = leave_one_panel_out(
        build_m1(),
        X,
        y,
        panels,
    )

    assert len(predictions) == len(X)
    assert predictions["y_pred"].notna().all()


def test_panel_metrics_matches_hand_calculation():
    predictions = pd.DataFrame(
        {
            "panel_id": ["A", "A", "B", "B"],
            "y_true": [5.0, 7.0, 4.0, 6.0],
            "y_pred": [6.0, 6.0, 5.0, 5.0],
        }
    )

    summary = panel_metrics(
        predictions
    ).set_index("panel_id")

    # Both panels have errors of +1 and -1.
    assert summary.loc["A", "mae"] == pytest.approx(1.0)
    assert summary.loc["A", "bias"] == pytest.approx(0.0)
    assert summary.loc["A", "n"] == 2

    assert summary.loc["B", "mae"] == pytest.approx(1.0)
    assert summary.loc["B", "bias"] == pytest.approx(0.0)
    assert summary.loc["B", "n"] == 2


def test_panel_metrics_keeps_bias_direction():
    # Predictions are one tonne per hectare above the truth.
    predictions = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 5.0],
            "y_pred": [6.0, 6.0],
        }
    )

    summary = panel_metrics(
        predictions
    ).set_index("panel_id")

    assert summary.loc["A", "mae"] == pytest.approx(1.0)
    assert summary.loc["A", "bias"] == pytest.approx(1.0)


def test_panel_metrics_computes_skill_over_baseline():
    candidate = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 5.0],
            "y_pred": [5.5, 5.5],
        }
    )
    baseline = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 5.0],
            "y_pred": [6.0, 6.0],
        }
    )

    summary = panel_metrics(
        candidate,
        baseline,
    ).set_index("panel_id")

    # The candidate cuts MAE from 1.0 to 0.5.
    assert summary.loc[
        "A",
        "skill_over_baseline",
    ] == pytest.approx(0.5)


def test_panel_metrics_zero_baseline_mae_is_undefined():
    candidate = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 6.0],
            "y_pred": [5.5, 6.5],
        }
    )
    perfect_baseline = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 6.0],
            "y_pred": [5.0, 6.0],
        }
    )

    summary = panel_metrics(
        candidate,
        perfect_baseline,
    ).set_index("panel_id")

    assert pd.isna(
        summary.loc["A", "skill_over_baseline"]
    )


def test_panel_metrics_rejects_different_baseline_rows():
    candidate = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 6.0],
            "y_pred": [5.5, 6.5],
        }
    )
    baseline = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 6.0],
            "y_pred": [5.0, 6.0],
        },
        index=[10, 11],
    )

    with pytest.raises(ValueError):
        panel_metrics(candidate, baseline)


def test_panel_metrics_rejects_changed_baseline_truth():
    candidate = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 6.0],
            "y_pred": [5.5, 6.5],
        }
    )
    baseline = pd.DataFrame(
        {
            "panel_id": ["A", "A"],
            "y_true": [5.0, 7.0],
            "y_pred": [5.0, 7.0],
        }
    )

    with pytest.raises(ValueError):
        panel_metrics(candidate, baseline)


def test_panel_metrics_rejects_missing_columns():
    incomplete = pd.DataFrame(
        {
            "panel_id": ["A"],
            "y_true": [1.0],
        }
    )

    with pytest.raises(ValueError):
        panel_metrics(incomplete)
