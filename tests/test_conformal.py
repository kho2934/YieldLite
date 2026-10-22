"""Tests for yieldlite.conformal."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from yieldlite.conformal import (
    LevelUnsupportedError,
    PanelCVPlus,
    interval_coverage,
    max_supported_level,
    min_panels_for_level,
)
from yieldlite.models import SeasonDistrictTable, build_m1


class _Passthrough(RegressorMixin, BaseEstimator):
    """Predict the ``pred`` column, so every residual can be chosen by hand."""

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return X["pred"].to_numpy(dtype=float)


class _FoldShifted(RegressorMixin, BaseEstimator):
    """Copy ``pred``, but add 5 when trained on fewer than 6 rows."""

    def fit(self, X, y):
        self.n_train_ = len(X)
        return self

    def predict(self, X):
        jump = 5.0 if self.n_train_ < 6 else 0.0
        return X["pred"].to_numpy(dtype=float) + jump


def _hand_frame(rows):
    """Split (panel, pred, y) tuples into the three inputs of ``fit``."""
    frame = pd.DataFrame(rows, columns=["panel", "pred", "y"])
    return frame[["pred"]], frame["y"], frame["panel"]


def _draw_panel(rng, size, shock_sd=0.8, noise_sd=0.4):
    """Draw one simulated panel whose households share a single shock."""
    features = pd.DataFrame(
        {"a": rng.normal(size=size), "b": rng.uniform(size=size)}
    )
    shock = rng.normal(scale=shock_sd)
    target = (
        3.0
        + 1.5 * features["a"].to_numpy()
        - features["b"].to_numpy()
        + shock
        + rng.normal(scale=noise_sd, size=size)
    )
    return features, target


def _simulated_panels(rng, n_panels):
    """Stack simulated panels into one training set."""
    frames, targets, labels = [], [], []
    for panel in range(n_panels):
        size = int(rng.integers(10, 40))
        features, target = _draw_panel(rng, size)
        frames.append(features)
        targets.append(target)
        labels.extend([panel] * size)

    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(targets),
        np.array(labels),
    )


def _survey_frame():
    """Build six small survey-like panels, two for each season."""
    rng = np.random.default_rng(3)
    specs = [
        ("DX_2020", "DX", 0.3),
        ("DX_2021", "DX", -0.2),
        ("HT_2020", "HT", 0.1),
        ("HT_2022", "HT", -0.1),
        ("TD_2020", "TD", 0.0),
        ("TD_2021", "TD", 0.25),
    ]
    season_effect = {"DX": 0.5, "HT": 0.0, "TD": -0.5}

    rows = []
    for panel, season, shock in specs:
        for district in ("OM", "CD", "TN"):
            for _ in range(6):
                npk = rng.uniform(80.0, 220.0)
                variety = rng.choice(["OM5451", "IR50404"])
                noise = rng.normal(scale=0.3)
                rows.append(
                    (
                        panel,
                        season,
                        district,
                        variety,
                        npk,
                        5.0 + season_effect[season] + shock + 0.004 * npk + noise,
                    )
                )

    return pd.DataFrame(
        rows,
        columns=[
            "panel_id",
            "season",
            "district_code",
            "variety_pooled",
            "npk_total_kg_ha",
            "yield_t_ha",
        ],
    )


def test_max_supported_level_is_k_over_k_plus_one():
    assert max_supported_level(8) == pytest.approx(8 / 9)
    assert max_supported_level(12) == pytest.approx(12 / 13)
    assert max_supported_level(1) == pytest.approx(0.5)


@pytest.mark.parametrize("n_panels", [0, -3, 2.5, True])
def test_max_supported_level_rejects_bad_counts(n_panels):
    with pytest.raises(ValueError):
        max_supported_level(n_panels)


@pytest.mark.parametrize(
    "level, expected",
    [(0.5, 1), (0.8, 4), (0.9, 9), (0.95, 19)],
)
def test_min_panels_for_level_known_values(level, expected):
    assert min_panels_for_level(level) == expected


def test_min_panels_is_the_smallest_count_that_reaches_the_level():
    for level in (0.1, 0.5, 0.66, 0.75, 0.8, 0.85, 0.9, 0.97):
        panels = min_panels_for_level(level)

        assert max_supported_level(panels) >= level - 1e-9
        if panels > 1:
            assert max_supported_level(panels - 1) < level


@pytest.mark.parametrize("level", [0, 1, -0.2, 1.2, float("nan"), True])
def test_min_panels_rejects_bad_levels(level):
    with pytest.raises(ValueError):
        min_panels_for_level(level)


def test_every_panel_counts_equally_whatever_its_size():
    # Panel A has two rows with errors of 10 and 20. Panel B has one row with
    # an error of 30. B holds half of the weight, so the 40% point already
    # reaches 30. Counting rows instead of panels would stop at 20.
    X, y, panels = _hand_frame(
        [("A", 0.0, 10.0), ("A", 0.0, 20.0), ("B", 0.0, 30.0)]
    )
    model = PanelCVPlus(_Passthrough()).fit(X, y, panels)

    lower, upper = model.predict_interval(
        pd.DataFrame({"pred": [100.0]}),
        level=0.4,
    )

    assert upper[0] == pytest.approx(130.0)
    assert lower[0] == pytest.approx(70.0)


def test_one_row_per_panel_matches_the_jackknife_plus_formula():
    rng = np.random.default_rng(1)
    n = 12
    level = 0.8
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 2.0 + X["a"] - 0.5 * X["b"] + rng.normal(scale=0.3, size=n)
    new = pd.DataFrame({"a": [0.3, -1.0], "b": [0.1, 0.8]})

    model = PanelCVPlus(LinearRegression()).fit(X, y, np.arange(n))
    lower, upper = model.predict_interval(new, level)

    # Jackknife+ written out from its definition (Barber et al., 2021).
    leave_out_predictions = []
    leave_out_errors = []
    for i in range(n):
        keep = np.arange(n) != i
        fitted = LinearRegression().fit(X[keep], y[keep])
        leave_out_errors.append(abs(y[i] - fitted.predict(X.iloc[[i]])[0]))
        leave_out_predictions.append(fitted.predict(new))
    predictions = np.array(leave_out_predictions)
    errors = np.array(leave_out_errors)[:, None]

    upper_rank = math.ceil(level * (n + 1))
    lower_rank = math.floor((1 - level) * (n + 1))
    expected_upper = np.sort(predictions + errors, axis=0)[upper_rank - 1]
    expected_lower = np.sort(predictions - errors, axis=0)[lower_rank - 1]

    estimate = model.predict(new)
    np.testing.assert_allclose(upper, np.maximum(expected_upper, estimate))
    np.testing.assert_allclose(lower, np.minimum(expected_lower, estimate))


def test_higher_levels_give_wider_intervals():
    rng = np.random.default_rng(2)
    X, y, panels = _simulated_panels(rng, 10)
    new, _ = _draw_panel(rng, 25)
    model = PanelCVPlus(LinearRegression()).fit(X, y, panels)

    widths = []
    for level in (0.3, 0.5, 0.8, 0.9):
        lower, upper = model.predict_interval(new, level)
        widths.append(float(np.mean(upper - lower)))

    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def test_fewer_panels_give_wider_intervals():
    rng = np.random.default_rng(4)
    new, _ = _draw_panel(rng, 100)
    mean_width = {}

    for n_panels in (4, 24):
        X, y, panels = _simulated_panels(rng, n_panels)
        model = PanelCVPlus(LinearRegression()).fit(X, y, panels)
        lower, upper = model.predict_interval(new, 0.8)
        mean_width[n_panels] = float(np.mean(upper - lower))

    assert mean_width[4] > mean_width[24]


def test_level_above_what_the_panels_support_is_refused():
    X, y, panels = _hand_frame([("A", 0.0, 1.0), ("B", 0.0, 2.0), ("C", 0.0, 3.0)])
    model = PanelCVPlus(_Passthrough()).fit(X, y, panels)

    assert model.max_level_ == pytest.approx(0.75)
    with pytest.raises(LevelUnsupportedError, match="75.0%"):
        model.predict_interval(X, level=0.8)


def test_a_level_exactly_at_the_cap_is_allowed():
    X, y, panels = _hand_frame([("A", 0.0, 1.0), ("B", 0.0, 2.0), ("C", 0.0, 3.0)])
    model = PanelCVPlus(_Passthrough()).fit(X, y, panels)

    lower, upper = model.predict_interval(X, level=0.75)

    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()


def test_default_level_needs_four_panels():
    rows = [("A", 0.0, 1.0), ("B", 0.0, 2.0), ("C", 0.0, 3.0), ("D", 0.0, 4.0)]

    X, y, panels = _hand_frame(rows)
    four = PanelCVPlus(_Passthrough()).fit(X, y, panels)
    lower, upper = four.predict_interval(X)
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()

    X, y, panels = _hand_frame(rows[:3])
    three = PanelCVPlus(_Passthrough()).fit(X, y, panels)
    with pytest.raises(LevelUnsupportedError):
        three.predict_interval(X)


def test_interval_contains_the_estimate():
    rng = np.random.default_rng(6)
    X, y, panels = _simulated_panels(rng, 8)
    new, _ = _draw_panel(rng, 200)
    model = PanelCVPlus(LinearRegression()).fit(X, y, panels)

    lower, upper = model.predict_interval(new, 0.5)
    estimate = model.predict(new)

    assert (lower <= estimate).all()
    assert (estimate <= upper).all()


def test_interval_is_widened_to_hold_the_estimate():
    # Fold models jump by 5 but the final model does not. The held-out errors
    # are all zero, so CV+ alone gives a point interval 5 above the estimate.
    X, y, panels = _hand_frame(
        [
            ("A", 1.0, 6.0),
            ("A", 2.0, 7.0),
            ("B", 3.0, 8.0),
            ("B", 4.0, 9.0),
            ("C", 5.0, 10.0),
            ("C", 6.0, 11.0),
        ]
    )
    model = PanelCVPlus(_FoldShifted()).fit(X, y, panels)

    lower, upper = model.predict_interval(X, level=0.5)

    estimate = model.predict(X)
    np.testing.assert_allclose(lower, estimate)
    np.testing.assert_allclose(upper, estimate + 5.0)


def test_rows_beyond_one_block_match_row_by_row_results():
    rng = np.random.default_rng(5)
    X, y, panels = _simulated_panels(rng, 6)
    model = PanelCVPlus(LinearRegression()).fit(X, y, panels)
    new = pd.DataFrame(
        {"a": rng.normal(size=600), "b": rng.uniform(size=600)}
    )

    lower, upper = model.predict_interval(new, 0.7)

    for row in (0, 255, 256, 511, 512, 599):
        single_lower, single_upper = model.predict_interval(new.iloc[[row]], 0.7)
        assert float(lower[row]) == pytest.approx(float(single_lower[0]))
        assert float(upper[row]) == pytest.approx(float(single_upper[0]))


def test_coverage_on_new_panels_is_near_the_nominal_level():
    rng = np.random.default_rng(11)
    X, y, panels = _simulated_panels(rng, 24)
    model = PanelCVPlus(LinearRegression()).fit(X, y, panels)

    coverages = []
    for _ in range(100):
        new, observed = _draw_panel(rng, 20)
        lower, upper = model.predict_interval(new, 0.8)
        coverages.append(interval_coverage(observed, lower, upper))

    # Pooling over 24 panels adds a small safety margin, so the average
    # sits a little above 80% instead of below it.
    assert 0.74 <= float(np.mean(coverages)) <= 0.95


@pytest.mark.parametrize("make_estimator", [SeasonDistrictTable, build_m1])
def test_works_with_the_real_model_ladder(make_estimator):
    frame = _survey_frame()

    model = PanelCVPlus(make_estimator()).fit(
        frame,
        frame["yield_t_ha"],
        frame["panel_id"],
    )
    lower, upper = model.predict_interval(frame, level=0.8)
    estimate = model.predict(frame)

    assert model.n_panels_ == 6
    assert len(model.fold_estimators_) == 6
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()
    assert (lower <= estimate).all()
    assert (estimate <= upper).all()
    assert (lower < upper).all()


def test_prototype_estimator_is_left_unfitted():
    prototype = LinearRegression()
    rng = np.random.default_rng(8)
    X, y, panels = _simulated_panels(rng, 4)

    PanelCVPlus(prototype).fit(X, y, panels)

    assert not hasattr(prototype, "coef_")


def _small_inputs():
    X = pd.DataFrame({"pred": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    panels = pd.Series(["A", "A", "B", "B"])
    return X, y, panels


def test_x_must_be_a_dataframe():
    _, y, panels = _small_inputs()

    with pytest.raises(TypeError):
        PanelCVPlus(_Passthrough()).fit(np.zeros((4, 1)), y, panels)


def test_inputs_must_have_the_same_length():
    X, y, panels = _small_inputs()

    with pytest.raises(ValueError, match="same number of rows"):
        PanelCVPlus(_Passthrough()).fit(X, y.iloc[:3], panels)


def test_pandas_inputs_must_share_the_index_of_x():
    X, y, panels = _small_inputs()
    y.index = [10, 11, 12, 13]

    with pytest.raises(ValueError, match="index"):
        PanelCVPlus(_Passthrough()).fit(X, y, panels)


def test_y_must_be_finite():
    X, y, panels = _small_inputs()
    y.iloc[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        PanelCVPlus(_Passthrough()).fit(X, y, panels)


def test_panel_ids_cannot_be_missing():
    X, y, _ = _small_inputs()
    panels = pd.Series(["A", "A", None, "B"])

    with pytest.raises(ValueError, match="missing"):
        PanelCVPlus(_Passthrough()).fit(X, y, panels)


def test_at_least_two_panels_are_needed():
    X, y, _ = _small_inputs()
    panels = pd.Series(["A"] * 4)

    with pytest.raises(ValueError, match="at least two"):
        PanelCVPlus(_Passthrough()).fit(X, y, panels)


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1, float("nan"), True])
def test_level_must_be_strictly_between_zero_and_one(level):
    X, y, panels = _small_inputs()
    model = PanelCVPlus(_Passthrough()).fit(X, y, panels)

    with pytest.raises(ValueError, match="between 0 and 1"):
        model.predict_interval(X, level)


def test_predicting_before_fitting_fails():
    X, _, _ = _small_inputs()
    model = PanelCVPlus(_Passthrough())

    with pytest.raises(NotFittedError):
        model.predict_interval(X)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_requires_a_dataframe():
    X, y, panels = _small_inputs()
    model = PanelCVPlus(_Passthrough()).fit(X, y, panels)

    with pytest.raises(TypeError):
        model.predict(np.zeros((2, 1)))


def test_interval_coverage_counts_rows_inside_inclusive_bounds():
    observed = [1.0, 2.0, 3.0, 4.0]
    lower = [1.0, 2.5, 2.0, 0.0]
    upper = [1.0, 3.0, 3.0, 3.9]

    assert interval_coverage(observed, lower, upper) == pytest.approx(0.5)


def test_interval_coverage_rejects_misaligned_or_empty_input():
    with pytest.raises(ValueError):
        interval_coverage([1.0, 2.0], [0.0], [3.0, 3.0])
    with pytest.raises(ValueError):
        interval_coverage([], [], [])
    with pytest.raises(ValueError):
        interval_coverage([1.0], [np.nan], [2.0])
