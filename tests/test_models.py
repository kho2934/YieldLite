"""Tests for the yieldlite prediction models."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import (
    LeaveOneGroupOut,
    cross_val_score,
)

from yieldlite.models import SeasonDistrictTable, build_m1


# M0: season-and-district baseline

def _m0_toy_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small dataset with two seasons and two districts."""
    X = pd.DataFrame(
        {
            "season": ["DX", "DX", "DX", "HT", "HT"],
            "district_code": ["OM", "OM", "CD", "OM", "CD"],
        }
    )
    y = pd.Series([6.0, 7.0, 8.0, 5.0, 5.5])
    return X, y


def test_fit_matches_hand_computed_shrinkage():
    X, y = _m0_toy_data()
    model = SeasonDistrictTable(shrinkage=10.0).fit(X, y)

    # DX/OM has two rows with a mean of 6.5.
    # The average across all three DX rows is 7.0.
    expected = (2 * 6.5 + 10.0 * 7.0) / (2 + 10.0)

    assert model.group_means_[("DX", "OM")] == pytest.approx(
        expected
    )


def test_predict_matches_group_mean_when_seen():
    X, y = _m0_toy_data()
    model = SeasonDistrictTable(shrinkage=10.0).fit(X, y)

    prediction = model.predict(X.iloc[[0]])

    assert prediction.iloc[0] == pytest.approx(
        model.group_means_[("DX", "OM")]
    )


def test_predict_falls_back_to_season_mean_for_unseen_district():
    X, y = _m0_toy_data()
    model = SeasonDistrictTable().fit(X, y)

    unseen = pd.DataFrame(
        {
            "season": ["DX"],
            "district_code": ["BRAND_NEW"],
        }
    )
    prediction = model.predict(unseen)

    assert prediction.iloc[0] == pytest.approx(
        model.season_mean_["DX"]
    )


def test_predict_falls_back_to_overall_mean_for_unseen_season():
    X, y = _m0_toy_data()
    model = SeasonDistrictTable().fit(X, y)

    unseen = pd.DataFrame(
        {
            "season": ["BRAND_NEW"],
            "district_code": ["BRAND_NEW"],
        }
    )
    prediction = model.predict(unseen)

    assert prediction.iloc[0] == pytest.approx(
        model.overall_mean_
    )


def test_shrinkage_pulls_small_group_toward_season_mean():
    X, y = _m0_toy_data()

    loose = SeasonDistrictTable(shrinkage=0.01).fit(X, y)
    tight = SeasonDistrictTable(shrinkage=1000.0).fit(X, y)

    raw_group_mean = 6.5
    season_mean = 7.0

    assert loose.group_means_[("DX", "OM")] == pytest.approx(
        raw_group_mean,
        abs=0.1,
    )
    assert tight.group_means_[("DX", "OM")] == pytest.approx(
        season_mean,
        abs=0.1,
    )


def test_fit_rejects_missing_columns():
    X = pd.DataFrame({"season": ["DX"]})
    y = pd.Series([6.0])

    with pytest.raises(ValueError):
        SeasonDistrictTable().fit(X, y)


def test_predict_rejects_missing_columns():
    X, y = _m0_toy_data()
    model = SeasonDistrictTable().fit(X, y)

    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"season": ["DX"]}))


def test_fit_rejects_invalid_target_values():
    X, y = _m0_toy_data()
    y.iloc[0] = np.nan

    with pytest.raises(ValueError):
        SeasonDistrictTable().fit(X, y)


def test_fit_rejects_negative_shrinkage():
    X, y = _m0_toy_data()

    with pytest.raises(ValueError):
        SeasonDistrictTable(shrinkage=-1.0).fit(X, y)


def test_predict_before_fit_raises_not_fitted_error():
    model = SeasonDistrictTable()
    empty = pd.DataFrame(
        {
            "season": [],
            "district_code": [],
        }
    )

    with pytest.raises(NotFittedError):
        model.predict(empty)


def test_get_params_and_clone_work():
    model = SeasonDistrictTable(shrinkage=5.0)
    cloned = clone(model)

    assert model.get_params() == {"shrinkage": 5.0}
    assert cloned.shrinkage == 5.0
    assert cloned is not model


# M1: regularized feature model

_DISTRICT_EFFECT = {
    "OM": 0.3,
    "CD": -0.2,
    "TN": 0.1,
    "TL": 0.0,
    "VT": -0.1,
}
_SEASON_EFFECT = {
    "DX": 0.4,
    "HT": -0.1,
    "TD": 0.0,
}
_VARIETY_EFFECT = {
    "pooled": 0.0,
    "IR50404": -0.1,
    "OM5451": 0.15,
}

# These are the eight season-year panels reserved for development.
_DEVELOPMENT_PANELS = [
    ("DX", 2020),
    ("HT", 2020),
    ("TD", 2020),
    ("DX", 2021),
    ("TD", 2021),
    ("DX", 2022),
    ("HT", 2022),
    ("TD", 2022),
]


def _m1_toy_data(
    n_panels: int = 8,
    rows_per_panel: int = 20,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create synthetic data with the same panel structure as the project.

    Each panel represents one season-year and contains households from
    several districts. Yield includes district, season, variety, and NPK
    signals so M1 has real patterns to learn.
    """
    if not 1 <= n_panels <= len(_DEVELOPMENT_PANELS):
        raise ValueError(
            f"n_panels must be between 1 and "
            f"{len(_DEVELOPMENT_PANELS)}"
        )

    rng = np.random.default_rng(seed)
    districts = list(_DISTRICT_EFFECT)
    varieties = list(_VARIETY_EFFECT)

    rows = []
    targets = []
    panel_ids = []

    for season, year in _DEVELOPMENT_PANELS[:n_panels]:
        panel_effect = rng.normal(0, 0.05)

        for row_number in range(rows_per_panel):
            # Cycling guarantees that every panel has all five districts.
            district = districts[
                row_number % len(districts)
            ]
            variety = rng.choice(varieties)
            npk = rng.uniform(80, 220)

            yield_value = (
                6.0
                + _DISTRICT_EFFECT[district]
                + _SEASON_EFFECT[season]
                + _VARIETY_EFFECT[variety]
                + 0.01 * npk
                + panel_effect
                + rng.normal(0, 0.15)
            )

            rows.append(
                {
                    "district_code": district,
                    "season": season,
                    "variety_pooled": variety,
                    "npk_total_kg_ha": npk,
                }
            )
            targets.append(yield_value)
            panel_ids.append(f"{season}_{year}")

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="yield")
    panels = pd.Series(panel_ids, name="panel_id")

    return X, y, panels


def test_synthetic_data_uses_season_year_panels():
    X, _, panels = _m1_toy_data(
        n_panels=8,
        rows_per_panel=20,
    )

    assert panels.nunique() == 8

    districts_per_panel = (
        pd.DataFrame(
            {
                "panel_id": panels,
                "district_code": X["district_code"],
            }
        )
        .groupby("panel_id")["district_code"]
        .nunique()
    )

    assert (districts_per_panel > 1).all()


def test_build_m1_fits_and_predicts():
    X, y, _ = _m1_toy_data()

    model = build_m1().fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert predictions.min() > 3.0
    assert predictions.max() < 10.0


def test_build_m1_contains_season_main_effect():
    X, y, _ = _m1_toy_data()
    model = build_m1().fit(X, y)

    feature_names = (
        model.named_steps["encode"]
        .get_feature_names_out()
    )

    assert any(
        name.startswith("season__")
        for name in feature_names
    )


def test_build_m1_uses_quantile_knots():
    model = build_m1()

    npk_transformer = next(
        transformer
        for name, transformer, _ in
        model.named_steps["encode"].transformers
        if name == "npk"
    )

    assert npk_transformer.knots == "quantile"


def test_build_m1_handles_unseen_categories_at_prediction_time():
    X, y, _ = _m1_toy_data()
    model = build_m1().fit(X, y)

    unseen = pd.DataFrame(
        {
            "district_code": ["BRAND_NEW"],
            "season": ["BRAND_NEW"],
            "variety_pooled": ["NEVER_SEEN"],
            "npk_total_kg_ha": [150.0],
        }
    )

    predictions = model.predict(unseen)

    assert len(predictions) == 1
    assert np.isfinite(predictions[0])


def test_build_m1_interaction_shrink_changes_predictions():
    X, y, _ = _m1_toy_data()

    loose = build_m1(
        interaction_shrink=1.0
    ).fit(X, y)
    tight = build_m1(
        interaction_shrink=0.01
    ).fit(X, y)

    loose_predictions = loose.predict(X)
    tight_predictions = tight.predict(X)

    assert not np.allclose(
        loose_predictions,
        tight_predictions,
    )


def test_build_m1_rejects_missing_feature():
    X, y, _ = _m1_toy_data()
    X = X.drop(columns="variety_pooled")

    with pytest.raises(ValueError):
        build_m1().fit(X, y)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"interaction_shrink": 0},
            "interaction_shrink",
        ),
        (
            {"alpha": -1},
            "alpha",
        ),
        (
            {"n_knots": 1},
            "n_knots",
        ),
    ],
)
def test_build_m1_rejects_invalid_parameters(
    arguments,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_m1(**arguments)


def test_build_m1_is_clonable():
    model = build_m1(
        interaction_shrink=0.2,
        alpha=2.0,
        n_knots=3,
    )
    cloned = clone(model)

    assert cloned.get_params()["ridge__alpha"] == 2.0
    assert cloned is not model


def test_build_m1_works_with_leave_one_group_out():
    X, y, panels = _m1_toy_data(
        n_panels=8,
        rows_per_panel=15,
    )

    scores = cross_val_score(
        build_m1(),
        X,
        y,
        groups=panels,
        cv=LeaveOneGroupOut(),
        scoring="neg_mean_absolute_error",
    )

    assert len(scores) == panels.nunique()
    assert np.isfinite(scores).all()
