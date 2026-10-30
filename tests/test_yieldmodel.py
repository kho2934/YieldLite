"""Tests for fitting, explaining, and using YieldModel."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.result import PredictionResult
from yieldlite.yieldmodel import YieldModel

# Eight synthetic panels with the same season structure as the development
# data used by the project.
_PANELS = {
    "DX_2020": "DX",
    "HT_2020": "HT",
    "TD_2020": "TD",
    "DX_2021": "DX",
    "TD_2021": "TD",
    "DX_2022": "DX",
    "HT_2022": "HT",
    "TD_2022": "TD",
}

_DISTRICTS = [
    "CD",
    "OM",
    "TL",
    "TN",
    "VT",
]

_SEASON_EFFECT = {
    "DX": 6.5,
    "HT": 5.0,
    "TD": 4.5,
}


def _frame(
    panels=None,
    fertilizer_effect=0.04,
    district_effect=0.2,
    offsets=None,
    households=12,
    thin_district=None,
):
    """Build synthetic household data with known yield relationships."""
    rng = np.random.default_rng(1)
    rows = []

    for panel, season in (
        panels or _PANELS
    ).items():
        for district_number, district in enumerate(
            _DISTRICTS
        ):
            count = (
                3
                if district == thin_district
                else households
            )

            for household_number in range(count):
                npk = (
                    120.0
                    + 10.0
                    * (
                        household_number
                        + district_number
                    )
                )

                rows.append(
                    {
                        "panel_id": panel,
                        "season": season,
                        "district_code": district,
                        "variety_raw": (
                            "A"
                            if household_number % 2 == 0
                            else "B"
                        ),
                        "variety_flag": "known",
                        "npk_total_kg_ha": npk,
                        "flagged": False,
                        "yield_t_ha": (
                            _SEASON_EFFECT[season]
                            + district_effect
                            * district_number
                            + fertilizer_effect
                            * (npk - 170.0)
                            + (offsets or {}).get(
                                panel,
                                0.0,
                            )
                            + rng.normal(0, 0.2)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _fit(frame=None, **kwargs):
    """Fit YieldModel with settings suited to the synthetic data."""
    settings = {
        "min_variety_count": 5,
    }
    settings.update(kwargs)

    training_data = (
        _frame()
        if frame is None
        else frame
    )

    return YieldModel(
        **settings
    ).fit(training_data)


def _predict(model, **changes):
    """Make a standard prediction and replace any requested inputs."""
    arguments = {
        "district": "CD",
        "season": "DX",
        "variety": "A",
        "npk_total": 170.0,
    }
    arguments.update(changes)

    return model.predict(**arguments)


def test_data_with_useful_inputs_selects_m1():
    model = _fit()

    assert model.rung == "m1"
    assert model.intervals_ is not None
    assert model.profile_.n_panels == 8


def test_a_clean_prediction_has_an_estimate_inside_its_interval():
    result = _predict(_fit())

    assert isinstance(
        result,
        PredictionResult,
    )
    assert result.status == "ok"
    assert result.rung == "m1"
    assert result.district == "CD"
    assert result.season == "DX"

    # In this synthetic dataset, CD during DX at 170 kg/ha is designed to
    # produce about 6.5 t/ha.
    assert abs(result.estimate - 6.5) < 0.4
    assert (
        result.interval.lower
        <= result.estimate
        <= result.interval.upper
    )
    assert result.interval.level == 0.8
    assert result.history is not None
    assert result.history.n_rows == 36


def test_the_prediction_interval_is_narrower_than_the_past_range():
    result = _predict(_fit())

    prediction_width = (
        result.interval.upper
        - result.interval.lower
    )
    historical_width = (
        result.history.upper
        - result.history.lower
    )

    assert prediction_width < historical_width


def test_more_fertilizer_produces_a_higher_estimate():
    model = _fit()

    low = _predict(
        model,
        npk_total=130.0,
    )
    high = _predict(
        model,
        npk_total=200.0,
    )

    assert high.estimate > low.estimate + 1.0


def test_labels_are_cleaned_before_prediction():
    model = _fit()

    result = _predict(
        model,
        district=" cd ",
        season="dx",
        variety="a",
    )

    assert result.district == "CD"
    assert result.season == "DX"
    assert result.status == "ok"


def test_an_unseen_variety_is_pooled_and_flagged():
    result = _predict(
        _fit(),
        variety="NEWVARIETY",
    )

    assert result.status == "flagged"
    assert [
        reason.code
        for reason in result.reasons
    ] == ["variety_pooled"]
    assert result.estimate is not None


def test_a_missing_variety_is_pooled_and_flagged():
    result = _predict(
        _fit(),
        variety=None,
    )

    assert result.status == "flagged"
    assert result.reasons[0].code == "variety_pooled"


def test_an_unknown_district_is_refused_without_history():
    result = _predict(
        _fit(),
        district="XX",
    )

    assert result.status == "refused"
    assert result.reasons[0].code == "unknown_district"
    assert result.estimate is None
    assert result.history is None


def test_an_unknown_season_is_refused():
    result = _predict(
        _fit(),
        season="ZZ",
    )

    assert result.status == "refused"
    assert result.reasons[0].code == "season_unresolved"


def test_fertilizer_outside_the_training_range_is_refused_with_history():
    result = _predict(
        _fit(),
        npk_total=900.0,
    )

    assert result.status == "refused"
    assert result.reasons[0].code == "npk_out_of_range"
    assert result.reasons[0].params["npk"] == 900
    assert result.estimate is None
    assert result.history is not None


def test_an_interval_level_the_data_cannot_support_is_refused():
    # With eight panels, the highest supported finite level is 8/9,
    # or about 88.9%.
    result = _predict(
        _fit(),
        level=0.95,
    )

    assert result.status == "refused"

    reason = result.reasons[0]

    assert reason.code == "level_unsupported"
    assert reason.params == {
        "level": 95,
        "needed": 19,
        "panels": 8,
    }
    assert result.history is not None


def test_a_supported_interval_level_is_accepted():
    result = _predict(
        _fit(),
        level=0.85,
    )

    assert result.status == "ok"
    assert result.interval.level == 0.85


def test_an_uninformative_interval_is_refused_with_history():
    # Large panel-to-panel shocks should produce an interval that is too wide
    # to improve on the usual historical range.
    swings = {
        panel: (
            4.0
            if index % 2 == 0
            else -4.0
        )
        for index, panel in enumerate(_PANELS)
    }

    model = _fit(
        _frame(offsets=swings),
        refuse_if_uninformative=True,
    )
    result = _predict(model)

    assert result.status == "refused"
    assert (
        result.reasons[0].code
        == "interval_uninformative"
    )
    assert result.estimate is None
    assert result.history is not None


def test_the_interval_width_rule_is_off_by_default():
    swings = {
        panel: (
            4.0
            if index % 2 == 0
            else -4.0
        )
        for index, panel in enumerate(_PANELS)
    }

    model = _fit(_frame(offsets=swings))
    result = _predict(model)

    assert result.status == "ok"
    assert result.estimate is not None


def test_a_thin_historical_group_is_flagged_instead_of_refused():
    # Three VT households in each of the three DX panels give only nine
    # historical rows, which is below the required minimum of ten.
    model = _fit(
        _frame(thin_district="VT")
    )
    result = _predict(
        model,
        district="VT",
    )

    assert result.status == "flagged"
    assert [
        reason.code
        for reason in result.reasons
    ] == ["history_thin"]
    assert result.history is None
    assert result.estimate is not None


def test_too_few_panels_refuses_predictions_but_keeps_history():
    panels = {
        "DX_2020": "DX",
        "HT_2020": "HT",
        "TD_2020": "TD",
    }

    model = _fit(
        _frame(panels)
    )

    assert model.rung is None
    assert model.intervals_ is None

    result = _predict(model)

    assert result.status == "refused"
    assert result.reasons[0].code == "too_few_panels"
    assert result.reasons[0].params == {
        "panels": 3,
        "needed": 4,
    }
    assert result.history is not None


def test_m0_does_not_require_fertilizer_or_variety():
    thresholds = {
        "thresholds": {
            "P0": {
                "status": "reached",
                "value": 4,
            },
            "P1": {
                "status": "not_reached",
                "value": None,
            },
            "H1": {
                "status": "reached",
                "value": 20,
            },
            "W2": {
                "status": "not_evaluated",
                "value": None,
            },
            "N3": {
                "status": "not_evaluated",
                "value": None,
            },
            "P3": {
                "status": "not_evaluated",
                "value": None,
            },
        }
    }

    model = _fit(
        thresholds=thresholds
    )
    result = model.predict(
        district="CD",
        season="DX",
    )

    assert model.rung == "m0"
    assert result.status in (
        "ok",
        "refused",
    )
    assert not any(
        reason.code == "variety_pooled"
        for reason in result.reasons
    )


def test_m1_requires_a_fertilizer_amount():
    with pytest.raises(
        ValueError,
        match="npk_total is required",
    ):
        _fit().predict(
            district="CD",
            season="DX",
            variety="A",
        )


@pytest.mark.parametrize(
    "npk",
    [-5.0, np.nan, np.inf, True, "170"],
)
def test_invalid_fertilizer_values_are_rejected(npk):
    with pytest.raises(
        ValueError,
        match="npk_total",
    ):
        _predict(
            _fit(),
            npk_total=npk,
        )


@pytest.mark.parametrize(
    "level",
    [0, 1, -0.2, 1.5, np.nan, True, "0.8"],
)
def test_invalid_interval_levels_are_rejected(level):
    with pytest.raises(
        ValueError,
        match="level",
    ):
        _predict(
            _fit(),
            level=level,
        )


@pytest.mark.parametrize(
    "value",
    ["", "  ", None, 3],
)
def test_district_and_season_must_be_non_empty_text(value):
    model = _fit()

    with pytest.raises(
        ValueError,
        match="district",
    ):
        _predict(
            model,
            district=value,
        )

    with pytest.raises(
        ValueError,
        match="season",
    ):
        _predict(
            model,
            season=value,
        )


def test_explain_returns_each_decision_in_both_languages():
    model = _fit()

    english = model.explain("en")
    vietnamese = model.explain("vi")

    assert (
        len(english)
        == len(vietnamese)
        == len(model.selection_.reasons)
    )
    assert english[-1].startswith("Using")
    assert vietnamese[-1].startswith("Chọn")
    assert all(
        "{" not in line
        for line in english + vietnamese
    )


def test_explain_says_why_fitting_was_refused():
    panels = {
        "DX_2020": "DX",
        "HT_2020": "HT",
        "TD_2020": "TD",
    }

    lines = _fit(
        _frame(panels)
    ).explain("en")

    assert len(lines) == 1
    assert "only 3 seasons" in lines[0]


def test_using_yieldmodel_before_fitting_raises_an_error():
    model = YieldModel()

    with pytest.raises(
        RuntimeError,
        match="fit",
    ):
        model.predict(
            district="CD",
            season="DX",
        )

    with pytest.raises(
        RuntimeError,
        match="fit",
    ):
        model.explain()

    with pytest.raises(
        RuntimeError,
        match="fit",
    ):
        _ = model.rung


def test_fitting_does_not_modify_the_input_data():
    frame = _frame()
    before = frame.copy()

    _fit(frame)

    pd.testing.assert_frame_equal(
        frame,
        before,
    )


def test_fit_rejects_invalid_training_tables():
    with pytest.raises(TypeError):
        YieldModel().fit(
            _frame().to_numpy()
        )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        YieldModel().fit(
            _frame().drop(columns="flagged")
        )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        YieldModel().fit(
            _frame().iloc[0:0]
        )

    duplicated = pd.concat(
        [
            _frame().iloc[:5],
            _frame().iloc[:5],
        ]
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        YieldModel().fit(duplicated)

    missing_yield = _frame()
    missing_yield.loc[
        3,
        "yield_t_ha",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        YieldModel().fit(missing_yield)

    invalid_npk = _frame()
    invalid_npk["npk_total_kg_ha"] = (
        invalid_npk["npk_total_kg_ha"]
        .astype(object)
    )
    invalid_npk.loc[
        3,
        "npk_total_kg_ha",
    ] = "lots"

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        YieldModel().fit(invalid_npk)


def test_prediction_summary_is_available_in_both_languages():
    result = _predict(_fit())

    english = result.summary("en")
    vietnamese = result.summary("vi")

    assert english[0].startswith(
        "Expected yield"
    )
    assert vietnamese[0].startswith(
        "Năng suất dự kiến"
    )
    assert len(english) == len(vietnamese) == 2


def test_public_classes_are_available_from_the_package():
    import yieldlite

    assert yieldlite.YieldModel is YieldModel
    assert (
        yieldlite.PredictionResult
        is PredictionResult
    )
