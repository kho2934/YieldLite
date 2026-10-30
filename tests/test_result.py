"""Tests for prediction results, warnings, refusals, and summaries."""

import dataclasses
import math

import pytest

from yieldlite.messages import CATALOG
from yieldlite.result import (
    FLAG_CODES,
    REFUSAL_CODES,
    HistoryRange,
    Interval,
    PredictionResult,
)
from yieldlite.selection import Reason

_INTERVAL = Interval(
    lower=4.8,
    upper=6.1,
    level=0.8,
)

_HISTORY = HistoryRange(
    lower=5.0,
    upper=6.2,
    level=0.8,
    n_rows=40,
)


def _ok(**changes):
    """Build a valid prediction and replace any fields supplied by a test."""
    fields = {
        "estimate": 5.4,
        "interval": _INTERVAL,
        "rung": "m1",
        "district": "CD",
        "season": "DX",
    }
    fields.update(changes)
    return PredictionResult(**fields)


def _refused(**changes):
    """Build a basic refused result."""
    fields = {
        "reasons": (
            Reason(
                "unknown_district",
                {"district": "XX"},
            ),
        )
    }
    fields.update(changes)
    return PredictionResult(**fields)


def test_a_clean_prediction_is_ok():
    result = _ok()

    assert result.status == "ok"
    assert not result.refused
    assert result.reasons == ()


def test_a_warning_keeps_the_prediction_but_marks_it_as_flagged():
    reason = Reason(
        "variety_pooled",
        {"variety": "OM18"},
    )

    result = _ok(reasons=[reason])

    assert result.status == "flagged"
    assert not result.refused
    assert result.estimate == 5.4
    assert result.interval == _INTERVAL


def test_a_refusal_has_no_estimate_or_interval():
    result = _refused()

    assert result.status == "refused"
    assert result.refused
    assert result.estimate is None
    assert result.interval is None


def test_a_refusal_takes_priority_over_a_warning():
    result = _refused(
        reasons=(
            Reason(
                "variety_pooled",
                {"variety": "OM18"},
            ),
            Reason(
                "too_few_panels",
                {"panels": 3, "needed": 4},
            ),
        )
    )

    assert result.status == "refused"


def test_a_refused_result_cannot_include_a_prediction():
    with pytest.raises(ValueError, match="refused"):
        _refused(estimate=5.4)

    with pytest.raises(ValueError, match="refused"):
        _refused(interval=_INTERVAL)


def test_a_prediction_needs_an_estimate_interval_and_model():
    with pytest.raises(ValueError, match="estimate and an interval"):
        PredictionResult(rung="m0")

    with pytest.raises(ValueError, match="estimate and an interval"):
        _ok(interval=None)

    with pytest.raises(ValueError, match="model"):
        _ok(rung=None)


def test_the_interval_must_contain_the_estimate():
    with pytest.raises(ValueError, match="contain"):
        _ok(estimate=7.0)

    assert _ok(estimate=4.8).estimate == 4.8
    assert _ok(estimate=6.1).estimate == 6.1


@pytest.mark.parametrize(
    "estimate",
    [math.nan, math.inf, True, "5.4"],
)
def test_the_estimate_must_be_a_finite_number(estimate):
    with pytest.raises(ValueError, match="estimate"):
        _ok(estimate=estimate)


def test_a_whole_number_estimate_is_stored_as_a_float():
    result = _ok(
        estimate=5,
        interval=Interval(
            lower=4,
            upper=6,
            level=0.8,
        ),
    )

    assert result.estimate == 5.0
    assert isinstance(result.estimate, float)


def test_reasons_are_stored_as_a_tuple():
    reason = Reason(
        "variety_pooled",
        {"variety": "OM18"},
    )

    assert _ok(reasons=[reason]).reasons == (reason,)


def test_only_prediction_warning_and_refusal_codes_are_allowed():
    with pytest.raises(ValueError, match="no_gain"):
        _ok(
            reasons=(
                Reason("no_gain", {}),
            )
        )

    with pytest.raises(ValueError, match="not_a_code"):
        _ok(
            reasons=(
                Reason("not_a_code", {}),
            )
        )


def test_reasons_must_be_reason_objects():
    with pytest.raises(TypeError):
        _ok(reasons=("variety_pooled",))


def test_refusal_and_flag_codes_are_separate_and_have_messages():
    assert not REFUSAL_CODES & FLAG_CODES
    assert (REFUSAL_CODES | FLAG_CODES) <= set(CATALOG)


def test_each_refusal_code_refuses_and_each_flag_only_warns():
    for code in REFUSAL_CODES:
        result = _refused(
            reasons=(
                Reason(code, {}),
            )
        )
        assert result.status == "refused"

    for code in FLAG_CODES:
        result = _ok(
            reasons=(
                Reason(code, {}),
            )
        )
        assert result.status == "flagged"


def test_the_model_must_belong_to_the_ladder():
    with pytest.raises(ValueError, match="rung"):
        _ok(rung="m9")

    assert _refused(rung="m0").rung == "m0"


def test_history_needs_a_district_and_season():
    with pytest.raises(ValueError, match="district"):
        _ok(
            history=_HISTORY,
            district=None,
        )

    with pytest.raises(ValueError, match="season"):
        _ok(
            history=_HISTORY,
            season=None,
        )


def test_history_must_be_a_history_range():
    with pytest.raises(TypeError):
        _ok(history=(5.0, 6.2))


def test_interval_bounds_must_be_ordered_and_finite():
    with pytest.raises(ValueError, match="larger"):
        Interval(
            lower=6.0,
            upper=5.0,
            level=0.8,
        )

    with pytest.raises(ValueError, match="finite"):
        Interval(
            lower=math.nan,
            upper=5.0,
            level=0.8,
        )

    with pytest.raises(ValueError, match="finite"):
        HistoryRange(
            lower=5.0,
            upper=math.inf,
            level=0.8,
            n_rows=10,
        )


@pytest.mark.parametrize(
    "level",
    [0, 1, -0.1, 1.5, math.nan, True, "0.8"],
)
def test_level_must_be_strictly_between_zero_and_one(level):
    with pytest.raises(ValueError, match="level"):
        Interval(
            lower=4.0,
            upper=5.0,
            level=level,
        )

    with pytest.raises(ValueError, match="level"):
        HistoryRange(
            lower=4.0,
            upper=5.0,
            level=level,
            n_rows=10,
        )


@pytest.mark.parametrize(
    "n_rows",
    [0, -3, 2.5, True, "10", None],
)
def test_history_needs_a_positive_whole_number_of_rows(n_rows):
    with pytest.raises(ValueError, match="n_rows"):
        HistoryRange(
            lower=4.0,
            upper=5.0,
            level=0.8,
            n_rows=n_rows,
        )


def test_a_result_cannot_be_changed_after_creation():
    result = _ok()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.estimate = 9.0


def test_clean_summary_contains_the_prediction_and_history():
    lines = _ok(history=_HISTORY).summary("en")

    assert lines == [
        "Expected yield: 5.4 t/ha, most likely between 4.8 and 6.1 t/ha "
        "(80% range).",
        "Past yields for CD in DX were usually between 5.0 and 6.2 t/ha "
        "(80% of past records).",
    ]


def test_summary_places_warnings_between_prediction_and_history():
    reason = Reason(
        "variety_pooled",
        {"variety": "OM18"},
    )

    lines = _ok(
        reasons=(reason,),
        history=_HISTORY,
    ).summary("en")

    assert len(lines) == 3
    assert lines[0].startswith("Expected yield")
    assert "OM18" in lines[1]
    assert lines[2].startswith("Past yields")


def test_refusal_with_history_shows_only_reason_and_past_range():
    result = _refused(
        history=_HISTORY,
        district="CD",
        season="DX",
    )

    lines = result.summary("en")

    assert len(lines) == 2
    assert "no prediction" in lines[0]
    assert lines[1].startswith("Past yields")
    assert not any(
        "Expected yield" in line
        for line in lines
    )


def test_refusal_without_history_shows_only_the_reason():
    lines = _refused().summary("en")

    assert len(lines) == 1
    assert "XX" in lines[0]


def test_summary_can_be_written_in_fluent_vietnamese():
    lines = _ok(history=_HISTORY).summary("vi")

    assert lines[0] == (
        "Năng suất dự kiến: 5.4 t/ha, với khoảng dự đoán từ 4.8 đến "
        "6.1 t/ha (mức 80%)."
    )
    assert lines[1] == (
        "Năng suất trước đây tại huyện CD trong vụ DX thường nằm trong "
        "khoảng từ 5.0 đến 6.2 t/ha (bao gồm 80% số liệu lịch sử)."
    )


def test_levels_are_shown_as_whole_percentages():
    result = _ok(
        estimate=5.4,
        interval=Interval(
            lower=4.0,
            upper=7.0,
            level=0.889,
        ),
    )

    assert "(89% range)" in result.summary("en")[0]
