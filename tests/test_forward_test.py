"""Tests for the forward test on locked panels."""

from functools import cache

import numpy as np
import pandas as pd
import pytest

from yieldlite.forward_test import (
    HOUSEHOLD_COLUMNS,
    run_forward_test,
    summarize_forward,
)

_DEV = {
    "DX_2020": "DX",
    "HT_2020": "HT",
    "TD_2020": "TD",
    "DX_2021": "DX",
    "TD_2021": "TD",
    "DX_2022": "DX",
    "HT_2022": "HT",
    "TD_2022": "TD",
}
_LOCKED = {
    "DX_2023": "DX",
    "TD_2023": "TD",
    "HT_2024": "HT",
    "TD_2024": "TD",
}
_DISTRICTS = ["CD", "OM", "TL", "TN", "VT"]
_SEASON_EFFECT = {"DX": 6.5, "HT": 5.0, "TD": 4.5}


def _frame(panels, households=8, shift=0.0, seed=1):
    """Households whose yield depends on season, district and fertilizer."""
    rng = np.random.default_rng(seed)
    rows = []

    for panel, season in panels.items():
        for k, district in enumerate(_DISTRICTS):
            for i in range(households):
                npk = 120.0 + 10.0 * (i + k)
                rows.append(
                    {
                        "panel_id": panel,
                        "season": season,
                        "district_code": district,
                        "variety_raw": "A" if i % 2 == 0 else "B",
                        "variety_flag": "known",
                        "npk_total_kg_ha": npk,
                        "flagged": False,
                        "yield_t_ha": _SEASON_EFFECT[season]
                        + 0.2 * k
                        + 0.04 * (npk - 170.0)
                        + shift
                        + rng.normal(0, 0.2),
                    }
                )

    return pd.DataFrame(rows)


@cache
def _development():
    return _frame(_DEV)


@cache
def _holdout():
    return _frame(_LOCKED, households=3, seed=2)


@cache
def _households():
    return run_forward_test(_development(), _holdout(), min_variety_count=5)


def test_every_household_and_level_gets_a_row():
    households = _households()
    holdout = _holdout()

    assert list(households.columns) == list(HOUSEHOLD_COLUMNS)
    assert len(households) == 2 * len(holdout)
    assert set(households["level"]) == {0.5, 0.8}
    assert set(households["panel_id"]) == set(_LOCKED)


def test_the_observed_yield_is_carried_through_unchanged():
    households = _households()
    holdout = _holdout()

    first = households[households["level"] == 0.8]

    assert np.allclose(
        first["observed"].to_numpy(),
        holdout["yield_t_ha"].to_numpy(),
    )


def test_ordinary_households_get_an_estimate_inside_their_interval():
    households = _households()

    assert (households["status"] == "ok").all()
    assert (households["lower"] <= households["estimate"]).all()
    assert (households["estimate"] <= households["upper"]).all()


def test_the_wider_level_gives_the_wider_interval():
    households = _households()

    width = (households["upper"] - households["lower"]).groupby(
        households["level"]
    ).mean()

    assert width[0.8] > width[0.5]


def test_the_model_beats_the_baseline_when_fertilizer_matters():
    households = _households()

    summary = summarize_forward(households)
    overall = summary[
        (summary["panel_id"] == "mean of panels") & (summary["level"] == 0.8)
    ].iloc[0]

    assert overall["skill"] > 0.5
    assert overall["mae"] < overall["mae_m0"]


def test_the_summary_has_one_row_per_panel_and_level_plus_the_mean():
    households = _households()

    summary = summarize_forward(households)

    assert len(summary) == 2 * (len(_LOCKED) + 1)
    assert summary["panel_id"].tolist()[:5] == [
        "DX_2023",
        "HT_2024",
        "TD_2023",
        "TD_2024",
        "mean of panels",
    ]
    assert summary["n"].dtype.kind == "i"


def test_the_mean_row_averages_the_panel_scores():
    households = _households()

    summary = summarize_forward(households)
    level = summary[summary["level"] == 0.8]
    panels = level[level["panel_id"] != "mean of panels"]
    mean = level[level["panel_id"] == "mean of panels"].iloc[0]

    assert mean["mae"] == pytest.approx(panels["mae"].mean())
    assert mean["coverage"] == pytest.approx(panels["coverage"].mean())
    assert mean["n"] == panels["n"].sum()


def test_an_unknown_district_is_refused_and_counted():
    development = _development()
    holdout = _holdout()

    changed = holdout.head(12).copy()
    changed.loc[changed.index[:5], "district_code"] = "XX"

    result = run_forward_test(
        development,
        changed,
        levels=[0.8],
        min_variety_count=5,
    )
    refused = result[result["status"] == "refused"]

    assert len(refused) == 5
    assert refused["reasons"].eq("unknown_district").all()
    assert refused["estimate"].isna().all()

    summary = summarize_forward(result)
    total = summary[summary["panel_id"] == "mean of panels"].iloc[0]

    assert total["n_refused"] == 5
    assert total["n_scored"] == len(changed) - 5


def test_a_missing_fertilizer_amount_is_invalid_not_a_crash():
    development = _development()
    holdout = _holdout()

    changed = holdout.head(12).copy()
    changed.loc[changed.index[:3], "npk_total_kg_ha"] = np.nan

    result = run_forward_test(
        development,
        changed,
        levels=[0.8],
        min_variety_count=5,
    )

    assert (result["status"] == "invalid").sum() == 3
    assert result.loc[result["status"] == "invalid", "estimate"].isna().all()

    summary = summarize_forward(result)
    total = summary[summary["panel_id"] == "mean of panels"].iloc[0]

    assert total["n_invalid"] == 3


def test_fertilizer_outside_the_training_range_is_refused():
    development = _development()
    holdout = _holdout()

    changed = holdout.head(12).copy()
    changed.loc[changed.index[:4], "npk_total_kg_ha"] = 5000.0

    result = run_forward_test(
        development,
        changed,
        levels=[0.8],
        min_variety_count=5,
    )

    assert (result["reasons"] == "npk_out_of_range").sum() == 4


def test_an_unresolved_variety_flag_hides_the_variety():
    development = _development()
    holdout = _holdout()

    changed = holdout.head(12).copy()
    changed["variety_flag"] = "unresolved"

    result = run_forward_test(
        development,
        changed,
        levels=[0.8],
        min_variety_count=5,
    )

    assert (result["status"] == "flagged").all()
    assert result["reasons"].str.contains("variety_pooled").all()


def test_a_level_the_panels_cannot_support_is_a_refusal():
    development = _development()
    holdout = _holdout()

    result = run_forward_test(
        development,
        holdout,
        levels=[0.95],
        min_variety_count=5,
    )

    assert (result["status"] == "refused").all()
    assert result["reasons"].eq("level_unsupported").all()


def test_a_shifted_season_lowers_coverage():
    development = _development()

    shifted = _frame(_LOCKED, households=3, shift=3.0, seed=2)

    result = run_forward_test(
        development,
        shifted,
        levels=[0.8],
        min_variety_count=5,
    )
    summary = summarize_forward(result)
    mean = summary[summary["panel_id"] == "mean of panels"].iloc[0]

    assert mean["coverage"] < 0.5
    assert mean["bias"] < -2.0


def test_history_coverage_is_scored_where_a_range_exists():
    households = _households()

    summary = summarize_forward(households)

    assert summary["history_coverage"].between(0, 1).all()


def test_panels_shared_with_the_development_data_are_rejected():
    development = _development()

    with pytest.raises(ValueError, match="share panels"):
        run_forward_test(development, development.copy())


def test_bad_inputs_are_rejected():
    development = _development()
    holdout = _holdout()

    with pytest.raises(TypeError, match="DataFrame"):
        run_forward_test(development, [1, 2, 3])

    with pytest.raises(ValueError, match="missing columns"):
        run_forward_test(development, holdout.drop(columns=["yield_t_ha"]))

    with pytest.raises(ValueError, match="empty"):
        run_forward_test(development, holdout.iloc[0:0])

    with pytest.raises(ValueError, match="levels"):
        run_forward_test(development, holdout, levels=[])

    with pytest.raises(ValueError, match="between 0 and 1"):
        run_forward_test(development, holdout, levels=[1.2])


def test_summarizing_bad_rows_is_rejected():
    households = _households()

    with pytest.raises(ValueError, match="missing columns"):
        summarize_forward(households.drop(columns=["estimate"]))

    with pytest.raises(ValueError, match="empty"):
        summarize_forward(households.iloc[0:0])


def test_the_hand_computed_scores_are_right():
    rows = pd.DataFrame(
        {
            "panel_id": ["P"] * 4,
            "district": ["CD"] * 4,
            "season": ["DX"] * 4,
            "level": [0.8] * 4,
            "status": ["ok", "ok", "ok", "refused"],
            "reasons": ["", "", "", "interval_uninformative"],
            "observed": [6.0, 7.0, 8.0, 5.0],
            "estimate": [6.5, 7.0, 7.0, np.nan],
            "lower": [6.0, 6.5, 6.5, np.nan],
            "upper": [7.0, 7.5, 7.5, np.nan],
            "history_lower": [6.0, 6.0, 6.0, 6.0],
            "history_upper": [7.0, 7.0, 7.0, 7.0],
            "m0_estimate": [6.0, 6.0, 6.0, 6.0],
        }
    )
    summary = summarize_forward(rows)
    panel = summary[summary["panel_id"] == "P"].iloc[0]

    assert panel["n"] == 4
    assert panel["n_refused"] == 1
    assert panel["n_scored"] == 3
    assert panel["refusal_rate"] == pytest.approx(0.25)
    assert panel["mae"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert panel["mae_m0"] == pytest.approx((0.0 + 1.0 + 2.0) / 3)
    assert panel["skill"] == pytest.approx(1 - 0.5 / 1.0)
    assert panel["bias"] == pytest.approx((0.5 + 0.0 - 1.0) / 3)
    assert panel["coverage"] == pytest.approx(2 / 3)
    assert panel["width"] == pytest.approx(1.0)
    assert panel["history_coverage"] == pytest.approx(2 / 4)
