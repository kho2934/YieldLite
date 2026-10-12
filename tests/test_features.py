"""Tests for the non-weather features used by yieldlite."""

import pandas as pd
import pytest

from yieldlite.features import (
    SeasonCalendar,
    VarietyPooler,
    npk_total,
)


# NPK calculations

def _nutrient_data(
    nitrogen: float = 100.0,
    phosphorus: float = 50.0,
    potassium: float = 40.0,
) -> pd.DataFrame:
    """Create one row of nutrient data for the NPK tests."""
    return pd.DataFrame(
        {
            "nitrogen_kg_ha": [nitrogen],
            "phosphorus_kg_ha": [phosphorus],
            "potassium_kg_ha": [potassium],
        }
    )


def test_npk_total_oxide_form_sums_directly():
    result = npk_total(
        _nutrient_data(),
        p_form="oxide",
        k_form="oxide",
    )

    assert result.iloc[0] == pytest.approx(190.0)


def test_npk_total_elemental_form_converts_first():
    result = npk_total(
        _nutrient_data(),
        p_form="elemental",
        k_form="elemental",
    )

    expected = (
        100.0
        + 50.0 * (141.943 / 61.948)
        + 40.0 * (94.195 / 78.196)
    )
    assert result.iloc[0] == pytest.approx(expected)


def test_npk_total_mixed_forms():
    df = _nutrient_data(
        nitrogen=0.0,
        phosphorus=10.0,
        potassium=10.0,
    )

    result = npk_total(
        df,
        p_form="elemental",
        k_form="oxide",
    )

    expected = 10.0 * (141.943 / 61.948) + 10.0
    assert result.iloc[0] == pytest.approx(expected)


def test_npk_total_rejects_unknown_form():
    with pytest.raises(ValueError):
        npk_total(
            _nutrient_data(0.0, 0.0, 0.0),
            p_form="bogus",
            k_form="oxide",
        )


# Variety pooling

def _variety_data() -> tuple[pd.Series, pd.Series]:
    """Create common, rare, other, and unclear variety examples."""
    rows = (
        [{"variety_raw": "OM5451", "variety_flag": "known"}] * 40
        + [{"variety_raw": "OM18", "variety_flag": "known"}] * 5
        + [{"variety_raw": "KHAC", "variety_flag": "other"}] * 40
        + [
            {
                "variety_raw": "KHONGRO",
                "variety_flag": "unclear",
            }
        ]
        * 100
    )

    df = pd.DataFrame(rows)
    return df["variety_raw"], df["variety_flag"]


def test_variety_pooler_keeps_well_supported_variety():
    variety_raw, variety_flag = _variety_data()

    pooler = VarietyPooler(min_count=30).fit(
        variety_raw,
        variety_flag,
    )
    result = pooler.transform(variety_raw, variety_flag)

    assert (
        result[variety_raw == "OM5451"] == "OM5451"
    ).all()


def test_variety_pooler_pools_rare_known_variety():
    variety_raw, variety_flag = _variety_data()

    pooler = VarietyPooler(min_count=30).fit(
        variety_raw,
        variety_flag,
    )
    result = pooler.transform(variety_raw, variety_flag)

    assert (
        result[variety_raw == "OM18"]
        == VarietyPooler.POOLED
    ).all()


def test_variety_pooler_keeps_other_with_support():
    variety_raw, variety_flag = _variety_data()

    pooler = VarietyPooler(min_count=30).fit(
        variety_raw,
        variety_flag,
    )
    result = pooler.transform(variety_raw, variety_flag)

    assert (result[variety_raw == "KHAC"] == "KHAC").all()


def test_variety_pooler_always_pools_unclear():
    variety_raw, variety_flag = _variety_data()

    # KHONGRO has many rows, but it still does not identify one variety.
    pooler = VarietyPooler(min_count=30).fit(
        variety_raw,
        variety_flag,
    )
    result = pooler.transform(variety_raw, variety_flag)

    assert (
        result[variety_raw == "KHONGRO"]
        == VarietyPooler.POOLED
    ).all()


def test_variety_pooler_pools_unseen_variety_at_prediction_time():
    variety_raw, variety_flag = _variety_data()

    pooler = VarietyPooler(min_count=30).fit(
        variety_raw,
        variety_flag,
    )

    new_raw = pd.Series(["OM5451", "BRAND_NEW_VARIETY"])
    new_flag = pd.Series(["known", "known"])
    result = pooler.transform(new_raw, new_flag)

    assert result.iloc[0] == "OM5451"
    assert result.iloc[1] == VarietyPooler.POOLED


def test_variety_pooler_requires_fit_first():
    pooler = VarietyPooler()

    with pytest.raises(RuntimeError):
        pooler.transform(
            pd.Series(["OM5451"]),
            pd.Series(["known"]),
        )


def test_variety_pooler_rejects_nonpositive_minimum():
    with pytest.raises(ValueError):
        VarietyPooler(min_count=0)


# Season calendar

def _season_series(
    label: str,
    dates,
) -> tuple[pd.Series, pd.Series]:
    """Create matching season and sowing-date Series."""
    season = pd.Series([label] * len(dates))
    sowing_date = pd.Series(pd.to_datetime(list(dates)))
    return season, sowing_date


def test_season_calendar_non_wrapping_round_trip():
    # Example Hè Thu dates that remain within one calendar year.
    dates = pd.date_range(
        "2022-04-10",
        "2022-06-01",
        periods=20,
    )
    season, sowing_date = _season_series(
        "HT",
        dates.strftime("%Y-%m-%d"),
    )

    calendar = SeasonCalendar().fit(season, sowing_date)

    assert calendar.season_for("2022-05-01") == "HT"


def test_season_calendar_handles_new_year_wraparound():
    # Đông Xuân can begin late in one year and continue into the next.
    before_new_year = pd.date_range(
        "2021-10-05",
        "2021-12-31",
        periods=10,
    )
    after_new_year = pd.date_range(
        "2022-01-01",
        "2022-01-20",
        periods=10,
    )

    dates = (
        list(before_new_year.strftime("%Y-%m-%d"))
        + list(after_new_year.strftime("%Y-%m-%d"))
    )
    season, sowing_date = _season_series("DX", dates)

    calendar = SeasonCalendar().fit(season, sowing_date)

    # The year can change without changing the learned season.
    assert calendar.season_for("2023-12-01") == "DX"
    assert calendar.season_for("2023-01-05") == "DX"


def test_season_calendar_returns_none_outside_every_range():
    dates = pd.date_range(
        "2022-04-10",
        "2022-06-01",
        periods=20,
    )
    season, sowing_date = _season_series(
        "HT",
        dates.strftime("%Y-%m-%d"),
    )

    calendar = SeasonCalendar().fit(season, sowing_date)

    assert calendar.season_for("2022-09-15") is None


def test_season_calendar_valid_mask_excludes_bad_rows():
    # The September value represents a known placeholder or corrupted date.
    good_dates = pd.date_range(
        "2022-04-10",
        "2022-05-10",
        periods=15,
    )
    dates = list(good_dates.strftime("%Y-%m-%d")) + [
        "2022-09-01"
    ]

    season = pd.Series(["HT"] * len(dates))
    sowing_date = pd.Series(pd.to_datetime(dates))
    valid = pd.Series([True] * 15 + [False])

    calendar = SeasonCalendar().fit(
        season,
        sowing_date,
        valid=valid,
    )

    assert calendar.season_for("2022-09-01") is None


def test_season_calendar_ignores_missing_dates():
    season = pd.Series(["HT", "HT", "HT"])
    sowing_date = pd.Series(
        pd.to_datetime(
            ["2022-04-10", None, "2022-05-10"]
        )
    )

    calendar = SeasonCalendar(
        low_pct=0,
        high_pct=100,
    ).fit(season, sowing_date)

    assert calendar.season_for("2022-04-20") == "HT"


def test_season_calendar_returns_none_for_overlapping_ranges():
    dates = pd.date_range(
        "2022-04-10",
        "2022-05-10",
        periods=10,
    )

    season = pd.Series(["HT"] * 10 + ["TD"] * 10)
    sowing_date = pd.Series(
        list(dates) + list(dates)
    )

    calendar = SeasonCalendar().fit(season, sowing_date)

    assert calendar.season_for("2022-04-25") is None


def test_season_calendar_rejects_invalid_percentiles():
    with pytest.raises(ValueError):
        SeasonCalendar(low_pct=95, high_pct=5)


def test_season_calendar_requires_fit_first():
    with pytest.raises(RuntimeError):
        SeasonCalendar().season_for("2022-01-01")
