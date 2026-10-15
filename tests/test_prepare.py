"""Tests for preparing the cleaned survey data."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.prepare import flag_rows, prepare_frame


def _frame() -> pd.DataFrame:
    """Create rows containing one example of each quality concern."""
    return pd.DataFrame(
        {
            "code_flag": ["certain"] * 6 + ["inferred"],
            "variety_flag": [
                "known",
                "other",
                "unclear",
                "known",
                "known",
                "known",
                "known",
            ],
            "district_mismatch": [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            "commune_corrupted": [
                False,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
            "sowing_date": pd.to_datetime(["2020-01-10"] * 7),
            "harvest_date": pd.to_datetime(
                ["2020-04-10"] * 5
                + ["2020-01-10", "2020-04-10"]
            ),
            "nitrogen_kg_ha": [80.0] * 7,
            "phosphorus_kg_ha": [30.0] * 7,
            "potassium_kg_ha": [30.0] * 7,
        }
    )


def test_each_doubt_is_flagged_and_other_variety_is_not():
    flags = flag_rows(_frame())

    assert flags.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_unresolved_variety_is_flagged():
    df = _frame()
    df.loc[0, "variety_flag"] = "unresolved"

    flags = flag_rows(df)

    assert bool(flags.iloc[0])


def test_flags_keep_index_and_are_boolean():
    df = _frame()
    df.index = list(range(10, 17))

    flags = flag_rows(df)

    assert flags.index.equals(df.index)
    assert flags.dtype == bool


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError):
        flag_rows(_frame().drop(columns="code_flag"))


def test_missing_dates_are_rejected():
    df = _frame()
    df.loc[0, "sowing_date"] = pd.NaT

    with pytest.raises(ValueError):
        flag_rows(df)


def test_string_dates_are_rejected():
    df = _frame()
    df["sowing_date"] = df["sowing_date"].dt.strftime("%Y-%m-%d")

    with pytest.raises(ValueError):
        flag_rows(df)


def test_non_boolean_quality_columns_are_rejected():
    df = _frame()
    df["district_mismatch"] = df["district_mismatch"].astype(int)

    with pytest.raises(ValueError):
        flag_rows(df)


def test_prepare_adds_columns_without_changing_input():
    df = _frame()

    out = prepare_frame(
        df,
        p_form="oxide",
        k_form="oxide",
    )

    assert "npk_total_kg_ha" not in df.columns
    assert "flagged" not in df.columns
    assert out["npk_total_kg_ha"].iloc[0] == pytest.approx(140.0)
    assert out["flagged"].sum() == 5


def test_prepare_elemental_exceeds_oxide():
    df = _frame()

    elemental = prepare_frame(
        df,
        p_form="elemental",
        k_form="elemental",
    )
    oxide = prepare_frame(
        df,
        p_form="oxide",
        k_form="oxide",
    )

    assert (
        elemental["npk_total_kg_ha"]
        > oxide["npk_total_kg_ha"]
    ).all()


def test_prepare_rejects_bad_form():
    with pytest.raises(ValueError):
        prepare_frame(
            _frame(),
            p_form="kg",
            k_form="oxide",
        )


def test_prepare_rejects_missing_nutrient_column():
    with pytest.raises(ValueError):
        prepare_frame(
            _frame().drop(columns="phosphorus_kg_ha"),
            p_form="oxide",
            k_form="oxide",
        )


def test_prepare_rejects_non_numeric_nutrients():
    df = _frame()
    df["nitrogen_kg_ha"] = "unknown"

    with pytest.raises(ValueError):
        prepare_frame(
            df,
            p_form="oxide",
            k_form="oxide",
        )


def test_prepare_rejects_non_finite_nutrients():
    df = _frame()
    df.loc[0, "nitrogen_kg_ha"] = np.inf

    with pytest.raises(ValueError):
        prepare_frame(
            df,
            p_form="oxide",
            k_form="oxide",
        )


def test_prepare_rejects_duplicate_index():
    df = _frame()
    df.index = [0] * len(df)

    with pytest.raises(ValueError):
        prepare_frame(
            df,
            p_form="oxide",
            k_form="oxide",
        )
