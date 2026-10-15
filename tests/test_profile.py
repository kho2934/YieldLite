"""Tests for the training-data profile."""

import dataclasses

import pandas as pd
import pytest

from yieldlite.profile import DataProfile


def _toy_frame() -> pd.DataFrame:
    """Create three panels across two seasons and two districts."""
    return pd.DataFrame(
        {
            "panel_id": ["DX_2020"] * 4 + ["DX_2021"] * 4 + ["HT_2020"] * 2,
            "season": ["DX"] * 8 + ["HT"] * 2,
            "district_code": ["OM", "OM", "CD", "CD"] * 2 + ["OM", "CD"],
            "variety_pooled": ["A", "A", "pooled", "A"] * 2
            + ["pooled", "A"],
            "flagged": [False, False, True, False] * 2 + [True, False],
        }
    )


def test_counts_rows_panels_and_seasons():
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.n_rows == 10
    assert profile.n_panels == 3
    assert profile.n_seasons == 2


def test_panels_per_season():
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.panels_per_season == {"DX": 2, "HT": 1}


def test_within_season_contrasts_is_panels_minus_seasons():
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.within_season_contrasts == 1


def test_median_households_per_cell():
    # The four district-season groups contain 4, 4, 1, and 1 rows, so the
    # median is halfway between 1 and 4.
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.median_households_per_cell == pytest.approx(2.5)


def test_rows_per_variety():
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.rows_per_variety == {"A": 7, "pooled": 3}


def test_flagged_share():
    profile = DataProfile.from_frame(_toy_frame())

    assert profile.flagged_share == pytest.approx(0.3)


def test_profile_fields_cannot_be_reassigned():
    profile = DataProfile.from_frame(_toy_frame())

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.n_rows = 0


def test_custom_flag_column():
    df = _toy_frame().rename(columns={"flagged": "any_flag"})

    profile = DataProfile.from_frame(df, flag_column="any_flag")

    assert profile.flagged_share == pytest.approx(0.3)


def test_accepts_nullable_boolean_flags_without_missing_values():
    df = _toy_frame()
    df["flagged"] = df["flagged"].astype("boolean")

    profile = DataProfile.from_frame(df)

    assert profile.flagged_share == pytest.approx(0.3)


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        DataProfile.from_frame(_toy_frame().drop(columns="season"))


def test_rejects_duplicate_column_names():
    df = pd.concat([_toy_frame(), _toy_frame()[["season"]]], axis=1)

    with pytest.raises(ValueError):
        DataProfile.from_frame(df)


def test_rejects_empty_table():
    with pytest.raises(ValueError):
        DataProfile.from_frame(_toy_frame().iloc[0:0])


def test_rejects_missing_values():
    df = _toy_frame()
    df.loc[0, "district_code"] = None

    with pytest.raises(ValueError):
        DataProfile.from_frame(df)


def test_rejects_non_boolean_flags():
    df = _toy_frame()
    df["flagged"] = df["flagged"].astype(int)

    with pytest.raises(ValueError):
        DataProfile.from_frame(df)


def test_rejects_panel_spanning_two_seasons():
    df = _toy_frame()
    df.loc[0, "season"] = "HT"

    with pytest.raises(ValueError):
        DataProfile.from_frame(df)


def test_rejects_non_dataframe():
    with pytest.raises(TypeError):
        DataProfile.from_frame([1, 2, 3])
