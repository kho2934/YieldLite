"""Tests for the degradation-study sampling tools."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.degradation import cap_households_per_cell, sample_panel_subsets


def _dev_panels() -> pd.Series:
    """Return the eight panels currently available for development."""
    return pd.Series(
        {
            "DX_2020": "DX",
            "HT_2020": "HT",
            "TD_2020": "TD",
            "DX_2021": "DX",
            "TD_2021": "TD",
            "DX_2022": "DX",
            "HT_2022": "HT",
            "TD_2022": "TD",
        }
    )


def test_every_subset_has_p_panels_and_all_seasons():
    panels = _dev_panels()

    subsets = sample_panel_subsets(
        panels,
        p=4,
        n_draws=50,
        rng=np.random.default_rng(0),
    )

    assert subsets
    for subset in subsets:
        assert len(subset) == 4
        assert set(panels[list(subset)]) == {"DX", "HT", "TD"}


def test_subsets_are_distinct():
    subsets = sample_panel_subsets(
        _dev_panels(),
        p=5,
        n_draws=50,
        rng=np.random.default_rng(0),
    )

    assert len(subsets) == len(set(subsets))


def test_returns_all_valid_subsets_when_few_exist():
    # Choosing seven of eight panels means leaving out one panel. Every season
    # has at least two panels, so all eight possible choices remain valid.
    subsets = sample_panel_subsets(
        _dev_panels(),
        p=7,
        n_draws=50,
        rng=np.random.default_rng(0),
    )

    assert len(subsets) == 8


def test_caps_the_number_of_draws():
    subsets = sample_panel_subsets(
        _dev_panels(),
        p=4,
        n_draws=10,
        rng=np.random.default_rng(0),
    )

    assert len(subsets) == 10


def test_same_seed_gives_same_subsets():
    first = sample_panel_subsets(
        _dev_panels(),
        p=4,
        n_draws=10,
        rng=np.random.default_rng(3),
    )
    second = sample_panel_subsets(
        _dev_panels(),
        p=4,
        n_draws=10,
        rng=np.random.default_rng(3),
    )

    assert first == second


def test_p_below_number_of_seasons_is_rejected():
    with pytest.raises(ValueError):
        sample_panel_subsets(
            _dev_panels(),
            p=2,
            n_draws=10,
            rng=np.random.default_rng(0),
        )


def test_p_above_number_of_panels_is_rejected():
    with pytest.raises(ValueError):
        sample_panel_subsets(
            _dev_panels(),
            p=9,
            n_draws=10,
            rng=np.random.default_rng(0),
        )


def test_rejects_zero_draws():
    with pytest.raises(ValueError):
        sample_panel_subsets(
            _dev_panels(),
            p=4,
            n_draws=0,
            rng=np.random.default_rng(0),
        )


def test_rejects_duplicate_panel_ids():
    panels = _dev_panels()
    panels.index = ["same_panel"] * len(panels)

    with pytest.raises(ValueError):
        sample_panel_subsets(
            panels,
            p=4,
            n_draws=10,
            rng=np.random.default_rng(0),
        )


def test_rejects_missing_seasons():
    panels = _dev_panels()
    panels.iloc[0] = None

    with pytest.raises(ValueError):
        sample_panel_subsets(
            panels,
            p=4,
            n_draws=10,
            rng=np.random.default_rng(0),
        )


def test_rejects_empty_panel_series():
    with pytest.raises(ValueError):
        sample_panel_subsets(
            pd.Series(dtype="object"),
            p=1,
            n_draws=10,
            rng=np.random.default_rng(0),
        )


def _household_frame() -> pd.DataFrame:
    """Create three district-season cells of different sizes."""
    rows = [("OM", "DX")] * 6 + [("CD", "DX")] * 2 + [("OM", "HT")] * 4
    return pd.DataFrame(
        rows,
        columns=["district_code", "season"],
    )


def test_cap_limits_each_cell():
    df = _household_frame()

    capped = cap_households_per_cell(
        df,
        h=3,
        rng=np.random.default_rng(0),
    )

    sizes = capped.groupby(["district_code", "season"]).size()
    assert sizes.to_dict() == {
        ("CD", "DX"): 2,
        ("OM", "DX"): 3,
        ("OM", "HT"): 3,
    }


def test_cap_leaves_small_cells_whole():
    df = _household_frame()

    capped = cap_households_per_cell(
        df,
        h=3,
        rng=np.random.default_rng(0),
    )

    small_cell = df[
        (df["district_code"] == "CD")
        & (df["season"] == "DX")
    ]
    assert set(small_cell.index) <= set(capped.index)


def test_cap_keeps_original_index_and_order():
    df = _household_frame()

    capped = cap_households_per_cell(
        df,
        h=3,
        rng=np.random.default_rng(0),
    )

    assert capped.index.is_monotonic_increasing
    assert capped.index.isin(df.index).all()


def test_cap_none_keeps_everything_in_a_copy():
    df = _household_frame()

    capped = cap_households_per_cell(
        df,
        h=None,
        rng=np.random.default_rng(0),
    )

    assert capped.equals(df)
    assert capped is not df


def test_cap_same_seed_gives_same_rows():
    df = _household_frame()

    first = cap_households_per_cell(
        df,
        h=3,
        rng=np.random.default_rng(5),
    )
    second = cap_households_per_cell(
        df,
        h=3,
        rng=np.random.default_rng(5),
    )

    assert first.index.equals(second.index)


def test_cap_rejects_bad_h():
    with pytest.raises(ValueError):
        cap_households_per_cell(
            _household_frame(),
            h=0,
            rng=np.random.default_rng(0),
        )


def test_cap_rejects_non_integer_h():
    with pytest.raises(TypeError):
        cap_households_per_cell(
            _household_frame(),
            h=2.5,
            rng=np.random.default_rng(0),
        )


def test_cap_rejects_missing_cell_columns():
    with pytest.raises(ValueError):
        cap_households_per_cell(
            _household_frame().drop(columns="season"),
            h=3,
            rng=np.random.default_rng(0),
        )


def test_cap_rejects_missing_cell_values():
    df = _household_frame()
    df.loc[0, "season"] = None

    with pytest.raises(ValueError):
        cap_households_per_cell(
            df,
            h=3,
            rng=np.random.default_rng(0),
        )


def test_cap_rejects_duplicate_index():
    df = _household_frame()
    df.index = [0] * len(df)

    with pytest.raises(ValueError):
        cap_households_per_cell(
            df,
            h=3,
            rng=np.random.default_rng(0),
        )
