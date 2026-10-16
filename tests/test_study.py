"""Tests for the degradation study."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.study import (
    RESULT_COLUMNS,
    run_degradation,
    summarize_degradation,
)

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

_DISTRICTS = ["CD", "OM", "TL", "TN", "VT"]

_SEASON_EFFECT = {
    "DX": 6.5,
    "HT": 5.0,
    "TD": 4.5,
}


def _dev_frame(
    households_per_cell: int = 6,
) -> pd.DataFrame:
    """Build synthetic data with clear season and district patterns."""
    rng = np.random.default_rng(1)
    rows = []

    for panel, season in _PANELS.items():
        for district_number, district in enumerate(_DISTRICTS):
            for household_number in range(households_per_cell):
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
                        "npk_total_kg_ha": (
                            120.0
                            + 10.0
                            * (household_number + district_number)
                        ),
                        "yield_t_ha": (
                            _SEASON_EFFECT[season]
                            + 0.2 * district_number
                            + rng.normal(0, 0.2)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _run(
    dev: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Run a small study configuration used by the tests."""
    defaults = {
        "p_values": [3, 4],
        "h_values": [2, None],
        "n_draws": 3,
        "seed": 0,
        "min_variety_count": 5,
    }
    defaults.update(kwargs)
    return run_degradation(dev, **defaults)


def test_result_has_expected_columns_and_panels():
    results = _run(_dev_frame())

    assert list(results.columns) == list(RESULT_COLUMNS)
    assert set(results["held_out"]) == set(_PANELS)
    assert set(results["p"]) == {3, 4}
    assert set(results["model"]) == {"m0", "m1"}


def test_same_seed_gives_identical_results():
    first = _run(_dev_frame(), seed=7)
    second = _run(_dev_frame(), seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_the_draws():
    first = _run(_dev_frame(), seed=1)
    second = _run(_dev_frame(), seed=2)

    assert not first.equals(second)


def test_training_size_follows_p_when_uncapped():
    results = _run(_dev_frame())

    uncapped = results[np.isinf(results["h"])]
    assert (uncapped["n_train"] == 30 * uncapped["p"]).all()
    assert (results["n_test"] == 30).all()


def test_household_cap_limits_each_cell():
    results = _run(_dev_frame())

    capped = results[results["h"] == 2]

    # Five districts, three seasons, and two households in each cell.
    assert (capped["n_train"] == 30).all()


def test_errors_are_finite_and_mae_is_non_negative():
    results = _run(_dev_frame())

    assert np.isfinite(
        results[["mae", "bias"]].to_numpy()
    ).all()
    assert (results["mae"] >= 0).all()


def test_models_learn_the_season_pattern():
    results = _run(_dev_frame())

    # The small random noise should not hide the much larger season effect.
    assert results["mae"].max() < 1.0


def test_single_model_can_be_requested():
    results = _run(
        _dev_frame(),
        models=["m0"],
    )

    assert set(results["model"]) == {"m0"}


def test_p_below_number_of_seasons_is_rejected():
    with pytest.raises(ValueError):
        _run(
            _dev_frame(),
            p_values=[2],
        )


def test_p_using_every_panel_is_rejected():
    with pytest.raises(ValueError):
        _run(
            _dev_frame(),
            p_values=[8],
        )


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        _run(
            _dev_frame(),
            models=["m9"],
        )


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError):
        _run(
            _dev_frame(),
            h_values=[],
        )


def test_missing_column_is_rejected():
    with pytest.raises(ValueError):
        _run(
            _dev_frame().drop(columns="npk_total_kg_ha")
        )


def test_missing_values_are_rejected():
    dev = _dev_frame()
    dev.loc[0, "yield_t_ha"] = np.nan

    with pytest.raises(ValueError):
        _run(dev)


def test_duplicate_index_is_rejected():
    dev = _dev_frame()
    dev.index = [0] * len(dev)

    with pytest.raises(ValueError):
        _run(dev)


def test_panel_in_two_seasons_is_rejected():
    dev = _dev_frame()
    dev.loc[0, "season"] = "HT"

    with pytest.raises(ValueError):
        _run(dev)


def test_summary_has_one_row_per_model_p_and_h():
    summary = summarize_degradation(
        _run(_dev_frame())
    )

    assert len(summary) == 2 * 2 * 2
    assert set(summary["n_panels"]) == {8}
    assert summary.loc[
        summary["model"] == "m0",
        "skill_vs_m0",
    ].isna().all()


def test_summary_skill_formula():
    results = pd.DataFrame(
        {
            "held_out": ["A", "A", "A", "A"],
            "p": [3, 3, 3, 3],
            "draw": [0, 0, 1, 1],
            "h": [np.inf] * 4,
            "model": ["m0", "m1", "m0", "m1"],
            "n_train": [10] * 4,
            "n_test": [5] * 4,
            "mae": [2.0, 1.0, 2.0, 1.0],
            "bias": [0.0] * 4,
        }
    )

    summary = summarize_degradation(results)

    m1 = summary.loc[summary["model"] == "m1"].iloc[0]
    assert m1["skill_vs_m0"] == pytest.approx(0.5)


def test_summary_skill_is_nan_when_baseline_is_perfect():
    results = pd.DataFrame(
        {
            "held_out": ["A", "A"],
            "p": [3, 3],
            "draw": [0, 0],
            "h": [np.inf] * 2,
            "model": ["m0", "m1"],
            "n_train": [10] * 2,
            "n_test": [5] * 2,
            "mae": [0.0, 1.0],
            "bias": [0.0] * 2,
        }
    )

    summary = summarize_degradation(results)

    assert summary["skill_vs_m0"].isna().all()


def test_summary_rejects_missing_columns():
    with pytest.raises(ValueError):
        summarize_degradation(
            pd.DataFrame({"mae": [1.0]})
        )
