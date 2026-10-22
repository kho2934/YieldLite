"""Tests for the interval study."""

import functools

import numpy as np
import pandas as pd
import pytest

from yieldlite.interval_study import (
    INTERVAL_COLUMNS,
    run_interval_study,
    summarize_intervals,
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
_SEASON_EFFECT = {"DX": 6.5, "HT": 5.0, "TD": 4.5}

# The cheapest useful study: one panel count, one draw, one level, one model.
_CHEAP = {"p_values": [3], "levels": (0.5,), "n_draws": 1, "models": ["m0"]}


def _dev_frame(households_per_cell: int = 6, offsets=None) -> pd.DataFrame:
    """Create synthetic rows with a clear season and district signal."""
    offsets = offsets or {}
    rng = np.random.default_rng(1)
    rows = []
    for panel, season in _PANELS.items():
        for k, district in enumerate(_DISTRICTS):
            for i in range(households_per_cell):
                rows.append(
                    {
                        "panel_id": panel,
                        "season": season,
                        "district_code": district,
                        "variety_raw": "A" if i % 2 == 0 else "B",
                        "variety_flag": "known",
                        "npk_total_kg_ha": 120.0 + 10.0 * (i + k),
                        "yield_t_ha": _SEASON_EFFECT[season]
                        + 0.2 * k
                        + offsets.get(panel, 0.0)
                        + rng.normal(0, 0.2),
                    }
                )
    return pd.DataFrame(rows)


def _run(dev: pd.DataFrame, **kwargs) -> pd.DataFrame:
    settings = {
        "p_values": [3, 4],
        "h_values": [None],
        "levels": (0.5, 0.8),
        "n_draws": 2,
        "seed": 0,
        "min_variety_count": 5,
    }
    settings.update(kwargs)
    return run_interval_study(dev, **settings)


def _cheap(dev: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run(dev, h_values=[None], **{**_CHEAP, **kwargs})


@functools.cache
def _default_results() -> pd.DataFrame:
    """Run the standard small study once and share it between tests.

    Tests must treat the result as read-only.
    """
    return _run(_dev_frame())


def test_result_has_expected_columns_and_settings():
    results = _default_results()

    assert list(results.columns) == list(INTERVAL_COLUMNS)
    assert set(results["held_out"]) == set(_PANELS)
    assert set(results["p"]) == {3, 4}
    assert set(results["draw"]) == {0, 1}
    assert set(results["model"]) == {"m0", "m1"}
    assert set(results["level"]) == {0.5, 0.8}


def test_three_panels_cannot_support_an_80_percent_interval():
    results = _default_results()

    unsupported = results[(results["p"] == 3) & (results["level"] == 0.8)]
    assert not unsupported["supported"].any()
    scores = ["coverage", "width", "ref_width", "informative"]
    assert unsupported[scores].isna().all().all()


def test_four_panels_support_an_80_percent_interval():
    results = _default_results()

    supported = results[(results["p"] == 4) & (results["level"] == 0.8)]
    assert supported["supported"].all()
    scores = ["coverage", "width", "ref_width", "informative"]
    assert supported[scores].notna().all().all()


def test_lower_levels_are_supported_with_three_panels():
    results = _default_results()

    half = results[(results["p"] == 3) & (results["level"] == 0.5)]
    assert half["supported"].all()


def test_scores_are_in_range():
    results = _default_results().dropna(subset=["coverage"])

    assert results["coverage"].between(0, 1).all()
    assert results["informative"].between(0, 1).all()
    assert (results["width"] > 0).all()
    assert (results["ref_width"] > 0).all()


def test_wider_level_gives_a_wider_interval_in_the_same_run():
    results = _default_results()
    results = results[results["p"] == 4]

    keys = ["held_out", "draw", "model"]
    wide = results[results["level"] == 0.8].set_index(keys)["width"]
    narrow = results[results["level"] == 0.5].set_index(keys)["width"]
    assert (wide >= narrow).all()


def test_held_out_panel_is_never_used_for_fitting():
    # One panel is shifted far above every other panel. If the study ever
    # trained on it while scoring it, the intervals would follow the shift.
    dev = _dev_frame(offsets={"DX_2022": 50.0})

    results = _cheap(dev)
    shifted = results[results["held_out"] == "DX_2022"]

    assert (shifted["coverage"] == 0.0).all()


def test_intervals_are_not_informative_when_panels_differ_more_than_households():
    # Panels sit 8 t/ha apart while households within a cell differ by about
    # 0.5, so any interval that allows for a new panel is far wider than the
    # usual range.
    swings = {p: 4.0 if i % 2 == 0 else -4.0 for i, p in enumerate(_PANELS)}

    results = _cheap(_dev_frame(offsets=swings))

    assert (results["informative"] == 0.0).all()
    assert (results["width"] > results["ref_width"]).all()


def test_intervals_are_informative_when_the_model_explains_the_households():
    # Yield is an exact function of fertilizer, so the second rung predicts it
    # almost perfectly while yields within a cell still spread over 2 t/ha.
    dev = _dev_frame()
    dev["yield_t_ha"] = 3.0 + 0.04 * dev["npk_total_kg_ha"]

    results = _cheap(dev, models=["m1"])

    assert (results["informative"] == 1.0).all()
    assert (results["width"] < 0.5 * results["ref_width"]).all()


def test_training_size_follows_p_when_no_cap_is_applied():
    results = _default_results()

    assert (results["n_train"] == 30 * results["p"]).all()
    assert (results["n_test"] == 30).all()


def test_cap_limits_training_rows_per_cell():
    results = _run(_dev_frame(), **{**_CHEAP, "h_values": [2]})

    # Five districts times three seasons times two households per cell.
    assert (results["n_train"] == 30).all()
    assert (results["n_test"] == 30).all()
    assert (results["h"] == 2.0).all()


def test_no_cap_is_stored_as_infinity():
    assert np.isinf(_default_results()["h"]).all()


def test_same_seed_gives_identical_results():
    first = _cheap(_dev_frame(), seed=7)
    second = _cheap(_dev_frame(), seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_the_draws():
    first = _cheap(_dev_frame(), seed=1)
    second = _cheap(_dev_frame(), seed=2)

    assert not first.equals(second)


def test_single_model_can_be_requested():
    results = _cheap(_dev_frame(), models=["m0"])

    assert set(results["model"]) == {"m0"}


def test_progress_is_reported_once_for_each_held_out_panel():
    messages = []

    _cheap(_dev_frame(), progress=messages.append)

    assert len(messages) == len(_PANELS)
    assert messages[0].endswith("(1 of 8)")
    assert messages[-1].endswith("(8 of 8)")
    assert all(panel in " ".join(messages) for panel in _PANELS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_values": [2]},
        {"p_values": [8]},
        {"p_values": []},
        {"h_values": []},
        {"models": ["m9"]},
        {"levels": []},
        {"levels": [0.8, 1.0]},
        {"levels": [0.0]},
        {"levels": [0.8, 0.8]},
    ],
)
def test_bad_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _run(_dev_frame(), **kwargs)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing"):
        _run(_dev_frame().drop(columns="npk_total_kg_ha"))


def _hand_results() -> pd.DataFrame:
    """Panel A with three draws and panel B with one, scored by hand."""
    scores = {
        "A": [
            (1.0, 2.0, 3.0, 1.0),
            (1.0, 4.0, 3.0, 0.5),
            (0.0, 6.0, 3.0, 0.0),
        ],
        "B": [(0.4, 2.0, 5.0, 0.25)],
    }
    rows = []
    for panel, draws in scores.items():
        for draw, (coverage, width, ref_width, informative) in enumerate(draws):
            rows.append(
                {
                    "held_out": panel,
                    "p": 4,
                    "draw": draw,
                    "h": np.inf,
                    "model": "m1",
                    "level": 0.8,
                    "supported": True,
                    "n_train": 100,
                    "n_test": 30,
                    "coverage": coverage,
                    "width": width,
                    "ref_width": ref_width,
                    "informative": informative,
                }
            )
    return pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))


def test_summary_averages_draws_before_panels():
    row = summarize_intervals(_hand_results()).iloc[0]

    # Panel A averages to 2/3 and panel B is 0.4, so each panel counts once.
    # Pooling the four draws instead would give 0.6.
    panel_means = [2 / 3, 0.4]
    assert row["coverage"] == pytest.approx(np.mean(panel_means))
    assert row["coverage_sd"] == pytest.approx(np.std(panel_means, ddof=1))
    assert row["coverage_se"] == pytest.approx(row["coverage_sd"] / np.sqrt(2))
    assert row["width"] == pytest.approx(3.0)
    assert row["ref_width"] == pytest.approx(4.0)
    assert row["informative"] == pytest.approx(0.375)
    assert row["n_panels"] == 2


def test_summary_marks_unsupported_levels():
    results = _hand_results()
    results["supported"] = False
    results[["coverage", "width", "ref_width", "informative"]] = np.nan

    row = summarize_intervals(results).iloc[0]

    assert not row["supported"]
    assert row[["coverage", "width", "ref_width", "informative"]].isna().all()


def test_summary_requires_the_result_columns():
    with pytest.raises(ValueError, match="missing"):
        summarize_intervals(pd.DataFrame({"coverage": [1.0]}))


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        summarize_intervals(_hand_results().iloc[0:0])


def test_summary_of_a_real_run_has_one_row_per_setting():
    summary = summarize_intervals(_default_results())

    # Two models, two levels, two panel counts, one household setting.
    assert len(summary) == 8
    assert (summary["n_panels"] == len(_PANELS)).all()
