"""Tests for the prediction-interval study."""

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

# More than the ten rows required for a historical range.
_HOUSEHOLDS = 12
_PANEL_SIZE = (
    len(_DISTRICTS)
    * _HOUSEHOLDS
)

# Small settings for tests that need only one inexpensive study.
_CHEAP = {
    "p_values": [3],
    "levels": (0.5,),
    "n_draws": 1,
    "models": ["m0"],
}

_INTERVAL_SCORES = [
    "coverage",
    "width",
    "informative",
]

_HISTORY_SCORES = [
    "ref_width",
    "assessable",
    "history_coverage",
]


def _dev_frame(
    households_per_cell: int = _HOUSEHOLDS,
    offsets: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Create synthetic rows with season and district effects."""
    panel_offsets = (
        {}
        if offsets is None
        else offsets
    )
    rng = np.random.default_rng(1)
    rows = []

    for panel, season in _PANELS.items():
        for district_index, district in enumerate(
            _DISTRICTS
        ):
            for household_index in range(
                households_per_cell
            ):
                rows.append(
                    {
                        "panel_id": panel,
                        "season": season,
                        "district_code": district,
                        "variety_raw": (
                            "A"
                            if household_index % 2 == 0
                            else "B"
                        ),
                        "variety_flag": "known",
                        "npk_total_kg_ha": (
                            120.0
                            + 10.0
                            * (
                                household_index
                                + district_index
                            )
                        ),
                        "yield_t_ha": (
                            _SEASON_EFFECT[season]
                            + 0.2 * district_index
                            + panel_offsets.get(
                                panel,
                                0.0,
                            )
                            + rng.normal(0, 0.2)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _run(
    dev: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    settings = {
        "p_values": [3, 4],
        "h_values": [None],
        "levels": (0.5, 0.8),
        "n_draws": 2,
        "seed": 0,
        "min_variety_count": 5,
    }
    settings.update(kwargs)

    return run_interval_study(
        dev,
        **settings,
    )


def _cheap(
    dev: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    return _run(
        dev,
        h_values=[None],
        **{
            **_CHEAP,
            **kwargs,
        },
    )


@functools.cache
def _default_results() -> pd.DataFrame:
    """Run the standard synthetic study once.

    Tests must treat the cached result as read-only.
    """
    return _run(_dev_frame())


def test_result_has_expected_columns_and_settings():
    results = _default_results()

    assert list(results.columns) == list(
        INTERVAL_COLUMNS
    )
    assert set(results["held_out"]) == set(
        _PANELS
    )
    assert set(results["p"]) == {3, 4}
    assert set(results["draw"]) == {0, 1}
    assert set(results["model"]) == {
        "m0",
        "m1",
    }
    assert set(results["level"]) == {
        0.5,
        0.8,
    }


def test_three_panels_cannot_support_eighty_percent():
    results = _default_results()

    unsupported = results[
        (results["p"] == 3)
        & (results["level"] == 0.8)
    ]

    assert not unsupported["supported"].any()
    assert (
        unsupported[_INTERVAL_SCORES]
        .isna()
        .all()
        .all()
    )

    # Historical measurements remain available even when the model interval
    # is unsupported.
    assert (
        unsupported[_HISTORY_SCORES]
        .notna()
        .all()
        .all()
    )


def test_four_panels_support_eighty_percent():
    results = _default_results()

    supported = results[
        (results["p"] == 4)
        & (results["level"] == 0.8)
    ]

    assert supported["supported"].all()
    assert (
        supported[
            [
                *_INTERVAL_SCORES,
                *_HISTORY_SCORES,
            ]
        ]
        .notna()
        .all()
        .all()
    )


def test_lower_levels_are_supported_with_three_panels():
    results = _default_results()

    half = results[
        (results["p"] == 3)
        & (results["level"] == 0.5)
    ]

    assert half["supported"].all()


def test_scores_are_in_range():
    results = _default_results().dropna(
        subset=["coverage"]
    )

    assert results["coverage"].between(
        0,
        1,
    ).all()
    assert results["informative"].between(
        0,
        1,
    ).all()
    assert results["assessable"].between(
        0,
        1,
    ).all()
    assert results["history_coverage"].between(
        0,
        1,
    ).all()
    assert (results["width"] > 0).all()
    assert (results["ref_width"] > 0).all()


def test_wider_level_gives_wider_interval():
    results = _default_results()
    results = results[
        results["p"] == 4
    ]

    keys = [
        "held_out",
        "draw",
        "model",
    ]

    wide = (
        results[results["level"] == 0.8]
        .set_index(keys)["width"]
    )
    narrow = (
        results[results["level"] == 0.5]
        .set_index(keys)["width"]
    )

    assert (wide >= narrow).all()


def test_held_out_panel_is_never_used_for_fitting():
    dev = _dev_frame(
        offsets={"DX_2022": 50.0}
    )

    results = _cheap(dev)
    shifted = results[
        results["held_out"] == "DX_2022"
    ]

    assert (
        shifted["coverage"] == 0.0
    ).all()


def test_held_out_panel_is_never_in_its_history():
    dev = _dev_frame(
        offsets={"DX_2022": 50.0}
    )

    results = _cheap(dev)
    shifted = results[
        results["held_out"] == "DX_2022"
    ]

    assert (
        shifted["history_coverage"] == 0.0
    ).all()


def test_intervals_are_not_informative_when_panels_dominate():
    panel_swings = {
        panel: (
            4.0
            if index % 2 == 0
            else -4.0
        )
        for index, panel in enumerate(_PANELS)
    }

    results = _cheap(
        _dev_frame(
            offsets=panel_swings
        )
    )

    assert (
        results["informative"] == 0.0
    ).all()
    assert (
        results["width"]
        > results["ref_width"]
    ).all()


def test_intervals_are_informative_when_model_explains_households():
    dev = _dev_frame()
    dev["yield_t_ha"] = (
        3.0
        + 0.04
        * dev["npk_total_kg_ha"]
    )

    results = _cheap(
        dev,
        models=["m1"],
    )

    assert (
        results["informative"] == 1.0
    ).all()
    assert (
        results["width"]
        < 0.5 * results["ref_width"]
    ).all()


def test_history_coverage_is_close_to_level_for_similar_panels():
    results = _run(
        _dev_frame(),
        p_values=[4],
    )

    half = results.loc[
        results["level"] == 0.5,
        "history_coverage",
    ].mean()
    most = results.loc[
        results["level"] == 0.8,
        "history_coverage",
    ].mean()

    assert 0.3 < half < 0.6
    assert 0.6 < most < 0.9
    assert half < most


def test_history_range_is_same_for_every_model():
    results = _default_results()

    keys = [
        "held_out",
        "p",
        "draw",
        "h",
        "level",
    ]
    columns = [
        "assessable",
        "history_coverage",
        "ref_width",
    ]

    by_model = {
        name: group.set_index(keys)[columns]
        for name, group in results.groupby(
            "model"
        )
    }

    pd.testing.assert_frame_equal(
        by_model["m0"],
        by_model["m1"],
    )


def test_full_cells_are_all_assessable():
    assert (
        _default_results()["assessable"] == 1.0
    ).all()


def test_thin_cells_have_no_history_range():
    results = _run(
        _dev_frame(),
        **{
            **_CHEAP,
            "h_values": [2],
        },
    )

    assert (
        results["assessable"] == 0.0
    ).all()
    assert (
        results[
            [
                "ref_width",
                "informative",
                "history_coverage",
            ]
        ]
        .isna()
        .all()
        .all()
    )

    # Prediction intervals can still be scored without historical ranges.
    assert (
        results[
            [
                "coverage",
                "width",
            ]
        ]
        .notna()
        .all()
        .all()
    )


def test_informative_ignores_households_without_history():
    dev = _dev_frame()
    dev["yield_t_ha"] = (
        3.0
        + 0.04
        * dev["npk_total_kg_ha"]
    )

    # Leave only three VT households in every panel. VT is then too thin for
    # a historical range but should not count as uninformative.
    rows_to_remove = (
        dev[dev["district_code"] == "VT"]
        .groupby("panel_id")
        .head(9)
        .index
    )
    dev = dev.drop(
        index=rows_to_remove
    )

    results = _cheap(
        dev,
        models=["m1"],
    )

    expected_rows = (
        4 * _HOUSEHOLDS + 3
    )
    expected_assessable = (
        4
        * _HOUSEHOLDS
        / expected_rows
    )

    assert (
        results["n_test"] == expected_rows
    ).all()
    np.testing.assert_allclose(
        results["assessable"],
        expected_assessable,
    )
    assert (
        results["informative"] == 1.0
    ).all()


def test_training_size_follows_panel_count():
    results = _default_results()

    assert (
        results["n_train"]
        == _PANEL_SIZE * results["p"]
    ).all()
    assert (
        results["n_test"] == _PANEL_SIZE
    ).all()


def test_cap_limits_rows_per_cell():
    results = _run(
        _dev_frame(),
        **{
            **_CHEAP,
            "h_values": [2],
        },
    )

    # Five districts × three seasons × two households.
    assert (
        results["n_train"] == 30
    ).all()
    assert (
        results["n_test"] == _PANEL_SIZE
    ).all()
    assert (
        results["h"] == 2.0
    ).all()


def test_no_cap_is_stored_as_infinity():
    assert np.isinf(
        _default_results()["h"]
    ).all()


def test_same_seed_gives_same_results():
    first = _cheap(
        _dev_frame(),
        seed=7,
    )
    second = _cheap(
        _dev_frame(),
        seed=7,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_different_seeds_change_draws():
    first = _cheap(
        _dev_frame(),
        seed=1,
    )
    second = _cheap(
        _dev_frame(),
        seed=2,
    )

    assert not first.equals(second)


def test_single_model_can_be_requested():
    results = _cheap(
        _dev_frame(),
        models=["m0"],
    )

    assert set(results["model"]) == {
        "m0"
    }


def test_progress_is_reported_for_each_panel():
    messages = []

    _cheap(
        _dev_frame(),
        progress=messages.append,
    )

    assert len(messages) == len(_PANELS)
    assert messages[0].endswith(
        "(1 of 8)"
    )
    assert messages[-1].endswith(
        "(8 of 8)"
    )

    combined = " ".join(messages)
    assert all(
        panel in combined
        for panel in _PANELS
    )


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
        _run(
            _dev_frame(),
            **kwargs,
        )


def test_missing_columns_are_rejected():
    dev = _dev_frame().drop(
        columns="npk_total_kg_ha"
    )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        _run(dev)


def _hand_results() -> pd.DataFrame:
    """Create two panel results that can be checked by hand."""
    score_names = (
        "coverage",
        "width",
        "ref_width",
        "informative",
        "assessable",
        "history_coverage",
    )

    scores = {
        "A": [
            (1.0, 2.0, 3.0, 1.0, 1.0, 0.8),
            (1.0, 4.0, 3.0, 0.5, 1.0, 0.6),
            (0.0, 6.0, 3.0, 0.0, 0.5, 0.7),
        ],
        "B": [
            (0.4, 2.0, 5.0, 0.25, 1.0, 0.5)
        ],
    }

    rows = []

    for panel, draws in scores.items():
        for draw, values in enumerate(draws):
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
                    **dict(
                        zip(
                            score_names,
                            values,
                            strict=True,
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=list(INTERVAL_COLUMNS),
    )


def test_summary_averages_draws_before_panels():
    row = summarize_intervals(
        _hand_results()
    ).iloc[0]

    panel_coverage = [
        2 / 3,
        0.4,
    ]

    assert row["coverage"] == pytest.approx(
        np.mean(panel_coverage)
    )
    assert row["coverage_sd"] == pytest.approx(
        np.std(
            panel_coverage,
            ddof=1,
        )
    )
    assert row["coverage_se"] == pytest.approx(
        row["coverage_sd"]
        / np.sqrt(2)
    )
    assert row["width"] == pytest.approx(
        3.0
    )
    assert row["ref_width"] == pytest.approx(
        4.0
    )
    assert row["informative"] == pytest.approx(
        0.375
    )
    assert row["n_panels"] == 2

    history_coverage = [
        0.7,
        0.5,
    ]

    assert row[
        "history_coverage"
    ] == pytest.approx(
        np.mean(history_coverage)
    )

    history_sd = np.std(
        history_coverage,
        ddof=1,
    )
    assert row[
        "history_coverage_se"
    ] == pytest.approx(
        history_sd / np.sqrt(2)
    )
    assert row["assessable"] == pytest.approx(
        np.mean(
            [
                5 / 6,
                1.0,
            ]
        )
    )


def test_summary_keeps_history_when_interval_is_unsupported():
    results = _hand_results()
    results["supported"] = False
    results[_INTERVAL_SCORES] = np.nan

    row = summarize_intervals(
        results
    ).iloc[0]

    assert not row["supported"]
    assert row[
        _INTERVAL_SCORES
    ].isna().all()

    assert row[
        "history_coverage"
    ] == pytest.approx(0.6)
    assert row["ref_width"] == pytest.approx(
        4.0
    )


def test_summary_skips_nan_history_panel():
    results = _hand_results()
    panel_b = results["held_out"] == "B"

    results.loc[
        panel_b,
        [
            "ref_width",
            "informative",
            "history_coverage",
        ],
    ] = np.nan
    results.loc[
        panel_b,
        "assessable",
    ] = 0.0

    row = summarize_intervals(
        results
    ).iloc[0]

    assert row["ref_width"] == pytest.approx(
        3.0
    )
    assert row["informative"] == pytest.approx(
        0.5
    )
    assert row[
        "history_coverage"
    ] == pytest.approx(0.7)
    assert np.isnan(
        row["history_coverage_se"]
    )
    assert row["assessable"] == pytest.approx(
        5 / 12
    )

    # Model coverage still uses both panels.
    assert row["coverage"] == pytest.approx(
        np.mean(
            [
                2 / 3,
                0.4,
            ]
        )
    )
    assert row["n_panels"] == 2


def test_summary_requires_result_columns():
    with pytest.raises(
        ValueError,
        match="missing",
    ):
        summarize_intervals(
            pd.DataFrame(
                {"coverage": [1.0]}
            )
        )


def test_summary_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        summarize_intervals(
            _hand_results().iloc[0:0]
        )


def test_summary_has_one_row_per_setting():
    summary = summarize_intervals(
        _default_results()
    )

    # Two models × two levels × two panel counts × one household cap.
    assert len(summary) == 8
    assert (
        summary["n_panels"]
        == len(_PANELS)
    ).all()
