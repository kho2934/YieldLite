"""Tests for yieldlite.history."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.history import typical_interval, typical_range


def _history() -> pd.DataFrame:
    """Two complete cells of ten rows and one thin cell of three rows."""
    rows = []

    for value in range(1, 11):
        rows.append(
            {
                "district_code": "A",
                "season": "DX",
                "yield_t_ha": value,
            }
        )
        rows.append(
            {
                "district_code": "A",
                "season": "TD",
                "yield_t_ha": 10 * value,
            }
        )

    for value in (4.0, 5.0, 6.0):
        rows.append(
            {
                "district_code": "B",
                "season": "DX",
                "yield_t_ha": value,
            }
        )

    return pd.DataFrame(rows)


def _rows(*cells) -> pd.DataFrame:
    """Create rows that need historical comparisons."""
    return pd.DataFrame(
        cells,
        columns=["district_code", "season"],
    )


def test_range_is_the_gap_between_the_percentiles_of_the_same_cell():
    # Yields 1 through 10 have a 10th percentile of 1.9 and a
    # 90th percentile of 9.1.
    width = typical_range(
        _history(),
        _rows(("A", "DX")),
    )

    np.testing.assert_allclose(width, [7.2])


def test_level_sets_which_percentiles_are_used():
    # The central half of 1 through 10 runs from 3.25 to 7.75.
    width = typical_range(
        _history(),
        _rows(("A", "DX")),
        level=0.5,
    )

    np.testing.assert_allclose(width, [4.5])


def test_each_district_and_season_has_its_own_range():
    width = typical_range(
        _history(),
        _rows(
            ("A", "DX"),
            ("A", "TD"),
        ),
    )

    np.testing.assert_allclose(
        width,
        [7.2, 72.0],
    )


def test_widths_follow_row_order_and_ignore_the_index():
    rows = _rows(
        ("A", "TD"),
        ("A", "DX"),
        ("A", "TD"),
    )
    rows.index = [30, 10, 20]

    width = typical_range(
        _history(),
        rows,
    )

    np.testing.assert_allclose(
        width,
        [72.0, 7.2, 72.0],
    )


def test_thin_cell_has_no_range():
    width = typical_range(
        _history(),
        _rows(("B", "DX")),
    )

    assert np.isnan(width).all()


def test_unseen_cell_has_no_range():
    width = typical_range(
        _history(),
        _rows(
            ("Z", "HT"),
            ("A", "HT"),
        ),
    )

    assert np.isnan(width).all()


def test_cell_with_exactly_min_rows_keeps_its_range():
    history = _history()

    enough = typical_range(
        history,
        _rows(("B", "DX")),
        min_rows=3,
    )
    too_thin = typical_range(
        history,
        _rows(("B", "DX")),
        min_rows=4,
    )

    # Yields 4, 5, and 6 run from 4.2 to 5.8 at the 80% level.
    np.testing.assert_allclose(enough, [1.6])
    assert np.isnan(too_thin).all()


def test_whole_number_float_min_rows_is_allowed():
    width = typical_range(
        _history(),
        _rows(("B", "DX")),
        min_rows=3.0,
    )

    np.testing.assert_allclose(width, [1.6])


def test_no_rows_gives_an_empty_result():
    width = typical_range(
        _history(),
        _rows(),
    )

    assert width.shape == (0,)


def test_inputs_are_not_changed():
    history = _history()
    rows = _rows(("A", "DX"))

    history_before = history.copy()
    rows_before = rows.copy()

    typical_range(history, rows)

    pd.testing.assert_frame_equal(
        history,
        history_before,
    )
    pd.testing.assert_frame_equal(
        rows,
        rows_before,
    )


def test_other_target_name_can_be_used():
    history = _history().rename(
        columns={"yield_t_ha": "kg"}
    )

    width = typical_range(
        history,
        _rows(("A", "DX")),
        target="kg",
    )

    np.testing.assert_allclose(width, [7.2])


def test_numeric_string_targets_are_converted():
    history = _history()
    history["yield_t_ha"] = (
        history["yield_t_ha"].astype(str)
    )

    width = typical_range(
        history,
        _rows(("A", "DX")),
    )

    np.testing.assert_allclose(width, [7.2])


def test_interval_ends_are_percentiles_of_the_same_cell():
    lower, upper = typical_interval(
        _history(),
        _rows(("A", "DX")),
    )

    np.testing.assert_allclose(lower, [1.9])
    np.testing.assert_allclose(upper, [9.1])


def test_interval_level_sets_the_percentiles():
    lower, upper = typical_interval(
        _history(),
        _rows(("A", "DX")),
        level=0.5,
    )

    np.testing.assert_allclose(lower, [3.25])
    np.testing.assert_allclose(upper, [7.75])


def test_interval_follows_row_order_and_ignores_the_index():
    rows = _rows(
        ("A", "TD"),
        ("A", "DX"),
        ("A", "TD"),
    )
    rows.index = [30, 10, 20]

    lower, upper = typical_interval(
        _history(),
        rows,
    )

    np.testing.assert_allclose(
        lower,
        [19.0, 1.9, 19.0],
    )
    np.testing.assert_allclose(
        upper,
        [91.0, 9.1, 91.0],
    )


def test_thin_and_unseen_cells_have_no_interval():
    rows = _rows(
        ("A", "DX"),
        ("B", "DX"),
        ("Z", "HT"),
        ("A", "TD"),
    )

    lower, upper = typical_interval(
        _history(),
        rows,
    )

    np.testing.assert_allclose(
        lower,
        [1.9, np.nan, np.nan, 19.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        upper,
        [9.1, np.nan, np.nan, 91.0],
        equal_nan=True,
    )


def test_range_is_distance_between_interval_ends():
    rows = _rows(
        ("A", "DX"),
        ("A", "TD"),
        ("B", "DX"),
    )

    lower, upper = typical_interval(
        _history(),
        rows,
    )
    width = typical_range(
        _history(),
        rows,
    )

    np.testing.assert_allclose(
        width,
        upper - lower,
        equal_nan=True,
    )


def test_no_rows_gives_an_empty_interval():
    lower, upper = typical_interval(
        _history(),
        _rows(),
    )

    assert lower.shape == (0,)
    assert upper.shape == (0,)


@pytest.mark.parametrize(
    "level",
    [
        0.0,
        1.0,
        -0.2,
        1.5,
        np.nan,
        np.inf,
        True,
        "0.8",
        None,
    ],
)
def test_level_must_be_strictly_between_zero_and_one(level):
    with pytest.raises(
        ValueError,
        match="strictly between 0 and 1",
    ):
        typical_range(
            _history(),
            _rows(("A", "DX")),
            level=level,
        )


@pytest.mark.parametrize(
    "min_rows",
    [
        0,
        -3,
        2.5,
        np.nan,
        np.inf,
        True,
        "3",
        None,
    ],
)
def test_min_rows_must_be_a_positive_whole_number(min_rows):
    with pytest.raises(
        ValueError,
        match="min_rows",
    ):
        typical_range(
            _history(),
            _rows(("A", "DX")),
            min_rows=min_rows,
        )


def test_history_must_have_required_columns():
    with pytest.raises(
        ValueError,
        match="history is missing",
    ):
        typical_range(
            _history().drop(columns="season"),
            _rows(("A", "DX")),
        )


def test_rows_must_have_required_columns():
    incomplete = pd.DataFrame(
        {"district_code": ["A"]}
    )

    with pytest.raises(
        ValueError,
        match="rows is missing",
    ):
        typical_range(
            _history(),
            incomplete,
        )


def test_history_cannot_be_empty():
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        typical_range(
            _history().iloc[0:0],
            _rows(("A", "DX")),
        )


def test_missing_values_are_rejected():
    history = _history()
    history.loc[0, "yield_t_ha"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        typical_range(
            history,
            _rows(("A", "DX")),
        )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        typical_range(
            _history(),
            _rows(("A", None)),
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        "not-a-number",
        np.inf,
        -np.inf,
    ],
)
def test_non_numeric_or_infinite_targets_are_rejected(
    bad_value,
):
    history = _history()
    history["yield_t_ha"] = history["yield_t_ha"].astype(object)
    history.loc[0, "yield_t_ha"] = bad_value

    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        typical_range(
            history,
            _rows(("A", "DX")),
        )


def test_inputs_must_be_dataframes():
    with pytest.raises(TypeError):
        typical_range(
            _history().to_numpy(),
            _rows(("A", "DX")),
        )

    with pytest.raises(TypeError):
        typical_range(
            _history(),
            [("A", "DX")],
        )
