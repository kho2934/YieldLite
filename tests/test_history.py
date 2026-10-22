"""Tests for yieldlite.history."""

import numpy as np
import pandas as pd
import pytest

from yieldlite.history import typical_range


def _history() -> pd.DataFrame:
    """Two full cells of ten rows and one thin cell of three rows."""
    rows = []
    for value in range(1, 11):
        rows.append({"district_code": "A", "season": "DX", "yield_t_ha": value})
        rows.append({"district_code": "A", "season": "TD", "yield_t_ha": 10 * value})
    for value in (4.0, 5.0, 6.0):
        rows.append({"district_code": "B", "season": "DX", "yield_t_ha": value})
    return pd.DataFrame(rows)


def _rows(*cells) -> pd.DataFrame:
    return pd.DataFrame(cells, columns=["district_code", "season"])


def _overall(history: pd.DataFrame, level: float) -> float:
    values = history["yield_t_ha"].to_numpy(dtype=float)
    return float(
        np.quantile(values, (1 + level) / 2) - np.quantile(values, (1 - level) / 2)
    )


def test_range_is_the_gap_between_the_percentiles_of_the_same_cell():
    # Yields 1..10: the 10th percentile is 1.9 and the 90th is 9.1.
    width = typical_range(_history(), _rows(("A", "DX")))

    np.testing.assert_allclose(width, [7.2])


def test_level_sets_which_percentiles_are_used():
    # The central half of 1..10 runs from 3.25 to 7.75.
    width = typical_range(_history(), _rows(("A", "DX")), level=0.5)

    np.testing.assert_allclose(width, [4.5])


def test_each_district_and_season_has_its_own_range():
    width = typical_range(_history(), _rows(("A", "DX"), ("A", "TD")))

    np.testing.assert_allclose(width, [7.2, 72.0])


def test_widths_follow_the_order_of_the_rows_and_ignore_their_index():
    rows = _rows(("A", "TD"), ("A", "DX"), ("A", "TD"))
    rows.index = [30, 10, 20]

    width = typical_range(_history(), rows)

    np.testing.assert_allclose(width, [72.0, 7.2, 72.0])


def test_thin_cell_falls_back_to_the_range_over_all_history():
    history = _history()

    width = typical_range(history, _rows(("B", "DX")))

    np.testing.assert_allclose(width, [_overall(history, 0.8)])


def test_unseen_cell_falls_back_to_the_range_over_all_history():
    history = _history()

    width = typical_range(history, _rows(("Z", "HT"), ("A", "HT")))

    np.testing.assert_allclose(width, [_overall(history, 0.8)] * 2)


def test_cell_with_exactly_min_rows_keeps_its_own_range():
    history = _history()

    own = typical_range(history, _rows(("B", "DX")), min_rows=3)
    fallback = typical_range(history, _rows(("B", "DX")), min_rows=4)

    # Yields 4, 5, 6: the 10th percentile is 4.2 and the 90th is 5.8.
    np.testing.assert_allclose(own, [1.6])
    np.testing.assert_allclose(fallback, [_overall(history, 0.8)])


def test_no_rows_gives_an_empty_result():
    width = typical_range(_history(), _rows())

    assert width.shape == (0,)


def test_inputs_are_not_changed():
    history = _history()
    rows = _rows(("A", "DX"))
    history_before = history.copy()
    rows_before = rows.copy()

    typical_range(history, rows)

    pd.testing.assert_frame_equal(history, history_before)
    pd.testing.assert_frame_equal(rows, rows_before)


def test_other_target_name_can_be_used():
    history = _history().rename(columns={"yield_t_ha": "kg"})

    width = typical_range(history, _rows(("A", "DX")), target="kg")

    np.testing.assert_allclose(width, [7.2])


@pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5, np.nan, np.inf])
def test_level_must_be_strictly_between_zero_and_one(level):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        typical_range(_history(), _rows(("A", "DX")), level=level)


@pytest.mark.parametrize("min_rows", [0, -3, 2.5])
def test_min_rows_must_be_a_positive_whole_number(min_rows):
    with pytest.raises(ValueError, match="min_rows"):
        typical_range(_history(), _rows(("A", "DX")), min_rows=min_rows)


def test_history_must_have_the_required_columns():
    with pytest.raises(ValueError, match="history is missing"):
        typical_range(_history().drop(columns="season"), _rows(("A", "DX")))


def test_rows_must_have_the_required_columns():
    with pytest.raises(ValueError, match="rows is missing"):
        typical_range(_history(), pd.DataFrame({"district_code": ["A"]}))


def test_history_cannot_be_empty():
    with pytest.raises(ValueError, match="empty"):
        typical_range(_history().iloc[0:0], _rows(("A", "DX")))


def test_missing_values_are_rejected():
    history = _history()
    history.loc[0, "yield_t_ha"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        typical_range(history, _rows(("A", "DX")))

    with pytest.raises(ValueError, match="missing"):
        typical_range(_history(), _rows(("A", None)))


def test_inputs_must_be_dataframes():
    with pytest.raises(TypeError):
        typical_range(_history().to_numpy(), _rows(("A", "DX")))

    with pytest.raises(TypeError):
        typical_range(_history(), [("A", "DX")])
