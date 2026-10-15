"""Tests for the fixed development and holdout split."""

import pandas as pd
import pytest

from yieldlite.split import lock_holdout

# Use the 12 panels from the project so the tests reflect the structure of
# the real dataset instead of an unrelated example.
_PANELS = [
    ("DX", 2020),
    ("DX", 2021),
    ("DX", 2022),
    ("DX", 2023),
    ("HT", 2020),
    ("HT", 2022),
    ("HT", 2024),
    ("TD", 2020),
    ("TD", 2021),
    ("TD", 2022),
    ("TD", 2023),
    ("TD", 2024),
]


def _toy_dataset(rows_per_panel: int = 3) -> pd.DataFrame:
    """Create a small test dataset containing all 12 project panels."""
    rows = [
        {
            "season": season,
            "year": year,
            "panel_id": f"{season}_{year}",
        }
        for season, year in _PANELS
        for _ in range(rows_per_panel)
    ]
    return pd.DataFrame(rows)


def test_splits_by_year():
    development, holdout = lock_holdout(_toy_dataset())

    assert set(development["year"]) == {2020, 2021, 2022}
    assert set(holdout["year"]) == {2023, 2024}


def test_holdout_matches_locked_panels():
    _, holdout = lock_holdout(_toy_dataset())

    assert set(holdout["panel_id"]) == {
        "DX_2023",
        "HT_2024",
        "TD_2023",
        "TD_2024",
    }


def test_keeps_whole_panels_together():
    development, holdout = lock_holdout(_toy_dataset())

    assert set(development["panel_id"]).isdisjoint(holdout["panel_id"])


def test_row_counts_add_up():
    df = _toy_dataset(rows_per_panel=3)
    development, holdout = lock_holdout(df)

    assert len(development) + len(holdout) == len(df)
    assert len(development) == 8 * 3  # Eight development panels
    assert len(holdout) == 4 * 3  # Four final-test panels


def test_unexpected_panel_raises():
    """Catch a new panel instead of silently changing the final test set."""
    extra_panel = pd.DataFrame(
        [{"season": "DX", "year": 2025, "panel_id": "DX_2025"}]
    )
    df = pd.concat([_toy_dataset(), extra_panel], ignore_index=True)

    with pytest.raises(ValueError):
        lock_holdout(df)


def test_custom_cutoff_mismatch_raises():
    """Reject a cutoff that no longer produces the locked holdout set."""
    with pytest.raises(ValueError):
        lock_holdout(_toy_dataset(), cutoff_year=2024)
