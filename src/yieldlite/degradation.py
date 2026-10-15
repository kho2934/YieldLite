"""Create smaller training sets for the degradation study.

The study measures how model performance changes as the training data becomes
smaller along two axes: ``p``, the number of season-year panels, and ``h``, the
number of household rows kept in each district-season group.

These functions only work with the development rows passed to them. The caller
is responsible for keeping the locked final holdout separate.
"""

from __future__ import annotations

from itertools import combinations
from numbers import Integral

import numpy as np
import pandas as pd


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer or raise a clear validation error."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1")

    return value


def sample_panel_subsets(
    panel_season: pd.Series,
    p: int,
    n_draws: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """Choose distinct training sets containing ``p`` panels.

    Every returned set includes at least one panel from each season. This
    prevents a draw from removing an entire season from the training data.

    If the number of valid combinations is smaller than ``n_draws``, the
    function returns every valid combination. Otherwise, it randomly selects
    ``n_draws`` combinations without replacement.

    Parameters
    ----------
    panel_season : pandas.Series
        Season label for each panel, indexed by panel ID.
    p : int
        Number of panels to include in each draw.
    n_draws : int
        Maximum number of panel combinations to return.
    rng : numpy.random.Generator
        Random-number generator used to make the draws reproducible.

    Returns
    -------
    list of tuple of str
        Panel combinations. The panel IDs inside each tuple are sorted.
    """
    if not isinstance(panel_season, pd.Series):
        raise TypeError("panel_season must be a pandas Series")

    if panel_season.empty:
        raise ValueError("panel_season cannot be empty")

    if not panel_season.index.is_unique:
        raise ValueError("panel_season must have one row per panel")

    if panel_season.index.hasnans:
        raise ValueError("panel IDs cannot be missing")

    if panel_season.isna().any():
        raise ValueError("every panel needs a season")

    p = _positive_integer(p, "p")
    n_draws = _positive_integer(n_draws, "n_draws")

    n_panels = len(panel_season)
    n_seasons = int(panel_season.nunique())

    if not n_seasons <= p <= n_panels:
        raise ValueError(
            f"p must be between {n_seasons} (one panel per season) and "
            f"{n_panels} (all panels), got {p}"
        )

    panel_ids = sorted(panel_season.index)
    seasons = panel_season.to_dict()

    valid_subsets = [
        subset
        for subset in combinations(panel_ids, p)
        if len({seasons[panel] for panel in subset}) == n_seasons
    ]

    if len(valid_subsets) <= n_draws:
        return valid_subsets

    selected = rng.choice(
        len(valid_subsets),
        size=n_draws,
        replace=False,
    )
    return [valid_subsets[int(i)] for i in sorted(selected)]


def cap_households_per_cell(
    df: pd.DataFrame,
    h: int | None,
    rng: np.random.Generator,
    cell_columns: tuple[str, ...] = ("district_code", "season"),
) -> pd.DataFrame:
    """Keep at most ``h`` randomly selected rows in each cell.

    A cell is one combination of ``cell_columns``. By default, it represents
    one district in one season. Cells already containing ``h`` rows or fewer
    are kept unchanged.

    Sampling happens across all selected panels represented in a cell. The
    returned rows retain their original indexes and ordering.

    Parameters
    ----------
    df : pandas.DataFrame
        Development rows available for sampling.
    h : int or None
        Maximum number of rows to keep in each cell. ``None`` keeps all rows.
    rng : numpy.random.Generator
        Random-number generator used to make the draw reproducible.
    cell_columns : tuple of str
        Columns used to define a cell.

    Returns
    -------
    pandas.DataFrame
        Selected rows in their original order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not df.index.is_unique:
        raise ValueError("df must use a unique row index")

    if df.index.hasnans:
        raise ValueError("df index cannot contain missing values")

    if h is None:
        return df.copy()

    h = _positive_integer(h, "h")

    if not cell_columns:
        raise ValueError("cell_columns must contain at least one column")

    if len(set(cell_columns)) != len(cell_columns):
        raise ValueError("cell_columns cannot contain duplicates")

    missing = [column for column in cell_columns if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing cell columns: {missing}")

    if df.loc[:, list(cell_columns)].isna().any().any():
        raise ValueError("cell columns cannot contain missing values")

    keep: list[object] = []

    for _, cell in df.groupby(
        list(cell_columns),
        sort=True,
        observed=True,
    ):
        if len(cell) <= h:
            keep.extend(cell.index)
        else:
            selected = rng.choice(
                cell.index.to_numpy(),
                size=h,
                replace=False,
            )
            keep.extend(selected)

    return df.loc[df.index.isin(keep)].copy()
