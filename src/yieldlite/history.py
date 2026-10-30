"""Describe the past yields that a new prediction can be compared with.

A prediction interval is useful only when it says more than the historical
yields for similar farms. This module builds that comparison by finding the
range containing most past yields from the same district and season.

If there are too few comparable rows, the range is left unavailable instead
of borrowing information from a different district or season.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

_CELL = ["district_code", "season"]


def typical_interval(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    level: float = 0.8,
    target: str = "yield_t_ha",
    min_rows: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the usual historical yield interval for each row.

    Each row is matched with past observations from the same district and
    season. The returned interval contains the central ``level`` share of
    those historical yields. For example, ``level=0.8`` gives the range from
    the 10th to the 90th percentile.

    A district-season cell must contain at least ``min_rows`` historical
    observations. If the cell is too small or has never appeared before, both
    bounds are returned as ``NaN``. This lets the caller report that the
    historical comparison is not assessable instead of substituting a range
    from unrelated farms.

    Parameters
    ----------
    history : pandas.DataFrame
        Historical observations containing ``district_code``, ``season``,
        and the target column.
    rows : pandas.DataFrame
        Rows that need a historical comparison. Each row must contain
        ``district_code`` and ``season``.
    level : float, default 0.8
        Central share of historical yields that the interval should contain.
    target : str, default "yield_t_ha"
        Name of the yield column in ``history``.
    min_rows : int, default 10
        Minimum number of historical observations required for a
        district-season cell to have its own interval.

    Returns
    -------
    lower, upper : numpy.ndarray
        Lower and upper historical bounds in the same order as ``rows``.
        Both values are ``NaN`` when the interval cannot be assessed.

    Raises
    ------
    TypeError
        If ``history`` or ``rows`` is not a pandas DataFrame.
    ValueError
        If an argument is invalid, a required column is missing, the history
        is empty, or the required historical values are missing or nonnumeric.
    """
    if not isinstance(history, pd.DataFrame) or not isinstance(
        rows,
        pd.DataFrame,
    ):
        raise TypeError(
            "history and rows must be pandas DataFrames"
        )

    if (
        isinstance(level, bool)
        or not isinstance(level, Real)
        or not np.isfinite(level)
        or not 0 < level < 1
    ):
        raise ValueError(
            "level must be a finite number strictly between 0 and 1"
        )

    if (
        isinstance(min_rows, bool)
        or not isinstance(min_rows, Real)
        or not np.isfinite(min_rows)
        or min_rows < 1
        or not float(min_rows).is_integer()
    ):
        raise ValueError(
            "min_rows must be a positive whole number"
        )

    min_rows = int(min_rows)

    history_columns = [*_CELL, target]
    missing_history = [
        column
        for column in history_columns
        if column not in history.columns
    ]
    if missing_history:
        raise ValueError(
            f"history is missing columns: {missing_history}"
        )

    missing_rows = [
        column
        for column in _CELL
        if column not in rows.columns
    ]
    if missing_rows:
        raise ValueError(
            f"rows is missing columns: {missing_rows}"
        )

    if history.empty:
        raise ValueError("history cannot be empty")

    history_data = history.loc[
        :,
        history_columns,
    ].copy()

    if history_data.isna().any().any():
        raise ValueError(
            "history cannot contain missing values"
        )

    if rows.loc[:, _CELL].isna().any().any():
        raise ValueError(
            "rows cannot contain missing district or season values"
        )

    target_values = pd.to_numeric(
        history_data[target],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(target_values).all():
        raise ValueError(
            "history target must contain only finite numeric values"
        )

    history_data[target] = target_values

    cells = history_data.groupby(
        _CELL,
        observed=True,
    )[target]

    tail = (1 - level) / 2
    bounds = pd.DataFrame(
        {
            "lower": cells.quantile(tail),
            "upper": cells.quantile(1 - tail),
        }
    )

    # Keep only cells with enough observations to describe their own range.
    cell_sizes = cells.size()
    bounds = bounds.loc[
        cell_sizes >= min_rows
    ]

    # Reindexing preserves the original row order. Cells without a reliable
    # historical range receive NaN for both bounds.
    keys = pd.MultiIndex.from_frame(
        rows.loc[:, _CELL]
    )
    matched = bounds.reindex(keys)

    return (
        matched["lower"].to_numpy(dtype=float),
        matched["upper"].to_numpy(dtype=float),
    )


def typical_range(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    level: float = 0.8,
    target: str = "yield_t_ha",
    min_rows: int = 10,
) -> np.ndarray:
    """Return the width of the usual historical interval for each row.

    The width is the upper bound minus the lower bound from
    :func:`typical_interval`. It remains ``NaN`` when the district-season cell
    does not contain enough historical observations to be assessed.

    Parameters
    ----------
    history : pandas.DataFrame
        Historical observations containing ``district_code``, ``season``,
        and the target column.
    rows : pandas.DataFrame
        Rows that need a historical comparison.
    level : float, default 0.8
        Central share of historical yields that the interval should contain.
    target : str, default "yield_t_ha"
        Name of the yield column in ``history``.
    min_rows : int, default 10
        Minimum number of historical observations required for a
        district-season cell.

    Returns
    -------
    numpy.ndarray
        One interval width for each row, in the same order as ``rows``.
        A value is ``NaN`` when the range cannot be assessed.
    """
    lower, upper = typical_interval(
        history,
        rows,
        level=level,
        target=target,
        min_rows=min_rows,
    )

    return upper - lower
