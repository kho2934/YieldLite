"""Summaries of past yields that a prediction can be compared with.

A prediction interval is only useful if it tells a farmer more than the yields
farms in the same district and season have already shown. This module measures
that reference: how wide the usual range of yields is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CELL = ["district_code", "season"]


def typical_range(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    level: float = 0.8,
    target: str = "yield_t_ha",
    min_rows: int = 10,
) -> np.ndarray:
    """Return the width of the usual yield range for each row.

    The usual range for a row is the distance between the lower and upper ends
    of the central ``level`` share of past yields from the same district and
    season. With ``level=0.8`` it runs from the 10th to the 90th percentile.

    A district-season with fewer than ``min_rows`` rows in ``history`` is too
    thin to describe. Rows from such a district-season use the range over all
    of ``history`` instead.

    Parameters
    ----------
    history : pandas.DataFrame
        Past rows with ``district_code``, ``season`` and the target column.
    rows : pandas.DataFrame
        Rows to describe, with ``district_code`` and ``season``.
    level : float, default 0.8
        Share of past yields the range should contain.
    target : str, default "yield_t_ha"
        Name of the yield column in ``history``.
    min_rows : int, default 10
        Fewest past rows a district-season needs to have its own range.

    Returns
    -------
    numpy.ndarray
        One width per row of ``rows``, in the order of ``rows``.

    Raises
    ------
    TypeError
        If ``history`` or ``rows`` is not a DataFrame.
    ValueError
        If ``level`` or ``min_rows`` is out of range, if a required column is
        missing or empty, or if ``history`` has no rows.
    """
    if not isinstance(history, pd.DataFrame) or not isinstance(rows, pd.DataFrame):
        raise TypeError("history and rows must be pandas DataFrames")

    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be a finite number strictly between 0 and 1")

    if int(min_rows) != min_rows or min_rows < 1:
        raise ValueError("min_rows must be a positive whole number")

    history_columns = [*_CELL, target]
    missing = [c for c in history_columns if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing}")

    missing = [c for c in _CELL if c not in rows.columns]
    if missing:
        raise ValueError(f"rows is missing columns: {missing}")

    if history.empty:
        raise ValueError("history cannot be empty")

    if history.loc[:, history_columns].isna().any().any():
        raise ValueError("history cannot contain missing values")

    if rows.loc[:, _CELL].isna().any().any():
        raise ValueError("rows cannot contain missing district or season values")

    lower = (1 - level) / 2
    upper = (1 + level) / 2

    overall = float(history[target].quantile(upper) - history[target].quantile(lower))

    cells = history.groupby(_CELL, observed=True)[target]
    widths = cells.quantile(upper) - cells.quantile(lower)
    widths = widths[cells.size() >= min_rows]

    keys = pd.MultiIndex.from_frame(rows.loc[:, _CELL])
    found = widths.reindex(keys).to_numpy(dtype=float)

    return np.where(np.isnan(found), overall, found)
