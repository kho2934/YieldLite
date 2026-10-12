"""Keep the 2023-2024 panels separate for the final evaluation.

Model development and calibration use only the 2020-2022 panels. The later
panels should remain untouched until the final forward test in Week 8.
"""

import pandas as pd


# These four panels are reserved for final testing. Checking the exact set
# prevents a new or renamed panel from quietly changing the holdout data.
EXPECTED_HOLDOUT_PANELS = {"DX_2023", "HT_2024", "TD_2023", "TD_2024"}


def lock_holdout(
    df: pd.DataFrame, cutoff_year: int = 2023
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate the development data from the final holdout data.

    Every row from 2023 onward goes into the holdout set. Because the split
    uses the year, all households from the same season-panel remain together.

    Parameters
    ----------
    df : pandas.DataFrame
        A loaded dataset containing ``year`` and ``panel_id`` columns.
    cutoff_year : int, default 2023
        The first year included in the final holdout set.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The development data followed by the holdout data. Models should be
        trained, selected, and calibrated using only the development data.

    Raises
    ------
    ValueError
        Raised when the holdout does not contain the four panels that were
        originally reserved for final testing.
    """
    is_holdout = df["year"] >= cutoff_year

    development = df.loc[~is_holdout].copy()
    holdout = df.loc[is_holdout].copy()

    actual_panels = set(holdout["panel_id"].unique())

    if actual_panels != EXPECTED_HOLDOUT_PANELS:
        raise ValueError(
            "Holdout panels do not match the locked set. "
            f"expected={sorted(EXPECTED_HOLDOUT_PANELS)} "
            f"got={sorted(actual_panels)}"
        )

    return development, holdout
