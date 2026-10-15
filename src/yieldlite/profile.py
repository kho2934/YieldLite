"""Summarize how much usable information is available for model fitting.

The model-selection policy does not need every training row. It needs a few
clear measures of data support: the number of independent season-year panels,
the household count in each district-season group, the support for each pooled
variety, and the share of rows carrying a data-quality flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = (
    "panel_id",
    "season",
    "district_code",
    "variety_pooled",
)


@dataclass(frozen=True)
class DataProfile:
    """A compact description of the data available for training.

    Attributes
    ----------
    n_rows : int
        Number of household-season observations.
    n_panels : int
        Number of distinct season-year panels.
    n_seasons : int
        Number of distinct seasons.
    panels_per_season : dict[str, int]
        Number of season-year panels available for each season.
    median_households_per_cell : float
        Median number of observations in a district-season group.
    within_season_contrasts : int
        ``n_panels - n_seasons``. This summarizes the year-to-year variation
        left after each season is allowed to have its own baseline.
    rows_per_variety : dict[str, int]
        Number of observations supporting each pooled variety level.
    flagged_share : float
        Share of rows carrying at least one data-quality flag selected by the
        caller.
    """

    n_rows: int
    n_panels: int
    n_seasons: int
    panels_per_season: dict[str, int]
    median_households_per_cell: float
    within_season_contrasts: int
    rows_per_variety: dict[str, int]
    flagged_share: float

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        flag_column: str = "flagged",
    ) -> DataProfile:
        """Build a profile from a prepared training table.

        Parameters
        ----------
        df : pandas.DataFrame
            Prepared data containing ``panel_id``, ``season``,
            ``district_code``, ``variety_pooled``, and a boolean flag column.
            These columns cannot contain missing values, and each panel must
            belong to exactly one season.
        flag_column : str, default "flagged"
            Boolean column marking rows with any data-quality concern. The
            caller decides which source flags contribute to it; this class
            only summarizes the result.

        Returns
        -------
        DataProfile
            Counts and proportions used by the model-selection policy.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if not df.columns.is_unique:
            raise ValueError("df must have unique column names")

        required = (*_REQUIRED_COLUMNS, flag_column)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"df is missing required columns: {missing}")

        if df.empty:
            raise ValueError("cannot profile an empty table")

        profile_columns = df.loc[:, list(required)]
        if profile_columns.isna().any().any():
            raise ValueError("profile columns cannot contain missing values")

        flags = df[flag_column]
        if not pd.api.types.is_bool_dtype(flags.dtype):
            raise ValueError(f"{flag_column} must be a boolean column")

        # A panel represents one season and year, so two season labels inside
        # the same panel would make all panel-based counts unreliable.
        seasons_per_panel = df.groupby("panel_id")["season"].nunique()
        if (seasons_per_panel > 1).any():
            bad_panels = list(seasons_per_panel.index[seasons_per_panel > 1])
            raise ValueError(
                f"each panel must belong to one season; check {bad_panels}"
            )

        panel_season = df[["panel_id", "season"]].drop_duplicates()
        panels_per_season = {
            str(season): int(count)
            for season, count in panel_season["season"]
            .value_counts()
            .sort_index()
            .items()
        }

        cell_sizes = df.groupby(["district_code", "season"]).size()
        rows_per_variety = {
            str(variety): int(count)
            for variety, count in df["variety_pooled"]
            .value_counts()
            .sort_index()
            .items()
        }

        n_panels = int(df["panel_id"].nunique())
        n_seasons = int(df["season"].nunique())

        return cls(
            n_rows=int(len(df)),
            n_panels=n_panels,
            n_seasons=n_seasons,
            panels_per_season=panels_per_season,
            median_households_per_cell=float(np.median(cell_sizes)),
            within_season_contrasts=n_panels - n_seasons,
            rows_per_variety=rows_per_variety,
            flagged_share=float(flags.mean()),
        )
