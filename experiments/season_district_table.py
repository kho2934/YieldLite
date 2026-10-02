"""A simple Season × District baseline for rice-yield prediction.

The model predicts each household's yield using the average from its season
and district. When a group has only a few records, its estimate is pulled
toward the overall average for that season so a small or unusual sample does
not have too much influence.
"""

from __future__ import annotations

import pandas as pd


class SeasonDistrictTable:
    """Predict yield from season-and-district averages.

    Parameters
    ----------
    shrinkage : float, default 10
        Controls how strongly small groups are pulled toward their season
        average. A larger value gives the season average more influence.
    """

    def __init__(self, shrinkage: float = 10.0):
        self.shrinkage = shrinkage

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonDistrictTable":
        """Learn the averages from the training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing the "season" and "district_code" columns.
        y : pandas.Series
            Observed rice yields in tonnes per hectare, in the same row order
            as X.

        Returns
        -------
        SeasonDistrictTable
            The fitted model.
        """
        data = X[["season", "district_code"]].copy()
        data["y"] = y.values

        self.overall_mean_ = float(data["y"].mean())
        self.season_mean_ = (
            data.groupby("season")["y"]
            .mean()
            .to_dict()
        )

        group_stats = (
            data.groupby(["season", "district_code"])["y"]
            .agg(["mean", "count"])
        )

        shrunk = {}

        for (season, district), row in group_stats.iterrows():
            group_size = row["count"]
            season_mean = self.season_mean_.get(
                season,
                self.overall_mean_,
            )

            shrunk[(season, district)] = (
                group_size * row["mean"]
                + self.shrinkage * season_mean
            ) / (group_size + self.shrinkage)

        self.group_means_ = shrunk
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the yield for every row in X.

        The model first looks for an average from the same season and district.
        If that combination was not present during training, it uses the
        season average. If the season is also new, it uses the overall training
        average.
        """
        predictions = []

        for season, district in zip(
            X["season"],
            X["district_code"],
        ):
            if (season, district) in self.group_means_:
                prediction = self.group_means_[(season, district)]
            elif season in self.season_mean_:
                prediction = self.season_mean_[season]
            else:
                prediction = self.overall_mean_

            predictions.append(prediction)

        return pd.Series(
            predictions,
            index=X.index,
            name="predicted_yield",
        )
