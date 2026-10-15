"""Prediction models used by yieldlite.

M0 is the simple season-and-district baseline. M1 adds rice variety, total
NPK, season, and a regularized district-by-season effect. Both models follow
the scikit-learn estimator interface so they can share the same validation
code.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    SplineTransformer,
)
from sklearn.utils.validation import check_is_fitted

_M0_COLUMNS = ("season", "district_code")
_M1_COLUMNS = (
    "season",
    "district_code",
    "variety_pooled",
    "npk_total_kg_ha",
)


def _require_columns(
    X: pd.DataFrame,
    required: tuple[str, ...],
) -> None:
    """Check that a model received the DataFrame columns it needs."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    missing = [
        column
        for column in required
        if column not in X.columns
    ]
    if missing:
        raise ValueError(
            f"X is missing required columns: {missing}"
        )

    if X.loc[:, list(required)].isna().any().any():
        raise ValueError(
            "required model columns cannot contain missing values"
        )


class SeasonDistrictTable(RegressorMixin, BaseEstimator):
    """M0: predict yield from season-and-district averages.

    The model calculates an average for each season-district group. Groups
    with fewer rows are pulled toward their season average so that a small
    number of unusual households does not control the prediction.

    M0 has the lowest data requirements in the model ladder. The public
    selection policy can still refuse to fit when the full dataset contains
    too few independent panels.

    Parameters
    ----------
    shrinkage : float, default 10.0
        Controls how strongly each group average is pulled toward its season
        average. Larger values create more shrinkage.
    """

    def __init__(self, shrinkage: float = 10.0) -> None:
        self.shrinkage = shrinkage

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SeasonDistrictTable:
        """Learn the overall, season, and season-district averages."""
        _require_columns(X, _M0_COLUMNS)

        if (
            isinstance(self.shrinkage, bool)
            or not isinstance(self.shrinkage, Real)
            or not np.isfinite(self.shrinkage)
            or self.shrinkage < 0
        ):
            raise ValueError(
                "shrinkage must be a finite number greater "
                "than or equal to 0"
            )

        if len(X) == 0:
            raise ValueError("cannot fit the model with no rows")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows"
            )

        target = pd.to_numeric(
            pd.Series(y),
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(target).all():
            raise ValueError(
                "y must contain only finite numeric values"
            )

        data = X.loc[:, list(_M0_COLUMNS)].copy()
        data["y"] = target

        self.overall_mean_ = float(data["y"].mean())
        self.season_mean_ = (
            data.groupby("season", observed=True)["y"]
            .mean()
            .to_dict()
        )

        group_stats = (
            data.groupby(
                ["season", "district_code"],
                observed=True,
            )["y"]
            .agg(["mean", "count"])
        )

        group_means: dict[tuple[str, str], float] = {}

        for (season, district), row in group_stats.iterrows():
            season_mean = self.season_mean_.get(
                season,
                self.overall_mean_,
            )
            count = float(row["count"])

            group_means[(season, district)] = (
                count * float(row["mean"])
                + float(self.shrinkage) * season_mean
            ) / (count + float(self.shrinkage))

        self.group_means_ = group_means
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict yield for each row.

        A known season-district group uses its shrunk group average. An unseen
        district uses the season average, and an unseen season uses the overall
        training average.

        These fallbacks keep the estimator usable during grouped validation.
        The public yieldlite interface can still refuse an unknown district
        before calling this model.
        """
        check_is_fitted(self, "group_means_")
        _require_columns(X, _M0_COLUMNS)

        predictions = []

        for season, district in zip(
            X["season"],
            X["district_code"],
            strict=True,
        ):
            group = (season, district)

            if group in self.group_means_:
                prediction = self.group_means_[group]
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


def _prepare_m1_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate M1's inputs and add the district-season interaction."""
    _require_columns(X, _M1_COLUMNS)

    prepared = X.copy()
    prepared["npk_total_kg_ha"] = pd.to_numeric(
        prepared["npk_total_kg_ha"],
        errors="coerce",
    )

    if not np.isfinite(prepared["npk_total_kg_ha"]).all():
        raise ValueError(
            "npk_total_kg_ha must contain finite numeric values"
        )

    prepared["district_season"] = (
        prepared["district_code"].astype(str)
        + "_"
        + prepared["season"].astype(str)
    )

    return prepared


def build_m1(
    interaction_shrink: float = 0.1,
    alpha: float = 1.0,
    n_knots: int = 4,
) -> Pipeline:
    """Build the M1 ridge-regression pipeline.

    M1 uses district, season, district-by-season, pooled variety, and total
    NPK. The variety and NPK columns must already be prepared by
    ``yieldlite.features``.

    ``alpha`` and ``interaction_shrink`` are temporary settings until the
    degradation study produces the final values. They should be tuned using
    grouped, panel-aware validation.

    Row-based tuning is not appropriate here. RidgeCV's default
    leave-one-row-out procedure would still mix households from the same
    season-year panel between training and validation.

    Parameters
    ----------
    interaction_shrink : float, default 0.1
        Scales the encoded district-by-season columns before ridge regression.
        A smaller positive value gives those interactions a stronger effective
        penalty.
    alpha : float, default 1.0
        Strength of ridge's L2 penalty.
    n_knots : int, default 4
        Number of training-data quantile knots used for the NPK spline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted M1 model.
    """
    if (
        isinstance(interaction_shrink, bool)
        or not isinstance(interaction_shrink, Real)
        or not np.isfinite(interaction_shrink)
        or interaction_shrink <= 0
    ):
        raise ValueError(
            "interaction_shrink must be a finite number greater than 0"
        )

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not np.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError(
            "alpha must be a finite number greater than or equal to 0"
        )

    if (
        isinstance(n_knots, bool)
        or not isinstance(n_knots, Integral)
        or n_knots < 2
    ):
        raise ValueError(
            "n_knots must be an integer of at least 2"
        )

    encode = ColumnTransformer(
        transformers=[
            (
                "district",
                OneHotEncoder(handle_unknown="ignore"),
                ["district_code"],
            ),
            (
                "season",
                OneHotEncoder(handle_unknown="ignore"),
                ["season"],
            ),
            (
                "district_season",
                OneHotEncoder(handle_unknown="ignore"),
                ["district_season"],
            ),
            (
                "variety",
                OneHotEncoder(handle_unknown="ignore"),
                ["variety_pooled"],
            ),
            (
                "npk",
                SplineTransformer(
                    n_knots=int(n_knots),
                    degree=1,
                    knots="quantile",
                    extrapolation="constant",
                    include_bias=False,
                ),
                ["npk_total_kg_ha"],
            ),
        ],
        transformer_weights={
            "district_season": float(interaction_shrink)
        },
    )

    return Pipeline(
        steps=[
            (
                "prepare_features",
                FunctionTransformer(
                    _prepare_m1_features,
                    validate=False,
                ),
            ),
            ("encode", encode),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
