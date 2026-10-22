"""Panel-based prediction intervals for datasets with only a few seasons.

The household rows in this project are not fully independent. Farms from the
same season-year panel share weather, prices, and other conditions, so the
number of independent examples is closer to the number of panels than the
number of households.

This module follows the same structure used to validate the models: it leaves
out one complete panel at a time. The method is CV+ from Barber et al. (2021).
Each calibration household is predicted by a model that was trained without
that household's panel, so its error is genuinely out of panel.

Large panels should not have more influence simply because they contain more
households. To prevent that, each household receives a weight of
``1 / (number of rows in its panel)``. This gives every panel a total weight
of one, following the pooling approach described by Dunn, Wasserman, and
Ramdas (2023).

With ``K`` panels, the calibration errors have a total weight of ``K`` and the
new panel adds one more unit. Therefore, the highest interval level that can
remain finite is ``K / (K + 1)``. If a user requests a higher level, the model
refuses the request instead of returning an artificially stretched interval.

Coverage becomes more reliable as the number of panels grows. This is not a
finite-sample guarantee, so the actual coverage should still be reported for
each held-out panel.

References
----------
Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. (2021).
Predictive inference with the jackknife+. Annals of Statistics, 49(1).

Dunn, R., Wasserman, L., and Ramdas, A. (2023). Distribution-free prediction
sets for two-layer hierarchical models. Journal of the American Statistical
Association, 118(544).
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

# Leave a small amount of room for floating-point rounding when a requested
# level is exactly at the supported limit, such as 80% with four panels.
_TOLERANCE = 1e-9

# Process new observations in small groups so interval prediction does not
# use too much memory on a large dataset.
_CHUNK_ROWS = 256


class LevelUnsupportedError(ValueError):
    """Raised when too few panels exist for the requested interval level."""


def _check_level(level: float) -> None:
    """Make sure the requested level is one finite number between 0 and 1."""
    if (
        isinstance(level, bool)
        or not isinstance(level, Real)
        or not np.isfinite(level)
        or not 0 < level < 1
    ):
        raise ValueError(
            "level must be a finite number strictly between 0 and 1"
        )


def max_supported_level(n_panels: int) -> float:
    """Calculate the highest interval level supported by the panel count.

    Parameters
    ----------
    n_panels : int
        Number of independent panels used for calibration.

    Returns
    -------
    float
        The maximum supported level, calculated as
        ``n_panels / (n_panels + 1)``. For example, four panels support an
        80% interval, eight support up to 88.9%, and twelve support up to
        92.3%.
    """
    if (
        isinstance(n_panels, bool)
        or not isinstance(n_panels, Integral)
        or n_panels < 1
    ):
        raise ValueError("n_panels must be an integer of at least 1")

    return n_panels / (n_panels + 1)


def min_panels_for_level(level: float) -> int:
    """Calculate how many panels are needed for an interval level.

    Parameters
    ----------
    level : float
        Desired interval level, strictly between 0 and 1. For example,
        ``0.8`` requests an 80% interval.

    Returns
    -------
    int
        The smallest number of panels for which
        ``K / (K + 1)`` reaches the requested level. An 80% interval needs
        at least four panels, while a 90% interval needs at least nine.
    """
    _check_level(level)

    return max(
        1,
        math.ceil(
            level / (1.0 - level) - _TOLERANCE
        ),
    )


def interval_coverage(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> float:
    """Measure how often observed values fall inside their intervals.

    Values equal to either boundary are counted as covered.

    Parameters
    ----------
    y_true : array-like
        Observed target values.
    lower, upper : array-like
        Lower and upper interval bounds for each observation.

    Returns
    -------
    float
        The share of observations inside their intervals, between 0 and 1.

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional, have different lengths,
        are empty, or contain non-finite values.
    """
    observed = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)

    if (
        observed.ndim != 1
        or not (observed.shape == low.shape == high.shape)
    ):
        raise ValueError(
            "y_true, lower, and upper must be one-dimensional "
            "and have the same length"
        )

    if observed.size == 0:
        raise ValueError("cannot calculate coverage for an empty dataset")

    if not (
        np.isfinite(observed).all()
        and np.isfinite(low).all()
        and np.isfinite(high).all()
    ):
        raise ValueError(
            "observed values and interval bounds must all be finite"
        )

    is_covered = (observed >= low) & (observed <= high)
    return float(np.mean(is_covered))


def _as_predictions(
    values: object,
    expected_rows: int,
    description: str,
) -> np.ndarray:
    """Convert model output into a checked one-dimensional prediction array."""
    predictions = np.asarray(values, dtype=float)

    if (
        predictions.ndim != 1
        or len(predictions) != expected_rows
    ):
        raise ValueError(
            "predict() must return exactly one value for each of the "
            f"{expected_rows} rows in {description}"
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"the model returned a non-finite prediction for {description}"
        )

    return predictions


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    mass: float,
) -> np.ndarray:
    """Find the first value in each row that reaches a target weight.

    Parameters
    ----------
    values : numpy.ndarray
        A two-dimensional array. Each row represents one new household and
        each column represents one calibration household.
    weights : numpy.ndarray
        Weight assigned to each calibration household. The same weights are
        used for every new row.
    mass : float
        The cumulative weight that must be reached, starting from the
        smallest value.

    Returns
    -------
    numpy.ndarray
        One weighted-quantile value for each row in ``values``.
    """
    order = np.argsort(values, axis=1, kind="stable")
    ordered_values = np.take_along_axis(
        values,
        order,
        axis=1,
    )
    cumulative_weight = np.cumsum(
        weights[order],
        axis=1,
    )

    first_reaching = np.argmax(
        cumulative_weight >= mass - _TOLERANCE,
        axis=1,
    )

    return ordered_values[
        np.arange(len(values)),
        first_reaching,
    ]


class PanelCVPlus(BaseEstimator):
    """Build CV+ intervals using leave-one-panel-out model fits.

    During ``fit``, the class trains one model for every panel. Each model is
    fitted on all panels except one, and its errors are measured on the panel
    it did not see. A final model is also fitted on all available rows to
    provide the point predictions.

    During ``predict_interval``, predictions from the fold models are combined
    with their held-out errors to form an interval for each new household.

    Parameters
    ----------
    estimator : sklearn-compatible regressor
        An unfitted model or pipeline. It is cloned before every fit. Any
        preprocessing that learns from data, such as variety pooling, should
        be included inside this estimator so that each fold learns only from
        its own training rows.

    Attributes
    ----------
    estimator_ : estimator
        Final model fitted on all training rows. This model provides the point
        predictions.
    fold_estimators_ : list of estimator
        One fitted model for each panel. Each model was trained without its
        corresponding panel.
    panels_ : numpy.ndarray
        Sorted panel labels. Their order matches ``fold_estimators_``.
    codes_ : numpy.ndarray
        Integer position in ``panels_`` for every calibration row.
    residuals_ : numpy.ndarray
        Absolute error for every calibration row, produced by the model that
        did not see that row's panel.
    weights_ : numpy.ndarray
        Weight of each calibration row. A row receives
        ``1 / (number of rows in its panel)``, so each panel contributes a
        total weight of one.
    n_panels_ : int
        Number of panels used for calibration.
    max_level_ : float
        Highest interval level that remains finite, calculated as
        ``K / (K + 1)``.

    Notes
    -----
    If a fold removes the only available panel for one season, the model must
    predict a season it has never seen. This can create large held-out errors
    and wider intervals. That conservative behavior is appropriate, and it
    becomes less important once every season has at least two panels.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        panels: pd.Series | np.ndarray,
    ) -> PanelCVPlus:
        """Fit the fold models, final model, and held-out residuals.

        Parameters
        ----------
        X : pandas.DataFrame
            Model features with one row for each household.
        y : array-like
            Observed yield for each household.
        panels : array-like
            Season-year panel assigned to each row. When pandas Series are
            provided, their indexes must match the index of ``X``.

        Returns
        -------
        PanelCVPlus
            The fitted interval model.

        Raises
        ------
        TypeError
            If ``X`` is not a pandas DataFrame.
        ValueError
            If the inputs have different lengths or indexes, contain missing
            panel IDs or invalid target values, or include fewer than two
            distinct panels.
        """
        target, codes, labels = self._check_fit_inputs(
            X,
            y,
            panels,
        )

        n_panels = len(labels)
        residuals = np.empty(len(X), dtype=float)
        fold_estimators = []

        # Fit one model for every leave-one-panel-out split.
        for code in range(n_panels):
            held_out = np.flatnonzero(codes == code)
            training = np.flatnonzero(codes != code)

            model = clone(self.estimator)
            model.fit(
                X.iloc[training],
                target[training],
            )

            predicted = _as_predictions(
                model.predict(X.iloc[held_out]),
                len(held_out),
                f"held-out panel {labels[code]}",
            )

            residuals[held_out] = np.abs(
                target[held_out] - predicted
            )
            fold_estimators.append(model)

        # Fit one final model on all rows for the point estimates.
        final_model = clone(self.estimator)
        final_model.fit(X, target)

        panel_sizes = np.bincount(
            codes,
            minlength=n_panels,
        )

        self.estimator_ = final_model
        self.fold_estimators_ = fold_estimators
        self.panels_ = labels
        self.codes_ = codes
        self.residuals_ = residuals

        # Each panel receives total weight one, regardless of how many
        # household rows it contains.
        self.weights_ = 1.0 / panel_sizes[codes]

        self.n_panels_ = n_panels
        self.max_level_ = max_supported_level(n_panels)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point predictions from the model fitted on all rows.

        Parameters
        ----------
        X : pandas.DataFrame
            New household rows to predict.

        Returns
        -------
        numpy.ndarray
            One predicted yield for each row.
        """
        check_is_fitted(self, "estimator_")
        self._check_frame(X)

        return _as_predictions(
            self.estimator_.predict(X),
            len(X),
            "the new data",
        )

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a prediction interval for every new row.

        The interval is expanded when necessary to include the final model's
        point prediction. A standard CV+ interval does not always contain that
        estimate, but a farmer should not receive a point estimate that sits
        outside its own reported range.

        Parameters
        ----------
        X : pandas.DataFrame
            New household rows to predict.
        level : float, default 0.8
            Requested interval level, strictly between 0 and 1.

        Returns
        -------
        lower, upper : numpy.ndarray
            Lower and upper bounds for every new row.

        Raises
        ------
        LevelUnsupportedError
            If the requested level is higher than the number of available
            panels can support.
        """
        check_is_fitted(self, "estimator_")
        _check_level(level)

        if level > self.max_level_ + _TOLERANCE:
            raise LevelUnsupportedError(
                f"{self.n_panels_} panels can support intervals only up to "
                f"{self.max_level_:.1%}; the requested level was {level:.1%}"
            )

        estimate = self.predict(X)

        # Every column contains predictions from one fold model for all of
        # the new household rows.
        fold_predictions = np.column_stack(
            [
                _as_predictions(
                    model.predict(X),
                    len(X),
                    "the new data",
                )
                for model in self.fold_estimators_
            ]
        )

        mass = level * (self.n_panels_ + 1)

        lower = np.empty(len(X), dtype=float)
        upper = np.empty(len(X), dtype=float)

        for start in range(0, len(X), _CHUNK_ROWS):
            rows = slice(
                start,
                start + _CHUNK_ROWS,
            )

            # Match each calibration household with the fold model that was
            # trained without that household's panel.
            block = fold_predictions[rows][:, self.codes_]

            upper[rows] = _weighted_quantile(
                block + self.residuals_,
                self.weights_,
                mass,
            )

            lower[rows] = -_weighted_quantile(
                self.residuals_ - block,
                self.weights_,
                mass,
            )

        # Make sure the reported interval includes the final point estimate.
        return (
            np.minimum(lower, estimate),
            np.maximum(upper, estimate),
        )

    @staticmethod
    def _check_frame(X: pd.DataFrame) -> None:
        """Require model features to be supplied as a pandas DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

    def _check_fit_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        panels: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check the training data and encode panel labels as integers."""
        self._check_frame(X)

        if not (len(X) == len(y) == len(panels)):
            raise ValueError(
                "X, y, and panels must contain the same number of rows"
            )

        # Pandas inputs must line up exactly with the feature table.
        for name, values in (
            ("y", y),
            ("panels", panels),
        ):
            if (
                isinstance(values, pd.Series)
                and not values.index.equals(X.index)
            ):
                raise ValueError(
                    f"{name} must use the same index as X"
                )

        target = pd.to_numeric(
            pd.Series(np.asarray(y)),
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(target).all():
            raise ValueError(
                "y must contain only finite numeric values"
            )

        panel_ids = pd.Series(np.asarray(panels))
        if panel_ids.isna().any():
            raise ValueError("panel IDs cannot be missing")

        codes, labels = pd.factorize(
            panel_ids,
            sort=True,
        )

        if len(labels) < 2:
            raise ValueError(
                "panel-based intervals need at least two distinct panels; "
                f"the training data contains {len(labels)}"
            )

        return target, codes, np.asarray(labels)
