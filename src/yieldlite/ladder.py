"""Build ready-to-fit pipelines for each model in the yieldlite ladder.

Every model receives the same input table: the original survey columns plus
``npk_total_kg_ha``. Variety pooling happens inside each pipeline, which means
it learns only from the rows used to fit that particular model.

This is especially important during leave-one-panel-out validation and
uncertainty calibration. If variety counts were calculated from the complete
dataset before splitting, information from the held-out panel could leak into
the training process.

To add another model to the ladder, define its estimator in
``yieldlite.models`` and register its builder in ``_BUILDERS`` below.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from yieldlite.features import VarietyPooler
from yieldlite.models import SeasonDistrictTable, build_m1

_VARIETY_COLUMNS = ("variety_raw", "variety_flag")


class PoolVarieties(TransformerMixin, BaseEstimator):
    """Add a pooled variety column without leaking held-out information.

    A recognized variety keeps its name only when it appears at least
    ``min_count`` times in the training rows. Rare, unclear, unresolved, and
    previously unseen varieties are grouped under ``"pooled"``.

    Parameters
    ----------
    min_count : int, default 30
        Minimum number of training rows required for a variety to keep its
        original name.

    Attributes
    ----------
    pooler_ : yieldlite.features.VarietyPooler
        The variety pooler learned from the training rows.
    """

    def __init__(self, min_count: int = 30) -> None:
        self.min_count = min_count

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> PoolVarieties:
        """Learn which varieties have enough support to remain separate.

        Parameters
        ----------
        X : pandas.DataFrame
            Training rows containing ``variety_raw`` and ``variety_flag``.
        y : object, optional
            Not used. This argument is included so the transformer works
            normally inside a scikit-learn pipeline.

        Returns
        -------
        PoolVarieties
            This fitted transformer.
        """
        self._check_columns(X)

        self.pooler_ = VarietyPooler(
            min_count=self.min_count
        ).fit(
            X["variety_raw"],
            X["variety_flag"],
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the rows with a new ``variety_pooled`` column.

        Parameters
        ----------
        X : pandas.DataFrame
            Rows containing ``variety_raw`` and ``variety_flag``.

        Returns
        -------
        pandas.DataFrame
            A copy of ``X`` with ``variety_pooled`` added.
        """
        check_is_fitted(self, "pooler_")
        self._check_columns(X)

        pooled = self.pooler_.transform(
            X["variety_raw"],
            X["variety_flag"],
        )

        return X.assign(variety_pooled=pooled)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        """Check that the input is a DataFrame with both variety columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        missing = [
            column
            for column in _VARIETY_COLUMNS
            if column not in X.columns
        ]
        if missing:
            raise ValueError(
                f"X is missing required columns: {missing}"
            )


# Each name points to a function or class that creates an unfitted model.
_BUILDERS: dict[str, Callable[[], BaseEstimator]] = {
    "m0": SeasonDistrictTable,
    "m1": build_m1,
}

RUNGS = tuple(_BUILDERS)


def build_rung(
    name: str,
    min_variety_count: int = 30,
) -> Pipeline:
    """Create an unfitted pipeline for one model rung.

    The pipeline first learns how to pool varieties and then fits the selected
    model. Because both steps stay together, the same fitted pipeline can be
    used directly for prediction without preparing the variety column by
    hand.

    Parameters
    ----------
    name : {"m0", "m1"}
        Name of the model rung to build.
    min_variety_count : int, default 30
        Minimum number of training rows required for a variety to remain its
        own category.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline containing the ``pool`` and ``model`` steps.

    Raises
    ------
    ValueError
        If the requested rung has not been registered.
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown rung {name!r}; choose from {list(RUNGS)}"
        )

    return Pipeline(
        steps=[
            (
                "pool",
                PoolVarieties(
                    min_count=min_variety_count
                ),
            ),
            (
                "model",
                _BUILDERS[name](),
            ),
        ]
    )
