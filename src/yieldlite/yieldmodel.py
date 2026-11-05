"""Fit yieldlite to survey data and make predictions with uncertainty.

``YieldModel`` brings the main parts of the package together:

1. ``fit`` profiles the training data, asks ``SelectionPolicy`` to choose the
   simplest model the data can support, and calibrates its prediction
   intervals by holding out one panel at a time.
2. ``explain`` gives a plain-language explanation of the model choice.
3. ``predict`` returns a ``PredictionResult`` containing either an estimate
   and interval or a clear refusal. When possible, a refusal still includes
   the usual historical yield range for that district and season.

Only M0 and M1 are currently implemented. Weather-based and boosted-tree
models are not yet used for prediction.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from yieldlite.conformal import (
    LevelUnsupportedError,
    PanelCVPlus,
    min_panels_for_level,
)
from yieldlite.features import VarietyPooler
from yieldlite.history import typical_interval
from yieldlite.ladder import build_rung
from yieldlite.messages import render_reasons
from yieldlite.profile import DataProfile
from yieldlite.result import (
    HistoryRange,
    Interval,
    PredictionResult,
)
from yieldlite.selection import (
    LeaveOnePanelOutGain,
    Reason,
    Selection,
    SelectionPolicy,
)
from yieldlite.study import TARGET

_REQUIRED = (
    "panel_id",
    "season",
    "district_code",
    "variety_raw",
    "variety_flag",
    "npk_total_kg_ha",
    "flagged",
    TARGET,
)

# These are the columns passed to a fitted model.
_FEATURES = [
    "district_code",
    "season",
    "variety_raw",
    "variety_flag",
    "npk_total_kg_ha",
]

# The supported fertilizer range is taken from the 1st to 99th percentile of
# the training data. Values outside it are too far from what the model learned.
_NPK_RANGE = (1.0, 99.0)

_NOT_GIVEN = "(not given)"


def _check_level(level: object) -> float:
    """Check and return a prediction-interval level between zero and one."""
    if (
        isinstance(level, bool)
        or not isinstance(level, Real)
        or not math.isfinite(level)
        or not 0 < level < 1
    ):
        raise ValueError(
            "level must be a number strictly between 0 and 1"
        )

    return float(level)


def _label(value: object, name: str) -> str:
    """Clean a district or season label to match the training data."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")

    return value.strip().upper()


class YieldModel:
    """Predict rice yield from a small survey and show its uncertainty.

    Parameters
    ----------
    thresholds : mapping, optional
        Threshold document used by ``SelectionPolicy``. When omitted,
        yieldlite uses the threshold file included with the package.
    min_variety_count : int, default 30
        Minimum number of training records required for a variety to keep its
        own model effect. Varieties with fewer records are pooled.
    refuse_if_uninformative : bool, default False
        When True, yieldlite refuses a prediction if its interval is no
        narrower than the usual historical yield range for the same district
        and season. The historical range is shown instead.

    Attributes
    ----------
    profile_ : DataProfile
        Summary of the data used for fitting.
    selection_ : Selection
        Selected model and the reasons behind the choice.
    intervals_ : PanelCVPlus or None
        Fitted prediction and interval model. It is None when fitting was
        refused because the data were insufficient.
    """

    def __init__(
        self,
        thresholds: Any = None,
        min_variety_count: int = 30,
        refuse_if_uninformative: bool = False,
    ) -> None:
        self.policy = SelectionPolicy(thresholds)
        self.min_variety_count = min_variety_count
        self.refuse_if_uninformative = bool(
            refuse_if_uninformative
        )

    @property
    def rung(self) -> str | None:
        """Return the selected model rung, or None if fitting was refused."""
        self._check_fitted()

        return self.selection_.rung

    def fit(self, df: pd.DataFrame) -> YieldModel:
        """Choose and fit a model using a prepared survey table.

        Parameters
        ----------
        df : pandas.DataFrame
            Table produced by ``yieldlite.prepare.prepare_frame``. It must
            include panel, season, district, variety, fertilizer, quality-flag,
            and yield columns.

        Returns
        -------
        YieldModel
            The fitted object. If too little data are available, fitting still
            records the refusal so future prediction requests can explain why
            no estimate is returned.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        missing = [
            column
            for column in _REQUIRED
            if column not in df.columns
        ]
        if missing:
            raise ValueError(f"df is missing columns: {missing}")

        if df.empty:
            raise ValueError("df cannot be empty")

        if not df.index.is_unique:
            raise ValueError("df must use a unique row index")

        frame = df.loc[:, list(_REQUIRED)].copy()

        if frame.isna().any().any():
            raise ValueError(
                "df cannot contain missing values in required columns"
            )

        # Convert the numeric columns explicitly so invalid strings or
        # infinite values are caught before model fitting begins.
        for column in ("npk_total_kg_ha", TARGET):
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            if not np.isfinite(
                values.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"{column} must contain finite numbers"
                )

            frame[column] = values.astype(float)

        # Use one consistent form for labels throughout fitting and prediction.
        for column in (
            "district_code",
            "season",
            "variety_raw",
        ):
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        # Pool varieties here so the data profile follows the same variety
        # rule as the model ladder.
        pooler = VarietyPooler(
            self.min_variety_count
        ).fit(
            frame["variety_raw"],
            frame["variety_flag"],
        )

        profile = DataProfile.from_frame(
            frame.assign(
                variety_pooled=pooler.transform(
                    frame["variety_raw"],
                    frame["variety_flag"],
                )
            )
        )

        selection = self.policy.select(
            profile,
            LeaveOnePanelOutGain(
                frame,
                self.min_variety_count,
            ),
        )

        self.profile_ = profile
        self.selection_: Selection = selection
        self.train_ = frame
        self.districts_ = frozenset(
            frame["district_code"]
        )
        self.seasons_ = frozenset(
            frame["season"]
        )
        self.npk_range_ = (
            float(
                np.percentile(
                    frame["npk_total_kg_ha"],
                    _NPK_RANGE[0],
                )
            ),
            float(
                np.percentile(
                    frame["npk_total_kg_ha"],
                    _NPK_RANGE[1],
                )
            ),
        )
        self.intervals_ = None

        if not selection.refused:
            estimator = build_rung(
                selection.rung,
                self.min_variety_count,
            )

            self.intervals_ = PanelCVPlus(
                estimator
            ).fit(
                frame[_FEATURES],
                frame[TARGET],
                frame["panel_id"],
            )

        return self

    def explain(self, lang: str = "en", verbose: bool = False) -> list[str]:
        """Explain the model choice in English or Vietnamese.

        Parameters
        ----------
        lang : {"en", "vi"}, default "en"
            Language of the explanation.
        verbose : bool, default False
            Also list the models that were skipped because the data they
            need has not been studied yet (M2 and M3 in this version). They
            are hidden by default because they are not built, so a reader
            cannot act on them.

        Returns
        -------
        list of str
            One sentence per decision, ending with the model that was
            chosen.
        """
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be True or False")

        self._check_fitted()

        reasons = self.selection_.reasons
        if not verbose:
            reasons = [
                reason
                for reason in reasons
                if reason.code != "rung_not_evaluated"
            ]

        return render_reasons(reasons, lang)

    def predict(
        self,
        district: str,
        season: str,
        variety: str | None = None,
        npk_total: float | None = None,
        level: float = 0.8,
    ) -> PredictionResult:
        """Predict yield for one household.

        Parameters
        ----------
        district, season : str
            District and season codes used in the training data.
        variety : str, optional
            Rice variety. A missing, rare, or unseen variety uses the pooled
            variety effect and receives a warning.
        npk_total : float, optional
            Total N + P2O5 + K2O, measured as kilograms of pure nutrient per
            hectare. It is required when the selected model uses fertilizer.
        level : float, default 0.8
            Intended coverage of the prediction interval.

        Returns
        -------
        PredictionResult
            An ``ok`` prediction, a ``flagged`` prediction with warnings, or a
            ``refused`` result with no estimate. A historical range is included
            whenever enough past data exist.
        """
        self._check_fitted()

        level = _check_level(level)
        district = _label(district, "district")
        season = _label(season, "season")

        if self.selection_.refused:
            return self._refusal(
                self.selection_.reasons,
                district,
                season,
                level,
            )

        if district not in self.districts_:
            return self._refusal(
                [
                    Reason(
                        "unknown_district",
                        {"district": district},
                    )
                ],
                district,
                season,
                level,
            )

        if season not in self.seasons_:
            return self._refusal(
                [
                    Reason(
                        "season_unresolved",
                        {},
                    )
                ],
                district,
                season,
                level,
            )

        reasons: list[Reason] = []
        uses_fertilizer = self.selection_.rung != "m0"

        if uses_fertilizer:
            npk = self._npk(npk_total)
            low, high = self.npk_range_

            # Refuse to extrapolate beyond nearly all fertilizer amounts seen
            # during training.
            if not low <= npk <= high:
                return self._refusal(
                    [
                        Reason(
                            "npk_out_of_range",
                            {
                                "npk": round(npk),
                                "low": round(low),
                                "high": round(high),
                            },
                        )
                    ],
                    district,
                    season,
                    level,
                )
        else:
            # M0 does not use fertilizer, but the shared model input still
            # needs a finite placeholder.
            npk = self.npk_range_[0]

        variety_raw = (
            _NOT_GIVEN
            if variety is None
            else str(variety).strip().upper()
        )

        row = pd.DataFrame(
            {
                "district_code": [district],
                "season": [season],
                "variety_raw": [variety_raw],
                "variety_flag": [
                    "unresolved"
                    if variety is None
                    else "known"
                ],
                "npk_total_kg_ha": [npk],
            }
        )

        if uses_fertilizer and self._is_pooled(row):
            reasons.append(
                Reason(
                    "variety_pooled",
                    {"variety": variety_raw},
                )
            )

        try:
            lower, upper = self.intervals_.predict_interval(
                row,
                level,
            )
        except LevelUnsupportedError:
            return self._refusal(
                [
                    Reason(
                        "level_unsupported",
                        {
                            "level": round(level * 100),
                            "needed": min_panels_for_level(
                                level
                            ),
                            "panels": self.profile_.n_panels,
                        },
                    )
                ],
                district,
                season,
                level,
            )

        estimate = float(
            self.intervals_.predict(row)[0]
        )
        lower = float(lower[0])
        upper = float(upper[0])

        history = self._history(
            district,
            season,
            level,
        )

        if history is None:
            reasons.append(
                Reason(
                    "history_thin",
                    {
                        "district": district,
                        "season": season,
                    },
                )
            )
        elif (
            self.refuse_if_uninformative
            and upper - lower
            >= history.upper - history.lower
        ):
            return self._refusal(
                [
                    Reason(
                        "interval_uninformative",
                        {
                            "width": round(
                                upper - lower,
                                2,
                            ),
                            "usual": round(
                                history.upper - history.lower,
                                2,
                            ),
                        },
                    )
                ],
                district,
                season,
                level,
            )

        return PredictionResult(
            reasons=tuple(reasons),
            estimate=estimate,
            interval=Interval(
                lower,
                upper,
                level,
            ),
            history=history,
            rung=self.selection_.rung,
            district=district,
            season=season,
        )

    def _check_fitted(self) -> None:
        """Stop methods from being used before fitting."""
        if not hasattr(self, "selection_"):
            raise RuntimeError(
                "call fit() before using the model"
            )

    def _npk(self, npk_total: object) -> float:
        """Validate the fertilizer amount required by the selected model."""
        if npk_total is None:
            raise ValueError(
                "npk_total is required because "
                f"{self.selection_.rung.upper()} uses fertilizer"
            )

        if (
            isinstance(npk_total, bool)
            or not isinstance(npk_total, Real)
            or not math.isfinite(npk_total)
            or npk_total < 0
        ):
            raise ValueError(
                "npk_total must be a non-negative finite number"
            )

        return float(npk_total)

    def _is_pooled(self, row: pd.DataFrame) -> bool:
        """Return True when the fitted model pools this variety."""
        pool = self.intervals_.estimator_.named_steps[
            "pool"
        ]

        transformed = pool.transform(row)

        return bool(
            (
                transformed["variety_pooled"]
                == "pooled"
            ).iloc[0]
        )

    def _history(
        self,
        district: str,
        season: str,
        level: float,
    ) -> HistoryRange | None:
        """Return the historical yield range when enough records exist."""
        cell = pd.DataFrame(
            {
                "district_code": [district],
                "season": [season],
            }
        )

        lower, upper = typical_interval(
            self.train_,
            cell,
            level=level,
        )

        if not (
            np.isfinite(lower[0])
            and np.isfinite(upper[0])
        ):
            return None

        n_rows = int(
            (
                (
                    self.train_["district_code"]
                    == district
                )
                & (
                    self.train_["season"]
                    == season
                )
            ).sum()
        )

        return HistoryRange(
            float(lower[0]),
            float(upper[0]),
            level,
            n_rows,
        )

    def _refusal(
        self,
        reasons: Any,
        district: str,
        season: str,
        level: float,
    ) -> PredictionResult:
        """Build a refusal and include historical yields when available."""
        known = (
            district in self.districts_
            and season in self.seasons_
        )

        history = (
            self._history(
                district,
                season,
                level,
            )
            if known
            else None
        )

        return PredictionResult(
            reasons=tuple(reasons),
            history=history,
            rung=self.selection_.rung,
            district=district,
            season=season,
        )
