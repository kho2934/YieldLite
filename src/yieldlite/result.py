"""Store and explain the outcome of one yield prediction.

A request can have one of three outcomes:

* ``ok``: a prediction and interval with no warnings.
* ``flagged``: a prediction and interval with one or more warnings.
* ``refused``: no prediction because the available data cannot support one.

When a prediction is refused, yieldlite can still show the usual historical
yield range if enough past records exist for that district and season. This
range is clearly labeled as historical information rather than a prediction.

The status is determined from the recorded reasons, so a refused result can
never accidentally be labeled ``ok``. These reasons describe problems with
the prediction itself. The explanation of why a model was selected remains
in ``yieldlite.selection``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from yieldlite.messages import render, render_reasons
from yieldlite.selection import LADDER, Reason

Status = Literal["ok", "flagged", "refused"]

# Any of these reasons means yieldlite should not return a prediction.
REFUSAL_CODES = frozenset(
    {
        "too_few_panels",
        "unknown_district",
        "season_unresolved",
        "npk_unit_unresolved",
        "npk_out_of_range",
        "level_unsupported",
        "interval_uninformative",
    }
)

# These reasons allow a prediction, but the user should still be warned.
FLAG_CODES = frozenset(
    {
        "variety_pooled",
        "history_thin",
    }
)


def _finite(value: object, name: str) -> float:
    """Return ``value`` as a float after checking that it is finite."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")

    return float(value)


def _level(value: object, name: str) -> float:
    """Check and return a probability level between zero and one."""
    level = _finite(value, name)

    if not 0 < level < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")

    return level


def _percent(level: float) -> int:
    """Convert a probability level such as 0.8 to a whole percentage."""
    return int(round(level * 100))


@dataclass(frozen=True)
class Interval:
    """Prediction interval for one yield estimate, measured in t/ha.

    ``level`` is the intended coverage, such as 0.8 for an 80% prediction
    interval.
    """

    lower: float
    upper: float
    level: float

    def __post_init__(self) -> None:
        """Check that the interval bounds and level are valid."""
        lower = _finite(self.lower, "lower")
        upper = _finite(self.upper, "upper")

        if lower > upper:
            raise ValueError("lower cannot be larger than upper")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "level",
            _level(self.level, "level"),
        )


@dataclass(frozen=True)
class HistoryRange:
    """Usual historical yield range for one district and season.

    The bounds are measured in t/ha. ``level`` is the share of past records
    included in the range, while ``n_rows`` records how many observations
    were available when the range was calculated.
    """

    lower: float
    upper: float
    level: float
    n_rows: int

    def __post_init__(self) -> None:
        """Check the bounds, coverage level, and number of records."""
        lower = _finite(self.lower, "lower")
        upper = _finite(self.upper, "upper")

        if lower > upper:
            raise ValueError("lower cannot be larger than upper")

        if (
            isinstance(self.n_rows, bool)
            or not isinstance(self.n_rows, int)
            or self.n_rows < 1
        ):
            raise ValueError("n_rows must be a positive whole number")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(
            self,
            "level",
            _level(self.level, "level"),
        )


@dataclass(frozen=True)
class PredictionResult:
    """Complete result returned for one prediction request.

    Parameters
    ----------
    reasons : tuple of Reason
        Warnings or refusal reasons attached to the request. Every code must
        belong to ``REFUSAL_CODES`` or ``FLAG_CODES``.
    estimate : float, optional
        Predicted yield in t/ha. It is required for a prediction and forbidden
        for a refused result.
    interval : Interval, optional
        Prediction interval containing the estimate. It is required for a
        prediction and forbidden for a refused result.
    history : HistoryRange, optional
        Usual historical yield range for the requested district and season.
    rung : str, optional
        Model that produced the estimate. It must be one of the models in
        ``selection.LADDER``.
    district, season : str, optional
        District and season for the request. Both are required when a
        historical range is included.
    """

    reasons: tuple[Reason, ...] = ()
    estimate: float | None = None
    interval: Interval | None = None
    history: HistoryRange | None = None
    rung: str | None = None
    district: str | None = None
    season: str | None = None

    def __post_init__(self) -> None:
        """Check that every part of the result agrees with its status."""
        reasons = tuple(self.reasons)

        for reason in reasons:
            if not isinstance(reason, Reason):
                raise TypeError("reasons must be Reason objects")

            if reason.code not in REFUSAL_CODES | FLAG_CODES:
                raise ValueError(
                    f"{reason.code} is not a refusal or flag code"
                )

        object.__setattr__(self, "reasons", reasons)

        if self.rung is not None and self.rung not in LADDER:
            raise ValueError(f"rung must be one of {list(LADDER)}")

        if self.history is not None:
            if not isinstance(self.history, HistoryRange):
                raise TypeError("history must be a HistoryRange")

            if self.district is None or self.season is None:
                raise ValueError("history needs a district and a season")

        if self.status == "refused":
            if self.estimate is not None or self.interval is not None:
                raise ValueError(
                    "a refused result cannot have an estimate or an interval"
                )

            return

        if self.estimate is None or self.interval is None:
            raise ValueError(
                "a prediction needs an estimate and an interval"
            )

        if not isinstance(self.interval, Interval):
            raise TypeError("interval must be an Interval")

        if self.rung is None:
            raise ValueError("a prediction needs the model that made it")

        estimate = _finite(self.estimate, "estimate")

        if not self.interval.lower <= estimate <= self.interval.upper:
            raise ValueError("the interval must contain the estimate")

        object.__setattr__(self, "estimate", estimate)

    @property
    def status(self) -> Status:
        """Return the result status: ``ok``, ``flagged``, or ``refused``."""
        codes = {
            reason.code
            for reason in self.reasons
        }

        if codes & REFUSAL_CODES:
            return "refused"

        if codes:
            return "flagged"

        return "ok"

    @property
    def refused(self) -> bool:
        """Return True when the request did not produce a prediction."""
        return self.status == "refused"

    def summary(self, lang: str = "en") -> list[str]:
        """Turn the result into readable lines of text.

        For a successful prediction, the estimate appears first, followed by
        any warnings and then the historical range. A refused result has no
        estimate line.
        """
        lines = []

        if self.interval is not None:
            lines.append(
                render(
                    "result_estimate",
                    lang,
                    estimate=self.estimate,
                    lower=self.interval.lower,
                    upper=self.interval.upper,
                    level=_percent(self.interval.level),
                )
            )

        lines.extend(
            render_reasons(
                self.reasons,
                lang,
            )
        )

        if self.history is not None:
            lines.append(
                render(
                    "result_history",
                    lang,
                    district=self.district,
                    season=self.season,
                    lower=self.history.lower,
                    upper=self.history.upper,
                    level=_percent(self.history.level),
                )
            )

        return lines
