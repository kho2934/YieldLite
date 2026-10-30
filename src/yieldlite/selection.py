"""Choose the simplest model that the available data can support.

Yieldlite's model ladder starts with M0, a season-and-district average, and
ends with M3, a boosted-tree benchmark. Moving up the ladder requires more
data. The required amounts come from the degradation and interval studies
and are stored in ``data/thresholds.json``.

The policy follows three rules:

1. Refuse to fit when there are fewer than P0 panels because a reliable
   prediction interval cannot be produced.
2. Consider a model only when its required thresholds were reached in the
   studies and the current data meet them.
3. Start with the most complex eligible model, but keep it only when
   leave-one-panel-out validation shows a gain larger than one standard
   error over the next simpler option.

The policy can move a model down the ladder, but it never promotes a model
that the data do not support. Every decision is saved as a ``Reason`` so it
can later be explained in English or Vietnamese.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from yieldlite.ladder import build_rung
from yieldlite.profile import DataProfile
from yieldlite.study import TARGET
from yieldlite.validate import leave_one_panel_out, panel_metrics

LADDER = ("m0", "m1", "m2", "m3")

THRESHOLD_NAMES = ("P0", "P1", "H1", "W2", "N3", "P3")

_STATUSES = ("reached", "not_reached", "not_evaluated")

# Each entry gives the threshold a model needs, the matching quantity in the
# data profile, and the reason to report when the data fall short.
_REQUIREMENTS = {
    "m1": (
        ("P1", "panels", "rung_needs_more_panels"),
        ("H1", "households", "rung_needs_more_households"),
    ),
    "m2": (("W2", "contrasts", "rung_needs_more_contrasts"),),
    "m3": (
        ("N3", "rows", "rung_needs_more_rows"),
        ("P3", "panels", "rung_needs_more_panels"),
    ),
}


@dataclass(frozen=True)
class Reason:
    """One reason behind a model-selection decision.

    ``code`` is a stable label that can be translated by
    ``yieldlite.messages``. ``params`` contains the values needed to explain
    that reason, such as the available and required panel counts.
    """

    code: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanelGain:
    """Improvement of one model over another across held-out panels.

    ``mean`` is the average reduction in panel MAE, so a positive value favors
    the more complex model. ``se`` is the standard error of that gain, and
    ``n_panels`` records how many panels were compared.
    """

    mean: float
    se: float
    n_panels: int

    @property
    def assessable(self) -> bool:
        """Return True when the gain and its uncertainty can be evaluated."""
        return self.n_panels >= 2 and math.isfinite(self.mean) and math.isfinite(
            self.se
        )

    @property
    def shows_gain(self) -> bool:
        """Return True when the gain is larger than one standard error."""
        return self.assessable and self.mean > self.se


@dataclass(frozen=True)
class Selection:
    """Final result of applying the selection policy.

    ``rung`` is the chosen model, or ``None`` when the policy refuses to fit.
    ``reasons`` explains the decision, ``eligible`` lists the models that met
    their data thresholds, and ``gains`` stores any panel-level comparisons
    that were made.
    """

    rung: str | None
    reasons: tuple[Reason, ...]
    eligible: tuple[str, ...] = ()
    gains: dict[str, PanelGain] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        """Return True when no model was selected."""
        return self.rung is None


@dataclass(frozen=True)
class Threshold:
    """One threshold read from ``thresholds.json``."""

    status: str
    value: float | None

    @property
    def reached(self) -> bool:
        """Return True when the study found a usable threshold value."""
        return self.status == "reached"


def load_thresholds() -> dict[str, Any]:
    """Load the threshold document included with yieldlite."""
    path = resources.files("yieldlite") / "data" / "thresholds.json"

    return json.loads(path.read_text(encoding="utf-8"))


def _read_thresholds(document: Mapping[str, Any]) -> dict[str, Threshold]:
    """Validate a threshold document and return its six entries."""
    if not isinstance(document, Mapping) or not isinstance(
        document.get("thresholds"), Mapping
    ):
        raise ValueError("thresholds must be a document with a 'thresholds' mapping")

    entries = document["thresholds"]
    thresholds = {}

    for name in THRESHOLD_NAMES:
        entry = entries.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"thresholds are missing {name}")

        status = entry.get("status")
        if status not in _STATUSES:
            raise ValueError(
                f"threshold {name} has status {repr(status)}; expected one of "
                f"{list(_STATUSES)}"
            )

        value = entry.get("value")
        if status == "reached":
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"threshold {name} is reached but its value {repr(value)} is "
                    "not a non-negative number"
                )
        else:
            value = None

        thresholds[name] = Threshold(status=status, value=value)

    return thresholds


def _plain(value: float | None) -> int | float | None:
    """Display a whole-number threshold as an integer."""
    if value is None:
        return None

    if float(value).is_integer():
        return int(value)

    return float(value)


class SelectionPolicy:
    """Choose a model using the study thresholds and current data profile.

    Parameters
    ----------
    thresholds : mapping, optional
        Threshold document produced by ``experiments/make_thresholds.py``.
        When omitted, the copy packaged with yieldlite is used.
    """

    def __init__(self, thresholds: Mapping[str, Any] | None = None) -> None:
        document = load_thresholds() if thresholds is None else thresholds
        self.thresholds = _read_thresholds(document)

    def select(
        self,
        profile: DataProfile,
        gain: Callable[[str, str], PanelGain] | None = None,
    ) -> Selection:
        """Choose the best-supported model for one data profile.

        ``gain`` compares an eligible model with the next simpler eligible
        model. It is called from the top of the ladder downward and stops once
        a model shows a gain larger than one standard error.

        Parameters
        ----------
        profile : DataProfile
            Summary of the available training data.
        gain : callable, optional
            Function called as ``gain(better, baseline)``. It is required when
            any model above M0 is eligible.

        Returns
        -------
        Selection
            The selected model and the reasons behind the decision.
        """
        if not isinstance(profile, DataProfile):
            raise TypeError("profile must be a DataProfile")

        p0 = self.thresholds["P0"]
        if not p0.reached or profile.n_panels < p0.value:
            return Selection(
                rung=None,
                reasons=(
                    Reason(
                        "too_few_panels",
                        {
                            "panels": profile.n_panels,
                            "needed": _plain(p0.value),
                        },
                    ),
                ),
            )

        # Work upward through the ladder because M2 depends on M1.
        eligible = ["m0"]
        blockers: dict[str, list[Reason]] = {}

        for rung in LADDER[1:]:
            found = self._blockers(rung, profile, eligible)
            if found:
                blockers[rung] = found
            else:
                eligible.append(rung)

        # Start with the highest rung. A model keeps its place only when the
        # panel-level comparison supports it.
        reasons: list[Reason] = []
        gains: dict[str, PanelGain] = {}
        chosen = "m0"
        evidence: PanelGain | None = None
        simpler_than_chosen = None

        for rung in reversed(LADDER[1:]):
            if rung in blockers:
                reasons.extend(blockers[rung])
                continue

            if gain is None:
                raise ValueError(
                    f"{rung.upper()} is eligible, so a gain function is "
                    "needed to compare it with a simpler rung"
                )

            simpler = eligible[eligible.index(rung) - 1]
            result = gain(rung, simpler)
            gains[rung] = result

            if result.shows_gain:
                chosen = rung
                evidence = result
                simpler_than_chosen = simpler
                break

            reasons.append(self._guard_reason(rung, simpler, result))

        params: dict[str, Any] = {
            "rung": chosen,
            "panels": profile.n_panels,
        }

        if evidence is not None:
            params.update(
                simpler=simpler_than_chosen,
                gain=evidence.mean,
                se=evidence.se,
                compared=evidence.n_panels,
            )

        reasons.append(Reason("rung_selected", params))

        return Selection(
            rung=chosen,
            reasons=tuple(reasons),
            eligible=tuple(eligible),
            gains=gains,
        )

    def _blockers(
        self,
        rung: str,
        profile: DataProfile,
        eligible: list[str],
    ) -> list[Reason]:
        """Return the reasons a model cannot be used."""
        needs = _REQUIREMENTS[rung]
        statuses = {
            self.thresholds[name].status
            for name, _, _ in needs
        }

        # A model cannot use a threshold that the studies did not reach or
        # have not evaluated.
        if "not_reached" in statuses:
            return [Reason("rung_not_reached", {"rung": rung})]

        if "not_evaluated" in statuses:
            return [Reason("rung_not_evaluated", {"rung": rung})]

        have = {
            "panels": profile.n_panels,
            "households": profile.median_households_per_cell,
            "contrasts": profile.within_season_contrasts,
            "rows": profile.n_rows,
        }

        reasons = []

        for name, quantity, code in needs:
            needed = self.thresholds[name].value
            if have[quantity] < needed:
                reasons.append(
                    Reason(
                        code,
                        {
                            "rung": rung,
                            "have": _plain(have[quantity]),
                            "needed": _plain(needed),
                        },
                    )
                )

        # M2 adds weather to M1, so it cannot be considered when M1 itself is
        # not eligible.
        if not reasons and rung == "m2" and "m1" not in eligible:
            reasons.append(
                Reason(
                    "rung_blocked",
                    {"rung": rung, "needs": "m1"},
                )
            )

        return reasons

    @staticmethod
    def _guard_reason(
        rung: str,
        simpler: str,
        result: PanelGain,
    ) -> Reason:
        """Explain why validation moved past a model."""
        params = {
            "rung": rung,
            "simpler": simpler,
            "compared": result.n_panels,
        }

        if not result.assessable:
            return Reason("gain_not_assessable", params)

        return Reason(
            "no_gain",
            {
                **params,
                "gain": result.mean,
                "se": result.se,
            },
        )


class LeaveOnePanelOutGain:
    """Compare models using leave-one-panel-out validation.

    Each eligible panel is held out once. Both models are trained on the
    remaining panels, and their MAE is measured on the held-out panel. The
    gain is the simpler model's MAE minus the more complex model's MAE.

    A panel is compared only when its season appears in at least one other
    panel. Otherwise, both models would be predicting a season they had never
    seen, which would not show whether the additional inputs were useful.

    Parameters
    ----------
    frame : pandas.DataFrame
        Prepared training data containing ``panel_id``, ``season``, the target,
        and all columns required by the model ladder.
    min_variety_count : int, default 30
        Minimum number of training rows required for a variety to keep its own
        category.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        min_variety_count: int = 30,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")

        missing = [
            column
            for column in ("panel_id", "season", TARGET)
            if column not in frame
        ]
        if missing:
            raise ValueError(f"frame is missing columns: {missing}")

        self.frame = frame
        self.min_variety_count = min_variety_count

        panels = frame[["panel_id", "season"]].drop_duplicates()
        panels_in_season = panels.groupby("season")["panel_id"].transform("size")

        self.comparable_panels = sorted(
            panels.loc[panels_in_season >= 2, "panel_id"]
        )

        # Cache each rung's panel MAE so repeated comparisons do not refit the
        # same model unnecessarily.
        self._panel_mae: dict[str, pd.Series] = {}

    def __call__(
        self,
        better: str,
        baseline: str,
    ) -> PanelGain:
        """Measure the gain of one model over a simpler model."""
        n_panels = len(self.comparable_panels)

        if n_panels < 2:
            return PanelGain(
                mean=math.nan,
                se=math.nan,
                n_panels=n_panels,
            )

        gains = (
            self._mae(baseline).loc[self.comparable_panels]
            - self._mae(better).loc[self.comparable_panels]
        )

        return PanelGain(
            mean=float(gains.mean()),
            se=float(gains.std(ddof=1) / np.sqrt(n_panels)),
            n_panels=n_panels,
        )

    def _mae(self, rung: str) -> pd.Series:
        """Calculate and cache the MAE for each held-out panel."""
        if rung not in self._panel_mae:
            predictions = leave_one_panel_out(
                build_rung(
                    rung,
                    self.min_variety_count,
                ),
                self.frame,
                self.frame[TARGET],
                self.frame["panel_id"],
            )

            self._panel_mae[rung] = (
                panel_metrics(predictions)
                .set_index("panel_id")["mae"]
            )

        return self._panel_mae[rung]
