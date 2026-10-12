"""Create the non-weather features used by yieldlite.

This module calculates total NPK, pools rice varieties with limited support,
and learns season windows from sowing dates. Weather features are handled
separately in ``yieldlite.weather``.
"""

from typing import Literal

import numpy as np
import pandas as pd


NutrientForm = Literal["elemental", "oxide"]
_VALID_FORMS = {"elemental", "oxide"}

# Use these factors when phosphorus or potassium is recorded in elemental
# form. They convert P to P2O5 and K to K2O before the nutrients are added.
P_TO_P2O5 = 141.943 / 61.948
K_TO_K2O = 94.195 / 78.196

_DAYS_PER_YEAR = 365.25


def npk_total(
    df: pd.DataFrame,
    p_form: NutrientForm,
    k_form: NutrientForm,
) -> pd.Series:
    """Calculate total NPK in kilograms per hectare.

    The result uses the common fertilizer basis of N + P2O5 + K2O. The caller
    must state how phosphorus and potassium were recorded because assuming
    the wrong form could change the result substantially.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``nitrogen_kg_ha``, ``phosphorus_kg_ha``, and
        ``potassium_kg_ha``.
    p_form, k_form : {"elemental", "oxide"}
        Use ``"elemental"`` when the column stores P or K directly. Use
        ``"oxide"`` when it already stores P2O5 or K2O.

    Returns
    -------
    pandas.Series
        Total NPK for each row in kilograms per hectare.

    Raises
    ------
    ValueError
        If either nutrient form is not ``"elemental"`` or ``"oxide"``.
    """
    if p_form not in _VALID_FORMS:
        raise ValueError(
            "p_form must be 'elemental' or 'oxide', "
            f"not {p_form!r}"
        )

    if k_form not in _VALID_FORMS:
        raise ValueError(
            "k_form must be 'elemental' or 'oxide', "
            f"not {k_form!r}"
        )

    phosphorus = df["phosphorus_kg_ha"]
    if p_form == "elemental":
        phosphorus = phosphorus * P_TO_P2O5

    potassium = df["potassium_kg_ha"]
    if k_form == "elemental":
        potassium = potassium * K_TO_K2O

    total = df["nitrogen_kg_ha"] + phosphorus + potassium
    total.name = "npk_total_kg_ha"
    return total


class VarietyPooler:
    """Group varieties that do not have enough reliable training data.

    A known variety keeps its name only when it has at least ``min_count``
    training rows. KHAC can also remain separate when it has enough support.

    KHONGRO, unresolved values, rare varieties, and varieties that were not
    seen during training are placed in one ``"pooled"`` group.
    """

    POOLED = "pooled"
    _HAS_SUPPORT_FLAGS = {"known", "other"}

    def __init__(self, min_count: int = 30) -> None:
        if min_count < 1:
            raise ValueError("min_count must be at least 1")

        self.min_count = min_count
        self.kept_varieties_: set[str] | None = None

    @staticmethod
    def _check_alignment(
        variety_raw: pd.Series,
        variety_flag: pd.Series,
    ) -> None:
        """Make sure the variety values and flags describe the same rows."""
        if not variety_raw.index.equals(variety_flag.index):
            raise ValueError(
                "variety_raw and variety_flag must have the same index"
            )

    def fit(
        self,
        variety_raw: pd.Series,
        variety_flag: pd.Series,
    ) -> "VarietyPooler":
        """Learn which varieties have enough support to remain separate."""
        self._check_alignment(variety_raw, variety_flag)

        eligible = variety_flag.isin(self._HAS_SUPPORT_FLAGS)
        counts = variety_raw.loc[eligible].value_counts()

        self.kept_varieties_ = set(
            counts[counts >= self.min_count].index
        )
        return self

    def transform(
        self,
        variety_raw: pd.Series,
        variety_flag: pd.Series,
    ) -> pd.Series:
        """Pool rare, unclear, unresolved, and previously unseen varieties."""
        if self.kept_varieties_ is None:
            raise RuntimeError("call fit() before transform()")

        self._check_alignment(variety_raw, variety_flag)

        has_support = (
            variety_flag.isin(self._HAS_SUPPORT_FLAGS)
            & variety_raw.isin(self.kept_varieties_)
        )

        return variety_raw.where(has_support, self.POOLED)


def _circular_doy_range(
    day_of_year: pd.Series,
    low_pct: float,
    high_pct: float,
) -> tuple[int, int]:
    """Find a day-of-year range that is allowed to cross New Year.

    Calendar dates form a circle. For example, December 20 and January 10
    are close together even though their day-of-year numbers are far apart.
    This function first finds the center of that circle and then calculates
    the requested percentile range around it.
    """
    clean_days = pd.to_numeric(day_of_year, errors="coerce").dropna()

    if clean_days.empty:
        raise ValueError("cannot learn a season range without valid dates")

    positions = clean_days.to_numpy(dtype=float) - 1.0
    angles = positions / _DAYS_PER_YEAR * 2 * np.pi

    center_angle = np.arctan2(
        np.sin(angles).mean(),
        np.cos(angles).mean(),
    )
    center_position = (
        center_angle / (2 * np.pi) * _DAYS_PER_YEAR
    ) % _DAYS_PER_YEAR

    half_year = _DAYS_PER_YEAR / 2
    offsets = (
        positions - center_position + half_year
    ) % _DAYS_PER_YEAR - half_year

    low_offset, high_offset = np.percentile(
        offsets,
        [low_pct, high_pct],
    )

    low_position = (
        center_position + low_offset
    ) % _DAYS_PER_YEAR
    high_position = (
        center_position + high_offset
    ) % _DAYS_PER_YEAR

    low_doy = int(round(low_position + 1))
    high_doy = int(round(high_position + 1))

    return low_doy, high_doy


def _doy_in_range(doy: int, low: int, high: int) -> bool:
    """Return whether a date falls inside a possibly wrapped range."""
    if low <= high:
        return low <= doy <= high

    # A reversed range means it crosses New Year.
    return doy >= low or doy <= high


class SeasonCalendar:
    """Learn the usual sowing-date window for each season.

    Each season receives one date range based on the training data. The
    calculation uses a circular calendar so Đông Xuân can cross New Year.

    If a date falls outside every learned range, or matches more than one
    season, ``season_for`` returns ``None`` instead of guessing.
    """

    def __init__(
        self,
        low_pct: float = 5.0,
        high_pct: float = 95.0,
    ) -> None:
        if not 0 <= low_pct < high_pct <= 100:
            raise ValueError(
                "percentiles must satisfy "
                "0 <= low_pct < high_pct <= 100"
            )

        self.low_pct = low_pct
        self.high_pct = high_pct
        self.ranges_: dict[str, tuple[int, int]] | None = None

    def fit(
        self,
        season: pd.Series,
        sowing_date: pd.Series,
        valid: pd.Series | None = None,
    ) -> "SeasonCalendar":
        """Learn a sowing-date range for every season in the training data.

        The optional ``valid`` mask should mark dates that are safe to use.
        This allows known placeholder or corrupted dates, such as those found
        in the Thốt Nốt records, to be removed before learning the windows.
        """
        if not season.index.equals(sowing_date.index):
            raise ValueError(
                "season and sowing_date must have the same index"
            )

        data = pd.DataFrame(
            {
                "season": season,
                "sowing_date": pd.to_datetime(
                    sowing_date,
                    errors="coerce",
                ),
            }
        )

        if valid is not None:
            if not season.index.equals(valid.index):
                raise ValueError(
                    "valid must have the same index as season"
                )

            valid_mask = valid.fillna(False).astype(bool)
            data = data.loc[valid_mask]

        # Missing labels and missing dates cannot help define a season.
        data = data.dropna(subset=["season", "sowing_date"])

        if data.empty:
            raise ValueError(
                "no valid season and sowing-date rows were available"
            )

        self.ranges_ = {
            str(label): _circular_doy_range(
                group["sowing_date"].dt.dayofyear,
                self.low_pct,
                self.high_pct,
            )
            for label, group in data.groupby("season", sort=False)
        }

        return self

    def season_for(self, sowing_date) -> str | None:
        """Return the single season matching a sowing date.

        ``None`` means the date is missing, falls outside every learned
        window, or matches more than one season.
        """
        if self.ranges_ is None:
            raise RuntimeError("call fit() before season_for()")

        date = pd.to_datetime(sowing_date, errors="coerce")
        if pd.isna(date):
            return None

        doy = date.dayofyear
        matches = [
            label
            for label, (low, high) in self.ranges_.items()
            if _doy_in_range(doy, low, high)
        ]

        return matches[0] if len(matches) == 1 else None
