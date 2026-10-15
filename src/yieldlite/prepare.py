"""Turn the cleaned survey table into model-ready data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from yieldlite.features import NutrientForm, npk_total

_REQUIRED_FLAG_COLUMNS = (
    "code_flag",
    "variety_flag",
    "district_mismatch",
    "commune_corrupted",
    "sowing_date",
    "harvest_date",
)

_BOOLEAN_FLAG_COLUMNS = (
    "district_mismatch",
    "commune_corrupted",
)

_NUTRIENT_COLUMNS = (
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
)


def flag_rows(df: pd.DataFrame) -> pd.Series:
    """Mark rows that contain a possible data-quality problem.

    A row is flagged when its household code is not certain, its variety is
    unclear or unresolved, its district disagrees with the household code,
    its commune name is corrupted, or its harvest date is not later than its
    sowing date.

    A variety labeled ``"other"`` is not flagged because it is treated as a
    real variety outside the named list. Rare and unseen varieties are handled
    separately by the variety pooler.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned loader output.

    Returns
    -------
    pandas.Series
        Boolean flag for each row, using the same index as ``df``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not df.columns.is_unique:
        raise ValueError("df must have unique column names")

    missing = [
        column
        for column in _REQUIRED_FLAG_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"df is missing columns: {missing}")

    dates = df[["sowing_date", "harvest_date"]]
    if dates.isna().any().any():
        raise ValueError("sowing_date and harvest_date cannot be missing")

    non_datetime = [
        column
        for column in ("sowing_date", "harvest_date")
        if not pd.api.types.is_datetime64_any_dtype(df[column].dtype)
    ]
    if non_datetime:
        raise ValueError(
            f"date columns must use a datetime dtype: {non_datetime}"
        )

    for column in _BOOLEAN_FLAG_COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values")
        if not pd.api.types.is_bool_dtype(df[column].dtype):
            raise ValueError(f"{column} must be a boolean column")

    uncertain_code = ~df["code_flag"].eq("certain").fillna(False)

    # Only "known" and "other" currently represent usable variety labels.
    # Unclear, unresolved, missing, or unexpected labels are flagged.
    uncertain_variety = ~df["variety_flag"].isin({"known", "other"})

    bad_dates = df["harvest_date"] <= df["sowing_date"]

    flagged = (
        uncertain_code
        | uncertain_variety
        | df["district_mismatch"]
        | df["commune_corrupted"]
        | bad_dates
    )

    return flagged.astype(bool)


def prepare_frame(
    df: pd.DataFrame,
    p_form: NutrientForm,
    k_form: NutrientForm,
) -> pd.DataFrame:
    """Add total NPK and a combined data-quality flag.

    Phosphorus and potassium forms must be provided explicitly because the
    package should not guess whether the survey stored elemental nutrients or
    fertilizer-grade oxides.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned loader output with a unique row index.
    p_form, k_form : {"elemental", "oxide"}
        Forms used to record phosphorus and potassium.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with ``npk_total_kg_ha`` and ``flagged`` added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not df.columns.is_unique:
        raise ValueError("df must have unique column names")

    if not df.index.is_unique:
        raise ValueError("df must use a unique row index")

    missing = [
        column
        for column in _NUTRIENT_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"df is missing nutrient columns: {missing}")

    non_numeric = [
        column
        for column in _NUTRIENT_COLUMNS
        if pd.api.types.is_bool_dtype(df[column].dtype)
        or not pd.api.types.is_numeric_dtype(df[column].dtype)
    ]
    if non_numeric:
        raise ValueError(
            f"nutrient columns must be numeric: {non_numeric}"
        )

    nutrients = df.loc[:, list(_NUTRIENT_COLUMNS)]
    nutrient_values = nutrients.to_numpy(dtype=float)
    if not np.isfinite(nutrient_values).all():
        raise ValueError(
            "nutrient columns cannot contain missing or non-finite values"
        )

    out = df.copy()
    out["npk_total_kg_ha"] = npk_total(
        out,
        p_form=p_form,
        k_form=k_form,
    )
    out["flagged"] = flag_rows(out)

    return out
