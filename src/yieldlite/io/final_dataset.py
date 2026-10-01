"""Load the merged 2020–2024 ĐTGT dataset and add quality flags.

Some commune names contain literal "?" characters from an earlier export, so
their original Vietnamese spelling cannot be recovered from this CSV.
"""

from pathlib import Path

import pandas as pd

from yieldlite.io.codes import parse_household_code


# Original column name mapped to the name used by yieldlite.
COLUMNS = {
    "ma": "household_code",
    "vu": "season",
    "nam": "year",
    "huyen": "district_code",
    "xa": "commune_raw",
    "dt": "area_ha",
    "giong": "variety_raw",
    "giong_kg_ha": "seed_rate_kg_ha",
    "N": "nitrogen_kg_ha",
    "P": "phosphorus_kg_ha",
    "K": "potassium_kg_ha",
    "cp_phan": "fertilizer_cost_vnd",
    "ngay_gieo": "sowing_date",
    "ngay_thu": "harvest_date",
    "sinh_truong": "growth_duration_days",
    "nhankhau": "household_size",
    "laodong": "labor_count",
    "nang_suat": "yield_t_ha",
    "nguon": "source_file",
    "vu_nam": "panel_id",
    "bucxa_tichluy": "radiation_cumulative",
    "bucxa_trobong": "radiation_at_heading",
    "nhietdo_tb": "temp_mean",
    "bien_do_nhiet": "temp_range",
    "so_ngay_nong": "hot_days_count",
    "mua_tichluy": "rain_cumulative",
    "so_ngay_mua_to": "heavy_rain_days_count",
}


# Rice varieties confirmed in the merged dataset.
KNOWN_VARIETIES = {
    "OM5451",
    "OM380",
    "OM18",
    "IR50404",
    "DAITHOM8",
    "JASMINE85",
}


def load_final_dataset(path: str | Path) -> pd.DataFrame:
    """Load the merged CSV, clean key fields, and add quality flags."""
    df = pd.read_csv(path, encoding="latin1").rename(columns=COLUMNS)

    df["household_size"] = pd.to_numeric(
        df["household_size"].str.strip(),
        errors="coerce",
    )

    df["sowing_date"] = pd.to_datetime(
        df["sowing_date"],
        format="%m/%d/%Y",
    )

    df["harvest_date"] = pd.to_datetime(
        df["harvest_date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    parsed = df["household_code"].apply(parse_household_code)

    df["code_flag"] = [code.flag for code in parsed]
    df["district_mismatch"] = [
        code.district != district
        for code, district in zip(parsed, df["district_code"])
    ]
    df["commune_corrupted"] = df["commune_raw"].str.contains(
        "?",
        regex=False,
    )

    def variety_flag(raw: str) -> str:
        if raw in KNOWN_VARIETIES:
            return "known"
        if raw == "KHAC":
            return "other"
        if raw == "KHONGRO":
            return "unclear"
        return "unresolved"

    df["variety_flag"] = df["variety_raw"].apply(variety_flag)

    return df
