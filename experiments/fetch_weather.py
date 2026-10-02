"""
Download daily weather data for the five surveyed districts in Cần Thơ
from 2019 through 2024 using the Open-Meteo Historical Weather API.

This script is only for exploration and is not part of the tested yieldlite
package. I used it to see whether weather variables improved yield prediction.
With only 12 independent season-panels, they did not perform better than the
simpler Season × District baseline. I am keeping the script so the experiment
can be repeated later.

Run:
    python experiments/fetch_weather.py

Creates:
    experiments/weather_daily.csv

Source:
    https://archive-api.open-meteo.com/v1/archive
"""
from __future__ import annotations

import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "weather_daily.csv")

# Approximate center coordinates for each surveyed district
DISTRICTS = {
    "OM": ("Ô Môn", 10.118, 105.628),
    "TN": ("Thốt Nốt", 10.278, 105.543),
    "VT": ("Vĩnh Thạnh", 10.225, 105.363),
    "CD": ("Cờ Đỏ", 10.103, 105.418),
    "TL": ("Thới Lai", 10.048, 105.512),
}

START_DATE, END_DATE = "2019-01-01", "2024-12-31"

VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float, ctx: ssl.SSLContext) -> dict:
    """Get the daily weather history for one district."""
    url = (
        f"{API_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily={','.join(VARIABLES)}&timezone=Asia%2FBangkok"
    )

    with urllib.request.urlopen(url, timeout=120, context=ctx) as response:
        return json.load(response)


def get_ssl_context() -> ssl.SSLContext:
    """Use normal SSL verification unless the network prevents it."""
    try:
        ctx = ssl.create_default_context()
        urllib.request.urlopen(
            API_URL
            + "?latitude=10&longitude=105"
            + "&start_date=2024-01-01&end_date=2024-01-02"
            + "&daily=temperature_2m_mean&timezone=Asia%2FBangkok",
            timeout=60,
            context=ctx,
        ).read()
        return ctx
    except (ssl.SSLError, urllib.error.URLError) as error:
        print(
            f"  [!] SSL verification failed ({error}). "
            "Trying again without certificate verification."
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def main() -> int:
    """Download and save weather data for all five districts."""
    print(
        f"Fetching weather from {START_DATE} to {END_DATE} "
        f"for {len(DISTRICTS)} districts\n"
    )

    ctx = get_ssl_context()
    rows = []

    for code, (name, lat, lon) in DISTRICTS.items():
        for attempt in range(3):
            try:
                data = fetch(lat, lon, ctx)
                break
            except Exception as error:
                if attempt == 2:
                    print(f"  [ERROR] Could not download data for {name}: {error}")
                    return 1

                print(f"  Retrying {name} after an error: {error}")
                time.sleep(3)

        daily = data["daily"]
        number_of_days = len(daily["time"])

        for index in range(number_of_days):
            rows.append(
                {
                    "district": code,
                    "date": daily["time"][index],
                    "temp_mean": daily["temperature_2m_mean"][index],
                    "temp_max": daily["temperature_2m_max"][index],
                    "temp_min": daily["temperature_2m_min"][index],
                    "rain": daily["precipitation_sum"][index],
                    "radiation": daily["shortwave_radiation_sum"][index],
                    "humidity": daily["relative_humidity_2m_mean"][index],
                }
            )

        print(
            f"  {code} - {name:<12} {number_of_days:>5} days "
            f"({daily['time'][0]} to {daily['time'][-1]})"
        )

        # Give the public API a short break between districts.
        time.sleep(1)

    weather_columns = (
        "temp_mean",
        "temp_max",
        "temp_min",
        "rain",
        "radiation",
        "humidity",
    )

    missing = {
        column: sum(1 for row in rows if row[column] is None)
        for column in weather_columns
    }

    if any(missing.values()):
        print("\nMissing values found:")
        for column, count in missing.items():
            if count:
                print(f"  {column}: {count}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows):,} daily weather records to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
