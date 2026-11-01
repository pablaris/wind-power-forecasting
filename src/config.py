"""
Single source of truth for the study period, the provinces and the weather
variables. Every script imports this file, so changing a value here changes it
everywhere.

Why 2024-2026: Open-Meteo's Previous Runs API, which gives the weather forecast
that was actually available one day earlier, is only archived from January 2024.
Using it removes the data leakage of using observed (reanalysis) weather.
"""
import pandas as pd

# Archived day-ahead forecasts of 100 m wind start on 2024-02-16 06:00 UTC, so the
# study period starts the next day. Both weather sources use exactly these hours.
START = "2024-02-17"
END = "2026-06-30"            # inclusive
TEST_START = "2025-07-01"     # train: Jan 2024 - Jun 2025 | test: Jul 2025 - Jun 2026

# The 9 largest wind-producing provinces in 2024-2026 (see province_ranking.csv).
# Weather is sampled at one point per province (its capital)
PROVINCES = {
    "Cádiz":    (36.53, -6.29),
    "Burgos":   (42.34, -3.70),
    "Navarra":  (42.81, -1.64),
    "Zaragoza": (41.65, -0.88),
    "Lugo":     (43.01, -7.56),
    "A Coruña": (43.37, -8.40),
    "Albacete": (38.99, -1.86),
    "Soria":    (41.76, -2.47),
    "Cuenca":   (40.07, -2.13),
}

WEATHER_VARS = ["wind_speed_100m", "wind_direction_100m", "wind_gusts_10m",
                "temperature_2m", "pressure_msl"]

FULL_RANGE = pd.date_range(START, f"{END} 23:00", freq="h", tz="UTC")
MONTHS = pd.period_range(START, END, freq="M")
