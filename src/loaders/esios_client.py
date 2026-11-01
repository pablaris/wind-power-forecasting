"""
Download hourly wind generation by province from ESIOS (REE), indicator 10037.

Changes from the first version:
  * The 8 provinces are now chosen explicitly (PROVINCES) so they always match
    open_meteo_client.py. The national ranking is still saved so the "top 8"
    choice can be justified in the report.
  * Empty or incomplete months are detected and retried instead of being
    silently accepted.

Run from the project root:
    python src/loaders/esios_client.py
"""
import os
import time
from datetime import datetime

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

URL = "https://api.esios.ree.es/indicators/10037"
START_DATE = datetime(2021, 1, 1)
N_MONTHS = 36
MAX_RETRIES = 3
MIN_COVERAGE = 0.98          # a month needs >= 98 % of its hours per province

# Must match the keys of PROVINCIAS in open_meteo_client.py
PROVINCES = ["Zaragoza", "Burgos", "Lugo", "Albacete",
             "A Coruña", "Navarra", "Cádiz", "Soria"]

# If ESIOS spells a province differently, map "ESIOS name" -> "our name" here,
# e.g. {"La Coruña": "A Coruña"}. check the printed list of ESIOS names.
NAME_MAP = {}

# Only set to True if a month is still incomplete after re-running, i.e. the
# data is missing at the source. Then document the gap in the report.
ALLOW_INCOMPLETE = True

script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(script_dir, "..", "..", "data", "raw", "esios")


def get_headers():
    load_dotenv(os.path.join(script_dir, "..", "..", ".env"), encoding="utf-8-sig")
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise ValueError("ESIOS_TOKEN not found in .env")
    return {"x-api-key": token.strip(), "Accept": "application/json"}


def download_month(start, end, headers):
    params = {
        "start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_trunc": "hour",
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(URL, headers=headers, params=params, timeout=90)
            response.raise_for_status()
            df = pd.DataFrame(response.json()["indicator"]["values"])
            if df.empty:
                raise ValueError("empty response")
            df["geo_name"] = df["geo_name"].replace(NAME_MAP)
            return df
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    return None


def month_coverage(df, start, end):
    """Fraction of the month's hours present for each of our provinces."""
    expected = (end - start).total_seconds() / 3600
    sub = df[df["geo_name"].isin(PROVINCES)]
    hours = sub.groupby("geo_name")["datetime"].nunique()
    return (hours.reindex(PROVINCES, fill_value=0) / expected).clip(upper=1)


def main():
    headers = get_headers()
    monthly_dfs, failed, incomplete = [], [], []

    for i in range(N_MONTHS):
        start = START_DATE + relativedelta(months=i)
        end = start + relativedelta(months=1)
        print(f"Downloading {start:%Y-%m} ({i + 1}/{N_MONTHS})")

        month_df = None
        for attempt in range(2):                    # one extra try if incomplete
            month_df = download_month(start, end, headers)
            if month_df is None:
                break
            cov = month_coverage(month_df, start, end)
            if (cov >= MIN_COVERAGE).all():
                break
            print(f"  incomplete: {cov[cov < MIN_COVERAGE].round(3).to_dict()}")
            time.sleep(5)

        if month_df is None:
            failed.append(f"{start:%Y-%m}")
            continue
        if (cov < MIN_COVERAGE).any():
            incomplete.append(f"{start:%Y-%m}")
        monthly_dfs.append(month_df)
        time.sleep(1)

    if failed:
        raise RuntimeError(f"Could not download: {', '.join(failed)}. Nothing saved.")
    if incomplete and not ALLOW_INCOMPLETE:
        raise RuntimeError(
            f"Incomplete months: {', '.join(incomplete)}. Re-run the script; if they "
            "stay incomplete, the gap is in ESIOS itself: set ALLOW_INCOMPLETE = True "
            "and document it. Nothing saved.")

    df = pd.concat(monthly_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime", "geo_name"])
    df = df.drop(columns=["datetime_utc", "tz_time"], errors="ignore")

    # National ranking, saved to justify the choice of provinces
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ranking = df.groupby("geo_name")["value"].sum().sort_values(ascending=False)
    ranking.to_csv(os.path.join(OUTPUT_DIR, "province_ranking.csv"))
    print("\nTop 12 by total generation (* = used in this project):")
    for name, total in ranking.head(12).items():
        mark = "*" if name in PROVINCES else " "
        print(f" {mark} {name:30s} {total / 1e6:8.2f} TWh")

    not_found = sorted(set(PROVINCES) - set(df["geo_name"]))
    if not_found:
        raise ValueError(
            f"Not found in ESIOS: {not_found}. Names available: "
            f"{sorted(df['geo_name'].unique())}. Fix NAME_MAP or PROVINCES.")

    df = df[df["geo_name"].isin(PROVINCES)]
    output_path = os.path.join(OUTPUT_DIR, "generacion_eolica.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows ({df['geo_name'].nunique()} provinces) to {output_path}")
    if incomplete:
        print(f"WARNING: kept incomplete months {incomplete} (ALLOW_INCOMPLETE=True)")


if __name__ == "__main__":
    main()
