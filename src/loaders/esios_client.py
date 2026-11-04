"""
Download hourly wind generation by province from ESIOS (REE), indicator 10037,
for the period and provinces defined in src/config.py.

  * Provinces are chosen explicitly (config.PROVINCES); the national ranking is
    saved to data/raw/esios/province_ranking.csv to justify the choice.
  * Empty or incomplete months are detected and retried.

Run from the project root:
    python src/loaders/esios_client.py
"""
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import config  # noqa: E402

URL = "https://api.esios.ree.es/indicators/10037"
MAX_RETRIES = 3          # retries for network errors / empty responses
COVERAGE_ATTEMPTS = 4    # retries when a month comes back without our provinces
MIN_COVERAGE = 0.98
PROVINCES = list(config.PROVINCES)

# If ESIOS spells a province differently: {"ESIOS name": "our name"}
NAME_MAP = {}

# Set to True only if a month stays incomplete after re-running (gap at source)
ALLOW_INCOMPLETE = False

OUTPUT_DIR = os.path.join(script_dir, "..", "..", "data", "raw", "esios")
MONTH_DIR = os.path.join(OUTPUT_DIR, "monthly")   # one file per complete month


def get_headers():
    load_dotenv(os.path.join(script_dir, "..", "..", ".env"), encoding="utf-8-sig")
    token = os.getenv("ESIOS_TOKEN")
    if not token:
        raise ValueError("ESIOS_TOKEN not found in .env")
    return {"x-api-key": token.strip(), "Accept": "application/json"}


def download_month(start, end, headers):
    params = {"start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "time_trunc": "hour"}
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
    expected = (end - start).total_seconds() / 3600
    sub = df[df["geo_name"].isin(PROVINCES)]
    hours = sub.groupby("geo_name")["datetime"].nunique()
    return (hours.reindex(PROVINCES, fill_value=0) / expected).clip(upper=1)


def fetch_month(start, end, headers):
    """Download one month; retry, and as a last resort split it in two halves."""
    month_df, cov = None, None
    for attempt in range(COVERAGE_ATTEMPTS):
        month_df = download_month(start, end, headers)
        if month_df is None:
            return None, None
        cov = month_coverage(month_df, start, end)
        if (cov >= MIN_COVERAGE).all():
            return month_df, cov
        print(f"  incomplete (attempt {attempt + 1}): "
              f"{cov[cov < MIN_COVERAGE].round(3).to_dict()}")
        if (cov == 0).all():
            print(f"  names received: {sorted(month_df['geo_name'].unique())[:10]}")
        time.sleep(10 * (attempt + 1))

    print("  trying the month in two halves...")
    mid = start + (end - start) / 2
    mid = mid.floor("D")
    halves = [download_month(start, mid, headers), download_month(mid, end, headers)]
    if all(h is not None for h in halves):
        split_df = pd.concat(halves, ignore_index=True).drop_duplicates(["datetime", "geo_name"])
        split_cov = month_coverage(split_df, start, end)
        if split_cov.mean() > cov.mean():
            month_df, cov = split_df, split_cov
            print(f"  halves coverage: {cov.round(3).to_dict()}")
    return month_df, cov


def main():
    headers = get_headers()
    os.makedirs(MONTH_DIR, exist_ok=True)
    monthly_dfs, failed, incomplete = [], [], []

    for i, month in enumerate(config.MONTHS):
        start = month.start_time
        end = (month + 1).start_time
        cache = os.path.join(MONTH_DIR, f"{month}.csv")
        if os.path.exists(cache):
            print(f"Using saved {month} ({i + 1}/{len(config.MONTHS)})")
            monthly_dfs.append(pd.read_csv(cache))
            continue

        print(f"Downloading {month} ({i + 1}/{len(config.MONTHS)})")
        month_df, cov = fetch_month(start, end, headers)
        if month_df is None:
            failed.append(str(month))
            continue
        if (cov < MIN_COVERAGE).any():
            incomplete.append(str(month))
        else:
            month_df.to_csv(cache, index=False)       # only complete months are saved
        monthly_dfs.append(month_df)
        time.sleep(1)

    if failed:
        raise RuntimeError(f"Could not download: {', '.join(failed)}. Complete months "
                           "are saved: re-run to retry only the missing ones.")
    if incomplete and not ALLOW_INCOMPLETE:
        raise RuntimeError(
            f"Incomplete months: {', '.join(incomplete)}. Complete months are saved, so "
            "re-running only retries these. If they stay incomplete after a few runs "
            "(ideally at a different time of day), the gap is in ESIOS itself: set "
            "ALLOW_INCOMPLETE = True and document it.")

    df = pd.concat(monthly_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime", "geo_name"])
    df = df.drop(columns=["datetime_utc", "tz_time"], errors="ignore")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ranking = df.groupby("geo_name")["value"].sum().sort_values(ascending=False)
    ranking.to_csv(os.path.join(OUTPUT_DIR, "province_ranking.csv"))
    print(f"\nTop 12 by total generation {config.START} to {config.END} (* = used):")
    for name, total in ranking.head(12).items():
        mark = "*" if name in PROVINCES else " "
        print(f" {mark} {name:30s} {total / 1e6:8.2f} TWh")

    not_found = sorted(set(PROVINCES) - set(df["geo_name"]))
    if not_found:
        raise ValueError(f"Not found in ESIOS: {not_found}. Names available: "
                         f"{sorted(df['geo_name'].unique())}. Fix NAME_MAP.")

    df = df[df["geo_name"].isin(PROVINCES)]
    path = os.path.join(OUTPUT_DIR, "generacion_eolica.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} rows ({df['geo_name'].nunique()} provinces) to {path}")
    if incomplete:
        print(f"WARNING: kept incomplete months {incomplete} (ALLOW_INCOMPLETE=True)")


if __name__ == "__main__":
    main()
