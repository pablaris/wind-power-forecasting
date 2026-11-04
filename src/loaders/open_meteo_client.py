"""
Download hourly weather for each province from Open-Meteo.

Two sources, chosen with --weather:
  forecast    (default) Previous Runs API: the forecast issued about one day
              before each hour ("_previous_day1"). This is what a forecaster
              would really have had, so it introduces no data leakage.
              Archived from January 2024 only.
  reanalysis  Historical Weather API (ERA5-based): the weather that actually
              happened. Not available at forecast time, so it is used only as an
              upper bound ("perfect weather forecast") for comparison.

Run from the project root:
    python src/loaders/open_meteo_client.py --weather forecast
    python src/loaders/open_meteo_client.py --weather reanalysis
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import config  # noqa: E402

URLS = {
    "forecast": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "reanalysis": "https://archive-api.open-meteo.com/v1/archive",
}
LEAD = "previous_day1"          # forecast made ~1 day before the hour
MODEL = None                    # None = Open-Meteo "best match"; or e.g. "ecmwf_ifs025"
MAX_NAN_SHARE = 0.5             # stop only if a variable is mostly missing
OUTPUT_DIR = os.path.join(script_dir, "..", "..", "data", "raw", "open_meteo")


def chunks(months_per_chunk=3):
    """Split the study period into blocks of a few months (smaller requests)."""
    months = list(config.MONTHS)
    for i in range(0, len(months), months_per_chunk):
        block = months[i:i + months_per_chunk]
        yield block[0].start_time.date(), block[-1].end_time.date()


def request(source, lat, lon, start, end):
    names = [f"{v}_{LEAD}" if source == "forecast" else v for v in config.WEATHER_VARS]
    params = {"latitude": lat, "longitude": lon,
              "start_date": str(start), "end_date": str(end),
              "hourly": ",".join(names), "wind_speed_unit": "ms", "timezone": "UTC"}
    if MODEL and source == "forecast":
        params["models"] = MODEL
    for attempt in range(3):
        try:
            r = requests.get(URLS[source], params=params, timeout=120)
            if r.status_code == 400:   # bad request: show Open-Meteo's explanation
                raise ValueError(r.json().get("reason", r.text))
            r.raise_for_status()
            df = pd.DataFrame(r.json()["hourly"])
            # remove the "_previous_day1" suffix so downstream code is identical
            return df.rename(columns={f"{v}_{LEAD}": v for v in config.WEATHER_VARS})
        except ValueError:
            raise
        except (requests.RequestException, KeyError) as e:
            print(f"    attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed: {start} to {end} at ({lat}, {lon})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", choices=URLS, default="forecast")
    args = parser.parse_args()

    frames = []
    for prov, (lat, lon) in config.PROVINCES.items():
        print(f"{prov}: downloading {args.weather} weather")
        for start, end in chunks():
            df = request(args.weather, lat, lon, start, end)
            df["provincia"] = prov
            frames.append(df)
            time.sleep(1)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "provincia"])

    print("\nShare of missing values per variable:")
    nan_share = df[config.WEATHER_VARS].isna().mean()
    for var, share in nan_share.items():
        print(f"  {var:22s} {share:6.1%}")

    # Where are the gaps? (gaps of at least one day, all provinces together)
    print("\nGaps of 24 h or more:")
    times = pd.to_datetime(df["time"], utc=True)
    any_gap = False
    for var in config.WEATHER_VARS:
        missing = pd.Series(sorted(times[df[var].isna()].unique()))
        if missing.empty:
            continue
        block = (missing.diff() != pd.Timedelta(hours=1)).cumsum()
        for _, g in missing.groupby(block):
            if len(g) >= 24:
                any_gap = True
                print(f"  {var:22s} {g.iloc[0]:%Y-%m-%d %H:%M} -> "
                      f"{g.iloc[-1]:%Y-%m-%d %H:%M} ({len(g)} h)")
    if not any_gap:
        print("  none")

    if nan_share["wind_speed_100m"] > MAX_NAN_SHARE:
        raise RuntimeError("wind_speed_100m is mostly missing: try setting MODEL = "
                           "'ecmwf_ifs025' at the top of this file. Nothing saved.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"meteorology_{args.weather}.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
