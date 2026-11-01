"""
Read-only diagnosis of province coverage and missing hours.

Run from the project root:
    python src/processing/check_dataset.py --weather forecast
"""
import argparse
import os
import sys

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import config  # noqa: E402

ROOT = os.path.join(script_dir, "..", "..")


def load(path, time_col, prov_col):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[time_col], utc=True)
    return df.rename(columns={prov_col: "provincia"})


def missing_ranges(times):
    missing = config.FULL_RANGE.difference(pd.DatetimeIndex(times.unique()))
    if len(missing) == 0:
        return []
    s = pd.Series(missing)
    block_id = (s.diff() != pd.Timedelta(hours=1)).cumsum()
    return [(g.iloc[0], g.iloc[-1], len(g)) for _, g in s.groupby(block_id)]


def coverage_report(name, df):
    print(f"\n=== {name}: hours per province (expected {len(config.FULL_RANGE)}) ===")
    for prov, g in df.groupby("provincia"):
        blocks = missing_ranges(g["datetime"])
        n_missing = sum(b[2] for b in blocks)
        print(f"{prov:12s} hours={g['datetime'].isin(config.FULL_RANGE).sum():6d} "
              f"missing={n_missing:5d}")
        big = [b for b in blocks if b[2] >= 24]
        for start, end, n in big[:5]:
            print(f"{'':12s}   gap {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} ({n} h)")
        if len(blocks) - len(big):
            print(f"{'':12s}   + {len(blocks) - len(big)} short gap(s) < 24 h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", choices=["forecast", "reanalysis"], default="forecast")
    w = parser.parse_args().weather

    esios = load(os.path.join(ROOT, "data", "raw", "esios", "generacion_eolica.csv"),
                 "datetime", "geo_name")
    meteo = load(os.path.join(ROOT, "data", "raw", "open_meteo", f"meteorology_{w}.csv"),
                 "time", "provincia")
    merged = load(os.path.join(ROOT, "data", "processed", f"dataset_{w}.csv"),
                  "datetime", "provincia")

    e, m = set(esios["provincia"]), set(meteo["provincia"])
    print(f"Study period: {config.START} to {config.END} | weather: {w}")
    print("=== Provinces ===")
    print("ESIOS      :", sorted(e))
    print("Open-Meteo :", sorted(m))
    print("Merged     :", sorted(set(merged["provincia"])))
    print("In ESIOS but not in Open-Meteo:", sorted(e - m) or "none")
    print("In Open-Meteo but not in ESIOS:", sorted(m - e) or "none")
    print("\nMissing weather values (share):")
    print(meteo[config.WEATHER_VARS].isna().mean().map("{:.1%}".format).to_string())

    coverage_report("ESIOS (raw)", esios)
    coverage_report(f"Open-Meteo {w} (raw)", meteo)
    coverage_report("Merged dataset", merged)


if __name__ == "__main__":
    main()
