"""
Diagnose province coverage and missing hours in the raw and merged datasets.
Read-only: it never modifies any file.

Run from the project root:
    python src/processing/check_dataset.py
"""
import os

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(script_dir, "..", ".."))
ESIOS = os.path.join(root, "data", "raw", "esios", "generacion_eolica.csv")
METEO = os.path.join(root, "data", "raw", "open_meteo", "meteorology.csv")
MERGED = os.path.join(root, "data", "processed", "dataset.csv")

FULL_RANGE = pd.date_range("2021-01-01 00:00", "2023-12-31 23:00", freq="h", tz="UTC")


def load(path, time_col, prov_col):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[time_col], utc=True)
    return df.rename(columns={prov_col: "provincia"})


def missing_ranges(times):
    """Group the missing hours of one province into consecutive blocks."""
    missing = FULL_RANGE.difference(pd.DatetimeIndex(times.unique()))
    if len(missing) == 0:
        return []
    s = pd.Series(missing)
    block_id = (s.diff() != pd.Timedelta(hours=1)).cumsum()
    return [(g.iloc[0], g.iloc[-1], len(g)) for _, g in s.groupby(block_id)]


def coverage_report(name, df):
    print(f"\n=== {name}: hours per province (expected {len(FULL_RANGE)}) ===")
    for prov, g in df.groupby("provincia"):
        blocks = missing_ranges(g["datetime"])
        n_missing = sum(b[2] for b in blocks)
        print(f"{prov:12s} hours={g['datetime'].nunique():6d} missing={n_missing:5d}")
        big = [b for b in blocks if b[2] >= 24]           # gaps of a day or more
        for start, end, n in big[:5]:
            print(f"{'':12s}   gap {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} ({n} h)")
        small = len(blocks) - len(big)
        if small:
            print(f"{'':12s}   + {small} short gap(s) < 24 h")


def main():
    esios = load(ESIOS, "datetime", "geo_name")
    meteo = load(METEO, "time", "provincia")
    merged = load(MERGED, "datetime", "provincia")

    # 1. Province names in each source
    e, m = set(esios["provincia"]), set(meteo["provincia"])
    print("=== Provinces ===")
    print("ESIOS      :", sorted(e))
    print("Open-Meteo :", sorted(m))
    print("Merged     :", sorted(set(merged["provincia"])))
    print("In ESIOS but not in Open-Meteo (dropped by the merge):", sorted(e - m) or "none")
    print("In Open-Meteo but not in ESIOS (dropped by the merge):", sorted(m - e) or "none")

    # 2. Missing hours, source by source, to see WHERE the gap comes from
    coverage_report("ESIOS (raw)", esios)
    coverage_report("Open-Meteo (raw)", meteo)
    coverage_report("Merged dataset", merged)


if __name__ == "__main__":
    main()
