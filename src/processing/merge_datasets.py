"""
Merge ESIOS generation with one weather source into one row per province and hour.

Run from the project root:
    python src/processing/merge_datasets.py --weather forecast
    python src/processing/merge_datasets.py --weather reanalysis
Output: data/processed/dataset_<weather>.csv
"""
import argparse
import os
import sys

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import config  # noqa: E402

MIN_COVERAGE = 0.98
ROOT = os.path.join(script_dir, "..", "..")

parser = argparse.ArgumentParser()
parser.add_argument("--weather", choices=["forecast", "reanalysis"], default="forecast")
args = parser.parse_args()

df_esios = pd.read_csv(os.path.join(ROOT, "data", "raw", "esios", "generacion_eolica.csv"))
df_meteo = pd.read_csv(os.path.join(ROOT, "data", "raw", "open_meteo",
                                    f"meteorology_{args.weather}.csv"))

df_esios["datetime"] = pd.to_datetime(df_esios["datetime"], utc=True)
df_meteo["datetime"] = pd.to_datetime(df_meteo["time"], utc=True)
df_esios = df_esios.rename(columns={"geo_name": "provincia", "value": "generacion_mw"})
df_esios = df_esios.drop(columns=["geo_id"], errors="ignore")
df_meteo = df_meteo.drop(columns=["time"])

esios_provs, meteo_provs = set(df_esios["provincia"]), set(df_meteo["provincia"])
if esios_provs != meteo_provs:
    raise ValueError(
        "Province names differ between sources:\n"
        f"  only in ESIOS:      {sorted(esios_provs - meteo_provs)}\n"
        f"  only in Open-Meteo: {sorted(meteo_provs - esios_provs)}")

df = pd.merge(df_esios, df_meteo, on=["datetime", "provincia"], how="inner")
df = df[df["datetime"].isin(config.FULL_RANGE)]      # keep the study period only

expected = len(config.FULL_RANGE)
print(f"Hours per province after merge ({args.weather} weather):")
for prov, n in df.groupby("provincia")["datetime"].nunique().items():
    flag = "" if n >= MIN_COVERAGE * expected else "   <-- LOW COVERAGE"
    print(f"  {prov:12s} {n:6d} / {expected}{flag}")

path = os.path.join(ROOT, "data", "processed", f"dataset_{args.weather}.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_csv(path, index=False)
print(f"Dataset merged: {df.shape} -> {os.path.normpath(path)}")
print(f"Range: from {df['datetime'].min()} to {df['datetime'].max()}")
