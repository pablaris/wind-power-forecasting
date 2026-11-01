"""
Merge ESIOS generation and Open-Meteo weather into one row per province and hour.
Stops with a clear error if the two sources do not contain the same provinces,
instead of silently dropping one in the inner merge.

Run from the project root:
    python src/processing/merge_datasets.py
"""
import os

import pandas as pd

MIN_COVERAGE = 0.98
EXPECTED_HOURS = 3 * 365 * 24          # 2021-2023 (no leap year)

script_dir = os.path.dirname(os.path.abspath(__file__))
esios_path = os.path.join(script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv")
meteo_path = os.path.join(script_dir, "..", "..", "data", "raw", "open_meteo", "meteorology.csv")

df_esios = pd.read_csv(esios_path)
df_meteo = pd.read_csv(meteo_path)

df_esios["datetime"] = pd.to_datetime(df_esios["datetime"], utc=True)
df_meteo["datetime"] = pd.to_datetime(df_meteo["time"], utc=True)
df_esios = df_esios.rename(columns={"geo_name": "provincia", "value": "generacion_mw"})
df_esios = df_esios.drop(columns=["geo_id"], errors="ignore")
df_meteo = df_meteo.drop(columns=["time"])

# --- Check that both sources have the same provinces
esios_provs, meteo_provs = set(df_esios["provincia"]), set(df_meteo["provincia"])
if esios_provs != meteo_provs:
    raise ValueError(
        "Province names differ between sources:\n"
        f"  only in ESIOS:      {sorted(esios_provs - meteo_provs)}\n"
        f"  only in Open-Meteo: {sorted(meteo_provs - esios_provs)}\n"
        "Make PROVINCES in esios_client.py match PROVINCIAS in open_meteo_client.py.")

df = pd.merge(df_esios, df_meteo, on=["datetime", "provincia"], how="inner")

# --- Coverage per province after merging
hours = df.groupby("provincia")["datetime"].nunique()
print("Hours per province after merge:")
for prov, n in hours.items():
    flag = "" if n >= MIN_COVERAGE * EXPECTED_HOURS else "   <-- LOW COVERAGE"
    print(f"  {prov:12s} {n:6d} / {EXPECTED_HOURS}{flag}")

output_path = os.path.join(script_dir, "..", "..", "data", "processed", "dataset.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Dataset merged: {df.shape}")
print(f"Provinces ({df['provincia'].nunique()}): {sorted(df['provincia'].unique())}")
print(f"Range: from {df['datetime'].min()} to {df['datetime'].max()}")
