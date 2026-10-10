import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
esios_path = os.path.join(script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv")
meteo_path = os.path.join(script_dir, "..", "..", "data", "raw", "open_meteo", "meteorology.csv")

df_esios = pd.read_csv(esios_path)
df_meteo = pd.read_csv(meteo_path)

df_esios["datetime"] = pd.to_datetime(df_esios["datetime"], utc = True)
df_meteo["datetime"] = pd.to_datetime(df_meteo["time"], utc = True)
df_esios = df_esios.rename(columns = {"geo_name": "provincia", "value": "generacion_mw"})
df_esios = df_esios.drop(columns = ["geo_id"])
df_meteo = df_meteo.drop(columns = ["time"])

df = pd.merge(df_esios, df_meteo, on =["datetime", "provincia"], how = "inner")

output_path = os.path.join(script_dir, "..", "..", "data", "processed", "dataset.csv")
os.makedirs(os.path.dirname(output_path), exist_ok= True)
df.to_csv(output_path, index = False)

print(f"Dataset merged: {df.shape}")
print(f"Provinces: {df["provincia"].unique()}")
print(f"Range: from {df["datetime"].min()} to {df['datetime'].max()}")