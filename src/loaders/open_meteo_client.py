import os
import requests
import pandas as pd


PROVINCIAS = {
    "Cádiz":    {"lat": 36.53, "lon": -6.29},
    "Burgos":   {"lat": 42.34, "lon": -3.70},
    "Navarra":  {"lat": 42.81, "lon": -1.64},
    "Zaragoza": {"lat": 41.65, "lon": -0.88},
    "Lugo":     {"lat": 43.01, "lon": -7.56},
    "A Coruña": {"lat": 43.37, "lon": -8.40},
    "Albacete": {"lat": 38.99, "lon": -1.86},
    "Soria":   {"lat": 41.76, "lon": -2.47},
}

VARIABLES = {
  "wind_speed_100m",
  "wind_direction_100m",
  "wind_gusts_10m", 
  "temperature_2m",
  "pressure_msl"
}

START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

df_all = []

for provincia, coords in PROVINCIAS.items():
  print(f"Descargando datos de {provincia}")

  response = requests.get(
    url = "https://archive-api.open-meteo.com/v1/archive",
    params = {
      "latitude": coords["lat"], 
      "longitude": coords["lon"], 
      "start_date": START_DATE, 
      "end_date": END_DATE, 
      "hourly": ",".join(VARIABLES), 
      "wind_speed_unit": "ms", 
      "timezone": "Europe/Madrid"
    },
  )
  data = response.json()
  hourly = data["hourly"]

  df = pd.DataFrame(hourly)
  df["provincia"] = provincia
  df_all.append(df)


df_final = pd.concat(df_all, ignore_index = True)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "..", "data", "raw", "open_meteo", "meteorology.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_csv(output_path, index = False)

