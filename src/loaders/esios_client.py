from dotenv import load_dotenv
import os
import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime

load_dotenv()
esios_token = os.getenv("ESIOS_TOKEN")

df = pd.DataFrame()
inicio = datetime(2023, 1, 1)

for month in range(0, 12):
  print(f"Descargando mes {month +1}/12")
  inicio_del_mes = inicio + relativedelta(months = month)
  final_del_mes = inicio + relativedelta(months = month +1)

  start_str = inicio_del_mes.strftime("%Y-%m-%dT%H:%M:%SZ")
  end_str = final_del_mes.strftime("%Y-%m-%dT%H:%M:%SZ")
  response = requests.get(
    url = "https://api.esios.ree.es/indicators/10037",
    headers = {
      "x-api-key": esios_token,
      "Accept": "application/json; application/xls",
      "Content-Type": "application/json"},
    params = {
      "start_date": start_str,
      "end_date": end_str,
      "time_trunc": "hour"
  }
  )
  monthly_data = response.json()["indicator"]["values"]
  df = pd.concat([df, pd.DataFrame(monthly_data)])
  

df = df.drop(columns = ["datetime_utc", "tz_time"])
total_por_provincia = df.groupby("geo_name")["value"].sum()
top_8 = total_por_provincia.sort_values(ascending=False).head(8)
df = df[df["geo_name"].isin(top_8.index)]

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv")
df.to_csv(output_path , index = False)
