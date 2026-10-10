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

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, "..", "..", ".env"), encoding="utf-8-sig")

token = os.getenv("ESIOS_TOKEN")
if not token:
    raise ValueError("ESIOS_TOKEN not found in .env")

headers = {"x-api-key": token.strip(), "Accept": "application/json"}


def download_month(start, end):
    params = {
        "start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_trunc": "hour",
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(URL, headers=headers, params=params, timeout=90)
            response.raise_for_status()
            return pd.DataFrame(response.json()["indicator"]["values"])
        except (requests.RequestException, KeyError) as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    return None


monthly_dfs = []
failed_months = []

for i in range(N_MONTHS):
    start = START_DATE + relativedelta(months=i)
    end = start + relativedelta(months=1)
    print(f"Downloading {start:%Y-%m} ({i + 1}/{N_MONTHS})")

    month_df = download_month(start, end)
    if month_df is None:
        failed_months.append(f"{start:%Y-%m}")
    else:
        monthly_dfs.append(month_df)
    time.sleep(1)

if failed_months:
    raise RuntimeError(f"Could not download: {', '.join(failed_months)}. Nothing saved.")

df = pd.concat(monthly_dfs, ignore_index=True)
df = df.drop_duplicates(subset = ["datetime", "geo_name"])
df = df.drop(columns=["datetime_utc", "tz_time"])

# Keep the 8 provinces with the highest total generation
top_8 = df.groupby("geo_name")["value"].sum().nlargest(8).index
df = df[df["geo_name"].isin(top_8)]

output_path = os.path.join(script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")