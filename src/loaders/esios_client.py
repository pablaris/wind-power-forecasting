from datetime import datetime
import os
import time
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()
esios_token = os.getenv("ESIOS_TOKEN")

df = pd.DataFrame()
start_date = datetime(2021, 1, 1)

for month in range(0, 36):
    print(f"Downloading month {month + 1}/36...")
    month_start = start_date + relativedelta(months=month)
    month_end = start_date + relativedelta(months=month + 1)

    start_str = month_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = month_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "x-api-key": esios_token,
        "Accept": "application/json; application/xls",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    params = {
        "start_date": start_str,
        "end_date": end_str,
        "time_trunc": "hour",
    }

    max_retries = 3
    success = False

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url="https://api.esios.ree.es/indicators/10037",
                headers=headers,
                params=params,
                timeout=15,
            )

            if response.status_code == 200:
                monthly_data = response.json()["indicator"]["values"]
                df = pd.concat([df, pd.DataFrame(monthly_data)])
                success = True
                break
            else:
                print(
                    f"  ⚠️ Attempt {attempt}/{max_retries} failed with status {response.status_code}. Retrying in 2s..."
                )
        except (
            requests.exceptions.RequestException,
            requests.exceptions.JSONDecodeError,
            KeyError,
        ) as e:
            print(f"  ⚠️ Attempt {attempt}/{max_retries} failed with error: {e}")

        time.sleep(2)

    if not success:
        print(
            f"❌ Failed to download month {month + 1} after {max_retries} attempts."
        )

    time.sleep(1)

df = df.drop(columns=["datetime_utc", "tz_time"])
total_by_province = df.groupby("geo_name")["value"].sum()
top_8_provinces = total_by_province.sort_values(ascending=False).head(8)
df = df[df["geo_name"].isin(top_8_provinces.index)]

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv"
)
df.to_csv(output_path, index=False)
print("✅ Download completed and saved successfully.")