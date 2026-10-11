import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Path configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

data_path = os.path.join(project_root, "data", "processed", "dataset.csv")
figures_dir = os.path.join(project_root, "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

# 2. Load and basic date preparation
df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

# Global visual style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

# ==============================================================================
# FIGURE 1: WIND POWER CURVE (Wind at 100m vs MW Generation)
# ==============================================================================
print("Generating Figure 1: Wind power curve (v^3)...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df.sample(min(15000, len(df)), random_state=42),
    x="wind_speed_100m",
    y="generacion_mw",
    hue="provincia",
    alpha=0.3,
    s=15,
    palette="tab10",
)
plt.title(
    "Wind Power Curve: Wind Speed (100m) vs. Generation (MW)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Wind Speed at 100m (m/s)")
plt.ylabel("Wind Generation (MW)")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Province")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "01_wind_power_curve.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 2: METEOROLOGICAL VARIABLES CORRELATION
# ==============================================================================
print("Generating Figure 2: Meteorological correlation matrix...")
meteo_cols = [
    "generacion_mw",
    "wind_speed_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
]

corr_matrix = df[meteo_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Correlation between Wind Generation and Meteorological Variables",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "02_meteo_correlation.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 3: GEOGRAPHICAL DYNAMICS BY PROVINCE
# ==============================================================================
print("Generating Figure 3: Province comparison...")
plt.figure(figsize=(11, 5))
order = (
    df.groupby("provincia")["generacion_mw"]
    .mean()
    .sort_values(ascending=False)
    .index
)
sns.boxplot(
    data=df,
    x="provincia",
    y="generacion_mw",
    order=order,
    palette="viridis",
    hue="provincia",
    legend=False,
)
plt.title(
    "Hourly Wind Generation Distribution by Province (MW)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Province")
plt.ylabel("Generation (MW)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "03_geographical_dynamics.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 4: MULTILEVEL TEMPORAL CYCLES (Hour, Day, Month)
# ==============================================================================
print("Generating Figure 4: Temporal patterns (Hour, Day, Month)...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=False)

# A. Hour of the day
hourly = df.groupby(["hour", "provincia"])["generacion_mw"].mean().reset_index()
sns.lineplot(
    data=hourly, x="hour", y="generacion_mw", hue="provincia", ax=axes[0], alpha=0.8
)
axes[0].set_title("A. Daily Cycle (Hour 0 - 23 UTC)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Hour of the Day")
axes[0].set_ylabel("Average Generation (MW)")
axes[0].set_xticks(range(0, 24))
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")

# B. Day of the week
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby(["dayofweek", "provincia"])["generacion_mw"].mean().reset_index()
sns.lineplot(
    data=daily,
    x="dayofweek",
    y="generacion_mw",
    hue="provincia",
    ax=axes[1],
    marker="o",
    legend=False,
)
axes[1].set_title(
    "B. Weekly Cycle (Monday to Sunday)", fontsize=11, fontweight="bold"
)
axes[1].set_xlabel("Day of the Week")
axes[1].set_ylabel("Average Generation (MW)")
axes[1].set_xticks(range(0, 7))
axes[1].set_xticklabels(day_labels)

# C. Month of the year
monthly = df.groupby(["month", "provincia"])["generacion_mw"].mean().reset_index()
sns.lineplot(
    data=monthly,
    x="month",
    y="generacion_mw",
    hue="provincia",
    ax=axes[2],
    marker="s",
    legend=False,
)
axes[2].set_title(
    "C. Monthly Seasonality (January to December)", fontsize=11, fontweight="bold"
)
axes[2].set_xlabel("Month of the Year")
axes[2].set_ylabel("Average Generation (MW)")
axes[2].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "04_temporal_cycles.png"), dpi=300)
plt.close()