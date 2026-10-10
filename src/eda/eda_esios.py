import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Configuración de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(
    script_dir, "..", "..", "data", "raw", "esios", "generacion_eolica.csv"
)
figures_dir = os.path.join(script_dir, "..", "..", "reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

# 2. Carga y preparación de datos
df = pd.read_csv(data_path)
if "value" not in df.columns:
    df.columns = ["value", "datetime", "geo_id", "geo_name"]

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values(["geo_name", "datetime"]).reset_index(drop=True)

# Crear variables temporales y de desfase temporal (1h, 24h y 168h / 1 semana)
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

df["val_lag1"] = df.groupby("geo_name")["value"].shift(1)
df["val_lag24"] = df.groupby("geo_name")["value"].shift(24)
df["val_lag168"] = df.groupby("geo_name")["value"].shift(168)  # 7 días * 24h

df["ramp_24h"] = df.groupby("geo_name")["value"].diff(24)

# ==========================================
# 3. CÁLCULO DE RATIOS Y DEGRADACIÓN TEMPORAL
# ==========================================
print("===================================================================")
print("   RATIOS ENERGÉTICOS Y ESTRUCTURA TEMPORAL (1h vs 24h vs 168h)")
print("===================================================================\n")

summary_list = []
for geo, group in df.groupby("geo_name"):
    # Autocorrelaciones a 1h, 24h y 168h (1 semana)
    valid_l1 = group.dropna(subset=["value", "val_lag1"])
    r1 = np.corrcoef(valid_l1["value"], valid_l1["val_lag1"])[0, 1]

    valid_l24 = group.dropna(subset=["value", "val_lag24"])
    r24 = np.corrcoef(valid_l24["value"], valid_l24["val_lag24"])[0, 1]

    valid_l168 = group.dropna(subset=["value", "val_lag168"])
    r168 = np.corrcoef(valid_l168["value"], valid_l168["val_lag168"])[0, 1]

    mean_val = group["value"].mean()
    std_val = group["value"].std()
    p10 = group["value"].quantile(0.10)
    p50 = group["value"].quantile(0.50)
    p90 = group["value"].quantile(0.90)
    cv = std_val / mean_val if mean_val > 0 else 0
    mean_abs_ramp_24h = group["ramp_24h"].abs().mean()
    max_abs_ramp_24h = group["ramp_24h"].abs().max()

    summary_list.append(
        {
            "Provincia": geo,
            "Media (MW)": round(mean_val, 1),
            "P50 (MW)": round(p50, 1),
            "P10-P90 (MW)": f"{round(p10,1)} - {round(p90,1)}",
            "CV": round(cv, 2),
            "r_1h": round(r1, 3),
            "r_24h": round(r24, 3),
            "r_168h": round(r168, 3),
            "Rampa Media 24h": round(mean_abs_ramp_24h, 1),
            "Rampa Max 24h": round(max_abs_ramp_24h, 1),
        }
    )

summary_df = pd.DataFrame(summary_list).sort_values(
    "Media (MW)", ascending=False
)
print(summary_df.to_string(index=False))
print("\n" + "=" * 75 + "\n")

# Estilo visual
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)

# ==========================================
# FIGURA 1: Resumen Geográfico con KPIs
# ==========================================
plt.figure(figsize=(11, 6))
ax = sns.barplot(
    data=summary_df,
    x="Media (MW)",
    y="Provincia",
    hue="Provincia",
    palette="viridis",
    legend=False,
)
plt.title(
    "Generación Eólica Media Horaria por Provincia (2021-2023)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Generación Media (MW)")
plt.ylabel("Provincia")

# Añadir etiquetas con métricas clave
for i, row in enumerate(summary_df.itertuples()):
    ax.text(
        row._2 + 10,
        i,
        f"Media: {row._2} MW | CV: {row.CV} | r24: {row.r_24h}",
        va="center",
        fontsize=9,
        fontweight="normal",
    )

plt.xlim(0, summary_df["Media (MW)"].max() * 1.35)
plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "01_resumen_geografico_provincias.png"), dpi=300
)
plt.close()

# ==========================================
# FIGURA 2: Patrón Temporal Multinivel (3x1)
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=False)

hourly = df.groupby(["hour", "geo_name"])["value"].mean().reset_index()
sns.lineplot(
    data=hourly, x="hour", y="value", hue="geo_name", ax=axes[0], alpha=0.8
)
axes[0].set_title(
    "A. Ciclo Diario (Hora 0 - 23 UTC)", fontsize=11, fontweight="bold"
)
axes[0].set_xlabel("Hora del Día")
axes[0].set_ylabel("Generación Promedio (MW)")
axes[0].set_xticks(range(0, 24))
axes[0].legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")

day_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
daily = df.groupby(["dayofweek", "geo_name"])["value"].mean().reset_index()
sns.lineplot(
    data=daily,
    x="dayofweek",
    y="value",
    hue="geo_name",
    ax=axes[1],
    marker="o",
    legend=False,
)
axes[1].set_title(
    "B. Ciclo Semanal (Lunes a Domingo)", fontsize=11, fontweight="bold"
)
axes[1].set_xlabel("Día de la Semana")
axes[1].set_ylabel("Generación Promedio (MW)")
axes[1].set_xticks(range(0, 7))
axes[1].set_xticklabels(day_labels)

monthly = df.groupby(["month", "geo_name"])["value"].mean().reset_index()
sns.lineplot(
    data=monthly,
    x="month",
    y="value",
    hue="geo_name",
    ax=axes[2],
    marker="s",
    legend=False,
)
axes[2].set_title(
    "C. Estacionalidad Mensual (Enero a Diciembre)",
    fontsize=11,
    fontweight="bold",
)
axes[2].set_xlabel("Mes del Año")
axes[2].set_ylabel("Generación Promedio (MW)")
axes[2].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "02_patron_temporal_multinivel.png"), dpi=300
)
plt.close()

# ==========================================
# FIGURA 3: Eventos Críticos y Rampas
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(
    df["ramp_24h"].dropna(),
    bins=100,
    kde=True,
    color="teal",
    stat="density",
)
plt.axvline(0, color="red", linestyle="--", alpha=0.7)
plt.title(
    "Distribución de la Rampa de Generación a 24 Horas (MW_t - MW_t-24)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Variación de Potencia a 24 Horas (MW)")
plt.ylabel("Densidad")
plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "03_eventos_criticos_y_rampas.png"), dpi=300
)
plt.close()

print(
    f"✅ EDA completado. Figuras guardadas en: {os.path.abspath(figures_dir)}"
)