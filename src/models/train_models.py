"""
Day-ahead (24 h) wind generation forecasting: preliminary model comparison.

Models: Persistence, Climatology, Ridge, Lasso, XGBoost.
Train/validation: 2021-2022 (TimeSeriesSplit for hyper-parameters).
Test (held out, touched once): 2023.

Run from the project root, after merge_datasets.py:
    python src/models/train_models.py

Outputs (reports/results/ and reports/figures/):
    data_quality.txt          missing hours, duplicates, negatives per province
    metrics_overall.csv       RMSE / MAE / nRMSE / skill per model (test 2023)
    metrics_by_province.csv   same, per province
    metrics_aggregate.csv     errors on the sum of the 8 provinces
    feature_importance.csv    XGBoost gain + Lasso coefficients
    results_table.tex         LaTeX table ready to \\input{} in the report
    05_model_comparison_by_province.png
    06_forecast_example_week.png
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # fallback so the script still runs without xgboost
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False

# ------------------------------------------------------------------ config
HORIZON = 24                    # forecast horizon in hours
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
N_SPLITS = 4                    # folds for time-series CV
SEED = 42

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA = os.path.join(root, "data", "processed", "dataset.csv")
RES_DIR = os.path.join(root, "reports", "results")
FIG_DIR = os.path.join(root, "reports", "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

WEATHER = ["wind_speed_100m", "wind_gusts_10m", "temperature_2m",
           "pressure_msl", "wind_direction_100m"]


# ------------------------------------------------------------- 1. loading
def load_data():
    df = pd.read_csv(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values(["provincia", "datetime"]).reset_index(drop=True)


def data_quality_report(df):
    """Missing hours, duplicates and negative values per province."""
    full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    lines = [f"Rows: {len(df)}  |  expected per province: {len(full_range)}"]
    for prov, g in df.groupby("provincia"):
        dups = g["datetime"].duplicated().sum()
        missing = len(full_range) - g["datetime"].nunique()
        neg = (g["generacion_mw"] < 0).sum()
        nan = g[["generacion_mw"] + WEATHER].isna().sum().sum()
        lines.append(f"{prov:10s} missing_hours={missing:5d} duplicates={dups:4d} "
                     f"negative_gen={neg:4d} NaN_values={nan:4d}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES_DIR, "data_quality.txt"), "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------- 2. feature engineering
def regularise_grid(df):
    """Put every province on a complete hourly grid so that shift(k) == k hours."""
    out = []
    for prov, g in df.groupby("provincia"):
        g = g.drop_duplicates("datetime").set_index("datetime").asfreq("h")
        g["provincia"] = prov
        g[WEATHER] = g[WEATHER].interpolate(limit=3)  # short gaps only
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def add_features(df):
    df = regularise_grid(df)
    g = df.groupby("provincia")

    # Target-history features: only information available >= HORIZON hours before
    df["gen_lag24"] = g["generacion_mw"].shift(HORIZON)
    df["gen_lag48"] = g["generacion_mw"].shift(HORIZON + 24)
    df["gen_lag168"] = g["generacion_mw"].shift(168)
    df["gen_roll24_mean"] = g["generacion_mw"].transform(
        lambda s: s.shift(HORIZON).rolling(24, min_periods=12).mean())

    # Physics-motivated weather transforms (weather at the target hour)
    ws = df["wind_speed_100m"]
    df["ws2"] = ws ** 2
    df["ws3_capped"] = ws.clip(upper=13) ** 3        # cubic up to ~rated speed
    df["in_operating_range"] = ((ws >= 3) & (ws <= 25)).astype(int)
    rad = np.deg2rad(df["wind_direction_100m"])      # direction is circular
    df["wdir_sin"], df["wdir_cos"] = np.sin(rad), np.cos(rad)
    df["dp_3h"] = g["pressure_msl"].diff(3)           # pressure tendency (fronts)
    df["ws_change_3h"] = g["wind_speed_100m"].diff(3)

    # Calendar (cyclical encoding)
    hour = df["datetime"].dt.hour
    doy = df["datetime"].dt.day_of_year
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    df["doy_sin"], df["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    df["month"] = df["datetime"].dt.month
    df["hour"] = hour

    df = pd.get_dummies(df, columns=["provincia"], prefix="prov", dtype=int)
    prov_cols = [c for c in df.columns if c.startswith("prov_")]
    df["provincia"] = df[prov_cols].idxmax(axis=1).str.replace("prov_", "")
    return df, prov_cols


FEATURES_BASE = ["wind_speed_100m", "ws2", "ws3_capped", "in_operating_range",
                 "wind_gusts_10m", "temperature_2m", "pressure_msl", "dp_3h",
                 "ws_change_3h", "wdir_sin", "wdir_cos",
                 "gen_lag24", "gen_lag48", "gen_lag168", "gen_roll24_mean",
                 "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


# ---------------------------------------------------------------- 3. metrics
def metrics(y, yhat):
    err = yhat - y
    return {"RMSE": np.sqrt(np.mean(err**2)), "MAE": np.mean(np.abs(err)),
            "Bias": np.mean(err)}


# ------------------------------------------------------------------ 4. main
def main():
    df = load_data()
    data_quality_report(df)
    df, prov_cols = add_features(df)
    features = FEATURES_BASE + prov_cols

    df = df.dropna(subset=features + ["generacion_mw"]).sort_values("datetime")
    train = df[df["datetime"] < TEST_START].copy()
    test = df[df["datetime"] >= TEST_START].copy()

    # Normalise target by each province's TRAINING max (capacity proxy) so the
    # pooled model treats provinces on a comparable 0-1 scale. Lags too.
    cap = train.groupby("provincia")["generacion_mw"].max()
    for d in (train, test):
        c = d["provincia"].map(cap)
        d["y_norm"] = d["generacion_mw"] / c
        for col in ["gen_lag24", "gen_lag48", "gen_lag168", "gen_roll24_mean"]:
            d[col] = d[col] / c

    X_tr, y_tr = train[features], train["y_norm"]
    X_te = test[features]
    n_prov = len(prov_cols)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=48 * n_prov)  # gap ~ 2 days

    preds, fit_time = {}, {}
    # --- Baselines
    preds["Persistence (t-24h)"] = test["gen_lag24"].values
    clim = train.groupby(["provincia", "month", "hour"])["y_norm"].mean()
    preds["Climatology"] = clim.reindex(
        pd.MultiIndex.from_frame(test[["provincia", "month", "hour"]])).values

    # --- Ridge / Lasso (scaled), alpha chosen by time-series CV
    cv_info = {}
    for name, est, grid in [
        ("Ridge", Ridge(), {"ridge__alpha": np.logspace(-3, 3, 13)}),
        ("Lasso", Lasso(max_iter=20000), {"lasso__alpha": np.logspace(-5, -1, 9)}),
    ]:
        gs = GridSearchCV(make_pipeline(StandardScaler(), est), grid, cv=tscv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        t0 = time.perf_counter()
        gs.fit(X_tr, y_tr)
        fit_time[name] = time.perf_counter() - t0
        preds[name] = gs.predict(X_te)
        cv_info[name] = (gs.best_params_, -gs.best_score_)
        if name == "Lasso":
            lasso_coef = pd.Series(gs.best_estimator_[-1].coef_, index=features)

    # --- Gradient boosting
    if HAS_XGB:
        gb = XGBRegressor(n_estimators=600, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                          tree_method="hist")
        grid = {"max_depth": [4, 6, 8], "min_child_weight": [1, 10]}
        gb_name = "XGBoost"
    else:
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                           random_state=SEED)
        grid = {"max_depth": [4, 8]}
        gb_name = "GradBoost (sklearn fallback)"
    gs = GridSearchCV(gb, grid, cv=tscv, scoring="neg_root_mean_squared_error")
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    fit_time[gb_name] = time.perf_counter() - t0
    preds[gb_name] = gs.predict(X_te)
    cv_info[gb_name] = (gs.best_params_, -gs.best_score_)
    if HAS_XGB:
        imp = pd.Series(gs.best_estimator_.feature_importances_, index=features)
    else:
        imp = pd.Series(np.nan, index=features)

    for k, (p, s) in cv_info.items():
        print(f"{k}: best {p}, CV RMSE (normalised) = {s:.4f}")

    # --- Back to MW, clip to physical range [0, cap]
    c_te = test["provincia"].map(cap).values
    for k in preds:
        test[k] = np.clip(preds[k], 0, 1) * c_te

    # --- Evaluation
    models = list(preds)
    rows, prov_rows = [], []
    base_rmse = metrics(test["generacion_mw"], test[models[0]])["RMSE"]
    for m in models:
        r = metrics(test["generacion_mw"], test[m])
        r["nRMSE_%"] = 100 * np.sqrt(np.mean(((test[m] - test["generacion_mw"]) / c_te) ** 2))
        r["Skill_vs_persistence_%"] = 100 * (1 - r["RMSE"] / base_rmse)
        r["Fit_time_s"] = fit_time.get(m, 0.0)   # includes CV grid search
        rows.append({"Model": m, **r})
        for prov, g in test.groupby("provincia"):
            prov_rows.append({"Model": m, "Province": prov,
                              **metrics(g["generacion_mw"], g[m])})
    overall = pd.DataFrame(rows)
    by_prov = pd.DataFrame(prov_rows)

    agg = test.groupby("datetime")[["generacion_mw"] + models].sum()
    agg_rows = [{"Model": m, **metrics(agg["generacion_mw"], agg[m])} for m in models]
    aggregate = pd.DataFrame(agg_rows)

    overall.to_csv(os.path.join(RES_DIR, "metrics_overall.csv"), index=False)
    by_prov.to_csv(os.path.join(RES_DIR, "metrics_by_province.csv"), index=False)
    aggregate.to_csv(os.path.join(RES_DIR, "metrics_aggregate.csv"), index=False)
    pd.DataFrame({"xgb_gain": imp, "lasso_coef": lasso_coef}).sort_values(
        "xgb_gain", ascending=False).to_csv(os.path.join(RES_DIR, "feature_importance.csv"))
    print(overall.round(2).to_string(index=False))
    print("\nAggregate (sum of 8 provinces):\n", aggregate.round(1).to_string(index=False))

    # --- LaTeX table
    tab = overall.merge(aggregate[["Model", "RMSE"]].rename(columns={"RMSE": "RMSE_agg"}),
                        on="Model")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & RMSE & MAE & nRMSE & Skill & RMSE$_{\Sigma 8}$ & Fit time \\",
             r" & (MW) & (MW) & (\%) & (\%) & (MW) & (s) \\", r"\midrule"]
    for _, r in tab.iterrows():
        lines.append(f"{r['Model']} & {r['RMSE']:.1f} & {r['MAE']:.1f} & "
                     f"{r['nRMSE_%']:.1f} & {r['Skill_vs_persistence_%']:.1f} & "
                     f"{r['RMSE_agg']:.0f} & {r['Fit_time_s']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RES_DIR, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Figure: RMSE by province and model
    piv = by_prov.pivot(index="Province", columns="Model", values="RMSE")[models]
    piv = piv.loc[piv[models[0]].sort_values(ascending=False).index]
    ax = piv.plot.bar(figsize=(10, 4.5), width=0.8)
    ax.set_ylabel("RMSE on 2023 test set (MW)")
    ax.set_xlabel("")
    ax.set_title("Day-ahead forecast error by province and model")
    plt.xticks(rotation=0)
    plt.legend(fontsize="small", ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_model_comparison_by_province.png"), dpi=300)
    plt.close()

    # --- Figure: example week, aggregate of 8 provinces
    week = agg.loc["2023-02-01":"2023-02-08"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week.index, week["generacion_mw"], color="black", lw=2, label="Observed")
    for m in [models[0], "Ridge", gb_name]:
        ax.plot(week.index, week[m], lw=1.2, label=m)
    ax.set_ylabel("Generation, sum of 8 provinces (MW)")
    ax.set_title("Example week of day-ahead forecasts (test set, Feb 2023)")
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_forecast_example_week.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()