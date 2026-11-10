"""
Day-ahead (24 h) wind generation forecasting: model comparison.

Following the proposal feedback, the script
  1. compares models on PROGRESSIVELY RICHER INPUTS (feature sets S1-S4), to see
     which information matters and whether complex models extract more from it;
  2. measures the ACCURACY-COMPLEXITY TRADE-OFF fairly: every model gets the same
     data, splits and time-series CV, and we record tuning, fitting and
     prediction time;
  3. gives 95 % CONFIDENCE INTERVALS (block bootstrap over test days), so we can
     tell whether differences between models are real.

Run from the project root (after merge_datasets.py):
    python src/models/train_models.py --weather forecast      # main, no leakage
    python src/models/train_models.py --weather reanalysis    # upper bound
Add --quick for a fast test run with smaller hyper-parameter grids.
Add --lstm to also train the LSTM (needs PyTorch); --lstm-sets full trains it
only on the full input set S4, which is much faster.
Add --device cuda to run XGBoost and the LSTM on an NVIDIA GPU (Ridge and Lasso
always run on the CPU).

Outputs: reports/results/<weather>/ and reports/figures/<weather>/
"""
import argparse
import os
import sys
import time

# CPU threads per model: all cores by default. On one of our laptops, all cores made
# XGBoost several times SLOWER (thread contention); if that happens, limit it for a
# run with:  N_THREADS=2 python src/models/train_models.py ...
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 1)))
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_var] = str(N_THREADS)       # must be set before numpy is imported

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Lasso, Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_XGB = False

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import config  # noqa: E402

# ------------------------------------------------------------------ config
HORIZON = 24
N_SPLITS = 4
N_BOOT = 1000
SEED = 42

HISTORY = ["gen_lag24", "gen_lag48", "gen_lag168", "gen_roll24_mean"]
CALENDAR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
WEATHER = ["wind_speed_100m", "wind_gusts_10m", "temperature_2m", "pressure_msl",
           "wdir_sin", "wdir_cos"]
PHYSICS = ["ws2", "ws3_capped", "in_operating_range", "dp_3h", "ws_change_3h"]

# Progressively richer inputs (province indicators are added to every set)
FEATURE_SETS = {
    "S1 History": HISTORY,
    "S2 +Calendar": HISTORY + CALENDAR,
    "S3 +Weather": HISTORY + CALENDAR + WEATHER,
    "S4 +Physics": HISTORY + CALENDAR + WEATHER + PHYSICS,
}
FULL_SET = "S4 +Physics"


# ------------------------------------------------------------ data & features
def load(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values(["provincia", "datetime"]).reset_index(drop=True)


def regularise_grid(df, weather_vars):
    """One row per province and hour, so that shift(k) means k hours."""
    out = []
    for prov, g in df.groupby("provincia"):
        g = g.drop_duplicates("datetime").set_index("datetime").asfreq("h")
        g["provincia"] = prov
        g[weather_vars] = g[weather_vars].interpolate(limit=3)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def add_features(df, weather_vars):
    df = regularise_grid(df, weather_vars)
    g = df.groupby("provincia")

    # History: only generation known at least HORIZON hours before
    df["gen_lag24"] = g["generacion_mw"].shift(HORIZON)
    df["gen_lag48"] = g["generacion_mw"].shift(HORIZON + 24)
    df["gen_lag168"] = g["generacion_mw"].shift(168)
    df["gen_roll24_mean"] = g["generacion_mw"].transform(
        lambda s: s.shift(HORIZON).rolling(24, min_periods=12).mean())

    # Weather (forecast for the target hour) and physics-motivated transforms
    ws = df["wind_speed_100m"]
    df["ws2"] = ws ** 2
    df["ws3_capped"] = ws.clip(upper=13) ** 3
    df["in_operating_range"] = ((ws >= 3) & (ws <= 25)).astype(int)
    rad = np.deg2rad(df["wind_direction_100m"])
    df["wdir_sin"], df["wdir_cos"] = np.sin(rad), np.cos(rad)
    df["dp_3h"] = g["pressure_msl"].diff(3) if "pressure_msl" in df else np.nan
    df["ws_change_3h"] = g["wind_speed_100m"].diff(3)

    # Calendar
    hour, doy = df["datetime"].dt.hour, df["datetime"].dt.day_of_year
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    df["doy_sin"], df["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    df["month"], df["hour"] = df["datetime"].dt.month, hour

    dummies = pd.get_dummies(df["provincia"], prefix="prov", dtype=int)
    return pd.concat([df, dummies], axis=1), list(dummies.columns)


# ------------------------------------------------------------------ models
def make_models(quick, device="cpu"):
    models = {
        "Ridge": (make_pipeline(StandardScaler(), Ridge()),
                  {"ridge__alpha": np.logspace(-3, 3, 7 if quick else 13)}),
        "Lasso": (make_pipeline(StandardScaler(), Lasso(max_iter=20000)),
                  {"lasso__alpha": np.logspace(-5, -1, 3 if quick else 9)}),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(n_estimators=150 if quick else 600, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                         n_jobs=N_THREADS, tree_method="hist", device=device),
            {"max_depth": [6] if quick else [4, 6, 8],
             "min_child_weight": [10] if quick else [1, 10]})
    else:
        models["XGBoost"] = (HistGradientBoostingRegressor(
            max_iter=100 if quick else 400, learning_rate=0.05, random_state=SEED),
            {"max_depth": [6] if quick else [4, 8]})
    return models


# ------------------------------------------------------------------ metrics
def metrics(y, yhat):
    err = yhat - y
    return {"RMSE": np.sqrt(np.mean(err**2)), "MAE": np.mean(np.abs(err)),
            "Bias": np.mean(err)}


def bootstrap(test, cols, n_boot=N_BOOT):
    """Block bootstrap over test DAYS (keeps within-day correlation).
    Returns a (n_boot x len(cols)) array of RMSE samples."""
    day = test["datetime"].dt.floor("D")
    sq = pd.DataFrame({c: (test[c] - test["generacion_mw"]) ** 2 for c in cols})
    sums = sq.groupby(day.values).sum().values
    counts = day.value_counts().sort_index().values
    rng = np.random.default_rng(SEED)
    out = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        idx = rng.integers(0, len(counts), len(counts))
        out[b] = np.sqrt(sums[idx].sum(axis=0) / counts[idx].sum())
    return out


def check_gpu():
    """Stop with a clear message if XGBoost cannot use the GPU."""
    if not HAS_XGB:
        sys.exit("--device cuda needs xgboost installed.")
    try:
        XGBRegressor(n_estimators=2, device="cuda", tree_method="hist").fit(
            np.random.rand(50, 3), np.random.rand(50))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"XGBoost cannot use the GPU: {e}\nCheck that 'nvidia-smi' works.")


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", choices=["forecast", "reanalysis"], default="forecast")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lstm", action="store_true", help="also train the LSTM")
    parser.add_argument("--lstm-sets", choices=["all", "full"], default="all",
                        help="input sets for the LSTM: all four, or only S4")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="cuda = run XGBoost and the LSTM on an NVIDIA GPU")
    args = parser.parse_args()
    if args.device == "cuda":
        check_gpu()
    if args.lstm:
        import lstm_model
        if not lstm_model.HAS_TORCH:
            sys.exit("PyTorch is not installed. Run: pip install torch "
                     "--index-url https://download.pytorch.org/whl/cpu")

    root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    res_dir = os.path.join(root, "reports", "results", args.weather)
    fig_dir = os.path.join(root, "reports", "figures", args.weather)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Using {N_THREADS} CPU threads per model | device for XGBoost/LSTM: {args.device}")
    df = load(os.path.join(root, "data", "processed", f"dataset_{args.weather}.csv"))

    # Drop weather variables that did not come back from the API
    weather_vars = [v for v in config.WEATHER_VARS if df[v].isna().mean() < 0.5]
    dropped = sorted(set(config.WEATHER_VARS) - set(weather_vars))
    if dropped:
        print(f"WARNING: dropping weather variables with no data: {dropped}")
    df, prov_cols = add_features(df, weather_vars)
    unusable = {"wind_gusts_10m": ["wind_gusts_10m"], "temperature_2m": ["temperature_2m"],
                "pressure_msl": ["pressure_msl", "dp_3h"]}
    skip = [c for v in dropped for c in unusable.get(v, [])]
    sets = {k: [f for f in v if f not in skip] + prov_cols for k, v in FEATURE_SETS.items()}

    all_feats = sorted(set().union(*sets.values()))
    hourly = df.copy()                 # complete hourly grid, used for LSTM sequences
    df = df.dropna(subset=all_feats + ["generacion_mw"]).sort_values("datetime")
    test_start = pd.Timestamp(config.TEST_START, tz="UTC")
    train = df[df["datetime"] < test_start].copy()
    test = df[df["datetime"] >= test_start].copy()
    print(f"Train: {train['datetime'].min():%Y-%m-%d} to {train['datetime'].max():%Y-%m-%d}"
          f" ({len(train)} rows) | Test: {test['datetime'].min():%Y-%m-%d} to "
          f"{test['datetime'].max():%Y-%m-%d} ({len(test)} rows)")

    # Normalise by each province's training maximum (capacity proxy)
    cap = train.groupby("provincia")["generacion_mw"].max()

    # The LSTM needs the previous hours to be complete. For a fair comparison, ALL
    # models are evaluated on the same test hours, so hours without a complete
    # window are removed from the test set (and from the LSTM's training data).
    train_lstm = None
    if args.lstm:
        is_train = lambda t: t < test_start  # noqa: E731
        arr_all, lookup_all = lstm_model.build_grid(hourly, all_feats, cap, is_train)
        ok_te = lstm_model.valid_windows(arr_all, lstm_model.row_index(test, lookup_all))
        ok_tr = lstm_model.valid_windows(arr_all, lstm_model.row_index(train, lookup_all))
        print(f"LSTM windows ({lstm_model.SEQ_LEN} h): dropping {(~ok_te).sum()} test rows "
              f"and {(~ok_tr).sum()} training rows without complete history")
        test = test[ok_te].copy()
        train_lstm = train[ok_tr].copy()
    for d in [x for x in (train, test, train_lstm) if x is not None]:
        c = d["provincia"].map(cap)
        d["y_norm"] = d["generacion_mw"] / c
        for col in HISTORY:
            d[col] = d[col] / c
    c_te = test["provincia"].map(cap).values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=48 * len(prov_cols))

    # --- Baselines
    pred_cols, info = [], {}
    test["Persistence"] = np.clip(test["gen_lag24"].values, 0, 1) * c_te
    clim = train.groupby(["provincia", "month", "hour"])["y_norm"].mean()
    test["Climatology"] = np.clip(clim.reindex(pd.MultiIndex.from_frame(
        test[["provincia", "month", "hour"]])).values, 0, 1) * c_te
    for b in ["Persistence", "Climatology"]:
        pred_cols.append(b)
        info[b] = {"Set": "-", "Model": b, "Device": "-", "Tune_s": 0.0, "Fit_s": 0.0,
                   "Pred_ms_per_1k": 0.0}

    # --- Every model on every feature set
    importances = {}
    for set_name, feats in sets.items():
        for model_name, (est, grid) in make_models(args.quick, args.device).items():
            col = f"{model_name} | {set_name}"
            print(f"Training {col} ({len(feats)} features)...", flush=True)
            gs = GridSearchCV(est, grid, cv=tscv, scoring="neg_root_mean_squared_error",
                              n_jobs=1 if model_name == "XGBoost" else N_THREADS)
            t0 = time.perf_counter()
            gs.fit(train[feats], train["y_norm"])
            tune = time.perf_counter() - t0
            t0 = time.perf_counter()
            p = gs.predict(test[feats])
            pred_ms = 1000 * (time.perf_counter() - t0) / len(test) * 1000
            test[col] = np.clip(p, 0, 1) * c_te
            pred_cols.append(col)
            dev = args.device if model_name == "XGBoost" else "cpu"
            info[col] = {"Set": set_name, "Model": model_name, "Device": dev, "Tune_s": tune,
                         "Fit_s": gs.refit_time_, "Pred_ms_per_1k": pred_ms,
                         "CV_nRMSE_%": -100 * gs.best_score_, "Best_params": gs.best_params_}
            if set_name == FULL_SET:
                best = gs.best_estimator_
                if model_name == "XGBoost" and HAS_XGB:
                    importances["xgb_gain"] = pd.Series(best.feature_importances_, index=feats)
                if model_name == "Lasso":
                    importances["lasso_coef"] = pd.Series(best[-1].coef_, index=feats)

    # --- LSTM last: PyTorch is only loaded now, so it never competes with XGBoost
    for set_name, feats in sets.items():
        if args.lstm and (args.lstm_sets == "all" or set_name == FULL_SET):
            col = f"LSTM | {set_name}"
            print(f"Training {col} ({len(feats)} features)...", flush=True)
            arr, lookup = lstm_model.build_grid(hourly, feats, cap, is_train)
            p, inf = lstm_model.tune_fit_predict(
                arr, lstm_model.row_index(train_lstm, lookup), train_lstm["y_norm"].values,
                train_lstm["datetime"], lstm_model.row_index(test, lookup), quick=args.quick,
                device=args.device)
            test[col] = np.clip(p, 0, 1) * c_te
            pred_cols.append(col)
            info[col] = {"Set": set_name, "Model": "LSTM", "Device": args.device, **inf}

    # --- Evaluation
    boot = bootstrap(test, pred_cols)
    agg = test.groupby("datetime")[["generacion_mw"] + pred_cols].sum()
    pers_rmse = metrics(test["generacion_mw"], test["Persistence"])["RMSE"]
    rows = []
    for i, col in enumerate(pred_cols):
        r = metrics(test["generacion_mw"], test[col])
        r["RMSE_CI_low"], r["RMSE_CI_high"] = np.percentile(boot[:, i], [2.5, 97.5])
        r["nRMSE_%"] = 100 * np.sqrt(np.mean(((test[col] - test["generacion_mw"]) / c_te)**2))
        r["Skill_%"] = 100 * (1 - r["RMSE"] / pers_rmse)
        r["RMSE_sum"] = metrics(agg["generacion_mw"], agg[col])["RMSE"]
        rows.append({**info[col], **r})
    res = pd.DataFrame(rows)
    res.drop(columns="Best_params").to_csv(os.path.join(res_dir, "metrics_all.csv"), index=False)

    # Does XGBoost extract more than a linear model? Gap per feature set, with CI
    gaps = []
    for set_name in sets:
        i_x = pred_cols.index(f"XGBoost | {set_name}")
        i_r = pred_cols.index(f"Ridge | {set_name}")
        d = boot[:, i_r] - boot[:, i_x]
        gaps.append({"Set": set_name, "Ridge_minus_XGB_MW": d.mean(),
                     "CI_low": np.percentile(d, 2.5), "CI_high": np.percentile(d, 97.5)})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(os.path.join(res_dir, "xgb_vs_ridge_gap.csv"), index=False)

    lstm_gaps = []
    for set_name in sets:
        if f"LSTM | {set_name}" in pred_cols:
            i_l = pred_cols.index(f"LSTM | {set_name}")
            i_x = pred_cols.index(f"XGBoost | {set_name}")
            d = boot[:, i_l] - boot[:, i_x]
            lstm_gaps.append({"Set": set_name, "LSTM_minus_XGB_MW": d.mean(),
                              "CI_low": np.percentile(d, 2.5),
                              "CI_high": np.percentile(d, 97.5)})
    lstm_gaps = pd.DataFrame(lstm_gaps)
    if len(lstm_gaps):
        lstm_gaps.to_csv(os.path.join(res_dir, "lstm_vs_xgb_gap.csv"), index=False)

    # Per province (full feature set)
    prov_rows = []
    for prov, g in test.groupby("provincia"):
        p_rmse = metrics(g["generacion_mw"], g["Persistence"])["RMSE"]
        for col in ["Persistence", "Climatology"] + [c for c in pred_cols
                                                      if c.endswith(f"| {FULL_SET}")]:
            r = metrics(g["generacion_mw"], g[col])
            prov_rows.append({"Province": prov, "Model": col.split(" | ")[0], **r,
                              "Skill_%": 100 * (1 - r["RMSE"] / p_rmse)})
    by_prov = pd.DataFrame(prov_rows)
    by_prov.to_csv(os.path.join(res_dir, "metrics_by_province.csv"), index=False)
    if importances:
        pd.DataFrame(importances).sort_values(list(importances)[0], ascending=False).to_csv(
            os.path.join(res_dir, "feature_importance.csv"))

    # --- Console summary
    pd.set_option("display.width", 160)
    show = res[["Set", "Model", "Device", "RMSE", "RMSE_CI_low", "RMSE_CI_high", "nRMSE_%",
                "Skill_%", "RMSE_sum", "Tune_s", "Fit_s", "Pred_ms_per_1k"]]
    print(f"\n=== Test results ({args.weather} weather) ===")
    print(show.round(2).to_string(index=False))
    print("\n=== Ridge RMSE minus XGBoost RMSE (MW, 95% CI): positive = XGBoost better ===")
    print(gaps.round(1).to_string(index=False))
    if len(lstm_gaps):
        print("\n=== LSTM RMSE minus XGBoost RMSE (MW, 95% CI): negative = LSTM better ===")
        print(lstm_gaps.round(1).to_string(index=False))

    # --- LaTeX tables
    def fmt_ci(r):
        return f"{r['RMSE']:.1f} [{r['RMSE_CI_low']:.0f}, {r['RMSE_CI_high']:.0f}]"

    main_rows = res[(res["Set"].isin(["-", FULL_SET]))]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & RMSE [95\% CI] & nRMSE & Skill & RMSE$_{\Sigma}$ & Tune & Fit \\",
             r" & (MW) & (\%) & (\%) & (MW) & (s) & (s) \\", r"\midrule"]
    for _, r in main_rows.iterrows():
        lines.append(f"{r['Model']} & {fmt_ci(r)} & {r['nRMSE_%']:.1f} & {r['Skill_%']:.1f}"
                     f" & {r['RMSE_sum']:.0f} & {r['Tune_s']:.0f} & {r['Fit_s']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(res_dir, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    piv = res[res["Set"] != "-"].pivot(index="Set", columns="Model", values="RMSE")
    model_names = [m for m in ["Ridge", "Lasso", "XGBoost", "LSTM"] if m in piv.columns]
    lines = [r"\begin{tabular}{l" + "r" * (len(model_names) + 1) + "}", r"\toprule",
             "Inputs & " + " & ".join(model_names) + r" & Ridge$-$XGB [95\% CI] \\",
             r"\midrule"]
    for s_name in sets:
        gp = gaps.set_index("Set").loc[s_name]
        cells = [f"{piv.loc[s_name, m]:.1f}" if pd.notna(piv.loc[s_name, m]) else "--"
                 for m in model_names]
        lines.append(f"{s_name} & " + " & ".join(cells) +
                     f" & {gp['Ridge_minus_XGB_MW']:.1f} "
                     f"[{gp['CI_low']:.0f}, {gp['CI_high']:.0f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(res_dir, "ablation_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --- Figures
    # 1) Progressive inputs
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in model_names:
        sub = res[res["Model"] == m].set_index("Set").reindex(list(sets)).dropna(subset=["RMSE"])
        ax.errorbar([list(sets).index(k) for k in sub.index], sub["RMSE"],
                    yerr=[sub["RMSE"] - sub["RMSE_CI_low"], sub["RMSE_CI_high"] - sub["RMSE"]],
                    marker="o", capsize=3, label=m)
    ax.axhline(pers_rmse, color="grey", ls="--", label="Persistence")
    ax.set_xticks(range(len(sets)), list(sets))
    ax.set_ylabel("Test RMSE (MW)")
    ax.set_title(f"Error with progressively richer inputs ({args.weather} weather)")
    ax.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "07_progressive_inputs.png"), dpi=300)
    plt.close()

    # 2) Accuracy vs computational cost
    full = res[res["Set"] == FULL_SET]
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in full.iterrows():
        ax.errorbar(r["Tune_s"], r["RMSE"],
                    yerr=[[r["RMSE"] - r["RMSE_CI_low"]], [r["RMSE_CI_high"] - r["RMSE"]]],
                    fmt="o", capsize=3)
        ax.annotate(r["Model"], (r["Tune_s"], r["RMSE"]), textcoords="offset points",
                    xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_xlabel("Tuning + training time (s, log scale)")
    ax.set_ylabel("Test RMSE (MW)")
    ax.set_title("Accuracy vs. computational cost (full inputs)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "08_accuracy_vs_cost.png"), dpi=300)
    plt.close()

    # 3) Skill by province
    piv = by_prov[by_prov["Model"].isin(["Ridge", "XGBoost"])].pivot(
        index="Province", columns="Model", values="Skill_%").sort_values("XGBoost")
    ax = piv.plot.barh(figsize=(7, 4))
    ax.set_xlabel("Skill vs. persistence (%)")
    ax.set_title("Skill by province (full inputs)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "09_skill_by_province.png"), dpi=300)
    plt.close()
    print(f"\nSaved results to {res_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
