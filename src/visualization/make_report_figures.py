"""
Make the figures for the report from the saved results (no retraining needed).

Run from the project root, after both training runs:
    python src/visualization/make_report_figures.py

Reads:  reports/results/forecast/   and   reports/results/reanalysis/
Writes: reports/figures/report/  (PDF for LaTeX, PNG for previews)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(script_dir, "..", ".."))
RES = os.path.join(ROOT, "reports", "results")
OUT = os.path.join(ROOT, "reports", "figures", "report")
os.makedirs(OUT, exist_ok=True)

SETS = ["S1 History", "S2 +Calendar", "S3 +Weather", "S4 +Physics"]
FULL = "S4 +Physics"
COLORS = {"Ridge": "#1f77b4", "Lasso": "#ff7f0e", "XGBoost": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200)
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


fc = pd.read_csv(os.path.join(RES, "forecast", "metrics_all.csv"))
re_path = os.path.join(RES, "reanalysis", "metrics_all.csv")
ra = pd.read_csv(re_path) if os.path.exists(re_path) else None
pers = fc.loc[fc["Model"] == "Persistence", "RMSE"].iloc[0]

# ---------------------------------------------------------------- Figure 1
# Error with progressively richer inputs; forecast (solid) vs observed weather (dotted)
fig, ax = plt.subplots(figsize=(6.0, 3.3))
x = range(len(SETS))
offsets = {"Ridge": -0.06, "Lasso": 0.0, "XGBoost": 0.06}   # so Ridge/Lasso don't hide
for model in ["Ridge", "Lasso", "XGBoost"]:
    sub = fc[fc["Model"] == model].set_index("Set").loc[SETS]
    xs = [i + offsets[model] for i in x]
    ax.errorbar(xs, sub["RMSE"],
                yerr=[sub["RMSE"] - sub["RMSE_CI_low"], sub["RMSE_CI_high"] - sub["RMSE"]],
                marker={"Ridge": "o", "Lasso": "s", "XGBoost": "^"}[model],
                ms=5, capsize=3, color=COLORS[model], label=f"{model} (forecast weather)",
                ls="-" if model != "Lasso" else "--")
    if ra is not None and model != "Lasso":
        sub_r = ra[ra["Model"] == model].set_index("Set").loc[SETS]
        ax.plot(xs, sub_r["RMSE"], ls=":", marker="o", mfc="white", ms=4,
                color=COLORS[model], label=f"{model} (observed weather)")
ax.axhline(pers, color="grey", ls="--", lw=1)
ax.text(len(SETS) - 1, pers + 4, "Persistence", ha="right", va="bottom",
        color="grey", fontsize=8)
ax.set_xticks(list(x), SETS)
ax.set_ylabel("Test RMSE (MW)")
ax.legend(loc="lower left", frameon=False)
save(fig, "fig1_progressive_inputs")

# ---------------------------------------------------------------- Figure 2
# Accuracy vs computational cost (full inputs, forecast weather)
full = fc[fc["Set"] == FULL]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
label_pos = {"Ridge": (8, 4), "Lasso": (8, -11), "XGBoost": (-8, 6)}
for _, r in full.iterrows():
    ax.errorbar(r["Tune_s"], r["RMSE"],
                yerr=[[r["RMSE"] - r["RMSE_CI_low"]], [r["RMSE_CI_high"] - r["RMSE"]]],
                fmt="o", ms=6, capsize=3, color=COLORS[r["Model"]])
    dx, dy = label_pos.get(r["Model"], (8, 4))
    ax.annotate(r["Model"], (r["Tune_s"], r["RMSE"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=8, ha="left" if dx > 0 else "right")
ax.set_xscale("log")
ax.set_xlim(full["Tune_s"].min() / 2.5, full["Tune_s"].max() * 2.5)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Tuning + training time (s, log scale)")
ax.set_ylabel("Test RMSE (MW)")
save(fig, "fig2_accuracy_vs_cost")

# ---------------------------------------------------------------- Figure 3
# Skill by province (full inputs, forecast weather)
bp = pd.read_csv(os.path.join(RES, "forecast", "metrics_by_province.csv"))
piv = bp[bp["Model"].isin(["Ridge", "XGBoost"])].pivot(
    index="Province", columns="Model", values="Skill_%").sort_values("XGBoost")
fig, ax = plt.subplots(figsize=(4.2, 3.0))
y = range(len(piv))
ax.barh([i - 0.2 for i in y], piv["Ridge"], height=0.4, color=COLORS["Ridge"], label="Ridge")
ax.barh([i + 0.2 for i in y], piv["XGBoost"], height=0.4, color=COLORS["XGBoost"],
        label="XGBoost")
ax.set_yticks(list(y), piv.index)
ax.set_xlabel("Skill vs. persistence (%)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig3_skill_by_province")
print(f"\nFigures saved to {OUT}")
