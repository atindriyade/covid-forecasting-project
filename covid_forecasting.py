"""
COVID Death Forecasting Model
==============================
Two-model approach:
  1. National-level: Facebook Prophet with external regressors (confirmed/cured cases)
  2. Global predictive: Elastic Net regression for cross-country mortality analysis

Author: Rishov | Inspired by Atindriya De's portfolio project
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from prophet import Prophet
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

PROPHET_COUNTRY  = "United States"
FORECAST_DAYS    = 60
TRAIN_CUTOFF     = "2022-06-01"
RANDOM_STATE     = 42
PALETTE          = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
                    "#0891B2", "#BE185D", "#65A30D", "#EA580C", "#6366F1"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "font.family":      "DejaVu Sans",
    "grid.color":       "#E2E8F0",
    "grid.linewidth":   0.7,
})

# ─────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────
print("=" * 60)
print("COVID Death Forecasting Model")
print("=" * 60)

df_raw = pd.read_csv("covid_data.csv", parse_dates=["date"])
df_raw.sort_values(["country", "date"], inplace=True)

# Rolling 7-day smoothing per country
for col in ["new_cases", "new_deaths", "new_recovered"]:
    df_raw[f"{col}_7d"] = (
        df_raw.groupby("country")[col]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

# Lag features
for lag in [7, 14]:
    df_raw[f"cases_lag{lag}"]     = df_raw.groupby("country")["new_cases_7d"].shift(lag)
    df_raw[f"deaths_lag{lag}"]    = df_raw.groupby("country")["new_deaths_7d"].shift(lag)
    df_raw[f"recovered_lag{lag}"] = df_raw.groupby("country")["new_recovered_7d"].shift(lag)

df_raw["cfr_7d"] = df_raw["new_deaths_7d"] / (df_raw["new_cases_7d"] + 1)
df_raw["cases_per_million"] = df_raw["new_cases_7d"] / (df_raw["population"] / 1e6)
df_raw["deaths_per_million"] = df_raw["new_deaths_7d"] / (df_raw["population"] / 1e6)

print(f"\n✔ Dataset loaded: {len(df_raw):,} rows | {df_raw['country'].nunique()} countries")
print(f"  Date range: {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")


# ─────────────────────────────────────────────
# FIGURE 1 – EDA: Cases & Deaths Overview
# ─────────────────────────────────────────────
print("\n[1/6] Generating EDA visualisations …")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("COVID-19 Global Overview  |  EDA", fontsize=15, fontweight="bold", y=0.98)

# Panel A – Total deaths by country
country_totals = df_raw.groupby("country")["new_deaths"].sum().sort_values(ascending=False)
ax = axes[0, 0]
bars = ax.barh(country_totals.index, country_totals.values / 1e3, color=PALETTE)
ax.set_xlabel("Total Deaths (thousands)")
ax.set_title("A  |  Total Deaths by Country")
for bar, val in zip(bars, country_totals.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val/1e3:.0f}K", va="center", fontsize=8)

# Panel B – 7-day death trend: top 5 countries
ax = axes[0, 1]
top5 = country_totals.head(5).index
for i, country in enumerate(top5):
    sub = df_raw[df_raw["country"] == country]
    ax.plot(sub["date"], sub["new_deaths_7d"], label=country, color=PALETTE[i], lw=1.8)
ax.set_title("B  |  Daily Deaths (7-day avg) – Top 5")
ax.set_xlabel("Date"); ax.set_ylabel("Deaths / day")
ax.legend(fontsize=8, loc="upper left")

# Panel C – CFR heatmap by country (monthly)
ax = axes[1, 0]
df_raw["month"] = df_raw["date"].dt.to_period("M").astype(str)
cfr_pivot = df_raw.groupby(["country", "month"])["cfr_7d"].mean().unstack()
cfr_pivot = cfr_pivot.iloc[:, ::3]  # every 3rd month for readability
sns.heatmap(cfr_pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "CFR"}, fmt=".3f", annot=False)
ax.set_title("C  |  Case Fatality Rate by Country & Quarter")
ax.set_xlabel(""); ax.set_ylabel("")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", labelsize=8)

# Panel D – Cases vs Deaths scatter (log scale)
ax = axes[1, 1]
for i, country in enumerate(df_raw["country"].unique()):
    sub = df_raw[df_raw["country"] == country].dropna()
    ax.scatter(sub["new_cases_7d"], sub["new_deaths_7d"],
               alpha=0.15, s=4, color=PALETTE[i], label=country)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("New Cases (7d avg, log)"); ax.set_ylabel("New Deaths (7d avg, log)")
ax.set_title("D  |  Cases vs Deaths (log scale)")
ax.legend(markerscale=3, fontsize=7, loc="upper left")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig1_eda_overview.png")


# ─────────────────────────────────────────────
# FIGURE 2 – National Trend: USA deep-dive
# ─────────────────────────────────────────────
usa = df_raw[df_raw["country"] == PROPHET_COUNTRY].copy()

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"National Trend Deep-Dive  |  {PROPHET_COUNTRY}", fontsize=14,
             fontweight="bold")

axes[0].fill_between(usa["date"], usa["new_cases_7d"], alpha=0.4, color=PALETTE[0])
axes[0].plot(usa["date"], usa["new_cases_7d"], color=PALETTE[0], lw=1.5)
axes[0].set_ylabel("New Cases (7d avg)")
axes[0].set_title("Daily New Cases")

axes[1].fill_between(usa["date"], usa["new_deaths_7d"], alpha=0.4, color=PALETTE[1])
axes[1].plot(usa["date"], usa["new_deaths_7d"], color=PALETTE[1], lw=1.5)
axes[1].set_ylabel("New Deaths (7d avg)")
axes[1].set_title("Daily New Deaths")

axes[2].plot(usa["date"], usa["cfr_7d"] * 100, color=PALETTE[3], lw=1.5)
axes[2].fill_between(usa["date"], usa["cfr_7d"] * 100, alpha=0.3, color=PALETTE[3])
axes[2].set_ylabel("CFR (%)")
axes[2].set_xlabel("Date")
axes[2].set_title("Case Fatality Rate (%)")

for ax in axes:
    ax.grid(True, alpha=0.5)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_national_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig2_national_trend.png")


# ─────────────────────────────────────────────
# MODEL 1 – Facebook Prophet (National)
# ─────────────────────────────────────────────
print("\n[2/6] Training Prophet model …")

usa_train = usa[usa["date"] < TRAIN_CUTOFF].copy()
usa_test  = usa[usa["date"] >= TRAIN_CUTOFF].copy()

prophet_df = usa_train.rename(columns={"date": "ds", "new_deaths_7d": "y"})
prophet_df = prophet_df[["ds", "y", "new_cases_7d", "new_recovered_7d"]].dropna()

model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,
)
model.add_regressor("new_cases_7d",     standardize=True)
model.add_regressor("new_recovered_7d", standardize=True)
model.fit(prophet_df)

# Forecast into test period
future = usa_test[["date", "new_cases_7d", "new_recovered_7d"]].copy()
future.rename(columns={"date": "ds"}, inplace=True)
future["new_cases_7d"]     = future["new_cases_7d"].ffill()
future["new_recovered_7d"] = future["new_recovered_7d"].ffill()

forecast = model.predict(future)
forecast.set_index("ds", inplace=True)

y_true = usa_test.set_index("date")["new_deaths_7d"].dropna()
y_pred = forecast.loc[y_true.index, "yhat"].clip(lower=0)

mae   = mean_absolute_error(y_true, y_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
r2    = r2_score(y_true, y_pred)

print(f"   Prophet  →  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

# ─── Prophet future projection ───────────────
all_future = model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
last_cases = usa["new_cases_7d"].iloc[-1]
last_recov = usa["new_recovered_7d"].iloc[-1]
all_future = all_future.merge(
    usa[["date", "new_cases_7d", "new_recovered_7d"]].rename(columns={"date": "ds"}),
    on="ds", how="left"
)
all_future["new_cases_7d"].fillna(last_cases, inplace=True)
all_future["new_recovered_7d"].fillna(last_recov, inplace=True)

full_forecast = model.predict(all_future)


# ─────────────────────────────────────────────
# FIGURE 3 – Prophet Forecast Plot
# ─────────────────────────────────────────────
print("\n[3/6] Plotting Prophet forecast …")

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle(f"Facebook Prophet  |  {PROPHET_COUNTRY}  |  Death Forecasting",
             fontsize=14, fontweight="bold")

# Panel A – full timeline
ax = axes[0]
ax.fill_between(full_forecast["ds"],
                full_forecast["yhat_lower"].clip(0),
                full_forecast["yhat_upper"].clip(0),
                alpha=0.25, color=PALETTE[0], label="95% CI")
ax.plot(full_forecast["ds"], full_forecast["yhat"].clip(0),
        color=PALETTE[0], lw=2, label="Forecast")
ax.scatter(usa["date"], usa["new_deaths_7d"], s=4, color="#64748B",
           alpha=0.4, label="Actuals")
ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color="red", ls="--", lw=1.5, label="Train cutoff")
ax.axvline(usa["date"].max(), color="green", ls="--", lw=1.5, label="Forecast start")
ax.set_ylabel("Daily Deaths (7d avg)")
ax.set_title("Full Timeline  +  Future Projection")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

# Panel B – test period zoom
ax = axes[1]
zoom_start = pd.Timestamp(TRAIN_CUTOFF)
zoom_mask  = full_forecast["ds"] >= zoom_start
fc_zoom    = full_forecast[zoom_mask]
ax.fill_between(fc_zoom["ds"],
                fc_zoom["yhat_lower"].clip(0),
                fc_zoom["yhat_upper"].clip(0),
                alpha=0.3, color=PALETTE[0], label="95% CI")
ax.plot(fc_zoom["ds"], fc_zoom["yhat"].clip(0),
        color=PALETTE[0], lw=2.2, label="Forecast")
ax.scatter(y_true.index, y_true.values, s=20, color=PALETTE[1],
           alpha=0.8, zorder=5, label="Actuals (test)")
ax.set_xlabel("Date"); ax.set_ylabel("Daily Deaths")
ax.set_title(f"Test Period Zoom  |  MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_prophet_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig3_prophet_forecast.png")


# ─────────────────────────────────────────────
# FIGURE 4 – Prophet Components
# ─────────────────────────────────────────────
fig = model.plot_components(full_forecast)
fig.set_size_inches(14, 10)
fig.suptitle(f"Prophet Components  |  {PROPHET_COUNTRY}", fontsize=13,
             fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_prophet_components.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig4_prophet_components.png")


# ─────────────────────────────────────────────
# MODEL 2 – Elastic Net (Global)
# ─────────────────────────────────────────────
print("\n[4/6] Training Elastic Net model …")

FEATURE_COLS = [
    "new_cases_7d", "new_recovered_7d", "cfr_7d",
    "cases_lag7",   "deaths_lag7",      "recovered_lag7",
    "cases_lag14",  "deaths_lag14",     "recovered_lag14",
    "cases_per_million", "deaths_per_million",
]

global_df = df_raw[df_raw["date"] < TRAIN_CUTOFF].dropna(subset=FEATURE_COLS + ["new_deaths_7d"]).copy()

X = global_df[FEATURE_COLS].values
y = global_df["new_deaths_7d"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tscv = TimeSeriesSplit(n_splits=5)
enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    alphas=np.logspace(-4, 2, 30),
    cv=tscv,
    max_iter=5000,
    random_state=RANDOM_STATE,
)
enet.fit(X_scaled, y)

print(f"   Best l1_ratio={enet.l1_ratio_:.2f}  alpha={enet.alpha_:.5f}")

# Evaluate on test set (after cutoff)
global_test = df_raw[df_raw["date"] >= TRAIN_CUTOFF].dropna(subset=FEATURE_COLS + ["new_deaths_7d"])
X_test  = scaler.transform(global_test[FEATURE_COLS].values)
y_test  = global_test["new_deaths_7d"].values
y_pred_enet = enet.predict(X_test).clip(min=0)

mae_e  = mean_absolute_error(y_test, y_pred_enet)
rmse_e = np.sqrt(mean_squared_error(y_test, y_pred_enet))
r2_e   = r2_score(y_test, y_pred_enet)
print(f"   Elastic Net  →  MAE={mae_e:.1f}  RMSE={rmse_e:.1f}  R²={r2_e:.4f}")


# ─────────────────────────────────────────────
# FIGURE 5 – Elastic Net Analysis
# ─────────────────────────────────────────────
print("\n[5/6] Plotting Elastic Net analysis …")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Elastic Net Regression  |  Global Death Forecasting",
             fontsize=14, fontweight="bold")

# Panel A – Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, y_pred_enet, alpha=0.3, s=10, color=PALETTE[4])
lim = max(y_test.max(), y_pred_enet.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="red", lw=2, label="Perfect fit")
ax.set_xlabel("Actual Deaths"); ax.set_ylabel("Predicted Deaths")
ax.set_title(f"A  |  Actual vs Predicted\n(R²={r2_e:.4f})")
ax.legend()

# Panel B – Feature Importance (coefficients)
ax = axes[1]
coefs = pd.Series(enet.coef_, index=FEATURE_COLS).sort_values()
colors = [PALETTE[1] if c < 0 else PALETTE[0] for c in coefs]
coefs.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("B  |  Feature Coefficients")
ax.set_xlabel("Coefficient Value")

# Panel C – Residuals
ax = axes[2]
residuals = y_test - y_pred_enet
ax.scatter(y_pred_enet, residuals, alpha=0.3, s=10, color=PALETTE[2])
ax.axhline(0, color="red", lw=1.5, ls="--")
ax.set_xlabel("Predicted Deaths"); ax.set_ylabel("Residuals")
ax.set_title(f"C  |  Residual Plot\nMAE={mae_e:.1f}  RMSE={rmse_e:.1f}")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_elasticnet_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig5_elasticnet_analysis.png")


# ─────────────────────────────────────────────
# FIGURE 6 – Model Comparison Dashboard
# ─────────────────────────────────────────────
print("\n[6/6] Building comparison dashboard …")

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Model Comparison Dashboard  |  COVID Death Forecasting",
             fontsize=15, fontweight="bold")

# Top row – metric comparison
ax1 = fig.add_subplot(gs[0, 0])
models = ["Prophet\n(National)", "Elastic Net\n(Global)"]
maes   = [mae, mae_e]
bars   = ax1.bar(models, maes, color=[PALETTE[0], PALETTE[4]], width=0.5, edgecolor="white")
for bar, v in zip(bars, maes):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", fontweight="bold")
ax1.set_title("Mean Absolute Error"); ax1.set_ylabel("MAE")

ax2 = fig.add_subplot(gs[0, 1])
rmses = [rmse, rmse_e]
bars  = ax2.bar(models, rmses, color=[PALETTE[0], PALETTE[4]], width=0.5, edgecolor="white")
for bar, v in zip(bars, rmses):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", fontweight="bold")
ax2.set_title("Root Mean Squared Error"); ax2.set_ylabel("RMSE")

ax3 = fig.add_subplot(gs[0, 2])
r2s  = [r2, r2_e]
bars = ax3.bar(models, r2s, color=[PALETTE[0], PALETTE[4]], width=0.5, edgecolor="white")
for bar, v in zip(bars, r2s):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{v:.4f}", ha="center", fontweight="bold")
ax3.set_title("R² Score"); ax3.set_ylabel("R²")
ax3.set_ylim(0, 1.1)

# Bottom row – per-country Elastic Net performance
ax4 = fig.add_subplot(gs[1, :])
country_metrics = []
for country in df_raw["country"].unique():
    sub = df_raw[(df_raw["country"] == country) & (df_raw["date"] >= TRAIN_CUTOFF)]
    sub = sub.dropna(subset=FEATURE_COLS + ["new_deaths_7d"])
    if len(sub) < 10:
        continue
    Xc   = scaler.transform(sub[FEATURE_COLS].values)
    yc   = sub["new_deaths_7d"].values
    ypred = enet.predict(Xc).clip(0)
    r2c   = r2_score(yc, ypred)
    country_metrics.append({"country": country, "R2": r2c,
                             "MAE": mean_absolute_error(yc, ypred)})

cm_df = pd.DataFrame(country_metrics).sort_values("R2", ascending=True)
colors_cm = [PALETTE[1] if r < 0.5 else PALETTE[2] if r < 0.75 else PALETTE[0]
             for r in cm_df["R2"]]
bars = ax4.barh(cm_df["country"], cm_df["R2"], color=colors_cm)
for bar, r2v in zip(bars, cm_df["R2"]):
    ax4.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{r2v:.4f}", va="center", fontsize=9)
ax4.axvline(0.75, color="orange", ls="--", lw=1.5, label="R²=0.75 threshold")
ax4.axvline(0.5,  color="red",    ls="--", lw=1.5, label="R²=0.50 threshold")
ax4.set_xlabel("R² Score"); ax4.set_title("Elastic Net – Per-Country R² on Test Set")
ax4.legend(); ax4.set_xlim(0, 1.1)

fig.savefig(FIGURES_DIR / "fig6_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✔ fig6_model_comparison.png")


# ─────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'MODEL':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 52)
print(f"{'Prophet (National)':<25} {mae:>8.1f} {rmse:>8.1f} {r2:>8.4f}")
print(f"{'Elastic Net (Global)':<25} {mae_e:>8.1f} {rmse_e:>8.1f} {r2_e:>8.4f}")
print("=" * 60)
print(f"\nFigures saved → {FIGURES_DIR.resolve()}")
print("\nDone ✔")
