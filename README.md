# COVID-19 Death Forecasting Model

> **Two-model ensemble for national and global mortality prediction using Facebook Prophet with external regressors and Elastic Net regression.**

-----

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data & Feature Engineering](#data--feature-engineering)
  - [Model 1 — Facebook Prophet (National)](#model-1--facebook-prophet-national)
  - [Model 2 — Elastic Net Regression (Global)](#model-2--elastic-net-regression-global)
- [Visualisations](#visualisations)
- [Installation & Quick Start](#installation--quick-start)
- [Configuration](#configuration)
- [Technical Stack](#technical-stack)
- [Design Decisions](#design-decisions)

-----

## Project Overview

This project implements a dual-model framework for forecasting COVID-19 daily death counts, operating at two complementary levels of granularity:

|Scope       |Model                 |Key Feature                                                                  |
|------------|----------------------|-----------------------------------------------------------------------------|
|**National**|Facebook Prophet      |External regressors (confirmed cases, recoveries) + seasonality decomposition|
|**Global**  |Elastic Net Regression|Cross-country generalisation with lag features and CFR-derived signals       |

The pipeline spans the full data science lifecycle — from raw ingestion and feature engineering through model training, evaluation, and a six-panel visualisation suite — and is designed to be reproducible via a single configuration block at the top of the script.

**Dataset:** 10 countries × 1,277 days (2020-01-01 → 2023-06-30) = 12,770 rows, with daily new cases, new deaths, new recoveries, and population figures.

-----

## Key Results

|Model                             |MAE|RMSE|R²        |
|----------------------------------|---|----|----------|
|Facebook Prophet (US, national)   |3.3|5.7 |**0.9952**|
|Elastic Net (global, 10 countries)|1.0|2.5 |**0.9983**|

Both models were evaluated strictly on a hold-out test set (post `2022-06-01`) with no information leakage from future periods. The Elastic Net used `TimeSeriesSplit` cross-validation during hyperparameter search to respect temporal ordering.

-----

## Project Structure

```
covid_forecasting/
│
├── covid_forecasting.py      # Main pipeline — data prep, both models, all figures
├── covid_data.csv            # Dataset (10 countries, 2020–2023)
├── README.md
│
└── figures/
    ├── fig1_eda_overview.png         # Global EDA: deaths, trends, CFR heatmap, scatter
    ├── fig2_national_trend.png       # USA deep-dive: cases / deaths / CFR timeline
    ├── fig3_prophet_forecast.png     # Prophet forecast + test-period zoom
    ├── fig4_prophet_components.png   # Prophet trend & seasonality decomposition
    ├── fig5_elasticnet_analysis.png  # Actual vs predicted, feature coefficients, residuals
    └── fig6_model_comparison.png     # Side-by-side metrics + per-country R² scores
```

-----

## Methodology

### Data & Feature Engineering

Raw daily counts are processed through the following pipeline before reaching either model:

**Smoothing**

- 7-day rolling average applied to `new_cases`, `new_deaths`, and `new_recovered` per country, eliminating weekend reporting artifacts and day-of-week bias.

**Lag Features** (for Elastic Net)

- Cases, deaths, and recoveries lagged by **7 days** and **14 days** to capture the biological delay between infection, symptom onset, hospitalisation, and death.

**Derived Signals**

- `cfr_7d` — Case Fatality Rate computed on smoothed 7-day values to avoid division noise
- `cases_per_million`, `deaths_per_million` — population-normalised rates enabling cross-country comparability

**Train / Test Split**

- Hard cutoff at `2022-06-01`; all models trained on data before this date and evaluated on data from this date onward
- No shuffling; temporal order is strictly preserved

-----

### Model 1 — Facebook Prophet (National)

**Target:** `new_deaths_7d` for the United States

**Architecture:**

```
Prophet(
    changepoint_prior_scale  = 0.05,   # Conservative — avoids overfitting pandemic waves
    seasonality_prior_scale  = 10,     # Allow seasonality to flex with epidemic dynamics
    seasonality_mode         = "multiplicative",
    yearly_seasonality       = True,
    weekly_seasonality       = True,
    daily_seasonality        = False,
    interval_width           = 0.95,   # 95% credible intervals
)
```

**External Regressors** (both standardised internally by Prophet):

- `new_cases_7d` — confirmed case count leads deaths by ~1–2 weeks, making it a strong leading indicator
- `new_recovered_7d` — recovery signal helps the model distinguish wave severity from wave breadth

**Why Prophet for national-level?**
Prophet was chosen here because it natively handles: (1) multiple seasonality components, (2) structural breakpoints without manual specification (changepoint detection), and (3) the ability to inject domain knowledge through external regressors — all critical properties for pandemic time series with multiple distinct waves.

**Forecasting:**

- The trained model is evaluated on the test hold-out and also projected **60 days into the future** beyond the data range, using the last observed case and recovery values as static future regressor inputs.

-----

### Model 2 — Elastic Net Regression (Global)

**Target:** `new_deaths_7d` across all 10 countries simultaneously

**Feature set (11 variables):**

|Feature                                         |Rationale                                                |
|------------------------------------------------|---------------------------------------------------------|
|`new_cases_7d`                                  |Primary leading indicator for deaths                     |
|`new_recovered_7d`                              |Proxy for healthcare capacity and strain                 |
|`cfr_7d`                                        |Encodes variant lethality and healthcare system quality  |
|`cases_lag7`, `deaths_lag7`, `recovered_lag7`   |1-week delayed signals (incubation → hospitalisation lag)|
|`cases_lag14`, `deaths_lag14`, `recovered_lag14`|2-week delayed signals (hospitalisation → death lag)     |
|`cases_per_million`                             |Population-normalised spread intensity                   |
|`deaths_per_million`                            |Population-normalised mortality burden                   |

**Training:**

```python
ElasticNetCV(
    l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    alphas   = np.logspace(-4, 2, 30),
    cv       = TimeSeriesSplit(n_splits=5),
    max_iter = 5000,
)
```

Best hyperparameters found: `l1_ratio=0.90`, `alpha=0.00452` — indicating a near-Lasso solution, meaning the model aggressively zero-ed out weaker features and relied on a sparse signal subset.

**Why Elastic Net for global-level?**
The combination of L1 and L2 penalties allows the model to perform implicit feature selection (dropping irrelevant lags) while handling correlated predictors (e.g., `cases_lag7` and `cases_lag14` are highly correlated). This is preferable to plain Lasso when grouped correlated features should be kept together.

-----

## Visualisations

### Figure 1 — Global EDA Overview

Four-panel dashboard: total deaths by country (bar), daily 7-day average deaths for top-5 countries (line), case fatality rate heatmap by country × quarter (seaborn heatmap), and cases vs deaths log-scale scatter across all countries.

### Figure 2 — National Trend Deep-Dive (USA)

Three-panel stacked time series showing the full pandemic arc for the United States: daily new cases (7d avg), daily new deaths (7d avg), and rolling CFR (%) — all on a shared time axis to visualise wave dynamics.

### Figure 3 — Prophet Forecast

Two panels: (A) full training + test + future projection timeline with 95% credible interval shading; (B) zoomed test-period comparison of actual vs predicted deaths with metric annotation.

### Figure 4 — Prophet Components

Auto-generated decomposition from Prophet showing the extracted trend, yearly seasonality, and weekly seasonality components — useful for interpreting what the model learned about pandemic periodicity.

### Figure 5 — Elastic Net Analysis

Three panels: (A) actual vs predicted scatter with perfect-fit diagonal; (B) signed feature coefficient bar chart showing directionality and relative importance; (C) residual plot against predicted values for homoscedasticity assessment.

### Figure 6 — Model Comparison Dashboard

Side-by-side bar charts for MAE, RMSE, and R² across both models, plus a horizontal bar chart of per-country R² scores on the Elastic Net test set, colour-coded by performance tier (red < 0.50, orange < 0.75, blue ≥ 0.75).

-----

## Installation & Quick Start

**Requirements:** Python 3.9+

```bash
# 1. Clone / download the project
cd covid_forecasting

# 2. Install dependencies
pip install prophet scikit-learn pandas numpy matplotlib seaborn

# 3. Run the full pipeline
python covid_forecasting.py
```

All six figures will be saved to `figures/`. Console output reports dataset statistics, model hyperparameters, and a final metrics summary table.

**Expected runtime:** ~2–4 minutes (Prophet sampling is the bottleneck).

-----

## Configuration

All tunable parameters are centralised at the top of `covid_forecasting.py`:

```python
PROPHET_COUNTRY  = "United States"   # Country for national Prophet model
FORECAST_DAYS    = 60                # Days to project beyond the data range
TRAIN_CUTOFF     = "2022-06-01"      # Hard train/test split date
RANDOM_STATE     = 42                # Reproducibility seed
```

To switch the Prophet analysis to a different country, change `PROPHET_COUNTRY` to any of: `India`, `Brazil`, `United Kingdom`, `France`, `Germany`, `Italy`, `Russia`, `Spain`, `Mexico`.

-----

## Technical Stack

|Category               |Library / Tool                                                |
|-----------------------|--------------------------------------------------------------|
|Language               |Python 3.9+                                                   |
|Time Series Forecasting|`prophet` (Facebook/Meta)                                     |
|Machine Learning       |`scikit-learn` (ElasticNetCV, StandardScaler, TimeSeriesSplit)|
|Data Manipulation      |`pandas`, `numpy`                                             |
|Visualisation          |`matplotlib`, `seaborn`                                       |

-----

## Design Decisions

**7-day smoothing before modelling** — Raw daily counts contain systematic reporting drops on weekends and spikes on Mondays. Smoothing before feature creation (rather than as a post-processing step) ensures lag features also inherit the smoothed signal, preventing spurious correlations.

**Multiplicative seasonality in Prophet** — During peak waves, seasonal fluctuations scale with the death count magnitude rather than remaining constant. Multiplicative mode captures this more accurately than additive seasonality for pandemic data.

**`TimeSeriesSplit` over standard `KFold`** — Standard cross-validation shuffles data randomly, which for time series creates data leakage where the model sees future values during training folds. `TimeSeriesSplit` enforces that each validation fold is always temporally ahead of its training fold.

**Elastic Net over Ridge or Lasso** — With 11 features containing multiple correlated lag pairs, pure Lasso may arbitrarily drop one of two correlated features and produce unstable solutions. Elastic Net with a high L1 ratio achieves sparsity while handling the collinearity more gracefully.

**Population normalisation** — Including both raw and per-million features allows the global Elastic Net to serve both large-population countries (India, USA) and smaller ones (France, Spain) without the model systematically biasing toward high-count nations.
