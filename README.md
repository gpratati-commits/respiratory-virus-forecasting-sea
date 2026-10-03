# Multi-Pathogen Respiratory Virus Forecasting in Southeast Asia

## Project overview

This independent research project investigates the epidemiological dynamics and short-term forecasting of major respiratory pathogens in Southeast Asia using real-world surveillance data.

The project focuses on:

- Influenza
- Respiratory syncytial virus (RSV)
- SARS-CoV-2

The central aim is to determine whether information on seasonality and recent activity of co-circulating respiratory pathogens improves short-term forecasts beyond models that use only the recent history of the target pathogen.

The project combines infectious-disease epidemiology, statistical modelling, machine learning, time-series forecasting, and reproducible computational research using Python and R.


## Research question

### Primary research question

**Do seasonal patterns and recent activity of co-circulating respiratory pathogens improve 1-, 2-, and 4-week-ahead forecasts of influenza, RSV, and SARS-CoV-2 beyond forecasts based only on the target pathogen's own recent history?**

The primary project therefore focuses on respiratory-virus surveillance signals rather than climatic predictors.


## Study objectives

### Primary objective

To develop and evaluate leakage-safe short-term forecasting models for influenza, RSV, and SARS-CoV-2 using real-world respiratory-virus surveillance data from Southeast Asia.

### Secondary objectives

1. Characterise the temporal and seasonal patterns of influenza, RSV, and SARS-CoV-2 across selected Southeast Asian settings.

2. Assess the completeness and quality of pathogen surveillance before including country-years in the analysis.

3. Investigate descriptive lagged temporal associations between co-circulating respiratory pathogens.

4. Develop 1-, 2-, and 4-week-ahead pathogen-specific forecasts.

5. Compare simple epidemiological forecasting baselines with statistical and machine-learning models.

6. Determine whether the addition of co-circulating pathogen information improves out-of-sample forecasting performance.

7. Develop probabilistic forecasts that quantify predictive uncertainty.

8. Explore whether hierarchical modelling can borrow information across countries while retaining country-specific epidemic characteristics.

9. Perform a dedicated Singapore analysis using pathogens for which sufficiently complete surveillance data are available.


## Main hypothesis

Models incorporating seasonal information and recent activity of co-circulating respiratory pathogens may provide additional predictive information beyond the target pathogen's own recent history.

This hypothesis will be evaluated using out-of-sample forecast performance rather than assuming in advance that multi-pathogen models are superior.


## Real-world data sources

### Influenza and RSV

Weekly influenza and RSV surveillance data are obtained from the World Health Organization respiratory-virus surveillance system.

The data include country-level epidemiological-week information and pathogen-specific surveillance variables.

Raw surveillance files are stored locally and are not distributed through this repository.


### SARS-CoV-2

Weekly SARS-CoV-2 surveillance data are obtained from publicly available World Health Organization COVID-19 surveillance data.

Because COVID-19 testing and reporting practices changed substantially over time, country-year reporting completeness is assessed before data are included in forecasting analyses.


## Geographic scope

The project evaluates Southeast Asian settings for which sufficiently complete surveillance data are available.

Candidate settings include:

- Singapore
- Malaysia
- Philippines
- Indonesia
- Thailand
- Viet Nam
- Brunei Darussalam
- Cambodia
- Myanmar
- Lao People's Democratic Republic

Final inclusion is determined by surveillance quality rather than by country preference alone.


## Singapore analysis

Singapore is retained as an important country-specific component of the project.

The available WHO surveillance extract provides useful influenza and SARS-CoV-2 observations for selected periods but does not provide a sufficiently complete RSV series for the primary Singapore analysis.

Therefore, missing Singapore RSV surveillance observations are not interpreted as zero RSV activity.

The Singapore analysis focuses primarily on influenza and SARS-CoV-2.


## Surveillance quality assessment

Before modelling, surveillance quality is assessed separately by country, year, and pathogen.

Quality-control procedures include:

- number of expected epidemiological weeks
- number of reported weeks
- missing observations
- duplicate country-week records
- multiple surveillance-source records
- pathogen-specific reporting completeness
- testing denominator availability
- negative SARS-CoV-2 reporting revisions
- identification of incomplete surveillance years

Missing surveillance values are retained as missing and are not automatically converted to zero.


## Study population definition

Country-years are classified separately for:

- influenza eligibility
- RSV eligibility
- SARS-CoV-2 eligibility
- influenza-RSV overlap
- influenza-SARS-CoV-2 overlap
- three-pathogen overlap

This allows pathogen-specific analyses to use appropriate surveillance periods without forcing every country and pathogen to share the same study period.


## Exploratory epidemiological analysis

The exploratory analysis examines:

- weekly pathogen activity
- surveillance coverage
- epidemic waves
- country-specific temporal patterns
- pandemic-era disruptions
- missingness patterns
- annual pathogen activity

Raw surveillance counts are not interpreted as directly comparable estimates of population incidence across countries because testing and surveillance intensity may differ.


## Seasonality analysis

Seasonality is investigated using epidemiological week.

Within-country annual activity is normalised before estimating seasonal profiles so that differences in surveillance intensity between countries do not dominate the analysis.

Seasonality outputs include:

- weekly seasonal profiles
- estimated peak epidemiological weeks
- country-specific respiratory-virus patterns
- Singapore-specific influenza and SARS-CoV-2 profiles


## Cross-pathogen analysis

Lagged temporal relationships are investigated between:

- influenza and RSV
- influenza and SARS-CoV-2
- RSV and SARS-CoV-2

Associations are evaluated over multiple weekly lags.

These relationships are interpreted as descriptive temporal associations and not as evidence that one pathogen causes, suppresses, or promotes another pathogen.


## Forecasting horizons

The primary forecasting horizons are:

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead


## Forecasting experiment

The central modelling experiment compares progressively richer information sets.

### M0 — Simple baseline

Examples include:

- persistence forecast
- recent moving-average forecast
- seasonal-naive forecast


### M1 — Target-pathogen history

Uses recent activity of the pathogen being forecast.

Examples include:

- lag 1
- lag 2
- lag 3
- lag 4
- lag 8
- rolling averages


### M2 — Target-pathogen history + seasonality

Adds seasonal information to M1.

Seasonality may be represented using cyclic sine and cosine functions of epidemiological week.


### M3 — Target-pathogen history + seasonality + other pathogens

Adds recent activity of co-circulating respiratory pathogens.

The primary scientific comparison is:

**M2 versus M3**

This comparison tests whether multi-pathogen surveillance provides additional predictive value after accounting for the target pathogen's recent history and seasonal pattern.


## Leakage-safe forecasting design

All forecasting analyses use chronological data splitting.

The intended structure is:

Training period  
→ model estimation

Validation period  
→ feature selection, model tuning, and hyperparameter selection

Test period  
→ final untouched out-of-sample evaluation

Random train-test splitting is not used for the primary forecasting analysis.

Predictors are created using only information that would have been available at the forecast origin.


## Planned forecasting models

### Epidemiological baselines

Simple forecasting models provide the reference performance that more complex models must improve upon.


### Statistical model

An interpretable statistical model such as a Generalised Additive Model or Negative Binomial regression will evaluate nonlinear seasonality and lagged pathogen relationships.


### XGBoost

Gradient-boosted decision trees will be used to investigate whether nonlinear relationships and interactions improve predictive performance.


### Bayesian hierarchical model

A hierarchical model will allow countries to retain country-specific characteristics while borrowing information across the regional dataset.


### Ensemble forecasting

Predictions from complementary models may be combined to assess whether an ensemble improves forecast robustness.


## Probabilistic forecasting

Where feasible, the project will generate predictive intervals in addition to point forecasts.

Forecast uncertainty will be evaluated using appropriate probabilistic forecasting metrics such as interval coverage and Weighted Interval Score.


## Forecast evaluation

Forecast performance will be assessed separately by:

- pathogen
- country
- forecast horizon
- model
- information set

Candidate evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)
- prediction-interval coverage
- Weighted Interval Score (WIS)


## Machine-learning interpretation

SHAP-based interpretation will be used for XGBoost models to investigate which variables contribute most strongly to predictions.

Potential predictors include:

- recent target-pathogen activity
- seasonal variables
- recent influenza activity
- recent RSV activity
- recent SARS-CoV-2 activity

SHAP values will be interpreted as measures of predictive contribution rather than causal effects.


## Sensitivity analyses

Sensitivity analyses will assess whether conclusions depend strongly on analytical choices such as:

- surveillance completeness thresholds
- lag definitions
- training windows
- inclusion or exclusion of selected pandemic periods
- country selection
- alternative outcome definitions
- pathogen-specific surveillance availability


## Reproducibility principles

The project follows several reproducibility rules:

1. Raw source data are never manually modified.
2. Raw surveillance datasets are excluded from GitHub.
3. Data cleaning is performed using reproducible scripts.
4. Missing surveillance observations are not automatically interpreted as zero.
5. Train, validation, and test periods are chronological.
6. Forecast features are constructed without future-data leakage.
7. Hyperparameter tuning does not use the final test period.
8. Random seeds are fixed where appropriate.
9. Analysis outputs are generated programmatically.
10. Major analysis decisions are documented in the repository.


## Repository structure

```text
respiratory-virus-forecasting-sea/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── python/
│   ├── data inspection and quality-control scripts
│   ├── surveillance cleaning scripts
│   ├── exploratory analysis
│   ├── seasonality analysis
│   ├── cross-pathogen analysis
│   └── forecasting scripts
│
├── R/
│   └── statistical and Bayesian modelling scripts
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── predictions/
│   └── models/
│
├── manuscript/
│   └── manuscript and research notes
│
├── requirements.txt
├── .gitignore
└── README.md