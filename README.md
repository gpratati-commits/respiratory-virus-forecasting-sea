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

8. Assess the robustness of forecasting conclusions across countries, pathogens, forecast horizons, and alternative evaluation conditions.

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


## Forecasting models

### Epidemiological baselines

Baseline forecasts were used as reference models against which the more complex statistical and machine-learning approaches were evaluated. These included persistence, recent moving-average, and seasonal-naive forecasts.

Baseline forecasts were generated using leakage-safe features, with MASE scaling estimated from training data only. Missing baseline predictions were retained as missing rather than replaced with zero.

### Regularized Negative Binomial model

An interpretable Negative Binomial generalized linear model was used for count forecasting.

Three nested predictor sets were evaluated:

- **M1:** target-pathogen activity at lags of 1, 2, 3, 4, and 8 weeks plus a 4-week rolling activity feature.
- **M2:** M1 plus annual seasonal sine and cosine terms.
- **M3:** M2 plus historical activity of the other available respiratory pathogens at lags of 1, 2, and 4 weeks and a 4-week rolling activity feature.

The Negative Binomial model used an NB2 variance parameterization. L2 ridge regularization was applied, with penalty strength selected using validation data only. The final model was refitted using the combined training and validation data before evaluation on the held-out test period.

### XGBoost

Gradient-boosted regression trees were used as a nonlinear machine-learning forecasting approach.

Six prespecified hyperparameter configurations were compared using validation-set performance. A maximum of 2000 boosting rounds was allowed, with early stopping after 100 rounds without validation improvement.

The test set was not used for hyperparameter selection or early stopping. After model selection, the final XGBoost model was refitted using the combined training and validation data and evaluated on the held-out test set.

### Ensemble forecasting

Negative Binomial and XGBoost forecasts were combined using a convex weighted ensemble.

Candidate Negative Binomial weights ranged from 0.0 to 1.0 in increments of 0.1, with the corresponding XGBoost weight equal to one minus the Negative Binomial weight.

Weights were selected using validation macro scaled mean absolute error. Validation selected an equal-weight combination for both M2 and M3:

- Negative Binomial weight = 0.5
- XGBoost weight = 0.5

The held-out test set was reserved for final evaluation and was not used to select ensemble weights.

## Probabilistic forecasting

Predictive uncertainty was evaluated in addition to point forecasting performance.

Prediction intervals were assessed using interval coverage and Weighted Interval Score (WIS). Calibration procedures were based on development data, while the test set remained reserved for final evaluation.

## Forecast evaluation

Forecast performance was evaluated separately by:

- pathogen
- country
- forecast horizon
- model
- predictor set

Evaluation metrics included:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Scaled Error (MASE)
- prediction-interval coverage
- Weighted Interval Score (WIS)

The principal M2-versus-M3 comparison assessed whether historical activity of other respiratory pathogens added predictive information beyond target-pathogen history and seasonality.

## Machine-learning interpretation

SHAP-based interpretation was used to examine the contribution of predictors to XGBoost forecasts.

Predictors assessed included recent target-pathogen activity, seasonal terms, and historical activity of co-circulating respiratory pathogens.

SHAP values were interpreted as measures of predictive contribution rather than evidence of causal relationships between pathogens.

## Sensitivity and robustness analyses

Sensitivity and robustness analyses were used to examine whether conclusions depended strongly on analytical choices and data availability.

These analyses considered factors including surveillance completeness, temporal availability, forecast horizons, pathogen-specific data availability, and alternative evaluation conditions.

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
├
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── predictions/
│   └── models/
│
├── manuscript/
│   └── manuscript and research materials
│
├── requirements.txt
├── .gitignore
└── README.md