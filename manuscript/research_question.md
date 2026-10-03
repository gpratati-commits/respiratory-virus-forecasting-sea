# Frozen Research Question

## Working title

**Does Multi-Pathogen Surveillance Improve Short-Term Respiratory Virus Forecasting? A Comparative Study of Influenza, RSV and SARS-CoV-2 in Southeast Asia**

## Primary research question

Does incorporating seasonality and recent activity of co-circulating respiratory pathogens improve 1-, 2-, and 4-week-ahead forecasts of influenza, RSV, and SARS-CoV-2 beyond forecasts based only on the target pathogen's own recent history?

## Primary objective

To develop and evaluate leakage-safe short-term forecasting models for influenza, RSV, and SARS-CoV-2 using real-world respiratory-virus surveillance data from Southeast Asia.

## Main hypothesis

Models incorporating seasonal information and recent activity of co-circulating respiratory pathogens may provide additional predictive information beyond the target pathogen's own recent history.

The hypothesis will be evaluated using out-of-sample forecast performance. The analysis will not assume beforehand that multi-pathogen models perform better.

## Forecast horizons

The primary forecast horizons are:

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead

## Main modelling experiment

Four levels of forecasting information will be compared.

### M0 — Baseline

Simple epidemiological forecasts such as persistence, recent moving average, and seasonal-naive forecasts.

### M1 — Target-pathogen history

Uses only previous observations of the pathogen being forecast.

Example predictors:

- lag 1
- lag 2
- lag 3
- lag 4
- lag 8
- recent rolling averages

### M2 — Target-pathogen history + seasonality

Adds seasonal information such as cyclic epidemiological-week predictors.

### M3 — Target-pathogen history + seasonality + co-circulating pathogens

Adds recent activity of other respiratory pathogens.

## Primary scientific comparison

The key comparison is:

**M2 versus M3**

This comparison will test whether information from co-circulating respiratory pathogens improves forecasting after accounting for the target pathogen's own recent history and seasonality.

## Pathogens

The primary pathogens are:

- Influenza
- Respiratory syncytial virus (RSV)
- SARS-CoV-2

Analyses will only be performed where pathogen-specific surveillance quality is sufficient.

## Geographic scope

The primary analysis includes Southeast Asian countries with sufficiently complete respiratory-virus surveillance.

Singapore will receive a dedicated secondary analysis using pathogens for which sufficiently complete surveillance data are available.

## Singapore analysis

The current WHO surveillance extract provides useful influenza and SARS-CoV-2 data for Singapore but does not provide a sufficiently complete Singapore RSV series.

Missing RSV surveillance will therefore not be interpreted as zero RSV activity.

The Singapore-specific analysis will focus primarily on influenza and SARS-CoV-2.

## Data sources

- WHO influenza surveillance
- WHO RSV surveillance
- WHO SARS-CoV-2 surveillance

## Forecasting principles

1. Training, validation, and test periods will be chronological.
2. Random train-test splitting will not be used for the primary forecasting analysis.
3. Future observations will never be used to construct predictors for earlier forecasts.
4. Hyperparameter tuning will use training and validation data only.
5. The final test period will remain untouched during model development.
6. Missing surveillance observations will not automatically be converted to zero.
7. Cross-pathogen correlations will be interpreted as temporal associations rather than causal effects.
8. Raw surveillance counts will not be assumed to represent directly comparable population incidence across countries.

## Planned model families

- epidemiological baselines
- Generalised Additive Model and/or Negative Binomial regression
- XGBoost
- Bayesian hierarchical model
- probabilistic forecasting
- ensemble forecasting

## Forecast evaluation

Forecast performance will be evaluated separately by:

- pathogen
- country
- forecast horizon
- model
- information set

Candidate metrics include:

- MAE
- RMSE
- MASE
- prediction interval coverage
- Weighted Interval Score

## Scope decision

Climate variables are not part of the primary project.

Previous ERA5 work is treated as an exploratory extension and is not required to answer the primary research question.

## Research-question freeze

The primary research question, forecast horizons, and central M2-versus-M3 comparison should remain unchanged during the primary analysis unless a major data limitation makes the planned analysis impossible.

Any substantial change should be documented before the affected analysis is performed.