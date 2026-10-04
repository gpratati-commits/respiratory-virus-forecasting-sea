# Frozen Forecasting Design

## Forecast horizons

The primary forecasting horizons are:

- 1 week ahead
- 2 weeks ahead
- 4 weeks ahead

## Primary regional forecasting cohort

The primary three-pathogen forecasting analysis includes:

- Indonesia
- Malaysia
- Brunei Darussalam

These countries provide sufficiently complete and consecutive overlapping surveillance for influenza, RSV, and SARS-CoV-2 and allow a common validation and test period.

### Indonesia

- Training: 2020–2022
- Validation: 2023
- Test: 2024

### Malaysia

- Training: 2020–2022
- Validation: 2023
- Test: 2024

### Brunei Darussalam

- Training: 2021–2022
- Validation: 2023
- Test: 2024

## Secondary analysis

### Philippines

- Training: 2020–2021
- Validation: 2022
- Test: 2023

The Philippines is analysed separately because sufficiently complete three-pathogen overlap does not continue into 2024.

### Thailand

- Training: 2023
- Validation: 2024
- Test: 2025

Thailand is treated as exploratory because only one year is available for model training under this design.

## Singapore case study

Singapore does not have sufficiently complete RSV observations in the current WHO dataset.

The Singapore case study therefore uses influenza and SARS-CoV-2 only.

- Training: 2020–2021
- Validation: 2022
- Test: 2023

Missing Singapore RSV observations will not be interpreted as zero RSV activity.

## Split rule

Forecasting splits will ultimately be assigned according to the target date rather than solely according to the forecast-origin date.

For example, if a 4-week-ahead forecast originates in December 2023 but targets January 2024, it belongs to the 2024 test period.

## Data-leakage rule

At forecast origin t, predictors may use information available at or before t.

Information from t+1 or later must never be used as a predictor.

Training and validation data may be used during model development.

The final test period must remain untouched until model development and tuning are complete.