# Publication Methods Audit

## Python analysis scripts

python/01_inspect_surveillance_data.py
python/02_assess_surveillance_quality.py
python/03_select_candidate_country_years.py
python/04_inspect_sarscov2_data.py
python/05_assess_sarscov2_quality.py
python/06_define_study_population.py
python/07_audit_flunet_weekly_sources.py
python/08_prepare_flunet_weekly.py
python/09_prepare_sarscov2_weekly.py
python/10_build_master_dataset.py
python/11_exploratory_analysis.py
python/12_seasonality_analysis.py
python/13_cross_pathogen_analysis.py
python/14_define_forecast_design.py
python/15_create_forecasting_features.py
python/16_baseline_forecasting.py
python/17_statistical_forecasting.py
python/18_xgboost_forecasting.py
python/19_xgboost_interpretation.py
python/20_probabilistic_forecasting.py
python/21_uncertainty_calibration_sensitivity.py
python/22_ensemble_forecasting.py
python/23_overall_model_comparison.py
python/24_robustness_synthesis.py
python/25_singapore_case_study.py
python/26_final_figures.py
python/27_finalize_manuscript_figures.py
python/28_manuscript_results_summary.py

## Forecast evaluation and leakage prevention

### XGBoost

Verified directly from `python/18_xgboost_forecasting.py`:

- Hyperparameters were selected using the validation set only.
- Early stopping used the validation set only.
- The test set was not used for hyperparameter tuning.
- After model selection, final models were refitted using the combined training and validation data.
- The test set was reserved for final performance evaluation.


### Baseline forecasting

Verified directly from `python/16_baseline_forecasting.py`:

- Test data were not used to tune any baseline forecast.
- MASE scaling was calculated using training data only.
- The seasonal-naive baseline used the observation exactly 52 weeks before the target date.
- Missing predictions were retained as missing and were not replaced with zero.


## Chronological forecast evaluation design

Forecast splits were assigned according to the forecast target date, with both the start and end dates of each interval included.

The verified analysis-specific periods were:

- Brunei Darussalam, primary regional analysis:
  training = 2021-01-01 to 2022-12-31;
  validation = 2023-01-01 to 2023-12-31;
  test = 2024-01-01 to 2024-12-31.

- Indonesia, primary regional analysis:
  training = 2020-01-01 to 2022-12-31;
  validation = 2023-01-01 to 2023-12-31;
  test = 2024-01-01 to 2024-12-31.

- Malaysia, primary regional analysis:
  training = 2020-01-01 to 2022-12-31;
  validation = 2023-01-01 to 2023-12-31;
  test = 2024-01-01 to 2024-12-31.

- Philippines, secondary regional analysis:
  training = 2020-01-01 to 2021-12-31;
  validation = 2022-01-01 to 2022-12-31;
  test = 2023-01-01 to 2023-12-31.

- Thailand, exploratory regional analysis:
  training = 2023-01-01 to 2023-12-31;
  validation = 2024-01-01 to 2024-12-31;
  test = 2025-01-01 to 2025-12-31.

- Singapore, secondary two-pathogen case study:
  training = 2020-01-01 to 2021-12-31;
  validation = 2022-01-01 to 2022-12-31;
  test = 2023-01-01 to 2023-12-31.

The validation period was used for model selection and tuning where applicable, while the test period was reserved for final out-of-sample evaluation. For XGBoost, hyperparameter selection and early stopping used validation data only, and final models were refitted on the combined training and validation data before test-set evaluation.


### Regularized statistical forecasting

Verified directly from `python/17_statistical_forecasting.py`:

- The statistical forecasting models used regularization.
- Ridge-penalty strength was selected using validation data only.
- The test set was never used to select the ridge penalty.
- Positive M3-versus-M2 improvement indicates lower forecast error for M3.
- Predictive improvement from cross-pathogen information was interpreted as a forecasting association and not as evidence of causal interaction between pathogens.


### Negative Binomial dispersion

The statistical forecasting models used a Negative Binomial GLM with an NB2 variance parameterization.

The dispersion parameter was estimated from the modelling outcome using a method-of-moments estimator:

Var(Y) = mu + alpha * mu^2

and therefore:

alpha = (Var(Y) - mu) / mu^2

where the sample variance was calculated with one degree of freedom (`ddof=1`).

For numerical stability, the dispersion parameter was constrained to a minimum value of 1e-6. A value of 1e-6 was also used when fewer than two usable observations were available, when the outcome mean was non-finite or non-positive, or when the calculated dispersion estimate was non-finite.


### Ridge-penalty tuning

The Negative Binomial GLMs used pure L2 (ridge) regularization.

The candidate ridge penalties were:

1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, and 1000.

The ridge strength was selected using validation-set performance only. The test set was not used for penalty selection.

The intercept was excluded from penalization.


### Predictor-set definitions: M1, M2 and M3

Three nested predictor sets were evaluated.

M1 represented target-pathogen history. It included the target pathogen's activity at lags of 1, 2, 3, 4 and 8 weeks, together with a 4-week rolling activity feature.

M2 extended M1 by adding annual seasonal terms represented by sine and cosine transformations of calendar time (`season_sin` and `season_cos`).

M3 extended M2 by incorporating contemporaneously available historical information from the other respiratory pathogen(s). For each other pathogen, activity at lags of 1, 2 and 4 weeks and a 4-week rolling activity feature were included.

The M2-versus-M3 comparison therefore evaluated whether adding historical activity of other respiratory pathogens improved forecasting beyond target-pathogen history and seasonality alone. Any predictive improvement was interpreted as forecasting association rather than evidence of causal interaction between pathogens.


### XGBoost forecasting

XGBoost models were fitted using gradient-boosted regression trees.

Six candidate hyperparameter configurations were evaluated:

- A: max_depth = 2, eta = 0.03, min_child_weight = 1, reg_lambda = 1
- B: max_depth = 2, eta = 0.05, min_child_weight = 1, reg_lambda = 1
- C: max_depth = 2, eta = 0.05, min_child_weight = 5, reg_lambda = 10
- D: max_depth = 3, eta = 0.03, min_child_weight = 1, reg_lambda = 1
- E: max_depth = 3, eta = 0.05, min_child_weight = 1, reg_lambda = 10
- F: max_depth = 3, eta = 0.05, min_child_weight = 5, reg_lambda = 10

Fixed XGBoost settings were:

- objective = reg:squarederror
- evaluation metric = RMSE
- tree method = hist
- L1 regularization (reg_alpha) = 0
- subsample = 0.8
- colsample_bytree = 0.8
- a fixed random seed was used
- nthread = 4

A maximum of 2000 boosting rounds was allowed during model development, with early stopping after 100 rounds without validation improvement.

Hyperparameter configuration and boosting duration were selected using validation data only. The test set was not used for tuning or early stopping.

After model selection, the final XGBoost model was refitted using the combined training and validation data with the selected hyperparameters and selected number of boosting rounds, and was then evaluated on the held-out test data.


### Ensemble forecasting

Negative Binomial and XGBoost point forecasts were combined using a convex weighted ensemble:

ensemble prediction = w_NB × NB prediction + w_XGB × XGBoost prediction,

where w_XGB = 1 - w_NB.

Candidate Negative Binomial weights ranged from 0.0 to 1.0 in increments of 0.1. Thus, a weight of 0 represented an XGBoost-only forecast, 1 represented a Negative-Binomial-only forecast, and 0.5 represented an equal-weight ensemble.

Ensemble weights were tuned using validation data only. Candidate weights were ranked primarily by validation macro scaled mean absolute error (`validation_macro_scaled_mae`). The scaling quantity was derived from training data only. If candidate weights were exactly tied on the primary criterion, the weight closest to 0.5 was preferred; if a further tie remained, the lower Negative Binomial weight was selected.

Validation selected equal weights for both predictor sets evaluated in the ensemble:
- M2: Negative Binomial weight = 0.5 and XGBoost weight = 0.5.
- M3: Negative Binomial weight = 0.5 and XGBoost weight = 0.5.

The held-out test set was not used for ensemble-weight selection and was reserved for final performance evaluation.

For Indonesia RSV, the training-only MASE scaling quantity was undefined, so that series was excluded from the MASE-scaled ensemble-weight tuning criterion. Its held-out test forecasts were nevertheless retained for unscaled MAE and RMSE evaluation.
