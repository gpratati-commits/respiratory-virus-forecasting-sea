"""
Leakage-safe XGBoost forecasting for respiratory-virus surveillance.

Research comparison
-------------------
M1 = target-pathogen history
M2 = M1 + seasonality
M3 = M2 + co-circulating respiratory-pathogen activity

Design
------
1. Use only common M2/M3-eligible observations.
2. TRAIN fits candidate XGBoost models.
3. VALIDATION selects hyperparameters and boosting rounds.
4. TEST is never used for tuning.
5. Final model is refit on TRAIN + VALIDATION.
6. Final TEST predictions are generated once.
7. Targets are modelled on log1p scale for numerical stability.
8. Evaluation is performed on the original count scale.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "respiratory_forecasting_features.csv"
)

MASE_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "baseline_mase_scales.csv"
)

TABLE_DIR = ROOT / "outputs" / "tables"
PREDICTION_DIR = ROOT / "outputs" / "predictions"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

METRIC_FILE = (
    TABLE_DIR
    / "xgboost_metrics.csv"
)

COMPARISON_FILE = (
    TABLE_DIR
    / "xgboost_m2_m3_comparison.csv"
)

TUNING_FILE = (
    TABLE_DIR
    / "xgboost_tuning.csv"
)

DIAGNOSTIC_FILE = (
    TABLE_DIR
    / "xgboost_model_diagnostics.csv"
)

PREDICTION_FILE = (
    PREDICTION_DIR
    / "xgboost_predictions.csv"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 20261004

np.random.seed(RANDOM_SEED)


# ============================================================
# 3. MODEL FEATURE DEFINITIONS
# ============================================================

MODEL_FEATURE_COLUMN = {
    "M1": "m1_feature_names",
    "M2": "m2_feature_names",
    "M3": "m3_feature_names",
}


# ============================================================
# 4. PRESPECIFIED HYPERPARAMETER GRID
# ============================================================

# Small grid because each epidemiological time series contains
# relatively few weekly observations.

CANDIDATE_CONFIGS = [
    {
        "config_id": "A",
        "max_depth": 2,
        "eta": 0.03,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "config_id": "B",
        "max_depth": 2,
        "eta": 0.05,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "config_id": "C",
        "max_depth": 2,
        "eta": 0.05,
        "min_child_weight": 5,
        "reg_lambda": 10.0,
    },
    {
        "config_id": "D",
        "max_depth": 3,
        "eta": 0.03,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    {
        "config_id": "E",
        "max_depth": 3,
        "eta": 0.05,
        "min_child_weight": 1,
        "reg_lambda": 10.0,
    },
    {
        "config_id": "F",
        "max_depth": 3,
        "eta": 0.05,
        "min_child_weight": 5,
        "reg_lambda": 10.0,
    },
]

MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("LEAKAGE-SAFE XGBOOST FORECASTING")
print("=" * 90)

print("\nXGBoost version:", xgb.__version__)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting feature file not found:\n{FEATURE_FILE}"
    )

if not MASE_FILE.exists():
    raise FileNotFoundError(
        f"MASE scale file not found:\n{MASE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

mase_scales = pd.read_csv(MASE_FILE)

for column in [
    "forecast_origin",
    "target_date",
]:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
    )

if df["forecast_origin"].isna().any():
    raise ValueError(
        "Invalid forecast_origin dates."
    )

if df["target_date"].isna().any():
    raise ValueError(
        "Invalid target_date dates."
    )


# ============================================================
# 6. KEEP COMMON M2/M3 SAMPLE
# ============================================================

eligible = (
    df["common_m2_m3_eligible"]
    .astype(str)
    .str.lower()
    .eq("true")
)

df = df[eligible].copy()

if df.empty:
    raise ValueError(
        "No common M2/M3 eligible observations."
    )


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def parse_features(value):

    if pd.isna(value):
        return []

    return [
        feature
        for feature in str(value).split("|")
        if feature
    ]


def get_mase_scale(
    country,
    pathogen,
):

    x = mase_scales[
        (mase_scales["country"] == country)
        & (
            mase_scales["target_pathogen"]
            == pathogen
        )
    ]

    if x.empty:
        return np.nan

    return x.iloc[0][
        "mase_scale_training_only"
    ]


def make_matrix(
    data,
    feature_names,
    include_label=True,
):

    X = (
        data[feature_names]
        .astype(float)
        .copy()
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing XGBoost predictors found."
        )

    if include_label:

        y = (
            data["target_value"]
            .astype(float)
            .to_numpy()
        )

        if (y < 0).any():
            raise ValueError(
                "Negative target values found."
            )

        y_log = np.log1p(y)

        return xgb.DMatrix(
            X,
            label=y_log,
            feature_names=feature_names,
        )

    return xgb.DMatrix(
        X,
        feature_names=feature_names,
    )


def inverse_log_prediction(
    prediction_log,
):

    prediction_log = np.asarray(
        prediction_log,
        dtype=float,
    )

    if not np.isfinite(
        prediction_log
    ).all():
        raise ValueError(
            "Non-finite log-scale prediction."
        )

    prediction = np.expm1(
        prediction_log
    )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Non-finite count-scale prediction."
        )

    # Surveillance counts cannot be negative.
    return np.maximum(
        prediction,
        0.0,
    )


def calculate_metrics(
    observed,
    predicted,
    mase_scale,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    errors = (
        predicted
        - observed
    )

    this_mae = np.mean(
        np.abs(errors)
    )

    this_rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    mean_error = np.mean(
        errors
    )

    if (
        pd.isna(mase_scale)
        or mase_scale == 0
    ):
        this_mase = np.nan
    else:
        this_mase = (
            this_mae
            / mase_scale
        )

    return {
        "mae": this_mae,
        "rmse": this_rmse,
        "mase": this_mase,
        "mean_error": mean_error,
    }


def base_params(config):

    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": config[
            "max_depth"
        ],
        "eta": config[
            "eta"
        ],
        "min_child_weight": config[
            "min_child_weight"
        ],
        "reg_lambda": config[
            "reg_lambda"
        ],
        "reg_alpha": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED,
        "nthread": 4,
    }


# ============================================================
# 8. MODEL FITTING
# ============================================================

prediction_rows = []
metric_rows = []
tuning_rows = []
diagnostic_rows = []

group_columns = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
]

for group_values, group in df.groupby(
    group_columns,
    dropna=False,
):

    (
        analysis_group,
        country,
        pathogen,
        horizon,
    ) = group_values

    group = (
        group
        .sort_values("target_date")
        .copy()
    )

    train = group[
        group["split"] == "train"
    ].copy()

    validation = group[
        group["split"] == "validation"
    ].copy()

    test = group[
        group["split"] == "test"
    ].copy()

    train_validation = group[
        group["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()

    mase_scale = get_mase_scale(
        country,
        pathogen,
    )

    if (
        train.empty
        or validation.empty
        or test.empty
    ):

        diagnostic_rows.append({
            "analysis_group": analysis_group,
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "model": "ALL",
            "status": "missing_split",
            "message": (
                "Train, validation or test split is empty."
            ),
        })

        continue

    for (
        model_name,
        feature_column,
    ) in MODEL_FEATURE_COLUMN.items():

        definitions = (
            group[feature_column]
            .dropna()
            .unique()
        )

        if len(definitions) != 1:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": "feature_definition_error",
                "message": (
                    "Expected one feature definition."
                ),
            })

            continue

        feature_names = parse_features(
            definitions[0]
        )

        try:

            dtrain = make_matrix(
                train,
                feature_names,
                include_label=True,
            )

            dvalidation = make_matrix(
                validation,
                feature_names,
                include_label=True,
            )

        except Exception as exc:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": "matrix_creation_failed",
                "message": str(exc),
            })

            continue


        # ====================================================
        # 9. HYPERPARAMETER TUNING
        # ====================================================

        candidate_results = []

        for config in CANDIDATE_CONFIGS:

            try:

                params = base_params(
                    config
                )

                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=MAX_BOOST_ROUNDS,
                    evals=[
                        (
                            dvalidation,
                            "validation",
                        )
                    ],
                    early_stopping_rounds=(
                        EARLY_STOPPING_ROUNDS
                    ),
                    verbose_eval=False,
                )

                best_iteration = int(
                    booster.best_iteration
                )

                best_rounds = (
                    best_iteration + 1
                )

                validation_log_pred = (
                    booster.predict(
                        dvalidation,
                        iteration_range=(
                            0,
                            best_rounds,
                        ),
                    )
                )

                validation_pred = (
                    inverse_log_prediction(
                        validation_log_pred
                    )
                )

                validation_mae = (
                    np.mean(
                        np.abs(
                            validation_pred
                            - validation[
                                "target_value"
                            ].to_numpy(
                                dtype=float
                            )
                        )
                    )
                )

                candidate_results.append({
                    "config": config,
                    "validation_mae": (
                        validation_mae
                    ),
                    "best_rounds": (
                        best_rounds
                    ),
                    "booster": booster,
                })

                tuning_rows.append({
                    "analysis_group": analysis_group,
                    "country": country,
                    "target_pathogen": pathogen,
                    "forecast_horizon_weeks": horizon,
                    "model": model_name,
                    "config_id": config[
                        "config_id"
                    ],
                    "max_depth": config[
                        "max_depth"
                    ],
                    "eta": config[
                        "eta"
                    ],
                    "min_child_weight": config[
                        "min_child_weight"
                    ],
                    "reg_lambda": config[
                        "reg_lambda"
                    ],
                    "best_rounds": best_rounds,
                    "validation_mae": (
                        validation_mae
                    ),
                    "status": "success",
                    "message": "",
                })

            except Exception as exc:

                tuning_rows.append({
                    "analysis_group": analysis_group,
                    "country": country,
                    "target_pathogen": pathogen,
                    "forecast_horizon_weeks": horizon,
                    "model": model_name,
                    "config_id": config[
                        "config_id"
                    ],
                    "max_depth": config[
                        "max_depth"
                    ],
                    "eta": config[
                        "eta"
                    ],
                    "min_child_weight": config[
                        "min_child_weight"
                    ],
                    "reg_lambda": config[
                        "reg_lambda"
                    ],
                    "best_rounds": np.nan,
                    "validation_mae": np.nan,
                    "status": "failed",
                    "message": str(exc),
                })


        if not candidate_results:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": "all_configs_failed",
                "message": (
                    "No XGBoost configuration "
                    "produced a usable validation model."
                ),
            })

            continue


        # Lowest original-scale validation MAE wins.
        candidate_results.sort(
            key=lambda result: (
                result["validation_mae"],
                result["best_rounds"],
                result["config"][
                    "config_id"
                ],
            )
        )

        best = candidate_results[0]

        chosen_config = best[
            "config"
        ]

        chosen_rounds = int(
            best["best_rounds"]
        )

        development_booster = best[
            "booster"
        ]


        # ====================================================
        # 10. DEVELOPMENT TRAIN / VALIDATION PREDICTIONS
        # ====================================================

        try:

            train_log_pred = (
                development_booster.predict(
                    dtrain,
                    iteration_range=(
                        0,
                        chosen_rounds,
                    ),
                )
            )

            validation_log_pred = (
                development_booster.predict(
                    dvalidation,
                    iteration_range=(
                        0,
                        chosen_rounds,
                    ),
                )
            )

            train_pred = (
                inverse_log_prediction(
                    train_log_pred
                )
            )

            validation_pred = (
                inverse_log_prediction(
                    validation_log_pred
                )
            )

        except Exception as exc:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": (
                    "development_prediction_failed"
                ),
                "message": str(exc),
            })

            continue


        # ====================================================
        # 11. FINAL TRAIN + VALIDATION MODEL
        # ====================================================

        try:

            dtrain_validation = make_matrix(
                train_validation,
                feature_names,
                include_label=True,
            )

            dtest = make_matrix(
                test,
                feature_names,
                include_label=True,
            )

            final_booster = xgb.train(
                params=base_params(
                    chosen_config
                ),
                dtrain=dtrain_validation,
                num_boost_round=chosen_rounds,
                verbose_eval=False,
            )

            test_log_pred = (
                final_booster.predict(
                    dtest
                )
            )

            test_pred = (
                inverse_log_prediction(
                    test_log_pred
                )
            )

        except Exception as exc:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": "final_model_failed",
                "message": str(exc),
            })

            continue


        # ====================================================
        # 12. SAVE METRICS + PREDICTIONS
        # ====================================================

        split_results = [
            (
                "train",
                train,
                train_pred,
                "development_train_only",
            ),
            (
                "validation",
                validation,
                validation_pred,
                "development_train_only",
            ),
            (
                "test",
                test,
                test_pred,
                "final_train_plus_validation",
            ),
        ]

        for (
            split_name,
            split_data,
            prediction,
            fit_stage,
        ) in split_results:

            result_metrics = calculate_metrics(
                split_data["target_value"],
                prediction,
                mase_scale,
            )

            metric_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "split": split_name,
                "model": model_name,
                "chosen_config_id": (
                    chosen_config[
                        "config_id"
                    ]
                ),
                "chosen_rounds": (
                    chosen_rounds
                ),
                "n": len(
                    split_data
                ),
                **result_metrics,
            })

            for i, (_, row) in enumerate(
                split_data.iterrows()
            ):

                prediction_rows.append({
                    "analysis_group": analysis_group,
                    "country": country,
                    "target_pathogen": pathogen,
                    "forecast_horizon_weeks": horizon,
                    "split": split_name,
                    "model": model_name,
                    "chosen_config_id": (
                        chosen_config[
                            "config_id"
                        ]
                    ),
                    "chosen_rounds": (
                        chosen_rounds
                    ),
                    "forecast_origin": row[
                        "forecast_origin"
                    ],
                    "target_date": row[
                        "target_date"
                    ],
                    "target_value": row[
                        "target_value"
                    ],
                    "prediction": (
                        prediction[i]
                    ),
                    "fit_stage": fit_stage,
                })


        diagnostic_rows.append({
            "analysis_group": analysis_group,
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "model": model_name,
            "status": "success",
            "message": "",
            "chosen_config_id": (
                chosen_config[
                    "config_id"
                ]
            ),
            "chosen_rounds": (
                chosen_rounds
            ),
            "validation_mae": (
                best[
                    "validation_mae"
                ]
            ),
            "n_train": len(
                train
            ),
            "n_validation": len(
                validation
            ),
            "n_test": len(
                test
            ),
        })


# ============================================================
# 13. COMBINE RESULTS
# ============================================================

predictions = pd.DataFrame(
    prediction_rows
)

metrics = pd.DataFrame(
    metric_rows
)

tuning = pd.DataFrame(
    tuning_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

if metrics.empty:
    raise RuntimeError(
        "No XGBoost metrics were produced."
    )


# ============================================================
# 14. M2 VS M3 COMPARISON
# ============================================================

source = metrics[
    metrics["model"].isin(
        ["M2", "M3"]
    )
].copy()

comparison = source.pivot_table(
    index=[
        "analysis_group",
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "split",
    ],
    columns="model",
    values=[
        "n",
        "mae",
        "rmse",
        "mase",
    ],
    aggfunc="first",
)

comparison.columns = [
    f"{metric}_{model.lower()}"
    for metric, model
    in comparison.columns
]

comparison = (
    comparison
    .reset_index()
)

for metric_name in [
    "mae",
    "rmse",
    "mase",
]:

    m2_col = (
        f"{metric_name}_m2"
    )

    m3_col = (
        f"{metric_name}_m3"
    )

    output_col = (
        f"m3_vs_m2_"
        f"{metric_name}_improvement_pct"
    )

    comparison[
        output_col
    ] = np.where(
        comparison[m2_col].notna()
        & comparison[m3_col].notna()
        & (
            comparison[m2_col]
            != 0
        ),
        100.0
        * (
            comparison[m2_col]
            - comparison[m3_col]
        )
        / comparison[m2_col],
        np.nan,
    )


# ============================================================
# 15. SAVE FILES
# ============================================================

predictions.to_csv(
    PREDICTION_FILE,
    index=False,
)

metrics.to_csv(
    METRIC_FILE,
    index=False,
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False,
)

tuning.to_csv(
    TUNING_FILE,
    index=False,
)

diagnostics.to_csv(
    DIAGNOSTIC_FILE,
    index=False,
)


# ============================================================
# 16. REPORT PRIMARY TEST RESULT
# ============================================================

primary_test = comparison[
    (
        comparison["analysis_group"]
        == "primary_regional"
    )
    & (
        comparison["split"]
        == "test"
    )
].copy()

display_columns = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "n_m2",
    "mae_m2",
    "mae_m3",
    "m3_vs_m2_mae_improvement_pct",
    "rmse_m2",
    "rmse_m3",
    "m3_vs_m2_rmse_improvement_pct",
]

display_columns = [
    column
    for column in display_columns
    if column in primary_test.columns
]

print("\n" + "=" * 100)
print("PRIMARY TEST-SET XGBOOST M2 VS M3")
print("=" * 100)

print(
    primary_test[
        display_columns
    ]
    .sort_values([
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
    ])
    .to_string(index=False)
)


# ============================================================
# 17. DIAGNOSTICS
# ============================================================

failed = diagnostics[
    diagnostics["status"]
    != "success"
]

print("\n" + "=" * 90)
print("XGBOOST MODEL DIAGNOSTICS")
print("=" * 90)

print(
    "Successful model specifications:",
    (
        diagnostics["status"]
        == "success"
    ).sum()
)

print(
    "Failed model specifications:",
    len(failed)
)

if not failed.empty:

    print("\nFailures:")

    print(
        failed[
            [
                "country",
                "target_pathogen",
                "forecast_horizon_weeks",
                "model",
                "status",
                "message",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\nSaved metrics:")
print(METRIC_FILE)

print("\nSaved M2/M3 comparison:")
print(COMPARISON_FILE)

print("\nSaved tuning results:")
print(TUNING_FILE)

print("\nSaved diagnostics:")
print(DIAGNOSTIC_FILE)

print("\nSaved predictions:")
print(PREDICTION_FILE)

print("\nImportant:")
print(
    "- Hyperparameters were selected using VALIDATION only."
)

print(
    "- Early stopping used VALIDATION only."
)

print(
    "- TEST was not used for tuning."
)

print(
    "- Final models were refitted on TRAIN + VALIDATION."
)

print(
    "- M2 and M3 use the common eligible forecasting sample."
)

print(
    "- Positive M3-vs-M2 improvement means M3 had lower error."
)

print(
    "- Predictive associations do not imply causation."
)

print("\nXGBOOST FORECASTING COMPLETE")