"""
Regularized Negative Binomial forecasting.

Primary research comparison
---------------------------
M1 = target-pathogen history
M2 = M1 + seasonality
M3 = M2 + co-circulating pathogen activity

Why regularization?
-------------------
The initial unregularized Negative Binomial GLMs showed numerical
instability and non-convergence for some highly overdispersed and
correlated respiratory-virus series.

This version uses ridge regularization.

Model-development rules
-----------------------
1. TRAIN is used to fit candidate models.
2. VALIDATION is used to select the ridge penalty.
3. TEST is never used to select the penalty.
4. After penalty selection, the model is refitted using
   TRAIN + VALIDATION.
5. The final frozen model is then evaluated once on TEST.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
PRED_DIR = ROOT / "outputs" / "predictions"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

METRIC_FILE = (
    TABLE_DIR
    / "statistical_forecasting_metrics.csv"
)

COMPARISON_FILE = (
    TABLE_DIR
    / "statistical_m2_m3_comparison.csv"
)

DIAGNOSTIC_FILE = (
    TABLE_DIR
    / "statistical_model_diagnostics.csv"
)

TUNING_FILE = (
    TABLE_DIR
    / "statistical_regularization_tuning.csv"
)

PREDICTION_FILE = (
    PRED_DIR
    / "statistical_predictions.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RIDGE_LAMBDAS = [
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

MODEL_FEATURE_COLUMN = {
    "M1": "m1_feature_names",
    "M2": "m2_feature_names",
    "M3": "m3_feature_names",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("REGULARIZED NEGATIVE BINOMIAL FORECASTING")
print("=" * 90)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

if not MASE_FILE.exists():
    raise FileNotFoundError(MASE_FILE)

df = pd.read_csv(FEATURE_FILE)
mase_scales = pd.read_csv(MASE_FILE)

for column in ["forecast_origin", "target_date"]:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
    )

if df["forecast_origin"].isna().any():
    raise ValueError("Invalid forecast_origin values.")

if df["target_date"].isna().any():
    raise ValueError("Invalid target_date values.")


# ============================================================
# 4. COMMON M2/M3 EVALUATION SET
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
        "No common M2/M3 eligible forecasting rows."
    )


# ============================================================
# 5. HELPERS
# ============================================================

def parse_features(value):

    if pd.isna(value):
        return []

    return [
        x for x in str(value).split("|")
        if x
    ]


def estimate_alpha(y):
    """
    Method-of-moments NB2 dispersion estimate:

        Var(Y) = mu + alpha * mu^2
    """

    y = pd.Series(y).dropna().astype(float)

    if len(y) < 2:
        return 1e-6

    mean_y = y.mean()
    var_y = y.var(ddof=1)

    if (
        not np.isfinite(mean_y)
        or mean_y <= 0
    ):
        return 1e-6

    alpha = (
        var_y - mean_y
    ) / (mean_y ** 2)

    if not np.isfinite(alpha):
        alpha = 1e-6

    return max(float(alpha), 1e-6)


def prepare_X(
    data,
    feature_names,
    means=None,
    stds=None,
):
    """
    Count/history predictors:
        log1p transform

    Seasonal predictors:
        retain original sine/cosine values

    All predictors are subsequently standardised using
    statistics learned from modelling data only.
    """

    X = (
        data[feature_names]
        .astype(float)
        .copy()
    )

    seasonal = {
        "season_sin",
        "season_cos",
    }

    for column in feature_names:

        if column not in seasonal:

            if (X[column].dropna() < 0).any():
                raise ValueError(
                    f"Negative predictor in {column}"
                )

            X[column] = np.log1p(
                X[column]
            )

    if means is None:
        means = X.mean()

    if stds is None:
        stds = X.std(ddof=0)

    stds = stds.copy()

    bad_sd = (
        (~np.isfinite(stds))
        | (stds == 0)
    )

    stds[bad_sd] = 1.0

    X = (
        X - means
    ) / stds

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    return X, means, stds


def fit_ridge_nb(
    data,
    feature_names,
    ridge_lambda,
):
    """
    Fit a ridge-regularized Negative Binomial GLM.
    """

    model_data = data.dropna(
        subset=[
            "target_value",
            *feature_names,
        ]
    ).copy()

    if len(model_data) <= len(feature_names) + 5:

        raise ValueError(
            "Too few observations for predictor count."
        )

    y = (
        model_data["target_value"]
        .astype(float)
    )

    if (y < 0).any():
        raise ValueError(
            "Negative target count found."
        )

    X, means, stds = prepare_X(
        model_data,
        feature_names,
    )

    alpha_nb = estimate_alpha(y)

    family = sm.families.NegativeBinomial(
        alpha=alpha_nb
    )

    model = sm.GLM(
        y,
        X,
        family=family,
    )

    # Do not penalise the intercept.
    penalty = np.full(
        X.shape[1],
        ridge_lambda,
        dtype=float,
    )

    penalty[0] = 0.0

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        result = model.fit_regularized(
            method="elastic_net",
            alpha=penalty,
            L1_wt=0.0,
            maxiter=2000,
            cnvrg_tol=1e-8,
            zero_tol=1e-10,
        )

    params = np.asarray(
        result.params,
        dtype=float,
    )

    if not np.isfinite(params).all():
        raise ValueError(
            "Regularized fit produced non-finite parameters."
        )

    return {
        "result": result,
        "means": means,
        "stds": stds,
        "alpha_nb": alpha_nb,
        "n": len(model_data),
    }


def predict_model(
    fitted,
    data,
    feature_names,
):

    X, _, _ = prepare_X(
        data,
        feature_names,
        means=fitted["means"],
        stds=fitted["stds"],
    )

    pred = np.asarray(
        fitted["result"].predict(X),
        dtype=float,
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            "Prediction contains NaN or infinite values."
        )

    return np.maximum(
        pred,
        0.0,
    )


def mae(y, pred):

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    return np.mean(
        np.abs(y - pred)
    )


def metrics(
    y,
    pred,
    mase_scale,
):

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    errors = pred - y

    this_mae = np.mean(
        np.abs(errors)
    )

    this_rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mean_error = np.mean(errors)

    if (
        pd.isna(mase_scale)
        or mase_scale == 0
    ):
        this_mase = np.nan
    else:
        this_mase = (
            this_mae / mase_scale
        )

    return {
        "mae": this_mae,
        "rmse": this_rmse,
        "mase": this_mase,
        "mean_error": mean_error,
    }


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


# ============================================================
# 6. MODEL DEVELOPMENT
# ============================================================

prediction_rows = []
metric_rows = []
diagnostic_rows = []
tuning_rows = []

group_cols = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
]

for keys, group in df.groupby(
    group_cols,
    dropna=False,
):

    (
        analysis_group,
        country,
        pathogen,
        horizon,
    ) = keys

    group = group.sort_values(
        "target_date"
    ).copy()

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
            ["train", "validation"]
        )
    ].copy()

    mase_scale = get_mase_scale(
        country,
        pathogen,
    )

    for model_name, feature_column in (
        MODEL_FEATURE_COLUMN.items()
    ):

        unique_features = (
            group[feature_column]
            .dropna()
            .unique()
        )

        if len(unique_features) != 1:

            diagnostic_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "model": model_name,
                "status": "feature_definition_error",
                "message": "Expected one feature definition.",
            })

            continue

        feature_names = parse_features(
            unique_features[0]
        )

        # ====================================================
        # 7. RIDGE TUNING: TRAIN -> VALIDATION
        # ====================================================

        candidate_results = []

        for ridge_lambda in RIDGE_LAMBDAS:

            try:

                fitted = fit_ridge_nb(
                    train,
                    feature_names,
                    ridge_lambda,
                )

                validation_pred = (
                    predict_model(
                        fitted,
                        validation,
                        feature_names,
                    )
                )

                validation_mae = mae(
                    validation["target_value"],
                    validation_pred,
                )

                candidate_results.append({
                    "ridge_lambda": ridge_lambda,
                    "validation_mae": validation_mae,
                    "fit": fitted,
                })

                tuning_rows.append({
                    "analysis_group": analysis_group,
                    "country": country,
                    "target_pathogen": pathogen,
                    "forecast_horizon_weeks": horizon,
                    "model": model_name,
                    "ridge_lambda": ridge_lambda,
                    "validation_mae": validation_mae,
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
                    "ridge_lambda": ridge_lambda,
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
                "status": "all_penalties_failed",
                "message": (
                    "No ridge penalty produced a usable "
                    "validation forecast."
                ),
            })

            continue

        # Lowest validation MAE wins.
        # If tied, use the smaller penalty.
        candidate_results.sort(
            key=lambda x: (
                x["validation_mae"],
                x["ridge_lambda"],
            )
        )

        best = candidate_results[0]

        chosen_lambda = best[
            "ridge_lambda"
        ]

        development_fit = best["fit"]

        # ====================================================
        # 8. TRAIN + VALIDATION -> FINAL TEST MODEL
        # ====================================================

        try:

            final_fit = fit_ridge_nb(
                train_validation,
                feature_names,
                chosen_lambda,
            )

            train_pred = predict_model(
                development_fit,
                train,
                feature_names,
            )

            validation_pred = predict_model(
                development_fit,
                validation,
                feature_names,
            )

            test_pred = predict_model(
                final_fit,
                test,
                feature_names,
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
        # 9. SAVE PREDICTIONS AND METRICS
        # ====================================================

        for split_name, split_data, pred, fit_stage in [
            (
                "train",
                train,
                train_pred,
                "train_only",
            ),
            (
                "validation",
                validation,
                validation_pred,
                "train_only",
            ),
            (
                "test",
                test,
                test_pred,
                "train_plus_validation",
            ),
        ]:

            metric_values = metrics(
                split_data["target_value"],
                pred,
                mase_scale,
            )

            metric_rows.append({
                "analysis_group": analysis_group,
                "country": country,
                "target_pathogen": pathogen,
                "forecast_horizon_weeks": horizon,
                "split": split_name,
                "model": model_name,
                "chosen_ridge_lambda": chosen_lambda,
                "n": len(split_data),
                **metric_values,
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
                    "chosen_ridge_lambda": chosen_lambda,
                    "forecast_origin": row[
                        "forecast_origin"
                    ],
                    "target_date": row[
                        "target_date"
                    ],
                    "target_value": row[
                        "target_value"
                    ],
                    "prediction": pred[i],
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
            "chosen_ridge_lambda": chosen_lambda,
            "validation_mae": best[
                "validation_mae"
            ],
            "n_train": development_fit[
                "n"
            ],
            "n_train_plus_validation": final_fit[
                "n"
            ],
            "alpha_train": development_fit[
                "alpha_nb"
            ],
            "alpha_final": final_fit[
                "alpha_nb"
            ],
        })


# ============================================================
# 10. COMBINE OUTPUTS
# ============================================================

predictions = pd.DataFrame(
    prediction_rows
)

metric_df = pd.DataFrame(
    metric_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

tuning = pd.DataFrame(
    tuning_rows
)

if metric_df.empty:
    raise RuntimeError(
        "No statistical model results generated."
    )


# ============================================================
# 11. M2 VS M3 COMPARISON
# ============================================================

source = metric_df[
    metric_df["model"].isin(
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

    m2 = f"{metric_name}_m2"
    m3 = f"{metric_name}_m3"

    out = (
        f"m3_vs_m2_"
        f"{metric_name}_improvement_pct"
    )

    comparison[out] = np.where(
        comparison[m2].notna()
        & comparison[m3].notna()
        & (comparison[m2] != 0),
        100
        * (
            comparison[m2]
            - comparison[m3]
        )
        / comparison[m2],
        np.nan,
    )


# ============================================================
# 12. SAVE FILES
# ============================================================

predictions.to_csv(
    PREDICTION_FILE,
    index=False,
)

metric_df.to_csv(
    METRIC_FILE,
    index=False,
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False,
)

diagnostics.to_csv(
    DIAGNOSTIC_FILE,
    index=False,
)

tuning.to_csv(
    TUNING_FILE,
    index=False,
)


# ============================================================
# 13. REPORT PRIMARY TEST RESULTS
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

cols = [
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

cols = [
    x for x in cols
    if x in primary_test.columns
]

print("\n" + "=" * 90)
print("PRIMARY TEST-SET M2 VS M3 COMPARISON")
print("=" * 90)

print(
    primary_test[
        cols
    ].to_string(
        index=False
    )
)


# ============================================================
# 14. DIAGNOSTICS
# ============================================================

failed = diagnostics[
    diagnostics["status"]
    != "success"
]

print("\n" + "=" * 90)
print("MODEL DIAGNOSTICS")
print("=" * 90)

print(
    "Successful final model specifications:",
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
    "- Ridge strength was selected using VALIDATION only."
)

print(
    "- TEST was never used to select the ridge penalty."
)

print(
    "- Positive M3-vs-M2 improvement means lower "
    "forecast error for M3."
)

print(
    "- Predictive improvement does not imply "
    "causal interaction between pathogens."
)

print(
    "\nREGULARIZED STATISTICAL FORECASTING COMPLETE"
)