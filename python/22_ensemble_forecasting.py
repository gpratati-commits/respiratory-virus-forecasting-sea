"""
Leakage-safe ensemble forecasting.

Combines the already-frozen Negative Binomial and XGBoost forecasts.

Scientific design
-----------------
M2:
    own-pathogen history + seasonality

M3:
    M2 + co-circulating pathogen activity

Ensemble:
    prediction =
        NB_weight * NegativeBinomial
        + (1 - NB_weight) * XGBoost

Weight selection
----------------
VALIDATION only:
    Choose one global ensemble weight for M2
    and one global ensemble weight for M3.

TEST:
    Apply those frozen weights once.

The test set is never used to select ensemble weights.

Indonesia RSV has an undefined training-only MASE scale.
It is excluded only from validation weight selection,
but ensemble predictions and MAE/RMSE are still produced for it.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NB_FILE = (
    ROOT
    / "outputs"
    / "predictions"
    / "statistical_predictions.csv"
)

XGB_FILE = (
    ROOT
    / "outputs"
    / "predictions"
    / "xgboost_predictions.csv"
)

MASE_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "baseline_mase_scales.csv"
)

TABLE_DIR = ROOT / "outputs" / "tables"
PREDICTION_DIR = ROOT / "outputs" / "predictions"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TUNING_FILE = (
    TABLE_DIR
    / "ensemble_weight_tuning.csv"
)

SELECTED_FILE = (
    TABLE_DIR
    / "ensemble_selected_weights.csv"
)

METRIC_FILE = (
    TABLE_DIR
    / "ensemble_metrics.csv"
)

M2_M3_FILE = (
    TABLE_DIR
    / "ensemble_m2_m3_comparison.csv"
)

COMPONENT_FILE = (
    TABLE_DIR
    / "ensemble_component_comparison.csv"
)

PREDICTION_FILE = (
    PREDICTION_DIR
    / "ensemble_predictions.csv"
)


# ============================================================
# 2. PRESPECIFIED WEIGHT GRID
# ============================================================

# Weight refers to the Negative Binomial component.
#
# 0.0 = 100% XGBoost
# 0.5 = equal-weight ensemble
# 1.0 = 100% Negative Binomial

WEIGHTS = np.round(
    np.arange(
        0.0,
        1.01,
        0.1,
    ),
    1,
)


# ============================================================
# 3. LOAD INPUTS
# ============================================================

print("\n" + "=" * 95)
print("LEAKAGE-SAFE NEGATIVE BINOMIAL + XGBOOST ENSEMBLE")
print("=" * 95)

for file in [
    NB_FILE,
    XGB_FILE,
    MASE_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


nb = pd.read_csv(
    NB_FILE
)

xgb = pd.read_csv(
    XGB_FILE
)

mase = pd.read_csv(
    MASE_FILE
)


# ============================================================
# 4. KEEP PRIMARY M2 / M3 VALIDATION + TEST
# ============================================================

def filter_predictions(data):

    return data[
        (data["analysis_group"] == "primary_regional")
        & (
            data["model"]
            .isin(["M2", "M3"])
        )
        & (
            data["split"]
            .isin(
                [
                    "validation",
                    "test",
                ]
            )
        )
    ].copy()


nb = filter_predictions(
    nb
)

xgb = filter_predictions(
    xgb
)


# ============================================================
# 5. EXACTLY MATCH NB AND XGBOOST FORECASTS
# ============================================================

KEYS = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "split",
    "model",
    "forecast_origin",
    "target_date",
]

nb_small = nb[
    KEYS
    + [
        "target_value",
        "prediction",
    ]
].rename(
    columns={
        "target_value":
            "nb_target_value",

        "prediction":
            "nb_prediction",
    }
)

xgb_small = xgb[
    KEYS
    + [
        "target_value",
        "prediction",
    ]
].rename(
    columns={
        "target_value":
            "xgb_target_value",

        "prediction":
            "xgb_prediction",
    }
)

matched = nb_small.merge(
    xgb_small,
    on=KEYS,
    how="outer",
    indicator=True,
    validate="one_to_one",
)


# ============================================================
# 6. MATCHING QC
# ============================================================

if not (
    matched["_merge"] == "both"
).all():

    print(
        "\nPrediction matching problem:"
    )

    print(
        matched["_merge"]
        .value_counts()
    )

    raise ValueError(
        "NB and XGBoost predictions do not match."
    )


if not np.allclose(
    matched["nb_target_value"],
    matched["xgb_target_value"],
    equal_nan=True,
):

    raise ValueError(
        "NB and XGBoost target values differ."
    )


matched["target_value"] = (
    matched[
        "nb_target_value"
    ]
)

matched = matched.drop(
    columns=[
        "nb_target_value",
        "xgb_target_value",
        "_merge",
    ]
)


# ============================================================
# 7. ADD TRAINING-ONLY MASE SCALE
# ============================================================

scale_table = mase[
    [
        "country",
        "target_pathogen",
        "mase_scale_training_only",
    ]
].drop_duplicates()

if scale_table.duplicated(
    subset=[
        "country",
        "target_pathogen",
    ]
).any():

    raise ValueError(
        "Duplicate MASE-scale rows."
    )


matched = matched.merge(
    scale_table,
    on=[
        "country",
        "target_pathogen",
    ],
    how="left",
    validate="many_to_one",
)


matched[
    "mase_scale_training_only"
] = pd.to_numeric(
    matched[
        "mase_scale_training_only"
    ],
    errors="coerce",
)


matched[
    "valid_mase_scale"
] = (
    matched[
        "mase_scale_training_only"
    ].notna()
    & np.isfinite(
        matched[
            "mase_scale_training_only"
        ]
    )
    & (
        matched[
            "mase_scale_training_only"
        ]
        > 0
    )
)


# ============================================================
# 8. REPORT INVALID SCALES
# ============================================================

invalid_scale_series = (
    matched.loc[
        ~matched[
            "valid_mase_scale"
        ],
        [
            "country",
            "target_pathogen",
        ],
    ]
    .drop_duplicates()
)

print(
    "\nSeries excluded from ensemble-weight "
    "selection because the training-only "
    "MASE scale is invalid:"
)

if invalid_scale_series.empty:

    print("None")

else:

    print(
        invalid_scale_series
        .to_string(index=False)
    )


# ============================================================
# 9. VALIDATION DATA FOR WEIGHT SELECTION
# ============================================================

validation = matched[
    matched["split"]
    == "validation"
].copy()


calibration_validation = validation[
    validation[
        "valid_mase_scale"
    ]
].copy()


if calibration_validation.empty:

    raise ValueError(
        "No validation rows with valid MASE scale."
    )


# ============================================================
# 10. VALIDATION WEIGHT TUNING
# ============================================================

tuning_rows = []


for model in [
    "M2",
    "M3",
]:

    model_validation = (
        calibration_validation[
            calibration_validation[
                "model"
            ]
            == model
        ]
        .copy()
    )

    if model_validation.empty:

        raise ValueError(
            f"No usable validation rows for {model}."
        )

    for nb_weight in WEIGHTS:

        xgb_weight = (
            1.0
            - nb_weight
        )

        temp = (
            model_validation
            .copy()
        )

        temp[
            "ensemble_prediction"
        ] = (
            nb_weight
            * temp[
                "nb_prediction"
            ]
            + xgb_weight
            * temp[
                "xgb_prediction"
            ]
        )


        # ----------------------------------------------------
        # Scaled validation absolute error
        # ----------------------------------------------------

        temp[
            "absolute_error"
        ] = np.abs(
            temp[
                "target_value"
            ]
            - temp[
                "ensemble_prediction"
            ]
        )

        temp[
            "scaled_absolute_error"
        ] = (
            temp[
                "absolute_error"
            ]
            / temp[
                "mase_scale_training_only"
            ]
        )


        # ----------------------------------------------------
        # First average within each forecasting series
        # ----------------------------------------------------

        per_series = (
            temp
            .groupby(
                [
                    "country",
                    "target_pathogen",
                    "forecast_horizon_weeks",
                ],
                as_index=False,
            )
            .agg(
                mean_scaled_absolute_error=(
                    "scaled_absolute_error",
                    "mean",
                ),
                mean_absolute_error=(
                    "absolute_error",
                    "mean",
                ),
                n=(
                    "target_value",
                    "size",
                ),
            )
        )


        # ----------------------------------------------------
        # Macro-average across epidemiological series
        # ----------------------------------------------------

        macro_scaled_mae = (
            per_series[
                "mean_scaled_absolute_error"
            ]
            .mean()
        )

        macro_unscaled_mae = (
            per_series[
                "mean_absolute_error"
            ]
            .mean()
        )


        tuning_rows.append({
            "model":
                model,

            "nb_weight":
                nb_weight,

            "xgb_weight":
                xgb_weight,

            "validation_macro_scaled_mae":
                macro_scaled_mae,

            "validation_macro_unscaled_mae":
                macro_unscaled_mae,

            "number_of_series":
                len(per_series),

            "number_of_validation_rows":
                len(temp),
        })


tuning = pd.DataFrame(
    tuning_rows
)


# ============================================================
# 11. SELECT WEIGHTS USING VALIDATION ONLY
# ============================================================

selected_rows = []


for model in [
    "M2",
    "M3",
]:

    candidate = tuning[
        tuning["model"] == model
    ].copy()


    # Primary criterion:
    # lowest macro scaled MAE.
    #
    # If exactly tied, prefer the weight closest to
    # 0.5 to avoid choosing an unnecessary extreme.

    candidate[
        "distance_from_equal_weight"
    ] = np.abs(
        candidate["nb_weight"]
        - 0.5
    )


    candidate = candidate.sort_values(
        [
            "validation_macro_scaled_mae",
            "distance_from_equal_weight",
            "nb_weight",
        ]
    )


    best = candidate.iloc[0]


    selected_rows.append({
        "model":
            model,

        "selected_nb_weight":
            best[
                "nb_weight"
            ],

        "selected_xgb_weight":
            best[
                "xgb_weight"
            ],

        "validation_macro_scaled_mae":
            best[
                "validation_macro_scaled_mae"
            ],

        "validation_macro_unscaled_mae":
            best[
                "validation_macro_unscaled_mae"
            ],

        "number_of_series_used":
            best[
                "number_of_series"
            ],
    })


selected = pd.DataFrame(
    selected_rows
)


print("\n" + "=" * 95)
print("VALIDATION-SELECTED ENSEMBLE WEIGHTS")
print("=" * 95)

print(
    selected
    .to_string(
        index=False
    )
)


# ============================================================
# 12. APPLY FROZEN WEIGHTS TO TEST
# ============================================================

test = matched[
    matched["split"]
    == "test"
].copy()


ensemble_parts = []


for model in [
    "M2",
    "M3",
]:

    row = selected[
        selected["model"]
        == model
    ].iloc[0]


    nb_weight = float(
        row[
            "selected_nb_weight"
        ]
    )

    xgb_weight = float(
        row[
            "selected_xgb_weight"
        ]
    )


    model_test = test[
        test["model"]
        == model
    ].copy()


    model_test[
        "selected_nb_weight"
    ] = nb_weight


    model_test[
        "selected_xgb_weight"
    ] = xgb_weight


    model_test[
        "ensemble_prediction"
    ] = (
        nb_weight
        * model_test[
            "nb_prediction"
        ]
        + xgb_weight
        * model_test[
            "xgb_prediction"
        ]
    )


    ensemble_parts.append(
        model_test
    )


ensemble_test = pd.concat(
    ensemble_parts,
    ignore_index=True,
)


# ============================================================
# 13. METRIC HELPER
# ============================================================

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


    mae = np.mean(
        np.abs(
            errors
        )
    )


    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )


    if (
        pd.isna(
            mase_scale
        )
        or not np.isfinite(
            mase_scale
        )
        or mase_scale <= 0
    ):

        mase_value = np.nan

    else:

        mase_value = (
            mae
            / mase_scale
        )


    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "mase":
            mase_value,

        "mean_error":
            np.mean(
                errors
            ),
    }


# ============================================================
# 14. ENSEMBLE TEST METRICS
# ============================================================

metric_rows = []


GROUP_COLUMNS = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "model",
]


for keys, group in ensemble_test.groupby(
    GROUP_COLUMNS,
    dropna=False,
):

    (
        country,
        pathogen,
        horizon,
        model,
    ) = keys


    scale_values = (
        group[
            "mase_scale_training_only"
        ]
        .dropna()
        .unique()
    )


    if len(scale_values) == 0:

        scale = np.nan

    elif len(scale_values) == 1:

        scale = float(
            scale_values[0]
        )

    else:

        raise ValueError(
            "Multiple MASE scales within one series."
        )


    metrics = calculate_metrics(
        observed=group[
            "target_value"
        ],
        predicted=group[
            "ensemble_prediction"
        ],
        mase_scale=scale,
    )


    metric_rows.append({
        "analysis_group":
            "primary_regional",

        "country":
            country,

        "target_pathogen":
            pathogen,

        "forecast_horizon_weeks":
            horizon,

        "model":
            model,

        "n":
            len(group),

        "selected_nb_weight":
            group[
                "selected_nb_weight"
            ].iloc[0],

        "selected_xgb_weight":
            group[
                "selected_xgb_weight"
            ].iloc[0],

        **metrics,
    })


ensemble_metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# 15. M2 VS M3 ENSEMBLE COMPARISON
# ============================================================

comparison = (
    ensemble_metrics
    .pivot_table(
        index=[
            "analysis_group",
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
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


for metric in [
    "mae",
    "rmse",
    "mase",
]:

    m2 = (
        f"{metric}_m2"
    )

    m3 = (
        f"{metric}_m3"
    )

    output = (
        f"m3_vs_m2_"
        f"{metric}_improvement_pct"
    )


    comparison[
        output
    ] = np.where(
        comparison[m2].notna()
        & comparison[m3].notna()
        & (
            comparison[m2]
            != 0
        ),

        100.0
        * (
            comparison[m2]
            - comparison[m3]
        )
        / comparison[m2],

        np.nan,
    )


# ============================================================
# 16. COMPONENT MODEL TEST COMPARISON
# ============================================================

component_rows = []


for keys, group in ensemble_test.groupby(
    GROUP_COLUMNS,
    dropna=False,
):

    (
        country,
        pathogen,
        horizon,
        model,
    ) = keys


    scale_values = (
        group[
            "mase_scale_training_only"
        ]
        .dropna()
        .unique()
    )


    scale = (
        float(scale_values[0])
        if len(scale_values) == 1
        else np.nan
    )


    nb_metrics = calculate_metrics(
        group["target_value"],
        group["nb_prediction"],
        scale,
    )


    xgb_metrics = calculate_metrics(
        group["target_value"],
        group["xgb_prediction"],
        scale,
    )


    ens_metrics = calculate_metrics(
        group["target_value"],
        group["ensemble_prediction"],
        scale,
    )


    best_component_mae = min(
        nb_metrics["mae"],
        xgb_metrics["mae"],
    )


    if best_component_mae > 0:

        ensemble_vs_best_pct = (
            100.0
            * (
                best_component_mae
                - ens_metrics["mae"]
            )
            / best_component_mae
        )

    else:

        ensemble_vs_best_pct = np.nan


    if (
        nb_metrics["mae"]
        < xgb_metrics["mae"]
    ):

        best_component = (
            "negative_binomial"
        )

    elif (
        xgb_metrics["mae"]
        < nb_metrics["mae"]
    ):

        best_component = (
            "xgboost"
        )

    else:

        best_component = "tie"


    component_rows.append({
        "country":
            country,

        "target_pathogen":
            pathogen,

        "forecast_horizon_weeks":
            horizon,

        "model":
            model,

        "n":
            len(group),

        "nb_mae":
            nb_metrics["mae"],

        "xgb_mae":
            xgb_metrics["mae"],

        "ensemble_mae":
            ens_metrics["mae"],

        "nb_rmse":
            nb_metrics["rmse"],

        "xgb_rmse":
            xgb_metrics["rmse"],

        "ensemble_rmse":
            ens_metrics["rmse"],

        "nb_mase":
            nb_metrics["mase"],

        "xgb_mase":
            xgb_metrics["mase"],

        "ensemble_mase":
            ens_metrics["mase"],

        "best_component_by_mae":
            best_component,

        "ensemble_vs_best_component_mae_improvement_pct":
            ensemble_vs_best_pct,

        "ensemble_beats_nb_mae":
            (
                ens_metrics["mae"]
                < nb_metrics["mae"]
            ),

        "ensemble_beats_xgb_mae":
            (
                ens_metrics["mae"]
                < xgb_metrics["mae"]
            ),

        "ensemble_beats_both_mae":
            (
                ens_metrics["mae"]
                < nb_metrics["mae"]
                and
                ens_metrics["mae"]
                < xgb_metrics["mae"]
            ),
    })


component_comparison = pd.DataFrame(
    component_rows
)


# ============================================================
# 17. SAVE PREDICTIONS
# ============================================================

prediction_columns = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "model",
    "split",
    "forecast_origin",
    "target_date",
    "target_value",
    "nb_prediction",
    "xgb_prediction",
    "selected_nb_weight",
    "selected_xgb_weight",
    "ensemble_prediction",
    "mase_scale_training_only",
    "valid_mase_scale",
]


ensemble_test[
    prediction_columns
].to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# 18. SAVE TABLES
# ============================================================

tuning.to_csv(
    TUNING_FILE,
    index=False,
)

selected.to_csv(
    SELECTED_FILE,
    index=False,
)

ensemble_metrics.to_csv(
    METRIC_FILE,
    index=False,
)

comparison.to_csv(
    M2_M3_FILE,
    index=False,
)

component_comparison.to_csv(
    COMPONENT_FILE,
    index=False,
)


# ============================================================
# 19. QC
# ============================================================

finite_predictions = (
    np.isfinite(
        ensemble_test[
            [
                "nb_prediction",
                "xgb_prediction",
                "ensemble_prediction",
                "target_value",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all()
)


weights_valid = (
    selected[
        "selected_nb_weight"
    ].between(
        0,
        1,
    ).all()
    and
    selected[
        "selected_xgb_weight"
    ].between(
        0,
        1,
    ).all()
)


weights_sum_to_one = np.allclose(
    (
        selected[
            "selected_nb_weight"
        ]
        +
        selected[
            "selected_xgb_weight"
        ]
    ),
    1.0,
)


same_n = (
    comparison[
        "n_m2"
    ]
    ==
    comparison[
        "n_m3"
    ]
).all()


# Boundary weights are legitimate:
# 0 = pure XGB
# 1 = pure NB.
#
# They simply indicate validation did not benefit
# from averaging for that model.

boundary_weights = selected[
    selected[
        "selected_nb_weight"
    ].isin(
        [
            0.0,
            1.0,
        ]
    )
]


# ============================================================
# 20. REPORT RESULTS
# ============================================================

print("\n" + "=" * 95)
print("FINAL ENSEMBLE QC")
print("=" * 95)

print(
    "All predictions finite:",
    finite_predictions,
)

print(
    "All selected weights within [0,1]:",
    weights_valid,
)

print(
    "NB + XGB weights sum to 1:",
    weights_sum_to_one,
)

print(
    "M2/M3 use identical test sample sizes:",
    same_n,
)

print(
    "Number of selected weights at pure-model boundary:",
    len(boundary_weights),
)


if not boundary_weights.empty:

    print(
        "\nBoundary selections "
        "(scientifically valid):"
    )

    print(
        boundary_weights
        .to_string(
            index=False
        )
    )


if (
    finite_predictions
    and weights_valid
    and weights_sum_to_one
    and same_n
):

    print(
        "\nENSEMBLE AUDIT: PASS"
    )

else:

    print(
        "\nENSEMBLE AUDIT: REVIEW REQUIRED"
    )


# ============================================================
# 21. MAIN TEST RESULTS
# ============================================================

print("\n" + "=" * 95)
print("PRIMARY ENSEMBLE M2 VS M3 TEST RESULTS")
print("=" * 95)

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

print(
    comparison[
        display_columns
    ]
    .sort_values(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 22. ENSEMBLE VS COMPONENTS
# ============================================================

print("\n" + "=" * 95)
print("ENSEMBLE VS NEGATIVE BINOMIAL / XGBOOST")
print("=" * 95)

print(
    component_comparison[
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "model",
            "nb_mae",
            "xgb_mae",
            "ensemble_mae",
            "best_component_by_mae",
            "ensemble_vs_best_component_mae_improvement_pct",
            "ensemble_beats_both_mae",
        ]
    ]
    .sort_values(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "model",
        ]
    )
    .to_string(
        index=False
    )
)


print("\nSUMMARY")

print(
    "Ensemble beats Negative Binomial:",
    component_comparison[
        "ensemble_beats_nb_mae"
    ].sum(),
    "of",
    len(component_comparison),
)

print(
    "Ensemble beats XGBoost:",
    component_comparison[
        "ensemble_beats_xgb_mae"
    ].sum(),
    "of",
    len(component_comparison),
)

print(
    "Ensemble beats BOTH:",
    component_comparison[
        "ensemble_beats_both_mae"
    ].sum(),
    "of",
    len(component_comparison),
)


# ============================================================
# 23. OUTPUT LOCATIONS
# ============================================================

print("\nSaved validation weight tuning:")
print(TUNING_FILE)

print("\nSaved selected weights:")
print(SELECTED_FILE)

print("\nSaved ensemble metrics:")
print(METRIC_FILE)

print("\nSaved M2/M3 comparison:")
print(M2_M3_FILE)

print("\nSaved component-model comparison:")
print(COMPONENT_FILE)

print("\nSaved ensemble predictions:")
print(PREDICTION_FILE)


print("\nImportant:")

print(
    "- Ensemble weights were selected using "
    "VALIDATION only."
)

print(
    "- TEST was used only for final evaluation."
)

print(
    "- Indonesia RSV was excluded only from "
    "MASE-scaled weight tuning because its "
    "training-only MASE scale is undefined."
)

print(
    "- Indonesia RSV test forecasts remain in "
    "MAE/RMSE evaluation."
)

print(
    "- A boundary weight of 0 or 1 is valid and "
    "means validation preferred a single model."
)

print(
    "\nENSEMBLE FORECASTING COMPLETE"
)