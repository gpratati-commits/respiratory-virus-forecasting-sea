"""
Leakage-safe baseline forecasting for the respiratory-virus project.

Baselines
---------
1. Persistence:
   current pathogen activity at the forecast origin.

2. Recent 4-week mean:
   mean activity during the four weeks preceding the forecast origin.

3. Seasonal naive:
   pathogen activity exactly 52 weeks before the target date.

Forecast horizons
-----------------
1, 2, and 4 weeks.

Important
---------
- No future disease observations are used as predictors.
- Forecast splits were already assigned by TARGET DATE.
- Missing surveillance observations remain missing.
- MASE scaling is estimated using TRAINING data only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_forecasting_features.csv"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_virus_master_weekly.csv"
)

PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_FILE = (
    PREDICTION_DIR
    / "baseline_predictions.csv"
)

METRIC_FILE = (
    TABLE_DIR
    / "baseline_metrics.csv"
)

MASE_FILE = (
    TABLE_DIR
    / "baseline_mase_scales.csv"
)


# ============================================================
# 2. PATHOGEN DEFINITIONS
# ============================================================

PATHOGEN_COLUMNS = {
    "influenza": "influenza_positive",
    "rsv": "rsv_positive",
    "sarscov2": "sarscov2_cases",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 85)
print("BASELINE RESPIRATORY-VIRUS FORECASTING")
print("=" * 85)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Forecast-feature dataset not found:\n{FEATURE_FILE}"
    )

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found:\n{MASTER_FILE}"
    )

features = pd.read_csv(FEATURE_FILE)
master = pd.read_csv(MASTER_FILE)

for col in ["forecast_origin", "target_date"]:
    features[col] = pd.to_datetime(
        features[col],
        errors="coerce"
    )

master["week_start"] = pd.to_datetime(
    master["week_start"],
    errors="coerce"
)

if features["forecast_origin"].isna().any():
    raise ValueError(
        "Some forecast_origin values are invalid."
    )

if features["target_date"].isna().any():
    raise ValueError(
        "Some target_date values are invalid."
    )


# ============================================================
# 4. CHECK UNIQUE MASTER COUNTRY-WEEKS
# ============================================================

duplicates = master.duplicated(
    subset=["country", "week_start"],
    keep=False
)

if duplicates.any():
    raise ValueError(
        "Duplicate country-week records found in master dataset."
    )


# ============================================================
# 5. INITIALISE BASELINE PREDICTIONS
# ============================================================

features["pred_persistence"] = np.nan
features["pred_recent_mean4"] = np.nan
features["pred_seasonal_naive52"] = np.nan

features["seasonal_reference_date"] = (
    features["target_date"]
    - pd.to_timedelta(
        52 * 7,
        unit="D"
    )
)


# ============================================================
# 6. CREATE EACH BASELINE
# ============================================================

for pathogen, pathogen_col in PATHOGEN_COLUMNS.items():

    mask = (
        features["target_pathogen"]
        == pathogen
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------
    #
    # Uses pathogen activity at forecast origin t.
    #
    # This value is available when the forecast is issued.

    features.loc[
        mask,
        "pred_persistence"
    ] = features.loc[
        mask,
        pathogen_col
    ]

    # --------------------------------------------------------
    # Recent 4-week mean
    # --------------------------------------------------------
    #
    # This feature was created in Step 15 using:
    #
    # t-1, t-2, t-3, t-4
    #
    # so it does not contain future information.

    rolling_col = f"{pathogen}_rolling4"

    features.loc[
        mask,
        "pred_recent_mean4"
    ] = features.loc[
        mask,
        rolling_col
    ]

    # --------------------------------------------------------
    # Seasonal naive
    # --------------------------------------------------------
    #
    # Prediction for target date uses the observed activity
    # exactly 52 weeks before that TARGET date.

    lookup = (
        master[
            ["country", "week_start", pathogen_col]
        ]
        .set_index(
            ["country", "week_start"]
        )[pathogen_col]
    )

    indices = features.index[mask]

    values = []

    for idx in indices:

        country = features.at[idx, "country"]

        reference_date = features.at[
            idx,
            "seasonal_reference_date"
        ]

        try:
            value = lookup.loc[
                (country, reference_date)
            ]
        except KeyError:
            value = np.nan

        values.append(value)

    features.loc[
        indices,
        "pred_seasonal_naive52"
    ] = values


# ============================================================
# 7. AUTOMATED LEAKAGE CHECK
# ============================================================

# Seasonal reference must always occur before forecast origin.

available_seasonal = (
    features[
        "pred_seasonal_naive52"
    ].notna()
)

bad_seasonal_dates = features[
    available_seasonal
    & (
        features["seasonal_reference_date"]
        >= features["forecast_origin"]
    )
]

if not bad_seasonal_dates.empty:
    raise ValueError(
        "Leakage detected in seasonal-naive baseline."
    )

# Targets must remain in the future.

if not (
    features["target_date"]
    > features["forecast_origin"]
).all():
    raise ValueError(
        "Target-date leakage check failed."
    )


# ============================================================
# 8. COMMON BASELINE EVALUATION SET
# ============================================================

features["baseline_common_eligible"] = (
    features["target_value"].notna()
    & features["pred_persistence"].notna()
    & features["pred_recent_mean4"].notna()
    & features["pred_seasonal_naive52"].notna()
)


# ============================================================
# 9. CALCULATE TRAINING-ONLY MASE SCALE
# ============================================================

mase_rows = []

for (country, pathogen), group in features.groupby(
    ["country", "target_pathogen"]
):

    pathogen_col = PATHOGEN_COLUMNS[pathogen]

    # Use h=1 training rows only.
    #
    # target_value = Y(t+1)
    # pathogen_col = Y(t)
    #
    # This gives the standard one-step naive difference.

    train = group[
        (group["split"] == "train")
        & (
            group["forecast_horizon_weeks"]
            == 1
        )
    ].copy()

    valid = train[
        train["target_value"].notna()
        & train[pathogen_col].notna()
    ]

    if valid.empty:
        scale = np.nan
    else:
        absolute_differences = np.abs(
            valid["target_value"]
            - valid[pathogen_col]
        )

        scale = absolute_differences.mean()

        if scale == 0:
            scale = np.nan

    mase_rows.append({
        "country": country,
        "target_pathogen": pathogen,
        "mase_scale_training_only": scale,
        "n_training_pairs_for_scale": len(valid),
    })


mase_scales = pd.DataFrame(mase_rows)

mase_scales.to_csv(
    MASE_FILE,
    index=False
)


# ============================================================
# 10. ADD MASE SCALE TO FORECAST DATA
# ============================================================

features = features.merge(
    mase_scales,
    on=[
        "country",
        "target_pathogen"
    ],
    how="left"
)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    observed,
    predicted,
    mase_scale
):
    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    errors = predicted - observed

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    mean_error = np.mean(errors)

    if (
        pd.isna(mase_scale)
        or mase_scale == 0
    ):
        mase = np.nan
    else:
        mase = mae / mase_scale

    return mae, rmse, mase, mean_error


# ============================================================
# 12. EVALUATE BASELINES
# ============================================================

BASELINES = {
    "persistence": "pred_persistence",
    "recent_mean4": "pred_recent_mean4",
    "seasonal_naive52": "pred_seasonal_naive52",
}

metric_rows = []

group_columns = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "split",
]

for group_values, group in features.groupby(
    group_columns,
    dropna=False
):

    (
        analysis_group,
        country,
        pathogen,
        horizon,
        split,
    ) = group_values

    mase_scale_values = (
        group[
            "mase_scale_training_only"
        ]
        .dropna()
        .unique()
    )

    if len(mase_scale_values) == 0:
        mase_scale = np.nan
    else:
        mase_scale = (
            mase_scale_values[0]
        )

    for model_name, prediction_col in BASELINES.items():

        valid = group[
            group["target_value"].notna()
            & group[prediction_col].notna()
        ].copy()

        if valid.empty:
            continue

        mae, rmse, mase, mean_error = (
            calculate_metrics(
                valid["target_value"],
                valid[prediction_col],
                mase_scale,
            )
        )

        metric_rows.append({
            "analysis_group": analysis_group,
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "split": split,
            "baseline_model": model_name,
            "n": len(valid),
            "mae": mae,
            "rmse": rmse,
            "mase": mase,
            "mean_error": mean_error,
            "evaluation_set": "model_available_rows",
        })


# ============================================================
# 13. ALSO EVALUATE ON COMMON BASELINE ROWS
# ============================================================

for group_values, group in features.groupby(
    group_columns,
    dropna=False
):

    (
        analysis_group,
        country,
        pathogen,
        horizon,
        split,
    ) = group_values

    common = group[
        group["baseline_common_eligible"]
    ].copy()

    if common.empty:
        continue

    mase_scale_values = (
        common[
            "mase_scale_training_only"
        ]
        .dropna()
        .unique()
    )

    if len(mase_scale_values) == 0:
        mase_scale = np.nan
    else:
        mase_scale = (
            mase_scale_values[0]
        )

    for model_name, prediction_col in BASELINES.items():

        mae, rmse, mase, mean_error = (
            calculate_metrics(
                common["target_value"],
                common[prediction_col],
                mase_scale,
            )
        )

        metric_rows.append({
            "analysis_group": analysis_group,
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "split": split,
            "baseline_model": model_name,
            "n": len(common),
            "mae": mae,
            "rmse": rmse,
            "mase": mase,
            "mean_error": mean_error,
            "evaluation_set": "common_baseline_rows",
        })


metrics = pd.DataFrame(metric_rows)

metrics = metrics.sort_values(
    [
        "analysis_group",
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "split",
        "evaluation_set",
        "baseline_model",
    ]
).reset_index(drop=True)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

prediction_columns = [
    "analysis_group",
    "country",
    "forecast_origin",
    "target_date",
    "target_pathogen",
    "forecast_horizon_weeks",
    "split",
    "target_value",
    "pred_persistence",
    "pred_recent_mean4",
    "pred_seasonal_naive52",
    "seasonal_reference_date",
    "baseline_common_eligible",
    "common_m2_m3_eligible",
]

features[
    prediction_columns
].to_csv(
    PREDICTION_FILE,
    index=False
)

metrics.to_csv(
    METRIC_FILE,
    index=False
)


# ============================================================
# 15. PRINT SUMMARY
# ============================================================

print("\nSaved baseline predictions:")
print(PREDICTION_FILE)

print("\nSaved baseline metrics:")
print(METRIC_FILE)

print("\nSaved training-only MASE scales:")
print(MASE_FILE)

print("\n" + "=" * 85)
print("TEST-SET BASELINE PERFORMANCE")
print("=" * 85)

test_summary = metrics[
    (metrics["split"] == "test")
    & (
        metrics["evaluation_set"]
        == "common_baseline_rows"
    )
][
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "baseline_model",
        "n",
        "mae",
        "rmse",
        "mase",
    ]
]

print(
    test_summary.to_string(
        index=False
    )
)

print("\nImportant:")
print(
    "- Test data were NOT used to tune any baseline."
)
print(
    "- MASE scaling used TRAINING data only."
)
print(
    "- Seasonal naive used data exactly 52 weeks before target date."
)
print(
    "- Missing predictions were not replaced with zero."
)

print("\nBASELINE FORECASTING COMPLETE")