"""
Interpret final XGBoost M3 models using XGBoost's built-in SHAP contributions.

Purpose
-------
For the primary regional forecasting cohort, quantify how much the final
M3 XGBoost forecasts depend on:

1. Own-pathogen history
2. Seasonality
3. Other-pathogen activity

Important
---------
- Final XGBoost hyperparameters and boosting rounds are read from the
  already-frozen Step-18 diagnostics.
- Models are refitted on TRAIN + VALIDATION exactly as in Step 18.
- SHAP values are calculated on TEST observations only.
- SHAP explains prediction behaviour; it does not establish causality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

DIAGNOSTIC_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "xgboost_model_diagnostics.csv"
)

TUNING_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "xgboost_tuning.csv"
)

TABLE_DIR = ROOT / "outputs" / "tables"

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "xgboost_interpretation"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_IMPORTANCE_FILE = (
    TABLE_DIR
    / "xgboost_shap_feature_importance.csv"
)

GROUP_IMPORTANCE_FILE = (
    TABLE_DIR
    / "xgboost_shap_grouped_importance.csv"
)

SHAP_QC_FILE = (
    TABLE_DIR
    / "xgboost_shap_qc.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_SEED = 20261004

np.random.seed(RANDOM_SEED)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("XGBOOST M3 INTERPRETATION")
print("=" * 90)

features = pd.read_csv(
    FEATURE_FILE
)

diagnostics = pd.read_csv(
    DIAGNOSTIC_FILE
)

tuning = pd.read_csv(
    TUNING_FILE
)

for column in [
    "forecast_origin",
    "target_date",
]:
    features[column] = pd.to_datetime(
        features[column],
        errors="coerce",
    )


# ============================================================
# 4. KEEP THE SAME COMMON MODELLING SAMPLE
# ============================================================

eligible = (
    features["common_m2_m3_eligible"]
    .astype(str)
    .str.lower()
    .eq("true")
)

features = features[
    eligible
].copy()


# ============================================================
# 5. SELECT FINAL PRIMARY M3 MODELS
# ============================================================

final_models = diagnostics[
    (diagnostics["analysis_group"] == "primary_regional")
    & (diagnostics["model"] == "M3")
    & (diagnostics["status"] == "success")
].copy()

if final_models.empty:
    raise ValueError(
        "No successful primary-regional M3 models found."
    )


# ============================================================
# 6. HELPERS
# ============================================================

def parse_features(value):

    if pd.isna(value):
        return []

    return [
        x
        for x in str(value).split("|")
        if x
    ]


def classify_feature(
    feature,
    target_pathogen,
):

    if feature in [
        "season_sin",
        "season_cos",
    ]:
        return "seasonality"

    if feature.startswith(
        f"{target_pathogen}_"
    ):
        return "own_pathogen_history"

    return "other_pathogen_activity"


def find_tuning_row(
    country,
    pathogen,
    horizon,
    config_id,
):

    x = tuning[
        (tuning["analysis_group"] == "primary_regional")
        & (tuning["country"] == country)
        & (tuning["target_pathogen"] == pathogen)
        & (
            tuning["forecast_horizon_weeks"]
            == horizon
        )
        & (tuning["model"] == "M3")
        & (
            tuning["config_id"]
            == config_id
        )
        & (tuning["status"] == "success")
    ]

    if len(x) != 1:
        raise ValueError(
            "Expected exactly one matching tuning row for "
            f"{country}, {pathogen}, h={horizon}, "
            f"config={config_id}; found {len(x)}."
        )

    return x.iloc[0]


def make_dmatrix(
    data,
    feature_names,
    include_label,
):

    X = (
        data[feature_names]
        .astype(float)
        .copy()
    )

    if X.isna().any().any():
        raise ValueError(
            "Missing predictor values found in SHAP dataset."
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


# ============================================================
# 7. REFIT FINAL M3 MODELS + CALCULATE SHAP
# ============================================================

feature_rows = []
group_rows = []
qc_rows = []

for _, model_row in final_models.iterrows():

    country = model_row["country"]
    pathogen = model_row["target_pathogen"]

    horizon = int(
        model_row["forecast_horizon_weeks"]
    )

    config_id = model_row[
        "chosen_config_id"
    ]

    chosen_rounds = int(
        model_row["chosen_rounds"]
    )

    group = features[
        (features["analysis_group"] == "primary_regional")
        & (features["country"] == country)
        & (
            features["target_pathogen"]
            == pathogen
        )
        & (
            features["forecast_horizon_weeks"]
            == horizon
        )
    ].copy()

    if group.empty:
        raise ValueError(
            f"No feature rows for {country}, "
            f"{pathogen}, h={horizon}."
        )

    feature_defs = (
        group["m3_feature_names"]
        .dropna()
        .unique()
    )

    if len(feature_defs) != 1:
        raise ValueError(
            "Expected exactly one M3 feature definition."
        )

    feature_names = parse_features(
        feature_defs[0]
    )

    train_validation = group[
        group["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()

    test = group[
        group["split"] == "test"
    ].copy()

    if (
        train_validation.empty
        or test.empty
    ):
        raise ValueError(
            f"Missing train+validation or test data for "
            f"{country}, {pathogen}, h={horizon}."
        )

    tuning_row = find_tuning_row(
        country,
        pathogen,
        horizon,
        config_id,
    )

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": int(
            tuning_row["max_depth"]
        ),
        "eta": float(
            tuning_row["eta"]
        ),
        "min_child_weight": float(
            tuning_row["min_child_weight"]
        ),
        "reg_lambda": float(
            tuning_row["reg_lambda"]
        ),
        "reg_alpha": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_SEED,
        "nthread": 4,
    }

    dtrain_validation = make_dmatrix(
        train_validation,
        feature_names,
        include_label=True,
    )

    dtest = make_dmatrix(
        test,
        feature_names,
        include_label=False,
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain_validation,
        num_boost_round=chosen_rounds,
        verbose_eval=False,
    )

    # --------------------------------------------------------
    # XGBoost built-in SHAP contributions
    # --------------------------------------------------------

    contributions = booster.predict(
        dtest,
        pred_contribs=True,
    )

    # Final column is the expected-value / bias contribution.
    shap_values = contributions[:, :-1]
    bias = contributions[:, -1]

    if shap_values.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "SHAP feature count does not match model features."
        )

    if not np.isfinite(
        shap_values
    ).all():
        raise ValueError(
            "Non-finite SHAP values produced."
        )

    # --------------------------------------------------------
    # Verify SHAP additivity
    # --------------------------------------------------------

    margin_from_shap = (
        shap_values.sum(axis=1)
        + bias
    )

    direct_margin = booster.predict(
        dtest,
        output_margin=True,
    )

    max_additivity_error = float(
        np.max(
            np.abs(
                margin_from_shap
                - direct_margin
            )
        )
    )

    qc_rows.append({
        "country": country,
        "target_pathogen": pathogen,
        "forecast_horizon_weeks": horizon,
        "chosen_config_id": config_id,
        "chosen_rounds": chosen_rounds,
        "n_test": len(test),
        "number_of_features": len(
            feature_names
        ),
        "max_shap_additivity_error": (
            max_additivity_error
        ),
    })

    # --------------------------------------------------------
    # Feature-level importance
    # --------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    total_importance = (
        mean_abs_shap.sum()
    )

    if total_importance <= 0:
        relative_importance = np.zeros(
            len(mean_abs_shap)
        )
    else:
        relative_importance = (
            100.0
            * mean_abs_shap
            / total_importance
        )

    local_feature_rows = []

    for feature, importance, relative in zip(
        feature_names,
        mean_abs_shap,
        relative_importance,
    ):

        feature_group = classify_feature(
            feature,
            pathogen,
        )

        row = {
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "feature": feature,
            "feature_group": feature_group,
            "mean_absolute_shap": float(
                importance
            ),
            "relative_importance_pct": float(
                relative
            ),
        }

        feature_rows.append(row)
        local_feature_rows.append(row)

    # --------------------------------------------------------
    # Grouped importance
    # --------------------------------------------------------

    local_df = pd.DataFrame(
        local_feature_rows
    )

    grouped = (
        local_df
        .groupby(
            "feature_group",
            as_index=False,
        )
        .agg(
            mean_absolute_shap=(
                "mean_absolute_shap",
                "sum",
            ),
            relative_importance_pct=(
                "relative_importance_pct",
                "sum",
            ),
        )
    )

    for _, grouped_row in grouped.iterrows():

        group_rows.append({
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "feature_group": grouped_row[
                "feature_group"
            ],
            "mean_absolute_shap": grouped_row[
                "mean_absolute_shap"
            ],
            "relative_importance_pct": (
                grouped_row[
                    "relative_importance_pct"
                ]
            ),
        })

    # --------------------------------------------------------
    # Plot top 12 features
    # --------------------------------------------------------

    plot_df = (
        local_df
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .head(12)
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["mean_absolute_shap"],
    )

    ax.set_xlabel(
        "Mean absolute SHAP contribution"
    )

    ax.set_ylabel(
        "Predictor"
    )

    ax.set_title(
        f"{country} | {pathogen} | "
        f"{horizon}-week XGBoost M3"
    )

    fig.tight_layout()

    safe_country = (
        country
        .lower()
        .replace(" ", "_")
    )

    figure_file = (
        FIGURE_DIR
        / (
            f"{safe_country}_"
            f"{pathogen}_"
            f"h{horizon}_"
            f"shap_importance.png"
        )
    )

    fig.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. SAVE TABLES
# ============================================================

feature_importance = pd.DataFrame(
    feature_rows
)

group_importance = pd.DataFrame(
    group_rows
)

shap_qc = pd.DataFrame(
    qc_rows
)

feature_importance.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False,
)

group_importance.to_csv(
    GROUP_IMPORTANCE_FILE,
    index=False,
)

shap_qc.to_csv(
    SHAP_QC_FILE,
    index=False,
)


# ============================================================
# 9. PRINT GROUPED RESULTS
# ============================================================

print("\n" + "=" * 90)
print("GROUPED XGBOOST M3 SHAP IMPORTANCE")
print("=" * 90)

display = (
    group_importance
    .sort_values(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "relative_importance_pct",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )
)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# 10. QC
# ============================================================

print("\n" + "=" * 90)
print("SHAP ADDITIVITY QC")
print("=" * 90)

print(
    shap_qc.to_string(
        index=False
    )
)

max_error = (
    shap_qc[
        "max_shap_additivity_error"
    ].max()
)

print(
    "\nMaximum SHAP additivity error:",
    max_error
)

if max_error < 1e-4:
    print("SHAP ADDITIVITY CHECK: PASS")
else:
    print("SHAP ADDITIVITY CHECK: REVIEW REQUIRED")


print("\nSaved feature-level importance:")
print(FEATURE_IMPORTANCE_FILE)

print("\nSaved grouped importance:")
print(GROUP_IMPORTANCE_FILE)

print("\nSaved SHAP QC:")
print(SHAP_QC_FILE)

print("\nSaved figures:")
print(FIGURE_DIR)

print("\nImportant:")
print(
    "- SHAP values describe predictive contributions."
)

print(
    "- They do NOT establish causal relationships."
)

print(
    "- Values are contributions on the model's "
    "log1p target scale."
)

print("\nXGBOOST INTERPRETATION COMPLETE")