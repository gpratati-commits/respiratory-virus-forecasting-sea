"""
Step 23: Overall matched model comparison.

Purpose
-------
Compare the forecasting approaches developed in the project:

1. Baseline forecasting methods
2. Negative Binomial
3. XGBoost
4. Negative Binomial + XGBoost ensemble

The comparison is performed on STRICTLY MATCHED TEST observations so
that one method cannot appear better simply because it was evaluated
on an easier set of weeks.

The script also separately evaluates the central scientific question:

    Does M3, which adds co-circulating pathogen activity,
    improve forecasting compared with M2?

Important
---------
- No model is fitted here.
- No hyperparameters are tuned here.
- TEST data are used only for final evaluation.
- Indonesia RSV has an undefined training-only MASE denominator.
  MAE and RMSE are retained for that series; MASE remains NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT
    / "outputs"
    / "predictions"
    / "baseline_predictions.csv"
)

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

ENSEMBLE_FILE = (
    ROOT
    / "outputs"
    / "predictions"
    / "ensemble_predictions.csv"
)

MASE_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "baseline_mase_scales.csv"
)

TABLE_DIR = ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_FILE = (
    TABLE_DIR
    / "overall_model_comparison.csv"
)

RANKING_FILE = (
    TABLE_DIR
    / "overall_model_ranking.csv"
)

WIN_FILE = (
    TABLE_DIR
    / "model_win_summary.csv"
)

PATHOGEN_FILE = (
    TABLE_DIR
    / "model_performance_by_pathogen.csv"
)

HORIZON_FILE = (
    TABLE_DIR
    / "model_performance_by_horizon.csv"
)

OVERALL_SUMMARY_FILE = (
    TABLE_DIR
    / "overall_model_summary.csv"
)

M2_M3_EFFECT_FILE = (
    TABLE_DIR
    / "cross_pathogen_information_effect.csv"
)

M2_M3_SUMMARY_FILE = (
    TABLE_DIR
    / "cross_pathogen_information_summary.csv"
)

MATCHED_SAMPLE_FILE = (
    TABLE_DIR
    / "overall_model_matched_sample_qc.csv"
)


# ============================================================
# 2. LOAD FILES
# ============================================================

print("\n" + "=" * 95)
print("STEP 23: OVERALL MATCHED MODEL COMPARISON")
print("=" * 95)

required_files = [
    BASELINE_FILE,
    NB_FILE,
    XGB_FILE,
    ENSEMBLE_FILE,
    MASE_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

baseline = pd.read_csv(BASELINE_FILE)
nb = pd.read_csv(NB_FILE)
xgb = pd.read_csv(XGB_FILE)
ensemble = pd.read_csv(ENSEMBLE_FILE)
mase = pd.read_csv(MASE_FILE)


# ============================================================
# 3. HELPERS
# ============================================================

def require_columns(df, columns, name):
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


def detect_column(
    df,
    candidates,
    description,
):
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def filter_primary_test(df):
    out = df.copy()

    if "analysis_group" in out.columns:
        if (
            out["analysis_group"]
            .eq("primary_regional")
            .any()
        ):
            out = out[
                out["analysis_group"]
                == "primary_regional"
            ].copy()

    if "split" in out.columns:
        out = out[
            out["split"] == "test"
        ].copy()

    return out


def clean_dates(df):
    out = df.copy()

    if "target_date" in out.columns:
        out["target_date"] = pd.to_datetime(
            out["target_date"],
            errors="coerce",
        )

    if "forecast_origin" in out.columns:
        out["forecast_origin"] = pd.to_datetime(
            out["forecast_origin"],
            errors="coerce",
        )

    return out


# ============================================================
# 4. REQUIRED IDENTIFIERS
# ============================================================

CORE_KEYS = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "target_date",
]

COMMON_REQUIRED = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "target_date",
    "target_value",
]

for df, name in [
    (baseline, "baseline_predictions.csv"),
    (nb, "statistical_predictions.csv"),
    (xgb, "xgboost_predictions.csv"),
    (ensemble, "ensemble_predictions.csv"),
]:
    require_columns(
        df,
        COMMON_REQUIRED,
        name,
    )


# ============================================================
# 5. PREPARE BASELINE PREDICTIONS
# ============================================================

baseline = filter_primary_test(
    baseline
)

baseline = clean_dates(
    baseline
)


# ------------------------------------------------------------
# The baseline prediction file is stored in WIDE format:
#
# pred_persistence
# pred_recent_mean4
# pred_seasonal_naive52
#
# Convert these separate prediction columns into LONG format
# so they can be compared fairly with NB, XGBoost and Ensemble.
# ------------------------------------------------------------

BASELINE_PREDICTION_COLUMNS = {
    "pred_persistence":
        "Baseline: Persistence",

    "pred_recent_mean4":
        "Baseline: Recent mean (4 weeks)",

    "pred_seasonal_naive52":
        "Baseline: Seasonal naive (52 weeks)",
}


missing_baseline_columns = [
    column
    for column in BASELINE_PREDICTION_COLUMNS
    if column not in baseline.columns
]

if missing_baseline_columns:
    raise ValueError(
        "Baseline prediction file is missing expected "
        "prediction columns: "
        + ", ".join(missing_baseline_columns)
    )


print("\nDetected baseline prediction columns:")

for column, label in BASELINE_PREDICTION_COLUMNS.items():
    print(
        f" - {column} -> {label}"
    )


baseline_parts = []


for prediction_column, method_name in (
    BASELINE_PREDICTION_COLUMNS.items()
):

    temp = baseline[
        CORE_KEYS
        + [
            "target_value",
            prediction_column,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            prediction_column:
                "prediction",
        }
    )

    temp[
        "forecast_method"
    ] = method_name

    baseline_parts.append(
        temp
    )


baseline_long = pd.concat(
    baseline_parts,
    ignore_index=True,
)


# Convert prediction values safely to numeric.
baseline_long[
    "prediction"
] = pd.to_numeric(
    baseline_long[
        "prediction"
    ],
    errors="coerce",
)


# Baselines do not use the M2/M3 information sets.
#
# We duplicate the same baseline forecasts under M2 and M3
# purely so that both information sets are compared against
# exactly the same benchmark.
baseline_m2 = baseline_long.copy()

baseline_m2[
    "information_set"
] = "M2"


baseline_m3 = baseline_long.copy()

baseline_m3[
    "information_set"
] = "M3"


baseline_long = pd.concat(
    [
        baseline_m2,
        baseline_m3,
    ],
    ignore_index=True,
)


print("\nBaseline methods entering Step 23:")

print(
    baseline_long[
        "forecast_method"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 6. PREPARE NEGATIVE BINOMIAL
# ============================================================

nb = filter_primary_test(
    nb
)

nb = clean_dates(
    nb
)

require_columns(
    nb,
    [
        "model",
        "prediction",
    ],
    "statistical_predictions.csv",
)

nb = nb[
    nb["model"].isin(
        ["M2", "M3"]
    )
].copy()

nb_long = nb[
    CORE_KEYS
    + [
        "target_value",
        "model",
        "prediction",
    ]
].copy()

nb_long = nb_long.rename(
    columns={
        "model":
            "information_set",
    }
)

nb_long[
    "forecast_method"
] = "Negative Binomial"


# ============================================================
# 7. PREPARE XGBOOST
# ============================================================

xgb = filter_primary_test(
    xgb
)

xgb = clean_dates(
    xgb
)

require_columns(
    xgb,
    [
        "model",
        "prediction",
    ],
    "xgboost_predictions.csv",
)

xgb = xgb[
    xgb["model"].isin(
        ["M2", "M3"]
    )
].copy()

xgb_long = xgb[
    CORE_KEYS
    + [
        "target_value",
        "model",
        "prediction",
    ]
].copy()

xgb_long = xgb_long.rename(
    columns={
        "model":
            "information_set",
    }
)

xgb_long[
    "forecast_method"
] = "XGBoost"


# ============================================================
# 8. PREPARE ENSEMBLE
# ============================================================

ensemble = filter_primary_test(
    ensemble
)

ensemble = clean_dates(
    ensemble
)

require_columns(
    ensemble,
    [
        "model",
        "ensemble_prediction",
    ],
    "ensemble_predictions.csv",
)

ensemble = ensemble[
    ensemble["model"].isin(
        ["M2", "M3"]
    )
].copy()

ensemble_long = ensemble[
    CORE_KEYS
    + [
        "target_value",
        "model",
        "ensemble_prediction",
    ]
].copy()

ensemble_long = ensemble_long.rename(
    columns={
        "model":
            "information_set",

        "ensemble_prediction":
            "prediction",
    }
)

ensemble_long[
    "forecast_method"
] = "Ensemble"


# ============================================================
# 9. COMBINE ALL METHODS
# ============================================================

all_predictions = pd.concat(
    [
        baseline_long,
        nb_long,
        xgb_long,
        ensemble_long,
    ],
    ignore_index=True,
)


all_predictions[
    "prediction"
] = pd.to_numeric(
    all_predictions[
        "prediction"
    ],
    errors="coerce",
)

all_predictions[
    "target_value"
] = pd.to_numeric(
    all_predictions[
        "target_value"
    ],
    errors="coerce",
)


all_predictions = all_predictions[
    all_predictions[
        "target_value"
    ].notna()
    & all_predictions[
        "prediction"
    ].notna()
].copy()


# ============================================================
# 10. DUPLICATE CHECK
# ============================================================

ROW_KEYS = (
    CORE_KEYS
    + [
        "information_set",
        "forecast_method",
    ]
)

duplicates = all_predictions.duplicated(
    subset=ROW_KEYS,
    keep=False,
)

if duplicates.any():

    print(
        "\nDuplicate prediction rows detected:"
    )

    print(
        all_predictions.loc[
            duplicates,
            ROW_KEYS
        ]
        .sort_values(
            ROW_KEYS
        )
        .head(50)
        .to_string(index=False)
    )

    raise ValueError(
        "Duplicate method predictions found."
    )


# ============================================================
# 11. CHECK TARGET CONSISTENCY
# ============================================================

OBSERVATION_KEYS = (
    CORE_KEYS
    + [
        "information_set",
    ]
)

target_check = (
    all_predictions
    .groupby(
        OBSERVATION_KEYS,
        dropna=False,
    )[
        "target_value"
    ]
    .nunique(
        dropna=False
    )
)

if (
    target_check > 1
).any():

    raise ValueError(
        "Methods disagree on target_value "
        "for the same test observation."
    )


# ============================================================
# 12. DEFINE EXPECTED METHODS
# ============================================================

baseline_methods = sorted(
    baseline_long[
        "forecast_method"
    ]
    .drop_duplicates()
    .tolist()
)

expected_methods = (
    baseline_methods
    + [
        "Negative Binomial",
        "XGBoost",
        "Ensemble",
    ]
)

expected_method_count = len(
    expected_methods
)

print("\nMethods entering strict comparison:")

for method in expected_methods:
    print(" -", method)

print(
    "\nExpected methods per test observation:",
    expected_method_count,
)


# ============================================================
# 13. STRICT MATCHED TEST SAMPLE
# ============================================================

availability = (
    all_predictions
    .groupby(
        OBSERVATION_KEYS,
        dropna=False,
    )[
        "forecast_method"
    ]
    .nunique()
    .rename(
        "available_method_count"
    )
    .reset_index()
)

availability[
    "strictly_matched"
] = (
    availability[
        "available_method_count"
    ]
    == expected_method_count
)

matched_ids = availability[
    availability[
        "strictly_matched"
    ]
][
    OBSERVATION_KEYS
].copy()


matched = all_predictions.merge(
    matched_ids,
    on=OBSERVATION_KEYS,
    how="inner",
    validate="many_to_one",
)


if matched.empty:
    raise ValueError(
        "No strictly matched test observations remain."
    )


# ============================================================
# 14. MATCHED-SAMPLE QC TABLE
# ============================================================

sample_qc = (
    matched
    .groupby(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size": "n"
        }
    )
)

sample_qc.to_csv(
    MATCHED_SAMPLE_FILE,
    index=False,
)


# Check that every method has identical n within
# each epidemiological forecasting problem.

n_check = (
    sample_qc
    .groupby(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "information_set",
        ]
    )[
        "n"
    ]
    .nunique()
)

identical_n = (
    n_check == 1
).all()


# ============================================================
# 15. ADD MASE SCALE
# ============================================================

require_columns(
    mase,
    [
        "country",
        "target_pathogen",
        "mase_scale_training_only",
    ],
    "baseline_mase_scales.csv",
)

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
        "Duplicate MASE scales found."
    )


scale_table[
    "mase_scale_training_only"
] = pd.to_numeric(
    scale_table[
        "mase_scale_training_only"
    ],
    errors="coerce",
)


# ============================================================
# 16. CALCULATE STRICTLY MATCHED METRICS
# ============================================================

metric_rows = []

METRIC_GROUPS = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "information_set",
    "forecast_method",
]


for keys, group in matched.groupby(
    METRIC_GROUPS,
    dropna=False,
):

    (
        country,
        pathogen,
        horizon,
        information_set,
        method,
    ) = keys

    observed = group[
        "target_value"
    ].to_numpy(
        dtype=float
    )

    predicted = group[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    errors = (
        predicted
        - observed
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    mean_error = np.mean(
        errors
    )


    scale_row = scale_table[
        (
            scale_table["country"]
            == country
        )
        & (
            scale_table[
                "target_pathogen"
            ]
            == pathogen
        )
    ]


    if scale_row.empty:

        scale = np.nan

    else:

        scale = scale_row[
            "mase_scale_training_only"
        ].iloc[0]


    if (
        pd.isna(scale)
        or not np.isfinite(scale)
        or scale <= 0
    ):

        mase_value = np.nan

    else:

        mase_value = (
            mae / scale
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

        "information_set":
            information_set,

        "forecast_method":
            method,

        "n":
            len(group),

        "mae":
            mae,

        "rmse":
            rmse,

        "mase":
            mase_value,

        "mean_error":
            mean_error,

        "mase_scale_training_only":
            scale,
    })


overall = pd.DataFrame(
    metric_rows
)


# ============================================================
# 17. RANK METHODS
# ============================================================

RANK_GROUPS = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "information_set",
]


ranking = overall.copy()


ranking[
    "mae_rank"
] = (
    ranking
    .groupby(
        RANK_GROUPS
    )[
        "mae"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


ranking[
    "rmse_rank"
] = (
    ranking
    .groupby(
        RANK_GROUPS
    )[
        "rmse"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


ranking[
    "mase_rank"
] = (
    ranking
    .groupby(
        RANK_GROUPS
    )[
        "mase"
    ]
    .rank(
        method="min",
        ascending=True,
    )
)


ranking[
    "mae_win_or_tie"
] = (
    ranking[
        "mae_rank"
    ]
    == 1
)

ranking[
    "rmse_win_or_tie"
] = (
    ranking[
        "rmse_rank"
    ]
    == 1
)

ranking[
    "mase_win_or_tie"
] = (
    ranking[
        "mase_rank"
    ]
    == 1
)


# ============================================================
# 18. WIN SUMMARY
# ============================================================

win_summary = (
    ranking
    .groupby(
        [
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .agg(
        number_of_series=(
            "mae",
            "size",
        ),

        mae_wins_or_ties=(
            "mae_win_or_tie",
            "sum",
        ),

        rmse_wins_or_ties=(
            "rmse_win_or_tie",
            "sum",
        ),

        mase_wins_or_ties=(
            "mase_win_or_tie",
            "sum",
        ),

        mean_mae_rank=(
            "mae_rank",
            "mean",
        ),

        mean_rmse_rank=(
            "rmse_rank",
            "mean",
        ),

        mean_mase_rank=(
            "mase_rank",
            "mean",
        ),
    )
)


# ============================================================
# 19. PERFORMANCE BY PATHOGEN
# ============================================================

by_pathogen = (
    overall
    .groupby(
        [
            "target_pathogen",
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .agg(
        number_of_series=(
            "mae",
            "size",
        ),

        macro_mean_mae=(
            "mae",
            "mean",
        ),

        median_mae=(
            "mae",
            "median",
        ),

        macro_mean_rmse=(
            "rmse",
            "mean",
        ),

        macro_mean_mase=(
            "mase",
            "mean",
        ),

        median_mase=(
            "mase",
            "median",
        ),
    )
)


# ============================================================
# 20. PERFORMANCE BY HORIZON
# ============================================================

by_horizon = (
    overall
    .groupby(
        [
            "forecast_horizon_weeks",
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .agg(
        number_of_series=(
            "mae",
            "size",
        ),

        macro_mean_mae=(
            "mae",
            "mean",
        ),

        median_mae=(
            "mae",
            "median",
        ),

        macro_mean_rmse=(
            "rmse",
            "mean",
        ),

        macro_mean_mase=(
            "mase",
            "mean",
        ),

        median_mase=(
            "mase",
            "median",
        ),
    )
)


# ============================================================
# 21. OVERALL METHOD SUMMARY
# ============================================================

overall_summary = (
    ranking
    .groupby(
        [
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .agg(
        number_of_series=(
            "mae",
            "size",
        ),

        mean_mae_rank=(
            "mae_rank",
            "mean",
        ),

        median_mae_rank=(
            "mae_rank",
            "median",
        ),

        mean_rmse_rank=(
            "rmse_rank",
            "mean",
        ),

        mean_mase_rank=(
            "mase_rank",
            "mean",
        ),

        macro_mean_mase=(
            "mase",
            "mean",
        ),

        median_mase=(
            "mase",
            "median",
        ),
    )
)


# ============================================================
# 22. CENTRAL RESEARCH QUESTION:
#     M2 VS M3
# ============================================================

# Baselines are identical under M2 and M3 and do not
# contain cross-pathogen predictors, so the scientific
# M2-vs-M3 comparison is restricted to:
#
# Negative Binomial
# XGBoost
# Ensemble

model_methods = [
    "Negative Binomial",
    "XGBoost",
    "Ensemble",
]

model_results = overall[
    overall[
        "forecast_method"
    ].isin(
        model_methods
    )
].copy()


m2 = model_results[
    model_results[
        "information_set"
    ]
    == "M2"
].copy()

m3 = model_results[
    model_results[
        "information_set"
    ]
    == "M3"
].copy()


PAIR_KEYS = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "forecast_method",
]


m2 = m2[
    PAIR_KEYS
    + [
        "n",
        "mae",
        "rmse",
        "mase",
    ]
].rename(
    columns={
        "n": "n_m2",
        "mae": "mae_m2",
        "rmse": "rmse_m2",
        "mase": "mase_m2",
    }
)


m3 = m3[
    PAIR_KEYS
    + [
        "n",
        "mae",
        "rmse",
        "mase",
    ]
].rename(
    columns={
        "n": "n_m3",
        "mae": "mae_m3",
        "rmse": "rmse_m3",
        "mase": "mase_m3",
    }
)


m2_m3 = m2.merge(
    m3,
    on=PAIR_KEYS,
    how="inner",
    validate="one_to_one",
)


if not (
    m2_m3["n_m2"]
    == m2_m3["n_m3"]
).all():

    raise ValueError(
        "M2/M3 comparison uses different "
        "matched sample sizes."
    )


for metric in [
    "mae",
    "rmse",
    "mase",
]:

    m2_col = f"{metric}_m2"
    m3_col = f"{metric}_m3"

    improvement_col = (
        f"m3_vs_m2_"
        f"{metric}_improvement_pct"
    )

    m2_m3[
        improvement_col
    ] = np.where(
        m2_m3[
            m2_col
        ].notna()
        & m2_m3[
            m3_col
        ].notna()
        & (
            m2_m3[
                m2_col
            ]
            != 0
        ),

        100.0
        * (
            m2_m3[
                m2_col
            ]
            - m2_m3[
                m3_col
            ]
        )
        / m2_m3[
            m2_col
        ],

        np.nan,
    )


m2_m3[
    "m3_better_mae"
] = (
    m2_m3["mae_m3"]
    < m2_m3["mae_m2"]
)


m2_m3[
    "m3_better_rmse"
] = (
    m2_m3["rmse_m3"]
    < m2_m3["rmse_m2"]
)


# ============================================================
# 23. M2 VS M3 SUMMARY
# ============================================================

m2_m3_summary = (
    m2_m3
    .groupby(
        "forecast_method",
        as_index=False,
    )
    .agg(
        number_of_comparisons=(
            "mae_m2",
            "size",
        ),

        m3_mae_wins=(
            "m3_better_mae",
            "sum",
        ),

        m3_rmse_wins=(
            "m3_better_rmse",
            "sum",
        ),

        mean_m3_vs_m2_mae_improvement_pct=(
            "m3_vs_m2_mae_improvement_pct",
            "mean",
        ),

        median_m3_vs_m2_mae_improvement_pct=(
            "m3_vs_m2_mae_improvement_pct",
            "median",
        ),

        mean_m3_vs_m2_rmse_improvement_pct=(
            "m3_vs_m2_rmse_improvement_pct",
            "mean",
        ),
    )
)


m2_m3_summary[
    "m3_mae_win_fraction"
] = (
    m2_m3_summary[
        "m3_mae_wins"
    ]
    / m2_m3_summary[
        "number_of_comparisons"
    ]
)


m2_m3_summary[
    "m3_rmse_win_fraction"
] = (
    m2_m3_summary[
        "m3_rmse_wins"
    ]
    / m2_m3_summary[
        "number_of_comparisons"
    ]
)


# ============================================================
# 24. QC
# ============================================================

all_predictions_finite = np.isfinite(
    matched[
        [
            "target_value",
            "prediction",
        ]
    ].to_numpy(
        dtype=float
    )
).all()


methods_per_observation = (
    matched
    .groupby(
        OBSERVATION_KEYS
    )[
        "forecast_method"
    ]
    .nunique()
)


strict_method_match = (
    methods_per_observation
    == expected_method_count
).all()


m2_m3_sample_match = (
    m2_m3["n_m2"]
    == m2_m3["n_m3"]
).all()


# ============================================================
# 25. SAVE OUTPUTS
# ============================================================

overall.to_csv(
    OVERALL_FILE,
    index=False,
)

ranking.to_csv(
    RANKING_FILE,
    index=False,
)

win_summary.to_csv(
    WIN_FILE,
    index=False,
)

by_pathogen.to_csv(
    PATHOGEN_FILE,
    index=False,
)

by_horizon.to_csv(
    HORIZON_FILE,
    index=False,
)

overall_summary.to_csv(
    OVERALL_SUMMARY_FILE,
    index=False,
)

m2_m3.to_csv(
    M2_M3_EFFECT_FILE,
    index=False,
)

m2_m3_summary.to_csv(
    M2_M3_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 26. PRINT QC
# ============================================================

print("\n" + "=" * 95)
print("FINAL STEP-23 QC")
print("=" * 95)

print(
    "All matched predictions finite:",
    all_predictions_finite,
)

print(
    "All observations contain every expected method:",
    strict_method_match,
)

print(
    "Identical sample size across methods within each series:",
    identical_n,
)

print(
    "M2/M3 model comparisons use identical sample sizes:",
    m2_m3_sample_match,
)

print(
    "Matched test prediction rows:",
    len(matched),
)

print(
    "Final series-method comparisons:",
    len(overall),
)


if (
    all_predictions_finite
    and strict_method_match
    and identical_n
    and m2_m3_sample_match
):

    print(
        "\nOVERALL MODEL COMPARISON AUDIT: PASS"
    )

else:

    print(
        "\nOVERALL MODEL COMPARISON AUDIT: "
        "REVIEW REQUIRED"
    )


# ============================================================
# 27. PRINT WIN SUMMARY
# ============================================================

print("\n" + "=" * 95)
print("MODEL WIN / RANK SUMMARY")
print("=" * 95)

print(
    win_summary
    .sort_values(
        [
            "information_set",
            "mean_mae_rank",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 28. PRINT CENTRAL M2 VS M3 RESULT
# ============================================================

print("\n" + "=" * 95)
print("CENTRAL RESEARCH QUESTION: DOES M3 IMPROVE OVER M2?")
print("=" * 95)

print(
    m2_m3_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 29. BEST METHODS BY SERIES
# ============================================================

best_by_mae = ranking[
    ranking[
        "mae_rank"
    ]
    == 1
][
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "information_set",
        "forecast_method",
        "n",
        "mae",
        "rmse",
        "mase",
    ]
].copy()


print("\n" + "=" * 95)
print("BEST METHOD(S) BY MAE")
print("=" * 95)

print(
    best_by_mae
    .sort_values(
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "information_set",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 30. OUTPUT LOCATIONS
# ============================================================

print("\nSaved:")
print(OVERALL_FILE)
print(RANKING_FILE)
print(WIN_FILE)
print(PATHOGEN_FILE)
print(HORIZON_FILE)
print(OVERALL_SUMMARY_FILE)
print(M2_M3_EFFECT_FILE)
print(M2_M3_SUMMARY_FILE)
print(MATCHED_SAMPLE_FILE)


print("\nImportant interpretation rules:")

print(
    "- Lower MAE, RMSE and MASE are better."
)

print(
    "- Rankings are based only on strictly "
    "matched TEST observations."
)

print(
    "- Positive M3-vs-M2 improvement means "
    "adding cross-pathogen information reduced error."
)

print(
    "- Negative M3-vs-M2 improvement means M3 "
    "performed worse than M2."
)

print(
    "- Indonesia RSV retains MAE/RMSE but MASE "
    "remains undefined because its training-only "
    "MASE scale is invalid."
)

print(
    "- Baselines are duplicated across M2/M3 only "
    "as common benchmarks; they do not actually "
    "use M2 or M3 predictor sets."
)

print(
    "\nOVERALL MODEL COMPARISON COMPLETE"
)