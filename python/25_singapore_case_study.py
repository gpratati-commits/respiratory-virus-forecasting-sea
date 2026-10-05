"""
Step 25: Singapore influenza–SARS-CoV-2 case-study synthesis.

Purpose
-------
Singapore is analysed separately from the primary regional
three-pathogen analysis because usable Singapore RSV observations
were not available in the FluNet extract used in this project.

The Singapore case study therefore evaluates:

    Influenza
    SARS-CoV-2

Forecast horizons:

    1 week
    2 weeks
    4 weeks

Forecasting methods:

    Baseline forecasts
    Negative Binomial
    XGBoost

For Negative Binomial and XGBoost, the central comparison is:

    M2 = own-pathogen history + seasonality
    M3 = M2 + activity of the other respiratory pathogen

Positive M3-vs-M2 improvement means M3 reduced forecast error.

Important
---------
Only TEST-set observations are used for final performance evaluation.
No test-set result is used to tune or refit a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
PREDICTION_DIR = ROOT / "outputs" / "predictions"

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "singapore_case_study"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_FILE = (
    DATA_DIR
    / "singapore_respiratory_weekly.csv"
)

BASELINE_FILE = (
    PREDICTION_DIR
    / "baseline_predictions.csv"
)

NB_FILE = (
    PREDICTION_DIR
    / "statistical_predictions.csv"
)

XGB_FILE = (
    PREDICTION_DIR
    / "xgboost_predictions.csv"
)


OUT_PATHOGEN_SUMMARY = (
    TABLE_DIR
    / "singapore_pathogen_summary.csv"
)

OUT_FORECAST_PERFORMANCE = (
    TABLE_DIR
    / "singapore_forecast_performance.csv"
)

OUT_CROSS_PATHOGEN = (
    TABLE_DIR
    / "singapore_cross_pathogen_effect.csv"
)

OUT_CASE_SUMMARY = (
    TABLE_DIR
    / "singapore_case_study_summary.csv"
)

OUT_MATCHED_QC = (
    TABLE_DIR
    / "singapore_m2_m3_matched_qc.csv"
)


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file missing:\n{path}"
        )


def singapore_rows(df):

    result = df.copy()

    if "country" in result.columns:
        result = result[
            result["country"].eq("Singapore")
        ].copy()

    if "analysis_group" in result.columns:

        case = result[
            result["analysis_group"].eq(
                "singapore_case_study"
            )
        ].copy()

        if not case.empty:
            result = case

    return result


def detect_prediction_column(
    df,
    candidates,
    name,
):

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify prediction column "
        f"in {name}.\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


def calculate_metrics(
    observed,
    predicted,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        & np.isfinite(predicted)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if len(observed) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
        }

    error = predicted - observed

    return {
        "n": len(observed),

        "mae":
            np.mean(
                np.abs(error)
            ),

        "rmse":
            np.sqrt(
                np.mean(
                    error ** 2
                )
            ),

        "mean_error":
            np.mean(error),
    }


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

for file in [
    RAW_FILE,
    BASELINE_FILE,
    NB_FILE,
    XGB_FILE,
]:
    require_file(file)


print("\n" + "=" * 95)
print("STEP 25: SINGAPORE INFLUENZA–SARS-CoV-2 CASE STUDY")
print("=" * 95)


# ============================================================
# 4. LOAD RAW SINGAPORE WEEKLY DATA
# ============================================================

raw = pd.read_csv(
    RAW_FILE
)

raw = singapore_rows(raw)


date_candidates = [
    "week_start",
    "date",
    "week",
]

date_column = None

for candidate in date_candidates:

    if candidate in raw.columns:
        date_column = candidate
        break


if date_column is None:
    raise ValueError(
        "Could not identify Singapore weekly date column."
    )


raw[date_column] = pd.to_datetime(
    raw[date_column],
    errors="coerce",
)


# Restrict descriptive case-study window to 2020–2023.

raw_case = raw[
    (
        raw[date_column]
        >= pd.Timestamp("2020-01-01")
    )
    &
    (
        raw[date_column]
        <= pd.Timestamp("2023-12-31")
    )
].copy()


# ============================================================
# 5. RAW PATHOGEN SUMMARY
# ============================================================

activity_candidates = {
    "influenza": [
        "influenza_positive",
        "influenza_activity",
    ],

    "sarscov2": [
        "sarscov2_cases",
        "covid_cases",
    ],
}


summary_rows = []


for pathogen, candidates in (
    activity_candidates.items()
):

    column = None

    for candidate in candidates:

        if candidate in raw_case.columns:
            column = candidate
            break

    if column is None:
        continue

    values = pd.to_numeric(
        raw_case[column],
        errors="coerce",
    )

    valid = values.dropna()

    summary_rows.append({
        "country":
            "Singapore",

        "pathogen":
            pathogen,

        "activity_column":
            column,

        "start_date":
            raw_case.loc[
                values.notna(),
                date_column,
            ].min(),

        "end_date":
            raw_case.loc[
                values.notna(),
                date_column,
            ].max(),

        "non_missing_weeks":
            int(
                values.notna().sum()
            ),

        "zero_weeks":
            int(
                (valid == 0).sum()
            ),

        "mean_activity":
            valid.mean(),

        "median_activity":
            valid.median(),

        "maximum_activity":
            valid.max(),
    })


pathogen_summary = pd.DataFrame(
    summary_rows
)


pathogen_summary.to_csv(
    OUT_PATHOGEN_SUMMARY,
    index=False,
)


# ============================================================
# 6. SINGAPORE TIME-SERIES FIGURES
# ============================================================

figure_specs = [
    (
        "influenza",
        [
            "influenza_positive",
            "influenza_activity",
        ],
        "Singapore influenza activity, 2020–2023",
        "Influenza activity",
        "singapore_influenza_2020_2023.png",
    ),

    (
        "sarscov2",
        [
            "sarscov2_cases",
            "covid_cases",
        ],
        "Singapore SARS-CoV-2 activity, 2020–2023",
        "Weekly reported cases",
        "singapore_sarscov2_2020_2023.png",
    ),
]


for (
    pathogen,
    candidates,
    title,
    ylabel,
    filename,
) in figure_specs:

    column = None

    for candidate in candidates:

        if candidate in raw_case.columns:
            column = candidate
            break

    if column is None:
        continue

    figure_data = raw_case[
        [
            date_column,
            column,
        ]
    ].copy()

    figure_data[column] = pd.to_numeric(
        figure_data[column],
        errors="coerce",
    )

    figure_data = figure_data.dropna()

    plt.figure(
        figsize=(10, 4.5)
    )

    plt.plot(
        figure_data[date_column],
        figure_data[column],
    )

    plt.xlabel("Week")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 7. BASELINE TEST PERFORMANCE
# ============================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

baseline = singapore_rows(
    baseline
)


if "split" in baseline.columns:

    baseline = baseline[
        baseline["split"].eq("test")
    ].copy()


baseline_methods = {
    "pred_persistence":
        "Baseline: Persistence",

    "pred_recent_mean4":
        "Baseline: Recent mean (4 weeks)",

    "pred_seasonal_naive52":
        "Baseline: Seasonal naive (52 weeks)",
}


missing_baselines = [
    column
    for column in baseline_methods
    if column not in baseline.columns
]


if missing_baselines:
    raise ValueError(
        "Missing Singapore baseline columns: "
        + ", ".join(
            missing_baselines
        )
    )


performance_rows = []


for (
    prediction_column,
    method_name,
) in baseline_methods.items():

    for (
        pathogen,
        horizon,
    ), group in baseline.groupby(
        [
            "target_pathogen",
            "forecast_horizon_weeks",
        ]
    ):

        metrics = calculate_metrics(
            group["target_value"],
            group[prediction_column],
        )

        performance_rows.append({
            "country":
                "Singapore",

            "forecast_method":
                method_name,

            "information_set":
                "Benchmark",

            "target_pathogen":
                pathogen,

            "forecast_horizon_weeks":
                horizon,

            **metrics,
        })


# ============================================================
# 8. LOAD NEGATIVE BINOMIAL + XGBOOST
# ============================================================

nb = pd.read_csv(
    NB_FILE
)

xgb = pd.read_csv(
    XGB_FILE
)


nb = singapore_rows(nb)
xgb = singapore_rows(xgb)


if "split" in nb.columns:

    nb = nb[
        nb["split"].eq("test")
    ].copy()


if "split" in xgb.columns:

    xgb = xgb[
        xgb["split"].eq("test")
    ].copy()


# ============================================================
# 9. DETECT POINT-PREDICTION COLUMNS
# ============================================================

nb_prediction_column = (
    detect_prediction_column(
        nb,
        [
            "prediction",
            "nb_prediction",
            "predicted_mean",
            "mean_prediction",
            "forecast_mean",
            "y_pred",
        ],
        "statistical_predictions.csv",
    )
)


xgb_prediction_column = (
    detect_prediction_column(
        xgb,
        [
            "prediction",
            "xgb_prediction",
            "predicted_mean",
            "mean_prediction",
            "forecast_mean",
            "y_pred",
        ],
        "xgboost_predictions.csv",
    )
)


print(
    "\nDetected Negative Binomial "
    "prediction column:",
    nb_prediction_column,
)

print(
    "Detected XGBoost prediction column:",
    xgb_prediction_column,
)


# ============================================================
# 10. MODEL TEST PERFORMANCE
# ============================================================

model_inputs = [
    (
        "Negative Binomial",
        nb,
        nb_prediction_column,
    ),

    (
        "XGBoost",
        xgb,
        xgb_prediction_column,
    ),
]


for (
    forecast_method,
    data,
    prediction_column,
) in model_inputs:

    for (
        pathogen,
        horizon,
        model,
    ), group in data.groupby(
        [
            "target_pathogen",
            "forecast_horizon_weeks",
            "model",
        ]
    ):

        metrics = calculate_metrics(
            group["target_value"],
            group[prediction_column],
        )

        performance_rows.append({
            "country":
                "Singapore",

            "forecast_method":
                forecast_method,

            "information_set":
                model,

            "target_pathogen":
                pathogen,

            "forecast_horizon_weeks":
                horizon,

            **metrics,
        })


performance = pd.DataFrame(
    performance_rows
)


performance.to_csv(
    OUT_FORECAST_PERFORMANCE,
    index=False,
)


# ============================================================
# 11. MATCHED M2 VS M3 COMPARISON
# ============================================================

cross_rows = []
matched_qc_rows = []


def compare_m2_m3(
    data,
    prediction_column,
    forecast_method,
):

    required = [
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
        "forecast_origin",
        "target_date",
        "target_value",
        prediction_column,
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{forecast_method} missing columns: "
            f"{missing}"
        )


    local = data.copy()

    local["forecast_origin"] = (
        pd.to_datetime(
            local["forecast_origin"],
            errors="coerce",
        )
    )

    local["target_date"] = (
        pd.to_datetime(
            local["target_date"],
            errors="coerce",
        )
    )


    for (
        pathogen,
        horizon,
    ), group in local.groupby(
        [
            "target_pathogen",
            "forecast_horizon_weeks",
        ]
    ):

        m2 = group[
            group["model"].eq("M2")
        ].copy()

        m3 = group[
            group["model"].eq("M3")
        ].copy()


        keys = [
            "forecast_origin",
            "target_date",
        ]


        if m2.duplicated(keys).any():

            raise ValueError(
                f"Duplicate M2 forecast rows for "
                f"{forecast_method}, "
                f"{pathogen}, horizon {horizon}."
            )


        if m3.duplicated(keys).any():

            raise ValueError(
                f"Duplicate M3 forecast rows for "
                f"{forecast_method}, "
                f"{pathogen}, horizon {horizon}."
            )


        matched = m2[
            keys
            + [
                "target_value",
                prediction_column,
            ]
        ].merge(
            m3[
                keys
                + [
                    "target_value",
                    prediction_column,
                ]
            ],
            on=keys,
            how="inner",
            suffixes=(
                "_m2",
                "_m3",
            ),
            validate="one_to_one",
        )


        targets_identical = np.allclose(
            matched[
                "target_value_m2"
            ],
            matched[
                "target_value_m3"
            ],
            equal_nan=True,
        )


        matched_qc_rows.append({
            "forecast_method":
                forecast_method,

            "target_pathogen":
                pathogen,

            "forecast_horizon_weeks":
                horizon,

            "m2_rows":
                len(m2),

            "m3_rows":
                len(m3),

            "matched_rows":
                len(matched),

            "targets_identical":
                targets_identical,
        })


        if not targets_identical:

            raise ValueError(
                "M2 and M3 target values differ "
                f"for {forecast_method}, "
                f"{pathogen}, horizon {horizon}."
            )


        metrics_m2 = calculate_metrics(
            matched[
                "target_value_m2"
            ],
            matched[
                f"{prediction_column}_m2"
            ],
        )


        metrics_m3 = calculate_metrics(
            matched[
                "target_value_m3"
            ],
            matched[
                f"{prediction_column}_m3"
            ],
        )


        mae_m2 = metrics_m2["mae"]
        mae_m3 = metrics_m3["mae"]

        rmse_m2 = metrics_m2["rmse"]
        rmse_m3 = metrics_m3["rmse"]


        if (
            np.isfinite(mae_m2)
            and mae_m2 > 0
        ):

            mae_improvement = (
                100.0
                * (
                    mae_m2
                    - mae_m3
                )
                / mae_m2
            )

        else:

            mae_improvement = np.nan


        if (
            np.isfinite(rmse_m2)
            and rmse_m2 > 0
        ):

            rmse_improvement = (
                100.0
                * (
                    rmse_m2
                    - rmse_m3
                )
                / rmse_m2
            )

        else:

            rmse_improvement = np.nan


        cross_rows.append({
            "country":
                "Singapore",

            "forecast_method":
                forecast_method,

            "target_pathogen":
                pathogen,

            "forecast_horizon_weeks":
                horizon,

            "n":
                len(matched),

            "mae_m2":
                mae_m2,

            "mae_m3":
                mae_m3,

            "m3_vs_m2_mae_improvement_pct":
                mae_improvement,

            "m3_better_mae":
                (
                    mae_m3
                    < mae_m2
                ),

            "rmse_m2":
                rmse_m2,

            "rmse_m3":
                rmse_m3,

            "m3_vs_m2_rmse_improvement_pct":
                rmse_improvement,

            "m3_better_rmse":
                (
                    rmse_m3
                    < rmse_m2
                ),
        })


compare_m2_m3(
    nb,
    nb_prediction_column,
    "Negative Binomial",
)


compare_m2_m3(
    xgb,
    xgb_prediction_column,
    "XGBoost",
)


cross_effect = pd.DataFrame(
    cross_rows
)


matched_qc = pd.DataFrame(
    matched_qc_rows
)


cross_effect.to_csv(
    OUT_CROSS_PATHOGEN,
    index=False,
)


matched_qc.to_csv(
    OUT_MATCHED_QC,
    index=False,
)


# ============================================================
# 12. CASE-STUDY SYNTHESIS
# ============================================================

case_summary = (
    cross_effect
    .groupby(
        "forecast_method",
        as_index=False,
    )
    .agg(
        comparisons=(
            "m3_better_mae",
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

        median_m3_vs_m2_mae_improvement_pct=(
            "m3_vs_m2_mae_improvement_pct",
            "median",
        ),

        mean_m3_vs_m2_mae_improvement_pct=(
            "m3_vs_m2_mae_improvement_pct",
            "mean",
        ),

        median_m3_vs_m2_rmse_improvement_pct=(
            "m3_vs_m2_rmse_improvement_pct",
            "median",
        ),
    )
)


case_summary[
    "m3_mae_win_fraction"
] = (
    case_summary[
        "m3_mae_wins"
    ]
    / case_summary[
        "comparisons"
    ]
)


case_summary[
    "m3_rmse_win_fraction"
] = (
    case_summary[
        "m3_rmse_wins"
    ]
    / case_summary[
        "comparisons"
    ]
)


case_summary.to_csv(
    OUT_CASE_SUMMARY,
    index=False,
)


# ============================================================
# 13. QC
# ============================================================

expected_pathogens = {
    "influenza",
    "sarscov2",
}


observed_pathogens = set(
    cross_effect[
        "target_pathogen"
    ]
    .dropna()
    .unique()
)


pathogens_correct = (
    observed_pathogens
    == expected_pathogens
)


expected_horizons = {
    1,
    2,
    4,
}


observed_horizons = set(
    cross_effect[
        "forecast_horizon_weeks"
    ]
    .dropna()
    .astype(int)
    .unique()
)


horizons_correct = (
    observed_horizons
    == expected_horizons
)


matched_targets_correct = (
    matched_qc[
        "targets_identical"
    ]
    .all()
)


matched_sample_sizes_correct = (
    (
        matched_qc[
            "matched_rows"
        ]
        == matched_qc[
            "m2_rows"
        ]
    )
    &
    (
        matched_qc[
            "matched_rows"
        ]
        == matched_qc[
            "m3_rows"
        ]
    )
).all()


cross_errors_finite = np.isfinite(
    cross_effect[
        [
            "mae_m2",
            "mae_m3",
            "rmse_m2",
            "rmse_m3",
        ]
    ]
    .to_numpy(
        dtype=float
    )
).all()


no_rsv = (
    "rsv"
    not in observed_pathogens
)


audit_pass = all([
    pathogens_correct,
    horizons_correct,
    matched_targets_correct,
    matched_sample_sizes_correct,
    cross_errors_finite,
    no_rsv,
])


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 95)
print("SINGAPORE CASE-STUDY QC")
print("=" * 95)

print(
    "Influenza + SARS-CoV-2 only:",
    pathogens_correct,
)

print(
    "Horizons 1/2/4 present:",
    horizons_correct,
)

print(
    "M2/M3 target values identical:",
    matched_targets_correct,
)

print(
    "M2/M3 matched sample sizes identical:",
    matched_sample_sizes_correct,
)

print(
    "All M2/M3 MAE/RMSE values finite:",
    cross_errors_finite,
)

print(
    "RSV excluded:",
    no_rsv,
)


if audit_pass:

    print(
        "\nSINGAPORE CASE-STUDY AUDIT: PASS"
    )

else:

    print(
        "\nSINGAPORE CASE-STUDY AUDIT: "
        "REVIEW REQUIRED"
    )


print("\n" + "=" * 95)
print("SINGAPORE M2 VS M3 RESULTS")
print("=" * 95)

print(
    cross_effect[
        [
            "forecast_method",
            "target_pathogen",
            "forecast_horizon_weeks",
            "n",
            "mae_m2",
            "mae_m3",
            "m3_vs_m2_mae_improvement_pct",
            "m3_better_mae",
        ]
    ]
    .sort_values(
        [
            "forecast_method",
            "target_pathogen",
            "forecast_horizon_weeks",
        ]
    )
    .to_string(
        index=False
    )
)


print("\n" + "=" * 95)
print("SINGAPORE CROSS-PATHOGEN SUMMARY")
print("=" * 95)

print(
    case_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 95)
print("SINGAPORE TEST FORECAST PERFORMANCE")
print("=" * 95)

print(
    performance[
        [
            "forecast_method",
            "information_set",
            "target_pathogen",
            "forecast_horizon_weeks",
            "n",
            "mae",
            "rmse",
        ]
    ]
    .sort_values(
        [
            "target_pathogen",
            "forecast_horizon_weeks",
            "forecast_method",
            "information_set",
        ]
    )
    .to_string(
        index=False
    )
)


print("\nSaved tables:")

for file in [
    OUT_PATHOGEN_SUMMARY,
    OUT_FORECAST_PERFORMANCE,
    OUT_CROSS_PATHOGEN,
    OUT_CASE_SUMMARY,
    OUT_MATCHED_QC,
]:

    print(file)


print("\nSaved figures:")
print(FIGURE_DIR)


print("\nInterpretation:")
print(
    "- Positive M3-vs-M2 improvement means "
    "the other pathogen improved forecasting."
)

print(
    "- Negative improvement means adding "
    "the other pathogen increased error."
)

print(
    "- Singapore is a two-pathogen secondary "
    "case study and does not include RSV."
)

print(
    "- Ensemble forecasts are not included because "
    "no Singapore ensemble predictions exist."
)

print(
    "- TEST observations are used only for final "
    "evaluation, never for tuning."
)


print(
    "\nSINGAPORE CASE STUDY COMPLETE"
)