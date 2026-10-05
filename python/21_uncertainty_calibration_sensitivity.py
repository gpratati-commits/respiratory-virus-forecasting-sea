"""
Validation-calibrated uncertainty sensitivity analysis.

Purpose
-------
The primary Negative Binomial probabilistic forecasts in Step 20
showed conservative / over-wide prediction intervals.

This sensitivity analysis asks whether uncertainty calibration can
be improved WITHOUT using the test set for tuning.

Design
------
VALIDATION:
    - Mean predictions come from models fitted on TRAIN only.
    - Negative Binomial dispersion is alpha_train.
    - A global dispersion multiplier is selected separately for
      M2 and M3 using validation performance only.

TEST:
    - Mean predictions come from the frozen final models fitted on
      TRAIN + VALIDATION.
    - Dispersion is alpha_final multiplied by the validation-selected
      calibration factor.
    - TEST is used only for final evaluation.

This is a sensitivity analysis and does not replace the primary
probabilistic analysis from Step 20.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAT_PRED_FILE = (
    ROOT
    / "outputs"
    / "predictions"
    / "statistical_predictions.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "statistical_model_diagnostics.csv"
)

MASE_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "baseline_mase_scales.csv"
)

PRIMARY_PROB_METRICS_FILE = (
    ROOT
    / "outputs"
    / "tables"
    / "probabilistic_interval_metrics.csv"
)

TABLE_DIR = ROOT / "outputs" / "tables"
PRED_DIR = ROOT / "outputs" / "predictions"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

TUNING_FILE = (
    TABLE_DIR
    / "uncertainty_calibration_tuning.csv"
)

SELECTED_FILE = (
    TABLE_DIR
    / "uncertainty_calibration_selected.csv"
)

CALIBRATED_METRICS_FILE = (
    TABLE_DIR
    / "uncertainty_calibrated_interval_metrics.csv"
)

CALIBRATED_COMPARISON_FILE = (
    TABLE_DIR
    / "uncertainty_calibrated_m2_m3_comparison.csv"
)

SENSITIVITY_SUMMARY_FILE = (
    TABLE_DIR
    / "uncertainty_calibration_sensitivity_summary.csv"
)

CALIBRATED_PREDICTION_FILE = (
    PRED_DIR
    / "uncertainty_calibrated_predictions.csv"
)


# ============================================================
# 2. PRESPECIFIED CALIBRATION GRID
# ============================================================

CALIBRATION_FACTORS = [
    0.05,
    0.10,
    0.20,
    0.32,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
]

INTERVALS = {
    "50": 0.50,
    "80": 0.20,
    "95": 0.05,
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 95)
print("VALIDATION-CALIBRATED UNCERTAINTY SENSITIVITY ANALYSIS")
print("=" * 95)

required_files = [
    STAT_PRED_FILE,
    DIAGNOSTIC_FILE,
    MASE_FILE,
    PRIMARY_PROB_METRICS_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

pred = pd.read_csv(STAT_PRED_FILE)
diag = pd.read_csv(DIAGNOSTIC_FILE)
mase = pd.read_csv(MASE_FILE)
primary_metrics = pd.read_csv(
    PRIMARY_PROB_METRICS_FILE
)

for column in [
    "forecast_origin",
    "target_date",
]:
    pred[column] = pd.to_datetime(
        pred[column],
        errors="coerce",
    )


# ============================================================
# 4. PRIMARY REGIONAL M2/M3 ONLY
# ============================================================

pred = pred[
    (pred["analysis_group"] == "primary_regional")
    & (pred["model"].isin(["M2", "M3"]))
].copy()

diag = diag[
    (diag["analysis_group"] == "primary_regional")
    & (diag["status"] == "success")
    & (diag["model"].isin(["M2", "M3"]))
].copy()

if pred.empty:
    raise ValueError(
        "No primary M2/M3 statistical predictions found."
    )


# ============================================================
# 5. DISPERSION TABLE
# ============================================================

dispersion = diag[
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
        "alpha_train",
        "alpha_final",
    ]
].drop_duplicates()

if dispersion.duplicated(
    subset=[
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
    ]
).any():
    raise ValueError(
        "Duplicate dispersion rows found."
    )


# ============================================================
# 6. MASE SCALE TABLE
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
        "Duplicate MASE scale rows found."
    )


# ============================================================
# 7. HELPERS
# ============================================================

def nb_quantile(mu, alpha, probability):
    """
    NB2:
        Var(Y) = mu + alpha * mu^2

    scipy parameterisation:
        n = 1 / alpha
        p = n / (n + mu)
    """

    mu = float(mu)
    alpha = float(alpha)

    if not np.isfinite(mu):
        return np.nan

    if not np.isfinite(alpha):
        return np.nan

    if mu <= 0:
        return 0.0

    alpha = max(
        alpha,
        1e-12,
    )

    n = 1.0 / alpha

    p = n / (
        n + mu
    )

    return float(
        nbinom.ppf(
            probability,
            n,
            p,
        )
    )


def interval_score(
    y,
    lower,
    upper,
    alpha_level,
):
    width = upper - lower

    below = (
        (2.0 / alpha_level)
        * (lower - y)
        if y < lower
        else 0.0
    )

    above = (
        (2.0 / alpha_level)
        * (y - upper)
        if y > upper
        else 0.0
    )

    return (
        width
        + below
        + above
    )


def row_wis(row):
    """
    Weighted Interval Score based on:
    median + 50%, 80%, 95% intervals.
    """

    y = float(
        row["target_value"]
    )

    median = float(
        row["median"]
    )

    total = (
        0.5
        * abs(
            y - median
        )
    )

    for label, alpha_level in (
        INTERVALS.items()
    ):

        score = interval_score(
            y=y,
            lower=float(
                row[f"lower_{label}"]
            ),
            upper=float(
                row[f"upper_{label}"]
            ),
            alpha_level=alpha_level,
        )

        total += (
            alpha_level / 2.0
        ) * score

    return (
        total
        / (
            len(INTERVALS)
            + 0.5
        )
    )


def make_probabilistic_predictions(
    data,
    alpha_column,
    factor,
):
    """
    Apply one dispersion calibration factor.

    factor < 1:
        narrower distribution

    factor > 1:
        wider distribution
    """

    out = data.copy()

    out["calibration_factor"] = factor

    out["calibrated_alpha"] = (
        out[alpha_column]
        * factor
    )

    out["median"] = [
        nb_quantile(
            mu,
            alpha,
            0.50,
        )
        for mu, alpha
        in zip(
            out["prediction"],
            out["calibrated_alpha"],
        )
    ]

    for label, alpha_level in (
        INTERVALS.items()
    ):

        lower_probability = (
            alpha_level / 2.0
        )

        upper_probability = (
            1.0
            - alpha_level / 2.0
        )

        out[
            f"lower_{label}"
        ] = [
            nb_quantile(
                mu,
                alpha,
                lower_probability,
            )
            for mu, alpha
            in zip(
                out["prediction"],
                out["calibrated_alpha"],
            )
        ]

        out[
            f"upper_{label}"
        ] = [
            nb_quantile(
                mu,
                alpha,
                upper_probability,
            )
            for mu, alpha
            in zip(
                out["prediction"],
                out["calibrated_alpha"],
            )
        ]

        out[
            f"covered_{label}"
        ] = (
            (
                out["target_value"]
                >= out[
                    f"lower_{label}"
                ]
            )
            & (
                out["target_value"]
                <= out[
                    f"upper_{label}"
                ]
            )
        )

        out[
            f"width_{label}"
        ] = (
            out[
                f"upper_{label}"
            ]
            - out[
                f"lower_{label}"
            ]
        )

    nested = (
        (
            out["lower_95"]
            <= out["lower_80"]
        )
        & (
            out["lower_80"]
            <= out["lower_50"]
        )
        & (
            out["lower_50"]
            <= out["upper_50"]
        )
        & (
            out["upper_50"]
            <= out["upper_80"]
        )
        & (
            out["upper_80"]
            <= out["upper_95"]
        )
    )

    if not nested.all():
        raise ValueError(
            "Prediction intervals are not nested."
        )

    out["wis"] = out.apply(
        row_wis,
        axis=1,
    )

    return out


def series_summary(data):
    """
    Summarise one probabilistic forecast dataset
    by country/pathogen/horizon/model.
    """

    rows = []

    group_columns = [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
    ]

    for keys, group in data.groupby(
        group_columns,
        dropna=False,
    ):

        (
            country,
            pathogen,
            horizon,
            model,
        ) = keys

        row = {
            "country": country,
            "target_pathogen": pathogen,
            "forecast_horizon_weeks": horizon,
            "model": model,
            "n": len(group),
            "mean_wis": group[
                "wis"
            ].mean(),
        }

        for label in INTERVALS:

            row[
                f"coverage_{label}"
            ] = (
                group[
                    f"covered_{label}"
                ].mean()
            )

            row[
                f"mean_width_{label}"
            ] = (
                group[
                    f"width_{label}"
                ].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. VALIDATION DATA
# ============================================================

validation = pred[
    pred["split"] == "validation"
].copy()

validation = validation.merge(
    dispersion,
    on=[
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
    ],
    how="left",
    validate="many_to_one",
)

validation = validation.merge(
    scale_table,
    on=[
        "country",
        "target_pathogen",
    ],
    how="left",
    validate="many_to_one",
)

# ============================================================
# VALIDATION INPUT QC
# ============================================================

core_validation_columns = [
    "target_value",
    "prediction",
    "alpha_train",
]

for column in core_validation_columns:
    validation[column] = pd.to_numeric(
        validation[column],
        errors="coerce",
    )

core_invalid = (
    validation[
        core_validation_columns
    ].isna().any(axis=1)
    | (
        ~np.isfinite(
            validation[
                core_validation_columns
            ].to_numpy(dtype=float)
        )
    ).any(axis=1)
)

if core_invalid.any():

    print(
        "\nInvalid core validation rows:"
    )

    print(
        validation.loc[
            core_invalid,
            [
                "country",
                "target_pathogen",
                "forecast_horizon_weeks",
                "model",
                *core_validation_columns,
            ],
        ].to_string(index=False)
    )

    raise ValueError(
        "Invalid target, prediction, or alpha_train "
        "found in validation data."
    )


validation[
    "mase_scale_training_only"
] = pd.to_numeric(
    validation[
        "mase_scale_training_only"
    ],
    errors="coerce",
)

valid_scale = (
    validation[
        "mase_scale_training_only"
    ].notna()
    & np.isfinite(
        validation[
            "mase_scale_training_only"
        ]
    )
    & (
        validation[
            "mase_scale_training_only"
        ]
        > 0
    )
)

excluded_for_scaling = (
    validation.loc[
        ~valid_scale,
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "model",
            "mase_scale_training_only",
        ],
    ]
    .drop_duplicates()
)

print(
    "\nValidation rows excluded from "
    "scaled-WIS calibration objective:",
    (~valid_scale).sum(),
)

if not excluded_for_scaling.empty:

    print(
        "\nSeries excluded from calibration-factor "
        "selection because MASE scale is undefined "
        "or non-positive:"
    )

    print(
        excluded_for_scaling
        .sort_values(
            [
                "country",
                "target_pathogen",
                "forecast_horizon_weeks",
                "model",
            ]
        )
        .to_string(index=False)
    )


calibration_validation = (
    validation.loc[
        valid_scale
    ].copy()
)

if calibration_validation.empty:
    raise ValueError(
        "No validation observations with a valid "
        "training-only MASE scale remain."
    )


# ============================================================
# 9. VALIDATION-ONLY CALIBRATION TUNING
# ============================================================

tuning_rows = []

for model in [
    "M2",
    "M3",
]:

    model_validation = calibration_validation[
    calibration_validation["model"] == model
].copy()

    for factor in CALIBRATION_FACTORS:

        candidate = (
            make_probabilistic_predictions(
                model_validation,
                alpha_column="alpha_train",
                factor=factor,
            )
        )

        candidate[
            "scaled_wis"
        ] = (
            candidate["wis"]
            / candidate[
                "mase_scale_training_only"
            ]
        )

        # First average within each epidemiological
        # forecasting series.
        per_series = (
            candidate
            .groupby(
                [
                    "country",
                    "target_pathogen",
                    "forecast_horizon_weeks",
                ],
                as_index=False,
            )
            .agg(
                mean_scaled_wis=(
                    "scaled_wis",
                    "mean",
                ),
                coverage_50=(
                    "covered_50",
                    "mean",
                ),
                coverage_80=(
                    "covered_80",
                    "mean",
                ),
                coverage_95=(
                    "covered_95",
                    "mean",
                ),
            )
        )

        # Macro averaging gives each country/pathogen/horizon
        # combination equal weight, preventing large COVID
        # counts from dominating the tuning objective.
        macro_scaled_wis = (
            per_series[
                "mean_scaled_wis"
            ].mean()
        )

        mean_coverage_50 = (
            per_series[
                "coverage_50"
            ].mean()
        )

        mean_coverage_80 = (
            per_series[
                "coverage_80"
            ].mean()
        )

        mean_coverage_95 = (
            per_series[
                "coverage_95"
            ].mean()
        )

        tuning_rows.append({
            "model": model,
            "calibration_factor": factor,
            "macro_mean_scaled_wis": (
                macro_scaled_wis
            ),
            "mean_coverage_50": (
                mean_coverage_50
            ),
            "mean_coverage_80": (
                mean_coverage_80
            ),
            "mean_coverage_95": (
                mean_coverage_95
            ),
            "number_of_series": (
                len(per_series)
            ),
        })


tuning = pd.DataFrame(
    tuning_rows
)


# ============================================================
# 10. SELECT FACTOR USING VALIDATION ONLY
# ============================================================

selected_rows = []

for model in [
    "M2",
    "M3",
]:

    x = tuning[
        tuning["model"] == model
    ].copy()

    x = x.sort_values(
        [
            "macro_mean_scaled_wis",
            "calibration_factor",
        ]
    )

    best = x.iloc[0]

    selected_rows.append({
        "model": model,
        "selected_calibration_factor": (
            best[
                "calibration_factor"
            ]
        ),
        "validation_macro_mean_scaled_wis": (
            best[
                "macro_mean_scaled_wis"
            ]
        ),
        "validation_mean_coverage_50": (
            best[
                "mean_coverage_50"
            ]
        ),
        "validation_mean_coverage_80": (
            best[
                "mean_coverage_80"
            ]
        ),
        "validation_mean_coverage_95": (
            best[
                "mean_coverage_95"
            ]
        ),
    })


selected = pd.DataFrame(
    selected_rows
)

print("\n" + "=" * 95)
print("VALIDATION-SELECTED CALIBRATION FACTORS")
print("=" * 95)

print(
    selected.to_string(
        index=False
    )
)


# ============================================================
# 11. TEST DATA
# ============================================================

test = pred[
    pred["split"] == "test"
].copy()

test = test.merge(
    dispersion,
    on=[
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
    ],
    how="left",
    validate="many_to_one",
)

if test[
    [
        "target_value",
        "prediction",
        "alpha_final",
    ]
].isna().any().any():
    raise ValueError(
        "Missing final test uncertainty inputs."
    )


# ============================================================
# 12. APPLY FROZEN VALIDATION FACTORS TO TEST
# ============================================================

calibrated_parts = []

for model in [
    "M2",
    "M3",
]:

    factor = float(
        selected.loc[
            selected["model"] == model,
            "selected_calibration_factor",
        ].iloc[0]
    )

    model_test = test[
        test["model"] == model
    ].copy()

    calibrated = (
        make_probabilistic_predictions(
            model_test,
            alpha_column="alpha_final",
            factor=factor,
        )
    )

    calibrated_parts.append(
        calibrated
    )


calibrated_test = pd.concat(
    calibrated_parts,
    ignore_index=True,
)


# ============================================================
# 13. TEST METRICS
# ============================================================

calibrated_metrics = (
    series_summary(
        calibrated_test
    )
)

calibrated_metrics.insert(
    0,
    "analysis_group",
    "primary_regional",
)


# ============================================================
# 14. CALIBRATED M2 VS M3 COMPARISON
# ============================================================

comparison = calibrated_metrics.pivot_table(
    index=[
        "analysis_group",
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
    ],
    columns="model",
    values=[
        "n",
        "mean_wis",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "mean_width_50",
        "mean_width_80",
        "mean_width_95",
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

comparison[
    "m3_vs_m2_wis_improvement_pct"
] = np.where(
    (
        comparison[
            "mean_wis_m2"
        ]
        > 0
    ),
    100.0
    * (
        comparison[
            "mean_wis_m2"
        ]
        - comparison[
            "mean_wis_m3"
        ]
    )
    / comparison[
        "mean_wis_m2"
    ],
    np.nan,
)


# ============================================================
# 15. PRIMARY VS CALIBRATED SENSITIVITY SUMMARY
# ============================================================

primary = primary_metrics[
    (primary_metrics["analysis_group"] == "primary_regional")
    & (
        primary_metrics["model"]
        .isin(["M2", "M3"])
    )
].copy()

summary_rows = []

for model in [
    "M2",
    "M3",
]:

    original = primary[
        primary["model"] == model
    ]

    calibrated = calibrated_metrics[
        calibrated_metrics["model"] == model
    ]

    factor = float(
        selected.loc[
            selected["model"] == model,
            "selected_calibration_factor",
        ].iloc[0]
    )

    summary_rows.append({
        "model": model,
        "selected_calibration_factor": factor,

        "primary_mean_coverage_50":
            original["coverage_50"].mean(),

        "calibrated_mean_coverage_50":
            calibrated["coverage_50"].mean(),

        "primary_mean_coverage_80":
            original["coverage_80"].mean(),

        "calibrated_mean_coverage_80":
            calibrated["coverage_80"].mean(),

        "primary_mean_coverage_95":
            original["coverage_95"].mean(),

        "calibrated_mean_coverage_95":
            calibrated["coverage_95"].mean(),

        "primary_mean_wis":
            original["mean_wis"].mean(),

        "calibrated_mean_wis":
            calibrated["mean_wis"].mean(),

        "primary_mean_width_80":
            original[
                "mean_width_80"
            ].mean(),

        "calibrated_mean_width_80":
            calibrated[
                "mean_width_80"
            ].mean(),
    })


sensitivity_summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 16. QC
# ============================================================

finite_columns = [
    "median",
    "lower_50",
    "upper_50",
    "lower_80",
    "upper_80",
    "lower_95",
    "upper_95",
    "wis",
]

finite = np.isfinite(
    calibrated_test[
        finite_columns
    ].to_numpy(dtype=float)
).all()

nested = (
    (
        calibrated_test[
            "lower_95"
        ]
        <= calibrated_test[
            "lower_80"
        ]
    )
    & (
        calibrated_test[
            "lower_80"
        ]
        <= calibrated_test[
            "lower_50"
        ]
    )
    & (
        calibrated_test[
            "lower_50"
        ]
        <= calibrated_test[
            "upper_50"
        ]
    )
    & (
        calibrated_test[
            "upper_50"
        ]
        <= calibrated_test[
            "upper_80"
        ]
    )
    & (
        calibrated_test[
            "upper_80"
        ]
        <= calibrated_test[
            "upper_95"
        ]
    )
).all()

same_n = (
    comparison["n_m2"]
    == comparison["n_m3"]
).all()


# ============================================================
# 17. SAVE OUTPUTS
# ============================================================

tuning.to_csv(
    TUNING_FILE,
    index=False,
)

selected.to_csv(
    SELECTED_FILE,
    index=False,
)

calibrated_test.to_csv(
    CALIBRATED_PREDICTION_FILE,
    index=False,
)

calibrated_metrics.to_csv(
    CALIBRATED_METRICS_FILE,
    index=False,
)

comparison.to_csv(
    CALIBRATED_COMPARISON_FILE,
    index=False,
)

sensitivity_summary.to_csv(
    SENSITIVITY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 18. REPORT RESULTS
# ============================================================

print("\n" + "=" * 95)
print("PRIMARY VS VALIDATION-CALIBRATED TEST PERFORMANCE")
print("=" * 95)

print(
    sensitivity_summary
    .to_string(index=False)
)


print("\n" + "=" * 95)
print("CALIBRATED TEST M2 VS M3")
print("=" * 95)

display_columns = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "coverage_80_m2",
    "coverage_80_m3",
    "coverage_95_m2",
    "coverage_95_m3",
    "mean_wis_m2",
    "mean_wis_m3",
    "m3_vs_m2_wis_improvement_pct",
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
    .to_string(index=False)
)


print("\n" + "=" * 95)
print("UNCERTAINTY CALIBRATION QC")
print("=" * 95)

print(
    "All calibrated quantities finite:",
    finite
)

print(
    "All calibrated intervals nested:",
    nested
)

print(
    "M2/M3 identical test sample:",
    same_n
)

boundary = selected[
    selected[
        "selected_calibration_factor"
    ].isin(
        [
            min(CALIBRATION_FACTORS),
            max(CALIBRATION_FACTORS),
        ]
    )
]

print(
    "Number of selected factors at tuning-grid boundary:",
    len(boundary)
)

if not boundary.empty:
    print("\nBoundary selections:")
    print(
        boundary.to_string(
            index=False
        )
    )


if (
    finite
    and nested
    and same_n
):
    print(
        "\nUNCERTAINTY CALIBRATION AUDIT: PASS"
    )
else:
    print(
        "\nUNCERTAINTY CALIBRATION AUDIT: "
        "REVIEW REQUIRED"
    )


print("\nSaved tuning results:")
print(TUNING_FILE)

print("\nSaved selected factors:")
print(SELECTED_FILE)

print("\nSaved calibrated metrics:")
print(CALIBRATED_METRICS_FILE)

print("\nSaved calibrated M2/M3 comparison:")
print(CALIBRATED_COMPARISON_FILE)

print("\nSaved sensitivity summary:")
print(SENSITIVITY_SUMMARY_FILE)

print("\nSaved calibrated predictions:")
print(CALIBRATED_PREDICTION_FILE)

print("\nImportant:")
print(
    "- Calibration factors were selected using "
    "VALIDATION only."
)

print(
    "- TEST was not used to choose calibration factors."
)

print(
    "- Step 20 remains the primary probabilistic analysis."
)

print(
    "- Step 21 is explicitly a sensitivity analysis."
)

print(
    "\nUNCERTAINTY CALIBRATION SENSITIVITY COMPLETE"
)