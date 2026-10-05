"""
Step 24: Robustness and sensitivity synthesis.

This script does NOT fit or tune forecasting models.

It synthesizes already-frozen test-set results to determine whether the
main conclusions are robust across:

- countries
- pathogens
- forecast horizons
- model families
- ensemble behaviour
- probabilistic calibration sensitivity

Primary scientific question
---------------------------
Does adding information from co-circulating respiratory pathogens (M3)
improve forecasting relative to own-pathogen history + seasonality (M2)?

Important:
Positive m3_vs_m2_mae_improvement_pct means M3 reduced MAE.
Negative values mean M3 performed worse.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "tables"

OVERALL_FILE = (
    TABLE_DIR / "overall_model_comparison.csv"
)

CROSS_FILE = (
    TABLE_DIR / "cross_pathogen_information_effect.csv"
)

ENSEMBLE_FILE = (
    TABLE_DIR / "ensemble_component_comparison.csv"
)

UNCERTAINTY_FILE = (
    TABLE_DIR
    / "uncertainty_calibration_sensitivity_summary.csv"
)


OUT_BY_COUNTRY = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_country.csv"
)

OUT_BY_PATHOGEN = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_pathogen.csv"
)

OUT_BY_HORIZON = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_horizon.csv"
)

OUT_BY_MODEL = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_model.csv"
)

OUT_MODEL_RANKS = (
    TABLE_DIR
    / "robustness_model_ranks.csv"
)

OUT_ENSEMBLE = (
    TABLE_DIR
    / "robustness_ensemble_summary.csv"
)

OUT_UNCERTAINTY = (
    TABLE_DIR
    / "robustness_uncertainty_summary.csv"
)

OUT_SYNTHESIS = (
    TABLE_DIR
    / "robustness_overall_synthesis.csv"
)


# ============================================================
# 2. LOAD INPUTS
# ============================================================

print("\n" + "=" * 95)
print("STEP 24: ROBUSTNESS AND SENSITIVITY SYNTHESIS")
print("=" * 95)

required_files = [
    OVERALL_FILE,
    CROSS_FILE,
    ENSEMBLE_FILE,
    UNCERTAINTY_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required Step-24 input missing:\n{file}"
        )


overall = pd.read_csv(OVERALL_FILE)
cross = pd.read_csv(CROSS_FILE)
ensemble = pd.read_csv(ENSEMBLE_FILE)
uncertainty = pd.read_csv(UNCERTAINTY_FILE)


# ============================================================
# 3. HELPER
# ============================================================

def require_columns(df, columns, name):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )


# ============================================================
# 4. CHECK CENTRAL M2 VS M3 TABLE
# ============================================================

require_columns(
    cross,
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "forecast_method",
        "mae_m2",
        "mae_m3",
        "rmse_m2",
        "rmse_m3",
        "m3_vs_m2_mae_improvement_pct",
    ],
    "cross_pathogen_information_effect.csv",
)


cross["m3_better_mae"] = (
    cross["mae_m3"]
    < cross["mae_m2"]
)

cross["m3_equal_mae"] = np.isclose(
    cross["mae_m3"],
    cross["mae_m2"],
    rtol=1e-10,
    atol=1e-10,
)

cross["m3_not_worse_mae"] = (
    cross["m3_better_mae"]
    | cross["m3_equal_mae"]
)

cross["raw_mae_change_m3_minus_m2"] = (
    cross["mae_m3"]
    - cross["mae_m2"]
)


# ============================================================
# 5. SUMMARY FUNCTION
# ============================================================

def summarize_cross_pathogen(
    data,
    group_columns,
):

    summary = (
        data
        .groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            comparisons=(
                "mae_m2",
                "size",
            ),

            m3_mae_wins=(
                "m3_better_mae",
                "sum",
            ),

            m3_mae_not_worse=(
                "m3_not_worse_mae",
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

            median_raw_mae_change_m3_minus_m2=(
                "raw_mae_change_m3_minus_m2",
                "median",
            ),

            mean_raw_mae_change_m3_minus_m2=(
                "raw_mae_change_m3_minus_m2",
                "mean",
            ),
        )
    )

    summary["m3_mae_win_fraction"] = (
        summary["m3_mae_wins"]
        / summary["comparisons"]
    )

    summary["m3_mae_not_worse_fraction"] = (
        summary["m3_mae_not_worse"]
        / summary["comparisons"]
    )

    return summary


# ============================================================
# 6. ROBUSTNESS BY COUNTRY
# ============================================================

by_country = summarize_cross_pathogen(
    cross,
    [
        "country",
        "forecast_method",
    ],
)

by_country.to_csv(
    OUT_BY_COUNTRY,
    index=False,
)


# ============================================================
# 7. ROBUSTNESS BY PATHOGEN
# ============================================================

by_pathogen = summarize_cross_pathogen(
    cross,
    [
        "target_pathogen",
        "forecast_method",
    ],
)

by_pathogen.to_csv(
    OUT_BY_PATHOGEN,
    index=False,
)


# ============================================================
# 8. ROBUSTNESS BY FORECAST HORIZON
# ============================================================

by_horizon = summarize_cross_pathogen(
    cross,
    [
        "forecast_horizon_weeks",
        "forecast_method",
    ],
)

by_horizon.to_csv(
    OUT_BY_HORIZON,
    index=False,
)


# ============================================================
# 9. ROBUSTNESS BY MODEL FAMILY
# ============================================================

by_model = summarize_cross_pathogen(
    cross,
    [
        "forecast_method",
    ],
)

by_model.to_csv(
    OUT_BY_MODEL,
    index=False,
)


# ============================================================
# 10. OVERALL MODEL-RANK ROBUSTNESS
# ============================================================

require_columns(
    overall,
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "information_set",
        "forecast_method",
        "mae",
        "rmse",
    ],
    "overall_model_comparison.csv",
)


ranked = overall.copy()

rank_groups = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "information_set",
]


ranked["mae_rank"] = (
    ranked
    .groupby(rank_groups)["mae"]
    .rank(
        method="min",
        ascending=True,
    )
)

ranked["rmse_rank"] = (
    ranked
    .groupby(rank_groups)["rmse"]
    .rank(
        method="min",
        ascending=True,
    )
)


rank_summary = (
    ranked
    .groupby(
        [
            "information_set",
            "forecast_method",
        ],
        as_index=False,
    )
    .agg(
        comparisons=(
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

        median_rmse_rank=(
            "rmse_rank",
            "median",
        ),

        mae_rank1_count=(
            "mae_rank",
            lambda x: int(
                np.sum(x == 1)
            ),
        ),

        rmse_rank1_count=(
            "rmse_rank",
            lambda x: int(
                np.sum(x == 1)
            ),
        ),
    )
)


rank_summary[
    "mae_rank1_fraction"
] = (
    rank_summary[
        "mae_rank1_count"
    ]
    / rank_summary[
        "comparisons"
    ]
)


rank_summary.to_csv(
    OUT_MODEL_RANKS,
    index=False,
)


# ============================================================
# 11. ENSEMBLE ROBUSTNESS
# ============================================================

require_columns(
    ensemble,
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
        "ensemble_beats_nb_mae",
        "ensemble_beats_xgb_mae",
        "ensemble_beats_both_mae",
        "ensemble_vs_best_component_mae_improvement_pct",
    ],
    "ensemble_component_comparison.csv",
)


ensemble_summary = (
    ensemble
    .groupby(
        "model",
        as_index=False,
    )
    .agg(
        comparisons=(
            "ensemble_beats_both_mae",
            "size",
        ),

        ensemble_beats_nb=(
            "ensemble_beats_nb_mae",
            "sum",
        ),

        ensemble_beats_xgb=(
            "ensemble_beats_xgb_mae",
            "sum",
        ),

        ensemble_beats_both=(
            "ensemble_beats_both_mae",
            "sum",
        ),

        median_vs_best_component_improvement_pct=(
            "ensemble_vs_best_component_mae_improvement_pct",
            "median",
        ),

        mean_vs_best_component_improvement_pct=(
            "ensemble_vs_best_component_mae_improvement_pct",
            "mean",
        ),
    )
)


ensemble_summary[
    "ensemble_beats_nb_fraction"
] = (
    ensemble_summary[
        "ensemble_beats_nb"
    ]
    / ensemble_summary[
        "comparisons"
    ]
)

ensemble_summary[
    "ensemble_beats_xgb_fraction"
] = (
    ensemble_summary[
        "ensemble_beats_xgb"
    ]
    / ensemble_summary[
        "comparisons"
    ]
)

ensemble_summary[
    "ensemble_beats_both_fraction"
] = (
    ensemble_summary[
        "ensemble_beats_both"
    ]
    / ensemble_summary[
        "comparisons"
    ]
)


ensemble_summary.to_csv(
    OUT_ENSEMBLE,
    index=False,
)


# ============================================================
# 12. UNCERTAINTY-CALIBRATION ROBUSTNESS
# ============================================================

require_columns(
    uncertainty,
    [
        "model",
        "selected_calibration_factor",
        "primary_mean_coverage_80",
        "calibrated_mean_coverage_80",
        "primary_mean_coverage_95",
        "calibrated_mean_coverage_95",
        "primary_mean_wis",
        "calibrated_mean_wis",
        "primary_mean_width_80",
        "calibrated_mean_width_80",
    ],
    "uncertainty_calibration_sensitivity_summary.csv",
)


uncertainty_summary = uncertainty.copy()


uncertainty_summary[
    "coverage80_change"
] = (
    uncertainty_summary[
        "calibrated_mean_coverage_80"
    ]
    - uncertainty_summary[
        "primary_mean_coverage_80"
    ]
)


uncertainty_summary[
    "coverage95_change"
] = (
    uncertainty_summary[
        "calibrated_mean_coverage_95"
    ]
    - uncertainty_summary[
        "primary_mean_coverage_95"
    ]
)


uncertainty_summary[
    "width80_change_pct"
] = (
    100.0
    * (
        uncertainty_summary[
            "calibrated_mean_width_80"
        ]
        - uncertainty_summary[
            "primary_mean_width_80"
        ]
    )
    / uncertainty_summary[
        "primary_mean_width_80"
    ]
)


uncertainty_summary[
    "wis_change_pct"
] = (
    100.0
    * (
        uncertainty_summary[
            "calibrated_mean_wis"
        ]
        - uncertainty_summary[
            "primary_mean_wis"
        ]
    )
    / uncertainty_summary[
        "primary_mean_wis"
    ]
)


uncertainty_summary.to_csv(
    OUT_UNCERTAINTY,
    index=False,
)


# ============================================================
# 13. CREATE OVERALL SYNTHESIS TABLE
# ============================================================

synthesis_rows = []


# ------------------------------------------------------------
# Cross-pathogen summary
# ------------------------------------------------------------

for _, row in by_model.iterrows():

    synthesis_rows.append({
        "domain":
            "cross_pathogen_information",

        "subgroup":
            row["forecast_method"],

        "metric":
            "m3_mae_win_fraction",

        "value":
            row["m3_mae_win_fraction"],

        "interpretation":
            (
                "M3 improved MAE in this fraction "
                "of matched M2-vs-M3 comparisons."
            ),
    })


# ------------------------------------------------------------
# Overall best model by information set
# ------------------------------------------------------------

for info in [
    "M2",
    "M3",
]:

    temp = (
        rank_summary[
            rank_summary[
                "information_set"
            ]
            == info
        ]
        .sort_values(
            "mean_mae_rank"
        )
    )

    if not temp.empty:

        best = temp.iloc[0]

        synthesis_rows.append({
            "domain":
                "overall_model_ranking",

            "subgroup":
                info,

            "metric":
                "best_mean_mae_rank_method",

            "value":
                best[
                    "mean_mae_rank"
                ],

            "interpretation":
                (
                    f"{best['forecast_method']} had "
                    f"the lowest mean MAE rank."
                ),
        })


# ------------------------------------------------------------
# Ensemble
# ------------------------------------------------------------

for _, row in ensemble_summary.iterrows():

    synthesis_rows.append({
        "domain":
            "ensemble",

        "subgroup":
            row["model"],

        "metric":
            "ensemble_beats_both_fraction",

        "value":
            row[
                "ensemble_beats_both_fraction"
            ],

        "interpretation":
            (
                "Fraction of test comparisons in which "
                "the ensemble beat both NB and XGBoost."
            ),
    })


# ------------------------------------------------------------
# Calibration
# ------------------------------------------------------------

for _, row in uncertainty_summary.iterrows():

    synthesis_rows.append({
        "domain":
            "uncertainty_calibration",

        "subgroup":
            row["model"],

        "metric":
            "wis_change_pct",

        "value":
            row[
                "wis_change_pct"
            ],

        "interpretation":
            (
                "Positive WIS change means validation-based "
                "recalibration worsened probabilistic performance."
            ),
    })


synthesis = pd.DataFrame(
    synthesis_rows
)

synthesis.to_csv(
    OUT_SYNTHESIS,
    index=False,
)


# ============================================================
# 14. QC
# ============================================================

finite_core_cross = np.isfinite(
    cross[
        [
            "mae_m2",
            "mae_m3",
            "rmse_m2",
            "rmse_m3",
        ]
    ].to_numpy(
        dtype=float
    )
).all()


valid_win_fractions = (
    by_country[
        "m3_mae_win_fraction"
    ].between(0, 1).all()
    and
    by_pathogen[
        "m3_mae_win_fraction"
    ].between(0, 1).all()
    and
    by_horizon[
        "m3_mae_win_fraction"
    ].between(0, 1).all()
    and
    by_model[
        "m3_mae_win_fraction"
    ].between(0, 1).all()
)


ensemble_fractions_valid = (
    ensemble_summary[
        [
            "ensemble_beats_nb_fraction",
            "ensemble_beats_xgb_fraction",
            "ensemble_beats_both_fraction",
        ]
    ]
    .apply(
        lambda x: x.between(0, 1)
    )
    .all()
    .all()
)


rank_values_valid = (
    ranked[
        [
            "mae_rank",
            "rmse_rank",
        ]
    ]
    .notna()
    .all()
    .all()
)


audit_pass = (
    finite_core_cross
    and valid_win_fractions
    and ensemble_fractions_valid
    and rank_values_valid
)


# ============================================================
# 15. PRINT QC
# ============================================================

print("\n" + "=" * 95)
print("STEP-24 ROBUSTNESS QC")
print("=" * 95)

print(
    "Core M2/M3 errors finite:",
    finite_core_cross,
)

print(
    "M3 win fractions valid:",
    valid_win_fractions,
)

print(
    "Ensemble fractions valid:",
    ensemble_fractions_valid,
)

print(
    "Model ranks valid:",
    rank_values_valid,
)


if audit_pass:

    print(
        "\nROBUSTNESS SYNTHESIS AUDIT: PASS"
    )

else:

    print(
        "\nROBUSTNESS SYNTHESIS AUDIT: "
        "REVIEW REQUIRED"
    )


# ============================================================
# 16. PRINT MAIN CROSS-PATHOGEN RESULT
# ============================================================

print("\n" + "=" * 95)
print("CROSS-PATHOGEN ROBUSTNESS BY MODEL")
print("=" * 95)

print(
    by_model[
        [
            "forecast_method",
            "comparisons",
            "m3_mae_wins",
            "m3_mae_win_fraction",
            "median_m3_vs_m2_mae_improvement_pct",
            "mean_m3_vs_m2_mae_improvement_pct",
            "median_raw_mae_change_m3_minus_m2",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 17. PRINT BY PATHOGEN
# ============================================================

print("\n" + "=" * 95)
print("CROSS-PATHOGEN ROBUSTNESS BY PATHOGEN")
print("=" * 95)

print(
    by_pathogen[
        [
            "target_pathogen",
            "forecast_method",
            "comparisons",
            "m3_mae_wins",
            "m3_mae_win_fraction",
            "median_m3_vs_m2_mae_improvement_pct",
        ]
    ]
    .sort_values(
        [
            "target_pathogen",
            "forecast_method",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 18. PRINT BY HORIZON
# ============================================================

print("\n" + "=" * 95)
print("CROSS-PATHOGEN ROBUSTNESS BY HORIZON")
print("=" * 95)

print(
    by_horizon[
        [
            "forecast_horizon_weeks",
            "forecast_method",
            "comparisons",
            "m3_mae_wins",
            "m3_mae_win_fraction",
            "median_m3_vs_m2_mae_improvement_pct",
        ]
    ]
    .sort_values(
        [
            "forecast_horizon_weeks",
            "forecast_method",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 19. PRINT MODEL RANKS
# ============================================================

print("\n" + "=" * 95)
print("ROBUST OVERALL MODEL RANKING")
print("=" * 95)

print(
    rank_summary[
        [
            "information_set",
            "forecast_method",
            "comparisons",
            "mean_mae_rank",
            "median_mae_rank",
            "mae_rank1_count",
            "mae_rank1_fraction",
        ]
    ]
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
# 20. PRINT ENSEMBLE SUMMARY
# ============================================================

print("\n" + "=" * 95)
print("ENSEMBLE ROBUSTNESS")
print("=" * 95)

print(
    ensemble_summary
    .to_string(
        index=False
    )
)


# ============================================================
# 21. PRINT UNCERTAINTY SUMMARY
# ============================================================

print("\n" + "=" * 95)
print("UNCERTAINTY-CALIBRATION SENSITIVITY")
print("=" * 95)

print(
    uncertainty_summary[
        [
            "model",
            "selected_calibration_factor",
            "primary_mean_coverage_80",
            "calibrated_mean_coverage_80",
            "primary_mean_coverage_95",
            "calibrated_mean_coverage_95",
            "width80_change_pct",
            "wis_change_pct",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 22. OUTPUTS
# ============================================================

print("\nSaved:")

for path in [
    OUT_BY_COUNTRY,
    OUT_BY_PATHOGEN,
    OUT_BY_HORIZON,
    OUT_BY_MODEL,
    OUT_MODEL_RANKS,
    OUT_ENSEMBLE,
    OUT_UNCERTAINTY,
    OUT_SYNTHESIS,
]:
    print(path)


print("\nInterpretation rules:")

print(
    "- Positive M3-vs-M2 improvement means M3 reduced error."
)

print(
    "- M3 win fraction above 0.50 means M3 won more often "
    "than M2 for that subgroup."
)

print(
    "- Lower model rank is better."
)

print(
    "- Positive calibration WIS change means recalibration "
    "made probabilistic performance worse."
)

print(
    "- Robustness analysis is descriptive; no TEST-set "
    "results are used to retune models."
)

print(
    "\nROBUSTNESS SYNTHESIS COMPLETE"
)