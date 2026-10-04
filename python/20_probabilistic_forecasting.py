"""
Probabilistic forecasting from the frozen regularized
Negative Binomial respiratory-virus models.

Purpose
-------
Convert the final Negative Binomial point forecasts into predictive
distributions and evaluate uncertainty using:

- 50% prediction intervals
- 80% prediction intervals
- 95% prediction intervals
- empirical interval coverage
- mean interval width
- Weighted Interval Score (WIS)

Important
---------
1. Point forecasts are the already-frozen TEST predictions from Step 17.
2. Negative Binomial dispersion comes from TRAIN + VALIDATION only.
3. TEST observations are used only for final evaluation.
4. No parameters are tuned using the TEST period.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
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

TABLE_DIR = ROOT / "outputs" / "tables"
PRED_DIR = ROOT / "outputs" / "predictions"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

PROBABILITY_FILE = (
    PRED_DIR
    / "probabilistic_predictions.csv"
)

INTERVAL_METRIC_FILE = (
    TABLE_DIR
    / "probabilistic_interval_metrics.csv"
)

M2_M3_FILE = (
    TABLE_DIR
    / "probabilistic_m2_m3_comparison.csv"
)


# ============================================================
# 2. INTERVAL SETTINGS
# ============================================================

# Central prediction intervals.
#
# coverage = 0.50 -> alpha = 0.50
# coverage = 0.80 -> alpha = 0.20
# coverage = 0.95 -> alpha = 0.05

INTERVALS = {
    "50": 0.50,
    "80": 0.20,
    "95": 0.05,
}


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 90)
print("PROBABILISTIC NEGATIVE BINOMIAL FORECASTING")
print("=" * 90)

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Statistical prediction file not found:\n"
        f"{PREDICTION_FILE}"
    )

if not DIAGNOSTIC_FILE.exists():
    raise FileNotFoundError(
        f"Statistical diagnostic file not found:\n"
        f"{DIAGNOSTIC_FILE}"
    )

pred = pd.read_csv(
    PREDICTION_FILE
)

diag = pd.read_csv(
    DIAGNOSTIC_FILE
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
# 4. KEEP FINAL TEST PREDICTIONS
# ============================================================

pred = pred[
    pred["split"] == "test"
].copy()

pred = pred[
    pred["model"].isin(
        ["M2", "M3"]
    )
].copy()

if pred.empty:
    raise ValueError(
        "No final M2/M3 test predictions found."
    )


# ============================================================
# 5. PREPARE FINAL DISPERSION PARAMETERS
# ============================================================

diag_final = diag[
    (diag["status"] == "success")
    & (
        diag["model"].isin(
            ["M2", "M3"]
        )
    )
].copy()

required_diag_columns = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "model",
    "alpha_final",
]

missing = [
    c
    for c in required_diag_columns
    if c not in diag_final.columns
]

if missing:
    raise ValueError(
        "Missing diagnostic columns: "
        + ", ".join(missing)
    )

diag_final = diag_final[
    required_diag_columns
].drop_duplicates()


# ============================================================
# 6. MERGE DISPERSION INTO TEST FORECASTS
# ============================================================

prob = pred.merge(
    diag_final,
    on=[
        "analysis_group",
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "model",
    ],
    how="left",
    validate="many_to_one",
)

if prob["alpha_final"].isna().any():
    bad = prob[
        prob["alpha_final"].isna()
    ][
        [
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "model",
        ]
    ].drop_duplicates()

    print(
        "\nMissing alpha_final for:"
    )

    print(
        bad.to_string(index=False)
    )

    raise ValueError(
        "Some final Negative Binomial dispersion "
        "parameters are missing."
    )


# ============================================================
# 7. NEGATIVE BINOMIAL QUANTILE FUNCTION
# ============================================================

def nb_quantile(
    mu,
    alpha,
    probability,
):
    """
    NB2 parameterisation:

        Var(Y) = mu + alpha * mu^2

    scipy.stats.nbinom uses:

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

    if alpha <= 0:
        alpha = 1e-8

    n = 1.0 / alpha

    p = n / (
        n + mu
    )

    q = nbinom.ppf(
        probability,
        n,
        p,
    )

    return float(q)


# ============================================================
# 8. GENERATE PREDICTION INTERVALS
# ============================================================

# NB median
prob["median"] = [
    nb_quantile(
        mu,
        alpha,
        0.50,
    )
    for mu, alpha
    in zip(
        prob["prediction"],
        prob["alpha_final"],
    )
]

for label, alpha_level in INTERVALS.items():

    lower_probability = (
        alpha_level / 2.0
    )

    upper_probability = (
        1.0
        - alpha_level / 2.0
    )

    prob[
        f"lower_{label}"
    ] = [
        nb_quantile(
            mu,
            alpha,
            lower_probability,
        )
        for mu, alpha
        in zip(
            prob["prediction"],
            prob["alpha_final"],
        )
    ]

    prob[
        f"upper_{label}"
    ] = [
        nb_quantile(
            mu,
            alpha,
            upper_probability,
        )
        for mu, alpha
        in zip(
            prob["prediction"],
            prob["alpha_final"],
        )
    ]


# ============================================================
# 9. BASIC INTERVAL VALIDITY CHECK
# ============================================================

for label in INTERVALS:

    lower = prob[
        f"lower_{label}"
    ]

    upper = prob[
        f"upper_{label}"
    ]

    if (
        (lower > upper)
        .any()
    ):
        raise ValueError(
            f"Invalid {label}% prediction interval."
        )

if not (
    (
        prob["lower_95"]
        <= prob["lower_80"]
    )
    & (
        prob["lower_80"]
        <= prob["lower_50"]
    )
    & (
        prob["upper_50"]
        <= prob["upper_80"]
    )
    & (
        prob["upper_80"]
        <= prob["upper_95"]
    )
).all():

    raise ValueError(
        "Prediction intervals are not correctly nested."
    )


# ============================================================
# 10. INTERVAL SCORE
# ============================================================

def interval_score(
    y,
    lower,
    upper,
    alpha_level,
):

    width = upper - lower

    below_penalty = (
        (2.0 / alpha_level)
        * (lower - y)
        if y < lower
        else 0.0
    )

    above_penalty = (
        (2.0 / alpha_level)
        * (y - upper)
        if y > upper
        else 0.0
    )

    return (
        width
        + below_penalty
        + above_penalty
    )


# ============================================================
# 11. WEIGHTED INTERVAL SCORE
# ============================================================

def calculate_wis(row):

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

    number_of_intervals = len(
        INTERVALS
    )

    for label, alpha_level in (
        INTERVALS.items()
    ):

        lower = float(
            row[
                f"lower_{label}"
            ]
        )

        upper = float(
            row[
                f"upper_{label}"
            ]
        )

        score = interval_score(
            y,
            lower,
            upper,
            alpha_level,
        )

        total += (
            alpha_level
            / 2.0
        ) * score

    denominator = (
        number_of_intervals
        + 0.5
    )

    return (
        total
        / denominator
    )


prob["wis"] = prob.apply(
    calculate_wis,
    axis=1,
)


# ============================================================
# 12. ROW-LEVEL COVERAGE
# ============================================================

for label in INTERVALS:

    prob[
        f"covered_{label}"
    ] = (
        (
            prob["target_value"]
            >= prob[
                f"lower_{label}"
            ]
        )
        & (
            prob["target_value"]
            <= prob[
                f"upper_{label}"
            ]
        )
    )

    prob[
        f"width_{label}"
    ] = (
        prob[
            f"upper_{label}"
        ]
        - prob[
            f"lower_{label}"
        ]
    )


# ============================================================
# 13. SUMMARISE PROBABILISTIC PERFORMANCE
# ============================================================

group_columns = [
    "analysis_group",
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "model",
]

metric_rows = []

for keys, group in prob.groupby(
    group_columns,
    dropna=False,
):

    (
        analysis_group,
        country,
        pathogen,
        horizon,
        model,
    ) = keys

    row = {
        "analysis_group": analysis_group,
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

    metric_rows.append(row)


metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# 14. M2 VS M3 PROBABILISTIC COMPARISON
# ============================================================

comparison = metrics.pivot_table(
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
    comparison[
        "mean_wis_m2"
    ].notna()
    & comparison[
        "mean_wis_m3"
    ].notna()
    & (
        comparison[
            "mean_wis_m2"
        ]
        != 0
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
# 15. SAVE OUTPUTS
# ============================================================

prob.to_csv(
    PROBABILITY_FILE,
    index=False,
)

metrics.to_csv(
    INTERVAL_METRIC_FILE,
    index=False,
)

comparison.to_csv(
    M2_M3_FILE,
    index=False,
)


# ============================================================
# 16. PRIMARY RESULTS
# ============================================================

primary = comparison[
    comparison["analysis_group"]
    == "primary_regional"
].copy()

display_columns = [
    "country",
    "target_pathogen",
    "forecast_horizon_weeks",
    "n_m2",
    "coverage_80_m2",
    "coverage_80_m3",
    "coverage_95_m2",
    "coverage_95_m3",
    "mean_wis_m2",
    "mean_wis_m3",
    "m3_vs_m2_wis_improvement_pct",
]

display_columns = [
    c
    for c in display_columns
    if c in primary.columns
]

print("\n" + "=" * 100)
print("PRIMARY PROBABILISTIC M2 VS M3 RESULTS")
print("=" * 100)

print(
    primary[
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


# ============================================================
# 17. QC
# ============================================================

all_finite = np.isfinite(
    prob[
        [
            "median",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
            "wis",
        ]
    ].to_numpy(
        dtype=float
    )
).all()

print("\n" + "=" * 90)
print("PROBABILISTIC FORECAST QC")
print("=" * 90)

print(
    "All probabilistic quantities finite:",
    all_finite
)

print(
    "All intervals correctly nested:",
    True
)

print(
    "Number of final test forecasts:",
    len(prob)
)


print("\nSaved probabilistic predictions:")
print(PROBABILITY_FILE)

print("\nSaved interval metrics:")
print(INTERVAL_METRIC_FILE)

print("\nSaved probabilistic M2/M3 comparison:")
print(M2_M3_FILE)


print("\nInterpretation:")
print(
    "- Coverage closer to the nominal level is better."
)

print(
    "- Narrower intervals are preferable only when "
    "coverage remains adequate."
)

print(
    "- Lower WIS is better."
)

print(
    "- Positive M3-vs-M2 WIS improvement means "
    "M3 had better probabilistic performance."
)

print(
    "- Test data were used only for final evaluation."
)

print(
    "\nPROBABILISTIC FORECASTING COMPLETE"
)