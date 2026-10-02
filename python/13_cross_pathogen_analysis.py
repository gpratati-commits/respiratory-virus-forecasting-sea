from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_virus_master_weekly.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "cross_pathogen"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

MAX_LAG = 8
MIN_OBSERVATIONS = 20

PATHOGENS = {
    "Influenza": {
        "value": "influenza_positive",
        "eligible": "influenza_eligible",
    },
    "RSV": {
        "value": "rsv_positive",
        "eligible": "rsv_eligible",
    },
    "SARS-CoV-2": {
        "value": "sarscov2_cases",
        "eligible": "covid_eligible",
    },
}

PAIRS = [
    ("Influenza", "RSV"),
    ("Influenza", "SARS-CoV-2"),
    ("RSV", "SARS-CoV-2"),
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("CROSS-PATHOGEN LAG ANALYSIS")
print("=" * 75)

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce"
)

df["ISO_YEAR"] = pd.to_numeric(
    df["ISO_YEAR"],
    errors="coerce"
)


# ============================================================
# 4. STANDARDISE WITHIN COUNTRY-YEAR
# ============================================================

# Raw surveillance counts should not be directly compared
# across countries or years because testing/reporting intensity
# can differ.
#
# We therefore:
# 1. log-transform counts using log1p
# 2. standardise within each country-year

for pathogen, info in PATHOGENS.items():

    value_col = info["value"]
    eligible_col = info["eligible"]

    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce"
    )

    df.loc[
        ~df[eligible_col].fillna(False),
        value_col
    ] = np.nan

    log_col = (
        value_col
        + "_log1p"
    )

    z_col = (
        value_col
        + "_z"
    )

    df[log_col] = np.log1p(
        df[value_col]
    )

    def zscore_group(x):

        if x.notna().sum() < 2:
            return pd.Series(
                np.nan,
                index=x.index
            )

        sd = x.std()

        if pd.isna(sd) or sd == 0:
            return pd.Series(
                np.nan,
                index=x.index
            )

        return (
            x - x.mean()
        ) / sd

    df[z_col] = (
        df
        .groupby(
            [
                "country",
                "ISO_YEAR",
            ]
        )[log_col]
        .transform(
            zscore_group
        )
    )


# ============================================================
# 5. LAGGED CORRELATION FUNCTION
# ============================================================

def lagged_analysis(
    country_df,
    pathogen_a,
    pathogen_b,
    lag
):

    a_col = (
        PATHOGENS[
            pathogen_a
        ]["value"]
        + "_z"
    )

    b_col = (
        PATHOGENS[
            pathogen_b
        ]["value"]
        + "_z"
    )

    a = country_df[
        [
            "week_start",
            a_col,
        ]
    ].copy()

    b = country_df[
        [
            "week_start",
            b_col,
        ]
    ].copy()

    a = a.rename(
        columns={
            a_col: "activity_a"
        }
    )

    b = b.rename(
        columns={
            b_col: "activity_b"
        }
    )

    # Positive lag means pathogen B occurs earlier.
    #
    # Example:
    # lag = 2 compares
    # Influenza(t) with RSV(t - 2 weeks)

    b[
        "week_start"
    ] = (
        b["week_start"]
        + pd.to_timedelta(
            lag,
            unit="W"
        )
    )

    merged = a.merge(
        b,
        on="week_start",
        how="inner"
    )

    merged = merged.dropna(
        subset=[
            "activity_a",
            "activity_b",
        ]
    )

    n = len(merged)

    if n < MIN_OBSERVATIONS:

        return {
            "n": n,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

    if (
        merged["activity_a"].nunique() < 2
        or
        merged["activity_b"].nunique() < 2
    ):

        return {
            "n": n,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

    rho, rho_p = spearmanr(
        merged["activity_a"],
        merged["activity_b"]
    )

    r, r_p = pearsonr(
        merged["activity_a"],
        merged["activity_b"]
    )

    return {
        "n": n,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "pearson_r": r,
        "pearson_p": r_p,
    }


# ============================================================
# 6. RUN COUNTRY-SPECIFIC ANALYSIS
# ============================================================

results = []

countries = sorted(
    df["country"]
    .dropna()
    .unique()
)

for country in countries:

    country_df = df[
        df["country"] == country
    ].copy()

    for pathogen_a, pathogen_b in PAIRS:

        for lag in range(
            0,
            MAX_LAG + 1
        ):

            stats = lagged_analysis(
                country_df,
                pathogen_a,
                pathogen_b,
                lag
            )

            results.append(
                {
                    "country": country,
                    "pathogen_a": pathogen_a,
                    "pathogen_b": pathogen_b,
                    "lag_weeks": lag,
                    **stats,
                }
            )


results = pd.DataFrame(
    results
)


# ============================================================
# 7. MULTIPLE-TESTING CORRECTION
# ============================================================

results[
    "spearman_q_fdr"
] = np.nan

results[
    "spearman_significant_fdr_0_05"
] = False


for (
    country,
    pathogen_a,
    pathogen_b
), group in results.groupby(
    [
        "country",
        "pathogen_a",
        "pathogen_b",
    ]
):

    valid = group[
        "spearman_p"
    ].notna()

    if valid.sum() == 0:
        continue

    indices = group.index[
        valid
    ]

    reject, qvals, _, _ = multipletests(
        group.loc[
            indices,
            "spearman_p"
        ],
        method="fdr_bh",
        alpha=0.05
    )

    results.loc[
        indices,
        "spearman_q_fdr"
    ] = qvals

    results.loc[
        indices,
        "spearman_significant_fdr_0_05"
    ] = reject


# ============================================================
# 8. IDENTIFY STRONGEST LAG
# ============================================================

best_rows = []

for (
    country,
    pathogen_a,
    pathogen_b
), group in results.groupby(
    [
        "country",
        "pathogen_a",
        "pathogen_b",
    ]
):

    valid = group.dropna(
        subset=[
            "spearman_rho"
        ]
    )

    if valid.empty:
        continue

    index = (
        valid[
            "spearman_rho"
        ]
        .abs()
        .idxmax()
    )

    best_rows.append(
        results.loc[
            index
        ].to_dict()
    )


best = pd.DataFrame(
    best_rows
)


# ============================================================
# 9. SAVE TABLES
# ============================================================

results_file = (
    TABLE_DIR
    / "cross_pathogen_lag_correlations.csv"
)

best_file = (
    TABLE_DIR
    / "cross_pathogen_best_lags.csv"
)

results.to_csv(
    results_file,
    index=False
)

best.to_csv(
    best_file,
    index=False
)


# ============================================================
# 10. FILENAME HELPER
# ============================================================

def clean_filename(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# ============================================================
# 11. FIGURES
# ============================================================

for (
    country,
    pathogen_a,
    pathogen_b
), group in results.groupby(
    [
        "country",
        "pathogen_a",
        "pathogen_b",
    ]
):

    valid = group.dropna(
        subset=[
            "spearman_rho"
        ]
    )

    if valid.empty:
        continue

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        valid["lag_weeks"],
        valid["spearman_rho"],
        marker="o"
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        f"Lag of {pathogen_b} (weeks)"
    )

    ax.set_ylabel(
        "Spearman correlation"
    )

    ax.set_title(
        f"{country}: {pathogen_a} vs {pathogen_b}"
    )

    fig.tight_layout()

    filename = (
        FIGURE_DIR
        / (
            "cross_pathogen_"
            + clean_filename(country)
            + "_"
            + clean_filename(pathogen_a)
            + "_"
            + clean_filename(pathogen_b)
            + ".png"
        )
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 12. PRINT BEST LAGS
# ============================================================

print("\n" + "=" * 75)
print("STRONGEST DESCRIPTIVE LAGGED ASSOCIATIONS")
print("=" * 75)

if best.empty:

    print(
        "\nNo usable cross-pathogen comparisons."
    )

else:

    display_columns = [
        "country",
        "pathogen_a",
        "pathogen_b",
        "lag_weeks",
        "n",
        "spearman_rho",
        "spearman_p",
        "spearman_q_fdr",
        "spearman_significant_fdr_0_05",
    ]

    print(
        best[
            display_columns
        ]
        .sort_values(
            [
                "country",
                "pathogen_a",
                "pathogen_b",
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# 13. FINISH
# ============================================================

print("\n" + "=" * 75)
print("CROSS-PATHOGEN ANALYSIS COMPLETE")
print("=" * 75)

print("\nSaved:")
print(results_file)
print(best_file)

print("\nFigures:")
print(FIGURE_DIR)

print(
    "\nInterpretation rule: positive lag L means pathogen B "
    "activity L weeks earlier is being compared with "
    "pathogen A activity in the current week."
)

print(
    "\nThese are descriptive temporal associations and "
    "should not be interpreted as causal pathogen interactions."
)