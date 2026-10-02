from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
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
    / "eda"
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
# 2. LOAD MASTER DATA
# ============================================================

print("\n" + "=" * 75)
print("RESPIRATORY VIRUS EXPLORATORY ANALYSIS")
print("=" * 75)

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce"
)

df = df.sort_values(
    [
        "country",
        "week_start",
    ]
).reset_index(drop=True)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print(f"\nRows:      {len(df):,}")
print(
    "Countries:",
    df["country"].nunique()
)

print(
    "First week:",
    df["week_start"].min()
)

print(
    "Last week:",
    df["week_start"].max()
)


# ============================================================
# 4. OUTCOME VARIABLES
# ============================================================

PATHOGENS = {
    "Influenza": "influenza_positive",
    "RSV": "rsv_positive",
    "SARS-CoV-2": "sarscov2_cases",
}

for col in PATHOGENS.values():

    if col not in df.columns:
        raise ValueError(
            f"Required column not found: {col}"
        )


# ============================================================
# 5. MISSINGNESS SUMMARY
# ============================================================

missingness = []

for label, col in PATHOGENS.items():

    missingness.append(
        {
            "pathogen": label,
            "nonmissing_weeks": int(
                df[col].notna().sum()
            ),
            "missing_weeks": int(
                df[col].isna().sum()
            ),
            "missing_percent": round(
                df[col].isna().mean() * 100,
                2
            ),
        }
    )

missingness = pd.DataFrame(
    missingness
)

missing_file = (
    TABLE_DIR
    / "eda_pathogen_missingness.csv"
)

missingness.to_csv(
    missing_file,
    index=False
)

print("\nPathogen missingness:\n")
print(
    missingness.to_string(
        index=False
    )
)


# ============================================================
# 6. COUNTRY-LEVEL AVAILABILITY
# ============================================================

country_availability = (
    df
    .groupby(
        "country",
        as_index=False
    )
    .agg(
        total_weeks=(
            "week_start",
            "size"
        ),
        first_week=(
            "week_start",
            "min"
        ),
        last_week=(
            "week_start",
            "max"
        ),
        influenza_weeks=(
            "influenza_positive",
            lambda x: x.notna().sum()
        ),
        rsv_weeks=(
            "rsv_positive",
            lambda x: x.notna().sum()
        ),
        sarscov2_weeks=(
            "sarscov2_cases",
            lambda x: x.notna().sum()
        ),
    )
)

availability_file = (
    TABLE_DIR
    / "eda_country_pathogen_availability.csv"
)

country_availability.to_csv(
    availability_file,
    index=False
)

print("\nCountry availability:\n")

print(
    country_availability
    .to_string(index=False)
)


# ============================================================
# 7. COUNTRY-YEAR SUMMARY
# ============================================================

country_year = (
    df
    .groupby(
        [
            "country",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        influenza_total=(
            "influenza_positive",
            lambda x: x.sum(min_count=1)
        ),
        rsv_total=(
            "rsv_positive",
            lambda x: x.sum(min_count=1)
        ),
        sarscov2_total=(
            "sarscov2_cases",
            lambda x: x.sum(min_count=1)
        ),
        influenza_observed_weeks=(
            "influenza_positive",
            lambda x: x.notna().sum()
        ),
        rsv_observed_weeks=(
            "rsv_positive",
            lambda x: x.notna().sum()
        ),
        sarscov2_observed_weeks=(
            "sarscov2_cases",
            lambda x: x.notna().sum()
        ),
    )
)

country_year_file = (
    TABLE_DIR
    / "eda_country_year_summary.csv"
)

country_year.to_csv(
    country_year_file,
    index=False
)


# ============================================================
# 8. SANITISE FILENAMES
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
# 9. COUNTRY-SPECIFIC TIME-SERIES FIGURES
# ============================================================

print("\nCreating country time-series figures...")

for country in sorted(
    df["country"]
    .dropna()
    .unique()
):

    country_df = df[
        df["country"] == country
    ].copy()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 9),
        sharex=True
    )

    axes[0].plot(
        country_df["week_start"],
        country_df["influenza_positive"]
    )

    axes[0].set_ylabel(
        "Influenza detections"
    )

    axes[0].set_title(
        f"{country}: weekly respiratory-virus surveillance"
    )


    axes[1].plot(
        country_df["week_start"],
        country_df["rsv_positive"]
    )

    axes[1].set_ylabel(
        "RSV detections"
    )


    axes[2].plot(
        country_df["week_start"],
        country_df["sarscov2_cases"]
    )

    axes[2].set_ylabel(
        "SARS-CoV-2 cases"
    )

    axes[2].set_xlabel(
        "Week"
    )


    fig.tight_layout()

    filename = (
        FIGURE_DIR
        / (
            "eda_timeseries_"
            + clean_filename(country)
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
# 10. SINGAPORE-SPECIFIC FIGURE
# ============================================================

singapore = df[
    df["country"].eq("Singapore")
].copy()

if not singapore.empty:

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7),
        sharex=True
    )

    axes[0].plot(
        singapore["week_start"],
        singapore["influenza_positive"]
    )

    axes[0].set_ylabel(
        "Influenza detections"
    )

    axes[0].set_title(
        "Singapore respiratory-virus surveillance"
    )


    axes[1].plot(
        singapore["week_start"],
        singapore["sarscov2_cases"]
    )

    axes[1].set_ylabel(
        "SARS-CoV-2 cases"
    )

    axes[1].set_xlabel(
        "Week"
    )

    fig.tight_layout()

    singapore_figure = (
        FIGURE_DIR
        / "eda_singapore_influenza_sarscov2.png"
    )

    fig.savefig(
        singapore_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 75)
print("EXPLORATORY ANALYSIS COMPLETE")
print("=" * 75)

print("\nTables:")
print(missing_file)
print(availability_file)
print(country_year_file)

print("\nFigures saved in:")
print(FIGURE_DIR)

print(
    "\nNext step: quantify pathogen seasonality "
    "using epidemiological week summaries."
)