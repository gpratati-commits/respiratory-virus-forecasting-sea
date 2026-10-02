from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
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
    / "seasonality"
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
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("RESPIRATORY VIRUS SEASONALITY ANALYSIS")
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

df["ISO_WEEK"] = pd.to_numeric(
    df["ISO_WEEK"],
    errors="coerce"
)


# ============================================================
# 3. PATHOGEN DEFINITIONS
# ============================================================

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


# ============================================================
# 4. BUILD LONG-FORMAT ELIGIBLE DATA
# ============================================================

long_data = []

for pathogen, info in PATHOGENS.items():

    value_col = info["value"]
    eligibility_col = info["eligible"]

    if value_col not in df.columns:
        raise ValueError(
            f"Missing required column: {value_col}"
        )

    if eligibility_col not in df.columns:
        raise ValueError(
            f"Missing required column: {eligibility_col}"
        )

    sub = df[
        df[eligibility_col].fillna(False)
        & df[value_col].notna()
    ].copy()

    sub["pathogen"] = pathogen

    sub["activity"] = pd.to_numeric(
        sub[value_col],
        errors="coerce"
    )

    # Negative biological activity should not occur.
    sub.loc[
        sub["activity"] < 0,
        "activity"
    ] = np.nan

    long_data.append(
        sub[
            [
                "country",
                "ISO_YEAR",
                "ISO_WEEK",
                "week_start",
                "pathogen",
                "activity",
            ]
        ]
    )


season = pd.concat(
    long_data,
    ignore_index=True
)


# ============================================================
# 5. YEAR-LEVEL TOTALS
# ============================================================

year_summary = (
    season
    .groupby(
        [
            "country",
            "pathogen",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        annual_activity=(
            "activity",
            lambda x: x.sum(min_count=1)
        ),
        observed_weeks=(
            "activity",
            lambda x: x.notna().sum()
        ),
    )
)


# ============================================================
# 6. MERGE YEAR TOTALS BACK
# ============================================================

season = season.merge(
    year_summary,
    on=[
        "country",
        "pathogen",
        "ISO_YEAR",
    ],
    how="left"
)


# ============================================================
# 7. RELATIVE WEEKLY ACTIVITY
# ============================================================

# Raw counts are not directly comparable between countries.
#
# For seasonality, each country's pathogen activity is
# normalised within each year:
#
# weekly activity / total activity during that year

season["relative_activity"] = np.where(
    season["annual_activity"] > 0,
    season["activity"]
    / season["annual_activity"],
    np.nan
)


# ============================================================
# 8. REMOVE YEARS WITH ZERO TOTAL ACTIVITY
# ============================================================

season = season[
    season["annual_activity"] > 0
].copy()


# ============================================================
# 9. SEASONAL PROFILE BY ISO WEEK
# ============================================================

profile = (
    season
    .groupby(
        [
            "country",
            "pathogen",
            "ISO_WEEK",
        ],
        as_index=False
    )
    .agg(
        mean_relative_activity=(
            "relative_activity",
            "mean"
        ),
        median_relative_activity=(
            "relative_activity",
            "median"
        ),
        sd_relative_activity=(
            "relative_activity",
            "std"
        ),
        number_of_years=(
            "ISO_YEAR",
            "nunique"
        ),
        number_of_observations=(
            "relative_activity",
            "count"
        ),
    )
)


# ============================================================
# 10. IDENTIFY PEAK WEEK
# ============================================================

peak_rows = []

for (
    country,
    pathogen
), group in profile.groupby(
    [
        "country",
        "pathogen",
    ]
):

    valid = group.dropna(
        subset=[
            "mean_relative_activity"
        ]
    )

    if valid.empty:
        continue

    peak_index = (
        valid[
            "mean_relative_activity"
        ]
        .idxmax()
    )

    row = valid.loc[
        peak_index
    ]

    peak_rows.append(
        {
            "country": country,
            "pathogen": pathogen,
            "peak_iso_week": int(
                row["ISO_WEEK"]
            ),
            "peak_mean_relative_activity": row[
                "mean_relative_activity"
            ],
            "years_contributing": int(
                row["number_of_years"]
            ),
        }
    )


peak_table = pd.DataFrame(
    peak_rows
)


# ============================================================
# 11. SAVE TABLES
# ============================================================

year_summary_file = (
    TABLE_DIR
    / "seasonality_country_year_summary.csv"
)

profile_file = (
    TABLE_DIR
    / "seasonality_weekly_profiles.csv"
)

peak_file = (
    TABLE_DIR
    / "seasonality_peak_weeks.csv"
)

year_summary.to_csv(
    year_summary_file,
    index=False
)

profile.to_csv(
    profile_file,
    index=False
)

peak_table.to_csv(
    peak_file,
    index=False
)


# ============================================================
# 12. CLEAN FILENAME HELPER
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
# 13. COUNTRY SEASONALITY FIGURES
# ============================================================

print("\nCreating seasonality figures...")

for country in sorted(
    profile["country"]
    .dropna()
    .unique()
):

    country_profile = profile[
        profile["country"] == country
    ].copy()

    pathogens_present = (
        country_profile[
            "pathogen"
        ]
        .unique()
        .tolist()
    )

    if not pathogens_present:
        continue

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for pathogen in [
        "Influenza",
        "RSV",
        "SARS-CoV-2",
    ]:

        sub = country_profile[
            country_profile[
                "pathogen"
            ] == pathogen
        ]

        if sub.empty:
            continue

        ax.plot(
            sub["ISO_WEEK"],
            sub[
                "mean_relative_activity"
            ],
            label=pathogen
        )

    ax.set_xlabel(
        "ISO epidemiological week"
    )

    ax.set_ylabel(
        "Mean proportion of annual activity"
    )

    ax.set_title(
        f"{country}: seasonal respiratory-virus profile"
    )

    ax.legend()

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / (
            "seasonality_"
            + clean_filename(country)
            + ".png"
        )
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 14. SINGAPORE-SPECIFIC FIGURE
# ============================================================

sg = profile[
    profile["country"]
    .eq("Singapore")
].copy()

if not sg.empty:

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for pathogen in [
        "Influenza",
        "SARS-CoV-2",
    ]:

        sub = sg[
            sg["pathogen"]
            == pathogen
        ]

        if sub.empty:
            continue

        ax.plot(
            sub["ISO_WEEK"],
            sub[
                "mean_relative_activity"
            ],
            label=pathogen
        )

    ax.set_xlabel(
        "ISO epidemiological week"
    )

    ax.set_ylabel(
        "Mean proportion of annual activity"
    )

    ax.set_title(
        "Singapore: influenza and SARS-CoV-2 seasonal profiles"
    )

    ax.legend()

    fig.tight_layout()

    sg_file = (
        FIGURE_DIR
        / "seasonality_singapore_focus.png"
    )

    fig.savefig(
        sg_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 15. PRINT PEAK-WEEK TABLE
# ============================================================

print("\n" + "=" * 75)
print("ESTIMATED PEAK ISO WEEKS")
print("=" * 75)

if peak_table.empty:

    print(
        "\nNo peak weeks could be estimated."
    )

else:

    print(
        peak_table
        .sort_values(
            [
                "country",
                "pathogen",
            ]
        )
        .to_string(index=False)
    )


# ============================================================
# 16. FINISH
# ============================================================

print("\n" + "=" * 75)
print("SEASONALITY ANALYSIS COMPLETE")
print("=" * 75)

print("\nSaved tables:")
print(year_summary_file)
print(profile_file)
print(peak_file)

print("\nFigures saved in:")
print(FIGURE_DIR)

print(
    "\nImportant: these are descriptive seasonal profiles, "
    "not evidence that season alone causes virus activity."
)