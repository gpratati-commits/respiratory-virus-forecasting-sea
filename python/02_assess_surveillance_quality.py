from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "VIW_FNT.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

CANDIDATE_COUNTRIES = [
    "Singapore",
    "Malaysia",
    "Thailand",
    "Philippines",
    "Indonesia",
    "Viet Nam",
    "Vietnam",
    "Cambodia",
    "Myanmar",
    "Brunei Darussalam",
    "Lao People's Democratic Republic",
]


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def iso_weeks_in_year(year):
    """
    Return the number of ISO epidemiological weeks in a year.

    ISO years contain either 52 or 53 weeks.
    December 28 always belongs to the final ISO week.
    """
    return date(int(year), 12, 28).isocalendar().week


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("WHO RESPIRATORY SURVEILLANCE QUALITY ASSESSMENT")
print("=" * 75)

print("\nLoading WHO data...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns:     {len(df.columns):,}")


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "COUNTRY_AREA_TERRITORY",
    "ISO_YEAR",
    "ISO_WEEK",
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Required columns missing: {missing_required}"
    )


# ============================================================
# 6. CLEAN YEAR/WEEK FIELDS
# ============================================================

df["ISO_YEAR"] = pd.to_numeric(
    df["ISO_YEAR"],
    errors="coerce"
)

df["ISO_WEEK"] = pd.to_numeric(
    df["ISO_WEEK"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "COUNTRY_AREA_TERRITORY",
        "ISO_YEAR",
        "ISO_WEEK",
    ]
).copy()

df["ISO_YEAR"] = df["ISO_YEAR"].astype(int)
df["ISO_WEEK"] = df["ISO_WEEK"].astype(int)


# ============================================================
# 7. IDENTIFY PATHOGEN-SPECIFIC VARIABLES
# ============================================================

influenza_columns = [
    col
    for col in [
        "INF_ALL",
        "INF_A",
        "INF_B",
        "SPEC_PROCESSED_NB",
        "SPEC_RECEIVED_NB",
        "INF_NEGATIVE",
    ]
    if col in df.columns
]

rsv_columns = [
    col
    for col in [
        "RSV",
        "RSV_PROCESSED",
    ]
    if col in df.columns
]

print("\nInfluenza-related reporting columns:")
for col in influenza_columns:
    print(f"  {col}")

print("\nRSV-related reporting columns:")
for col in rsv_columns:
    print(f"  {col}")


# ============================================================
# 8. CREATE ROW-LEVEL REPORTING FLAGS
# ============================================================

if influenza_columns:
    df["influenza_reported_row"] = (
        df[influenza_columns]
        .notna()
        .any(axis=1)
    )
else:
    df["influenza_reported_row"] = False


if rsv_columns:
    df["rsv_reported_row"] = (
        df[rsv_columns]
        .notna()
        .any(axis=1)
    )
else:
    df["rsv_reported_row"] = False


# ============================================================
# 9. CONVERT IMPORTANT NUMERIC FIELDS
# ============================================================

for col in [
    "INF_ALL",
    "INF_A",
    "INF_B",
    "RSV",
    "RSV_PROCESSED",
    "SPEC_PROCESSED_NB",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 10. WEEK-LEVEL DATA
# ============================================================

group_keys = [
    "COUNTRY_AREA_TERRITORY",
    "ISO_YEAR",
    "ISO_WEEK",
]

weekly = (
    df
    .groupby(group_keys, as_index=False)
    .agg(
        raw_rows=(
            "COUNTRY_AREA_TERRITORY",
            "size"
        ),
        influenza_reported=(
            "influenza_reported_row",
            "max"
        ),
        rsv_reported=(
            "rsv_reported_row",
            "max"
        ),
    )
)


# ============================================================
# 11. DUPLICATE COUNTRY-WEEK INFORMATION
# ============================================================

weekly["multiple_raw_rows"] = (
    weekly["raw_rows"] > 1
)

duplicate_summary = (
    weekly
    .groupby(
        [
            "COUNTRY_AREA_TERRITORY",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        unique_weeks=("ISO_WEEK", "nunique"),
        weeks_with_multiple_rows=(
            "multiple_raw_rows",
            "sum"
        ),
        maximum_rows_per_week=(
            "raw_rows",
            "max"
        ),
    )
)


# ============================================================
# 12. COUNTRY-YEAR REPORTING QUALITY
# ============================================================

quality = (
    weekly
    .groupby(
        [
            "COUNTRY_AREA_TERRITORY",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        reported_weeks=(
            "ISO_WEEK",
            "nunique"
        ),
        influenza_reporting_weeks=(
            "influenza_reported",
            "sum"
        ),
        rsv_reporting_weeks=(
            "rsv_reported",
            "sum"
        ),
    )
)


# ============================================================
# 13. CORRECT NUMBER OF EXPECTED ISO WEEKS
# ============================================================

quality["expected_iso_weeks"] = (
    quality["ISO_YEAR"]
    .apply(iso_weeks_in_year)
)


# ============================================================
# 14. CALCULATE COMPLETENESS
# ============================================================

quality["overall_reporting_percent"] = (
    quality["reported_weeks"]
    / quality["expected_iso_weeks"]
    * 100
)

quality["influenza_reporting_percent"] = (
    quality["influenza_reporting_weeks"]
    / quality["expected_iso_weeks"]
    * 100
)

quality["rsv_reporting_percent"] = (
    quality["rsv_reporting_weeks"]
    / quality["expected_iso_weeks"]
    * 100
)


# Avoid tiny floating-point display noise
percentage_columns = [
    "overall_reporting_percent",
    "influenza_reporting_percent",
    "rsv_reporting_percent",
]

quality[percentage_columns] = (
    quality[percentage_columns]
    .round(2)
)


# ============================================================
# 15. MERGE DUPLICATE INFORMATION
# ============================================================

quality = quality.merge(
    duplicate_summary,
    on=[
        "COUNTRY_AREA_TERRITORY",
        "ISO_YEAR",
    ],
    how="left",
    suffixes=("", "_duplicate_check"),
)


# ============================================================
# 16. BASIC SCREENING FLAGS
# ============================================================

quality["overall_ge_80pct"] = (
    quality["overall_reporting_percent"]
    >= 80
)

quality["influenza_ge_80pct"] = (
    quality["influenza_reporting_percent"]
    >= 80
)

quality["rsv_ge_80pct"] = (
    quality["rsv_reporting_percent"]
    >= 80
)


# ============================================================
# 17. SAVE FULL QUALITY TABLE
# ============================================================

quality_file = (
    OUTPUT_DIR
    / "surveillance_quality_country_year.csv"
)

quality.to_csv(
    quality_file,
    index=False
)


# ============================================================
# 18. CANDIDATE SOUTHEAST ASIAN COUNTRIES
# ============================================================

candidate_quality = quality[
    quality[
        "COUNTRY_AREA_TERRITORY"
    ].isin(CANDIDATE_COUNTRIES)
].copy()

candidate_quality = candidate_quality.sort_values(
    [
        "COUNTRY_AREA_TERRITORY",
        "ISO_YEAR",
    ]
)

candidate_file = (
    OUTPUT_DIR
    / "candidate_country_quality.csv"
)

candidate_quality.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# 19. COUNTRY-LEVEL SUMMARY
# ============================================================

if not candidate_quality.empty:

    candidate_summary = (
        candidate_quality
        .groupby(
            "COUNTRY_AREA_TERRITORY",
            as_index=False
        )
        .agg(
            first_year=(
                "ISO_YEAR",
                "min"
            ),
            last_year=(
                "ISO_YEAR",
                "max"
            ),
            number_of_years=(
                "ISO_YEAR",
                "nunique"
            ),
            median_overall_reporting_percent=(
                "overall_reporting_percent",
                "median"
            ),
            median_influenza_reporting_percent=(
                "influenza_reporting_percent",
                "median"
            ),
            median_rsv_reporting_percent=(
                "rsv_reporting_percent",
                "median"
            ),
            years_influenza_ge_80pct=(
                "influenza_ge_80pct",
                "sum"
            ),
            years_rsv_ge_80pct=(
                "rsv_ge_80pct",
                "sum"
            ),
        )
    )

else:

    candidate_summary = pd.DataFrame()


candidate_summary_file = (
    OUTPUT_DIR
    / "candidate_country_summary.csv"
)

candidate_summary.to_csv(
    candidate_summary_file,
    index=False
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("CANDIDATE COUNTRY SUMMARY")
print("=" * 75)

if candidate_summary.empty:

    print(
        "\nNo candidate Southeast Asian countries "
        "matched the WHO country names."
    )

else:

    print(
        candidate_summary
        .sort_values(
            "median_rsv_reporting_percent",
            ascending=False
        )
        .to_string(index=False)
    )


# ============================================================
# 21. SINGAPORE DETAIL
# ============================================================

print("\n" + "=" * 75)
print("SINGAPORE YEAR-BY-YEAR QUALITY")
print("=" * 75)

singapore = candidate_quality[
    candidate_quality[
        "COUNTRY_AREA_TERRITORY"
    ].eq("Singapore")
]

if singapore.empty:

    print("\nSingapore not found.")

else:

    display_columns = [
        "ISO_YEAR",
        "expected_iso_weeks",
        "reported_weeks",
        "overall_reporting_percent",
        "influenza_reporting_weeks",
        "influenza_reporting_percent",
        "rsv_reporting_weeks",
        "rsv_reporting_percent",
        "weeks_with_multiple_rows",
    ]

    print(
        singapore[
            display_columns
        ].to_string(index=False)
    )


# ============================================================
# 22. OUTPUT LOCATIONS
# ============================================================

print("\n" + "=" * 75)
print("QUALITY ASSESSMENT COMPLETE")
print("=" * 75)

print("\nCreated:")

print(quality_file)
print(candidate_file)
print(candidate_summary_file)

print(
    "\nDo not select the final countries yet. "
    "Review pathogen-specific reporting completeness first."
)