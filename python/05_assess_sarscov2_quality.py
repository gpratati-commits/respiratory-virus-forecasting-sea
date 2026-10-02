from pathlib import Path
from datetime import date

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WHO-COVID-19-global-data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. COUNTRIES OF INTEREST
# ============================================================

CANDIDATE_COUNTRIES = [
    "Singapore",
    "Malaysia",
    "Philippines",
    "Indonesia",
    "Thailand",
    "Viet Nam",
    "Brunei Darussalam",
]


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def iso_weeks_in_year(year):
    """
    Number of ISO weeks in a given ISO year.
    An ISO year contains either 52 or 53 weeks.
    """
    return date(
        int(year),
        12,
        28
    ).isocalendar().week


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("WHO SARS-CoV-2 SURVEILLANCE QUALITY ASSESSMENT")
print("=" * 75)

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"\nRows loaded: {len(df):,}")


# ============================================================
# 5. DATE PREPARATION
# ============================================================

df["Date_reported"] = pd.to_datetime(
    df["Date_reported"],
    errors="coerce"
)

df["New_cases"] = pd.to_numeric(
    df["New_cases"],
    errors="coerce"
)


# ============================================================
# 6. KEEP CANDIDATE COUNTRIES
# ============================================================

df = df[
    df["Country"].isin(
        CANDIDATE_COUNTRIES
    )
].copy()


# ============================================================
# 7. CREATE ISO YEAR AND WEEK
# ============================================================

iso = (
    df["Date_reported"]
    .dt.isocalendar()
)

df["ISO_YEAR"] = (
    iso["year"]
    .astype("Int64")
)

df["ISO_WEEK"] = (
    iso["week"]
    .astype("Int64")
)


# ============================================================
# 8. USE COMPLETE YEARS ONLY
# ============================================================

# 2026 is still incomplete.
df = df[
    df["ISO_YEAR"] <= 2025
].copy()


# ============================================================
# 9. CREATE ONE RECORD PER COUNTRY-WEEK
# ============================================================

weekly = (
    df
    .groupby(
        [
            "Country",
            "ISO_YEAR",
            "ISO_WEEK",
        ],
        as_index=False
    )
    .agg(
        raw_rows=(
            "Date_reported",
            "size"
        ),
        new_cases=(
            "New_cases",
            lambda x: x.sum(min_count=1)
        ),
    )
)


# ============================================================
# 10. WEEK-LEVEL FLAGS
# ============================================================

weekly["case_value_present"] = (
    weekly["new_cases"].notna()
)

weekly["case_value_missing"] = (
    weekly["new_cases"].isna()
)

weekly["zero_case_week"] = (
    weekly["new_cases"] == 0
)

weekly["positive_case_week"] = (
    weekly["new_cases"] > 0
)

weekly["negative_revision_week"] = (
    weekly["new_cases"] < 0
)

weekly["multiple_raw_rows"] = (
    weekly["raw_rows"] > 1
)


# ============================================================
# 11. COUNTRY-YEAR QUALITY
# ============================================================

quality = (
    weekly
    .groupby(
        [
            "Country",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        weeks_present=(
            "ISO_WEEK",
            "nunique"
        ),
        nonmissing_case_weeks=(
            "case_value_present",
            "sum"
        ),
        missing_case_weeks=(
            "case_value_missing",
            "sum"
        ),
        zero_case_weeks=(
            "zero_case_week",
            "sum"
        ),
        positive_case_weeks=(
            "positive_case_week",
            "sum"
        ),
        negative_revision_weeks=(
            "negative_revision_week",
            "sum"
        ),
        weeks_with_multiple_rows=(
            "multiple_raw_rows",
            "sum"
        ),
    )
)


# ============================================================
# 12. EXPECTED ISO WEEKS
# ============================================================

quality["expected_iso_weeks"] = (
    quality["ISO_YEAR"]
    .apply(iso_weeks_in_year)
)


# ============================================================
# 13. CASE-REPORTING COMPLETENESS
# ============================================================

quality["case_reporting_percent"] = (
    quality["nonmissing_case_weeks"]
    / quality["expected_iso_weeks"]
    * 100
).round(2)


# ============================================================
# 14. SCREENING FLAG
# ============================================================

# This is only a surveillance-completeness screen.
# It does NOT guarantee that reported case counts are
# epidemiologically comparable between countries or years.

quality["case_reporting_ge_80pct"] = (
    quality["case_reporting_percent"]
    >= 80
)


# ============================================================
# 15. SAVE COUNTRY-YEAR QUALITY TABLE
# ============================================================

quality_file = (
    OUTPUT_DIR
    / "sarscov2_country_year_quality.csv"
)

quality.to_csv(
    quality_file,
    index=False
)


# ============================================================
# 16. COUNTRY SUMMARY
# ============================================================

summary = (
    quality
    .groupby(
        "Country",
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
        years_ge_80pct=(
            "case_reporting_ge_80pct",
            "sum"
        ),
        median_case_reporting_percent=(
            "case_reporting_percent",
            "median"
        ),
    )
)

summary_file = (
    OUTPUT_DIR
    / "sarscov2_country_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. PRINT SINGAPORE
# ============================================================

print("\n" + "=" * 75)
print("SINGAPORE YEAR-BY-YEAR SARS-CoV-2 QUALITY")
print("=" * 75)

singapore = quality[
    quality["Country"]
    .eq("Singapore")
]

print(
    singapore[
        [
            "ISO_YEAR",
            "expected_iso_weeks",
            "nonmissing_case_weeks",
            "missing_case_weeks",
            "case_reporting_percent",
            "positive_case_weeks",
            "zero_case_weeks",
            "negative_revision_weeks",
            "case_reporting_ge_80pct",
        ]
    ].to_string(index=False)
)


# ============================================================
# 18. PRINT ALL COUNTRIES
# ============================================================

print("\n" + "=" * 75)
print("COUNTRY SUMMARY")
print("=" * 75)

print(
    summary
    .sort_values(
        "years_ge_80pct",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 19. PRINT ELIGIBLE YEARS
# ============================================================

print("\n" + "=" * 75)
print("YEARS WITH >=80% NON-MISSING CASE REPORTING")
print("=" * 75)

good = quality[
    quality[
        "case_reporting_ge_80pct"
    ]
]

for country, group in good.groupby("Country"):

    years = sorted(
        group["ISO_YEAR"]
        .astype(int)
        .tolist()
    )

    print(
        f"{country}: {years}"
    )


# ============================================================
# 20. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("SARS-CoV-2 QUALITY ASSESSMENT COMPLETE")
print("=" * 75)

print("\nSaved:")
print(quality_file)
print(summary_file)

print(
    "\nImportant: completeness of reported case counts "
    "does not imply complete ascertainment of infections."
)