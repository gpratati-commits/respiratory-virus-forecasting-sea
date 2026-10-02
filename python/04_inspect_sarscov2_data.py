from pathlib import Path
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
# 2. CANDIDATE COUNTRIES
# ============================================================

CANDIDATE_COUNTRIES = [
    "Singapore",
    "Malaysia",
    "Philippines",
    "Indonesia",
    "Thailand",
    "Viet Nam",
    "Vietnam",
    "Brunei Darussalam",
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("WHO SARS-CoV-2 DATA INSPECTION")
print("=" * 75)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"COVID-19 dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"\nRows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 4. SHOW COLUMNS
# ============================================================

print("\nColumns:")

for i, col in enumerate(
    df.columns,
    start=1
):
    print(f"{i:02d}. {col}")


# ============================================================
# 5. PARSE DATES
# ============================================================

df["Date_reported"] = pd.to_datetime(
    df["Date_reported"],
    errors="coerce"
)

print("\n" + "=" * 75)
print("DATE COVERAGE")
print("=" * 75)

print(
    "Earliest date:",
    df["Date_reported"].min()
)

print(
    "Latest date:",
    df["Date_reported"].max()
)


# ============================================================
# 6. COUNTRY AVAILABILITY
# ============================================================

print("\n" + "=" * 75)
print("CANDIDATE COUNTRY AVAILABILITY")
print("=" * 75)

available = set(
    df["Country"]
    .dropna()
    .astype(str)
)

for country in CANDIDATE_COUNTRIES:

    if country in available:
        print(f"FOUND:     {country}")

    else:
        print(f"NOT FOUND: {country}")


# ============================================================
# 7. FILTER CANDIDATE COUNTRIES
# ============================================================

candidate_df = df[
    df["Country"].isin(
        CANDIDATE_COUNTRIES
    )
].copy()


# ============================================================
# 8. CREATE YEAR AND ISO-WEEK VARIABLES
# ============================================================

iso = (
    candidate_df["Date_reported"]
    .dt.isocalendar()
)

candidate_df["ISO_YEAR"] = (
    iso["year"]
    .astype("Int64")
)

candidate_df["ISO_WEEK"] = (
    iso["week"]
    .astype("Int64")
)


# ============================================================
# 9. NUMERIC CASE VARIABLES
# ============================================================

candidate_df["New_cases"] = pd.to_numeric(
    candidate_df["New_cases"],
    errors="coerce"
)

candidate_df["Cumulative_cases"] = pd.to_numeric(
    candidate_df["Cumulative_cases"],
    errors="coerce"
)


# ============================================================
# 10. DUPLICATE COUNTRY-WEEK CHECK
# ============================================================

print("\n" + "=" * 75)
print("DUPLICATE COUNTRY-DATE CHECK")
print("=" * 75)

duplicates = candidate_df.duplicated(
    subset=[
        "Country",
        "Date_reported",
    ],
    keep=False
)

print(
    "Rows involved in duplicated country-date records:",
    duplicates.sum()
)


# ============================================================
# 11. NEGATIVE REVISION CHECK
# ============================================================

print("\n" + "=" * 75)
print("NEGATIVE CASE REVISION CHECK")
print("=" * 75)

negative_cases = candidate_df[
    candidate_df["New_cases"] < 0
]

print(
    "Rows with negative New_cases:",
    len(negative_cases)
)

if len(negative_cases) > 0:

    print("\nNegative revisions by country:")

    print(
        negative_cases["Country"]
        .value_counts()
        .to_string()
    )


# ============================================================
# 12. COUNTRY DATE RANGE
# ============================================================

country_dates = (
    candidate_df
    .groupby(
        "Country",
        as_index=False
    )
    .agg(
        first_date=(
            "Date_reported",
            "min"
        ),
        last_date=(
            "Date_reported",
            "max"
        ),
        number_rows=(
            "Date_reported",
            "size"
        ),
        unique_weeks=(
            "Date_reported",
            "nunique"
        ),
    )
)

print("\n" + "=" * 75)
print("COUNTRY DATE COVERAGE")
print("=" * 75)

print(
    country_dates
    .to_string(index=False)
)


# ============================================================
# 13. YEAR-BY-YEAR WEEK COVERAGE
# ============================================================

coverage = (
    candidate_df
    .dropna(
        subset=[
            "Country",
            "ISO_YEAR",
            "ISO_WEEK",
        ]
    )
    .groupby(
        [
            "Country",
            "ISO_YEAR",
        ],
        as_index=False
    )
    .agg(
        reported_weeks=(
            "ISO_WEEK",
            "nunique"
        ),
        nonmissing_case_weeks=(
            "New_cases",
            lambda x: x.notna().sum()
        ),
    )
)


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

country_dates_file = (
    OUTPUT_DIR
    / "sarscov2_country_date_coverage.csv"
)

coverage_file = (
    OUTPUT_DIR
    / "sarscov2_country_year_coverage.csv"
)

negative_file = (
    OUTPUT_DIR
    / "sarscov2_negative_case_revisions.csv"
)

country_dates.to_csv(
    country_dates_file,
    index=False
)

coverage.to_csv(
    coverage_file,
    index=False
)

negative_cases.to_csv(
    negative_file,
    index=False
)


# ============================================================
# 15. SINGAPORE DETAIL
# ============================================================

print("\n" + "=" * 75)
print("SINGAPORE SARS-CoV-2 COVERAGE")
print("=" * 75)

sg = coverage[
    coverage["Country"]
    .eq("Singapore")
]

if sg.empty:

    print(
        "\nSingapore not found in COVID dataset."
    )

else:

    print(
        sg.to_string(
            index=False
        )
    )


# ============================================================
# 16. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("SARS-CoV-2 INSPECTION COMPLETE")
print("=" * 75)

print("\nCreated:")

print(country_dates_file)
print(coverage_file)
print(negative_file)

print(
    "\nNext step: assess weekly SARS-CoV-2 "
    "reporting completeness before merging datasets."
)