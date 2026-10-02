from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WHO-COVID-19-global-data.csv"
)

ELIGIBILITY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_study_population.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("PREPARING WEEKLY SARS-CoV-2 DATA")
print("=" * 75)

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE
)

print(f"\nRaw WHO COVID rows: {len(df):,}")


# ============================================================
# 3. PREPARE DATES
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
# 4. CREATE ISO YEAR/WEEK
# ============================================================

iso = df["Date_reported"].dt.isocalendar()

df["ISO_YEAR"] = (
    iso["year"]
    .astype("Int64")
)

df["ISO_WEEK"] = (
    iso["week"]
    .astype("Int64")
)

df = df.dropna(
    subset=[
        "Country",
        "ISO_YEAR",
        "ISO_WEEK",
    ]
).copy()

df["ISO_YEAR"] = df["ISO_YEAR"].astype(int)
df["ISO_WEEK"] = df["ISO_WEEK"].astype(int)

df = df.rename(
    columns={
        "Country": "country"
    }
)


# ============================================================
# 5. KEEP COVID-ELIGIBLE COUNTRY-YEARS
# ============================================================

eligible = eligibility[
    eligibility["covid_eligible"]
].copy()

eligible = eligible[
    [
        "country",
        "ISO_YEAR",
    ]
].drop_duplicates()

df = df.merge(
    eligible,
    on=[
        "country",
        "ISO_YEAR",
    ],
    how="inner"
)

print(
    f"Rows after COVID eligibility filter: {len(df):,}"
)


# ============================================================
# 6. AGGREGATE TO ONE COUNTRY-WEEK
# ============================================================

weekly = (
    df
    .groupby(
        [
            "country",
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
        sarscov2_cases=(
            "New_cases",
            lambda x: x.sum(min_count=1)
        ),
    )
)


# ============================================================
# 7. CREATE WEEK START DATE
# ============================================================

weekly["week_start"] = pd.to_datetime(
    weekly["ISO_YEAR"].astype(str)
    + "-W"
    + weekly["ISO_WEEK"].astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)


# ============================================================
# 8. HANDLE NEGATIVE REPORTING REVISIONS
# ============================================================

weekly["negative_revision"] = (
    weekly["sarscov2_cases"] < 0
)

# Negative weekly values are administrative corrections,
# not negative biological incidence.
#
# Preserve the original value for audit, but create a modelling
# variable where negative corrections are treated as missing.

weekly["sarscov2_cases_original"] = (
    weekly["sarscov2_cases"]
)

weekly.loc[
    weekly["sarscov2_cases"] < 0,
    "sarscov2_cases"
] = np.nan


# ============================================================
# 9. MISSINGNESS FLAGS
# ============================================================

weekly["case_value_present"] = (
    weekly["sarscov2_cases"].notna()
)

weekly["zero_case_week"] = (
    weekly["sarscov2_cases"] == 0
)


# ============================================================
# 10. SORT
# ============================================================

weekly = weekly.sort_values(
    [
        "country",
        "week_start",
    ]
).reset_index(drop=True)


# ============================================================
# 11. QC SUMMARY
# ============================================================

qc = (
    weekly
    .groupby(
        "country",
        as_index=False
    )
    .agg(
        weeks=(
            "week_start",
            "size"
        ),
        missing_case_weeks=(
            "sarscov2_cases",
            lambda x: x.isna().sum()
        ),
        negative_revision_weeks=(
            "negative_revision",
            "sum"
        ),
        zero_case_weeks=(
            "zero_case_week",
            "sum"
        ),
        maximum_raw_rows_per_week=(
            "raw_rows",
            "max"
        ),
    )
)


# ============================================================
# 12. SAVE
# ============================================================

weekly_file = (
    OUTPUT_DIR
    / "sarscov2_weekly_clean.csv"
)

qc_file = (
    TABLE_DIR
    / "sarscov2_weekly_clean_qc.csv"
)

weekly.to_csv(
    weekly_file,
    index=False
)

qc.to_csv(
    qc_file,
    index=False
)


# ============================================================
# 13. PRINT
# ============================================================

print("\n" + "=" * 75)
print("WEEKLY SARS-CoV-2 DATASET COMPLETE")
print("=" * 75)

print(f"\nWeekly rows: {len(weekly):,}")
print(f"Countries: {weekly['country'].nunique()}")

print("\nQC summary:\n")

print(
    qc.to_string(index=False)
)

print("\nSaved:")
print(weekly_file)
print(qc_file)

print(
    "\nNegative reporting corrections are preserved in "
    "sarscov2_cases_original but set to missing in the "
    "modelling outcome."
)