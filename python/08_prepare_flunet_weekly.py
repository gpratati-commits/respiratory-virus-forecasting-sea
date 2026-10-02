from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "VIW_FNT.csv"
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

OUTPUT_DIR.mkdir(
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
print("PREPARING WEEKLY WHO FLUNET DATA")
print("=" * 75)

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE
)

print(f"\nRaw WHO rows: {len(df):,}")


# ============================================================
# 3. STANDARDISE COUNTRY/YEAR/WEEK
# ============================================================

df = df.rename(
    columns={
        "COUNTRY_AREA_TERRITORY": "country"
    }
)

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
        "country",
        "ISO_YEAR",
        "ISO_WEEK",
    ]
).copy()

df["ISO_YEAR"] = df["ISO_YEAR"].astype(int)
df["ISO_WEEK"] = df["ISO_WEEK"].astype(int)


# ============================================================
# 4. KEEP STUDY-ELIGIBLE COUNTRY-YEARS
# ============================================================

eligible = eligibility[
    eligibility["influenza_eligible"]
    | eligibility["rsv_eligible"]
].copy()

eligible = eligible[
    [
        "country",
        "ISO_YEAR",
        "influenza_eligible",
        "rsv_eligible",
    ]
]

df = df.merge(
    eligible,
    on=[
        "country",
        "ISO_YEAR",
    ],
    how="inner"
)

print(
    f"Rows after eligibility filter: {len(df):,}"
)


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "INF_ALL",
    "INF_A",
    "INF_B",
    "SPEC_PROCESSED_NB",
    "RSV",
    "RSV_PROCESSED",
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 6. ROW-LEVEL INFLUENZA POSITIVE COUNT
# ============================================================

# Use INF_ALL when WHO provides it.
# If INF_ALL is missing, derive influenza total from
# INF_A + INF_B.
#
# IMPORTANT:
# We do NOT add INF_ALL + INF_A + INF_B because
# INF_ALL already represents total influenza positives.

derived_flu = (
    df[
        [
            "INF_A",
            "INF_B",
        ]
    ]
    .sum(
        axis=1,
        min_count=1
    )
)

df["influenza_positive_row"] = (
    df["INF_ALL"]
    .where(
        df["INF_ALL"].notna(),
        derived_flu
    )
)


# ============================================================
# 7. ROW-LEVEL DENOMINATORS
# ============================================================

df["influenza_tested_row"] = (
    df["SPEC_PROCESSED_NB"]
)

df["rsv_positive_row"] = (
    df["RSV"]
)

df["rsv_tested_row"] = (
    df["RSV_PROCESSED"]
)


# ============================================================
# 8. SOURCE-LEVEL QC FLAGS
# ============================================================

df["rsv_source_denominator_problem"] = (
    df["rsv_positive_row"].notna()
    & df["rsv_tested_row"].notna()
    & (
        (df["rsv_positive_row"] > df["rsv_tested_row"])
        |
        (
            (df["rsv_tested_row"] == 0)
            & (df["rsv_positive_row"] > 0)
        )
    )
)

df["influenza_source_denominator_problem"] = (
    df["influenza_positive_row"].notna()
    & df["influenza_tested_row"].notna()
    & (
        df["influenza_positive_row"]
        > df["influenza_tested_row"]
    )
)


# ============================================================
# 9. AGGREGATE SOURCE ROWS TO COUNTRY-WEEK
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
        raw_source_rows=(
            "country",
            "size"
        ),

        influenza_positive=(
            "influenza_positive_row",
            lambda x: x.sum(min_count=1)
        ),

        influenza_tested=(
            "influenza_tested_row",
            lambda x: x.sum(min_count=1)
        ),

        rsv_positive=(
            "rsv_positive_row",
            lambda x: x.sum(min_count=1)
        ),

        rsv_tested=(
            "rsv_tested_row",
            lambda x: x.sum(min_count=1)
        ),

        influenza_source_qc_problem=(
            "influenza_source_denominator_problem",
            "max"
        ),

        rsv_source_qc_problem=(
            "rsv_source_denominator_problem",
            "max"
        ),

        influenza_eligible=(
            "influenza_eligible",
            "max"
        ),

        rsv_eligible=(
            "rsv_eligible",
            "max"
        ),
    )
)


# ============================================================
# 10. CREATE ISO WEEK START DATE
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
# 11. WEEK-LEVEL DENOMINATOR VALIDITY
# ============================================================

weekly["influenza_denominator_valid"] = (
    weekly["influenza_tested"].notna()
    & (weekly["influenza_tested"] > 0)
    & weekly["influenza_positive"].notna()
    & (
        weekly["influenza_positive"]
        <= weekly["influenza_tested"]
    )
)

weekly["rsv_denominator_valid"] = (
    weekly["rsv_tested"].notna()
    & (weekly["rsv_tested"] > 0)
    & weekly["rsv_positive"].notna()
    & (
        weekly["rsv_positive"]
        <= weekly["rsv_tested"]
    )
)


# ============================================================
# 12. POSITIVITY
# ============================================================

weekly["influenza_positivity"] = np.where(
    weekly["influenza_denominator_valid"],
    weekly["influenza_positive"]
    / weekly["influenza_tested"],
    np.nan
)

weekly["rsv_positivity"] = np.where(
    weekly["rsv_denominator_valid"],
    weekly["rsv_positive"]
    / weekly["rsv_tested"],
    np.nan
)


# ============================================================
# 13. MASK PATHOGEN DATA OUTSIDE ELIGIBLE YEARS
# ============================================================

influenza_columns = [
    "influenza_positive",
    "influenza_tested",
    "influenza_positivity",
]

rsv_columns = [
    "rsv_positive",
    "rsv_tested",
    "rsv_positivity",
]

weekly.loc[
    ~weekly["influenza_eligible"],
    influenza_columns
] = np.nan

weekly.loc[
    ~weekly["rsv_eligible"],
    rsv_columns
] = np.nan


# ============================================================
# 14. SORT
# ============================================================

weekly = weekly.sort_values(
    [
        "country",
        "week_start",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 15. QC SUMMARY
# ============================================================

qc_summary = (
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

        weeks_multiple_sources=(
            "raw_source_rows",
            lambda x: (x > 1).sum()
        ),

        influenza_positive_missing=(
            "influenza_positive",
            lambda x: x.isna().sum()
        ),

        rsv_positive_missing=(
            "rsv_positive",
            lambda x: x.isna().sum()
        ),

        invalid_influenza_denominator_weeks=(
            "influenza_denominator_valid",
            lambda x: (~x).sum()
        ),

        invalid_rsv_denominator_weeks=(
            "rsv_denominator_valid",
            lambda x: (~x).sum()
        ),
    )
)


# ============================================================
# 16. SAVE
# ============================================================

weekly_file = (
    OUTPUT_DIR
    / "flunet_weekly_clean.csv"
)

qc_file = (
    TABLE_DIR
    / "flunet_weekly_clean_qc.csv"
)

weekly.to_csv(
    weekly_file,
    index=False
)

qc_summary.to_csv(
    qc_file,
    index=False
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("WEEKLY FLUNET DATASET COMPLETE")
print("=" * 75)

print(
    f"\nWeekly rows: {len(weekly):,}"
)

print(
    f"Countries: {weekly['country'].nunique()}"
)

print(
    f"First week: {weekly['week_start'].min()}"
)

print(
    f"Last week:  {weekly['week_start'].max()}"
)

print("\nQC summary:\n")

print(
    qc_summary
    .to_string(index=False)
)

print("\nSaved:")
print(weekly_file)
print(qc_file)

print(
    "\nImportant: invalid denominator weeks retain "
    "positive counts but have positivity set to missing."
)