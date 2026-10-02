from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLUNET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flunet_weekly_clean.csv"
)

COVID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sarscov2_weekly_clean.csv"
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
print("BUILDING MASTER RESPIRATORY VIRUS DATASET")
print("=" * 75)

flu = pd.read_csv(
    FLUNET_FILE,
    low_memory=False
)

covid = pd.read_csv(
    COVID_FILE,
    low_memory=False
)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE,
    low_memory=False
)

print(f"\nInfluenza/RSV weekly rows: {len(flu):,}")
print(f"SARS-CoV-2 weekly rows:    {len(covid):,}")


# ============================================================
# 3. DEFINE MERGE KEY
# ============================================================

KEY = [
    "country",
    "ISO_YEAR",
    "ISO_WEEK",
]


# ============================================================
# 4. CHECK KEY UNIQUENESS
# ============================================================

flu_duplicates = flu.duplicated(
    subset=KEY
).sum()

covid_duplicates = covid.duplicated(
    subset=KEY
).sum()

print("\nDuplicate-key check:")
print(
    f"Influenza/RSV duplicate country-weeks: "
    f"{flu_duplicates}"
)

print(
    f"SARS-CoV-2 duplicate country-weeks: "
    f"{covid_duplicates}"
)

if flu_duplicates > 0:
    raise ValueError(
        "Influenza/RSV dataset contains duplicate "
        "country-year-week keys."
    )

if covid_duplicates > 0:
    raise ValueError(
        "SARS-CoV-2 dataset contains duplicate "
        "country-year-week keys."
    )


# ============================================================
# 5. PREPARE WEEK-START VARIABLES
# ============================================================

flu["week_start"] = pd.to_datetime(
    flu["week_start"],
    errors="coerce"
)

covid["week_start"] = pd.to_datetime(
    covid["week_start"],
    errors="coerce"
)

flu = flu.rename(
    columns={
        "week_start": "week_start_flu",
        "raw_source_rows": "flunet_source_rows",
    }
)

covid = covid.rename(
    columns={
        "week_start": "week_start_covid",
        "raw_rows": "covid_raw_rows",
    }
)


# ============================================================
# 6. OUTER MERGE
# ============================================================

master = flu.merge(
    covid,
    on=KEY,
    how="outer",
    suffixes=(
        "_flu",
        "_covid"
    ),
)


# ============================================================
# 7. CHECK WEEK-START CONSISTENCY
# ============================================================

both_dates = (
    master["week_start_flu"].notna()
    & master["week_start_covid"].notna()
)

date_mismatch = (
    both_dates
    & (
        master["week_start_flu"]
        != master["week_start_covid"]
    )
)

print(
    "\nRows with inconsistent week_start dates:",
    date_mismatch.sum()
)

if date_mismatch.sum() > 0:

    examples = master.loc[
        date_mismatch,
        KEY
        + [
            "week_start_flu",
            "week_start_covid",
        ]
    ]

    print(
        examples.head(20)
        .to_string(index=False)
    )

    raise ValueError(
        "Week-date mismatch found between datasets."
    )


# ============================================================
# 8. CREATE FINAL WEEK START
# ============================================================

master["week_start"] = (
    master["week_start_flu"]
    .combine_first(
        master["week_start_covid"]
    )
)

master = master.drop(
    columns=[
        "week_start_flu",
        "week_start_covid",
    ]
)


# ============================================================
# 9. MERGE ANALYSIS ELIGIBILITY FLAGS
# ============================================================

eligibility_columns = [
    "country",
    "ISO_YEAR",
    "influenza_eligible",
    "rsv_eligible",
    "covid_eligible",
    "influenza_rsv_eligible",
    "influenza_covid_eligible",
    "three_pathogen_eligible",
]

eligibility = eligibility[
    eligibility_columns
].drop_duplicates()

master = master.merge(
    eligibility,
    on=[
        "country",
        "ISO_YEAR",
    ],
    how="left",
    suffixes=(
        "",
        "_study"
    ),
)


# ============================================================
# 10. CLEAN ELIGIBILITY FLAGS
# ============================================================

eligibility_flags = [
    "influenza_eligible",
    "rsv_eligible",
    "covid_eligible",
    "influenza_rsv_eligible",
    "influenza_covid_eligible",
    "three_pathogen_eligible",
]

for col in eligibility_flags:

    if col not in master.columns:

        study_col = (
            col
            + "_study"
        )

        if study_col in master.columns:
            master[col] = master[study_col]

    master[col] = (
        master[col]
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# 11. WEEK-LEVEL DATA AVAILABILITY
# ============================================================

master["influenza_observed"] = (
    master["influenza_positive"]
    .notna()
)

master["rsv_observed"] = (
    master["rsv_positive"]
    .notna()
)

master["sarscov2_observed"] = (
    master["sarscov2_cases"]
    .notna()
)

master["influenza_rsv_observed"] = (
    master["influenza_observed"]
    & master["rsv_observed"]
)

master["three_pathogen_observed"] = (
    master["influenza_observed"]
    & master["rsv_observed"]
    & master["sarscov2_observed"]
)


# ============================================================
# 12. DO NOT TURN MISSING INTO ZERO
# ============================================================

# Missing surveillance values remain NaN.
#
# A missing value means surveillance information is unavailable.
# It must not be interpreted as zero virus activity.


# ============================================================
# 13. SORT
# ============================================================

master = master.sort_values(
    [
        "country",
        "week_start",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 14. FINAL DUPLICATE CHECK
# ============================================================

master_duplicates = master.duplicated(
    subset=KEY
).sum()

print(
    "\nMaster duplicate country-weeks:",
    master_duplicates
)

if master_duplicates > 0:

    raise ValueError(
        "Master dataset contains duplicate keys."
    )


# ============================================================
# 15. COUNTRY QC SUMMARY
# ============================================================

qc = (
    master
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

        influenza_observed_weeks=(
            "influenza_observed",
            "sum"
        ),

        rsv_observed_weeks=(
            "rsv_observed",
            "sum"
        ),

        sarscov2_observed_weeks=(
            "sarscov2_observed",
            "sum"
        ),

        influenza_rsv_observed_weeks=(
            "influenza_rsv_observed",
            "sum"
        ),

        three_pathogen_observed_weeks=(
            "three_pathogen_observed",
            "sum"
        ),
    )
)


# ============================================================
# 16. THREE-PATHOGEN ANALYSIS SUBSET
# ============================================================

three_pathogen = master[
    master["three_pathogen_eligible"]
].copy()


# ============================================================
# 17. SINGAPORE SUBSET
# ============================================================

singapore = master[
    master["country"]
    .eq("Singapore")
].copy()


# ============================================================
# 18. SAVE FILES
# ============================================================

master_file = (
    OUTPUT_DIR
    / "respiratory_virus_master_weekly.csv"
)

three_file = (
    OUTPUT_DIR
    / "three_pathogen_master_weekly.csv"
)

singapore_file = (
    OUTPUT_DIR
    / "singapore_respiratory_weekly.csv"
)

qc_file = (
    TABLE_DIR
    / "master_dataset_qc.csv"
)

master.to_csv(
    master_file,
    index=False
)

three_pathogen.to_csv(
    three_file,
    index=False
)

singapore.to_csv(
    singapore_file,
    index=False
)

qc.to_csv(
    qc_file,
    index=False
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("MASTER RESPIRATORY VIRUS DATASET COMPLETE")
print("=" * 75)

print(f"\nMaster rows: {len(master):,}")

print(
    "Countries:",
    master["country"].nunique()
)

print(
    "First week:",
    master["week_start"].min()
)

print(
    "Last week:",
    master["week_start"].max()
)

print(
    "\nThree-pathogen eligible rows:",
    len(three_pathogen)
)

print(
    "Singapore rows:",
    len(singapore)
)

print("\nCountry QC summary:\n")

print(
    qc.to_string(
        index=False
    )
)

print("\nSaved:")
print(master_file)
print(three_file)
print(singapore_file)
print(qc_file)

print(
    "\nImportant: missing pathogen observations remain missing "
    "and were not converted to zero."
)