from pathlib import Path
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
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def join_unique(series):
    values = (
        series
        .dropna()
        .astype(str)
        .unique()
    )

    values = sorted(values)

    return " | ".join(values)


# ============================================================
# 3. LOAD ELIGIBILITY TABLE
# ============================================================

print("\n" + "=" * 75)
print("FLUNET WEEKLY SOURCE AUDIT")
print("=" * 75)

eligibility = pd.read_csv(
    ELIGIBILITY_FILE
)

# Keep country-years useful for influenza or RSV analysis
eligible = eligibility[
    eligibility["influenza_eligible"]
    | eligibility["rsv_eligible"]
].copy()

selected_country_years = eligible[
    [
        "country",
        "ISO_YEAR",
    ]
].drop_duplicates()


# ============================================================
# 4. LOAD RAW WHO FLUNET DATA
# ============================================================

print("\nLoading WHO FluNet data...")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False
)

print(f"Raw rows: {len(df):,}")


# ============================================================
# 5. PREPARE COUNTRY/YEAR/WEEK VARIABLES
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

df = df.rename(
    columns={
        "COUNTRY_AREA_TERRITORY": "country"
    }
)


# ============================================================
# 6. KEEP ONLY STUDY-ELIGIBLE COUNTRY-YEARS
# ============================================================

df = df.merge(
    selected_country_years,
    on=[
        "country",
        "ISO_YEAR",
    ],
    how="inner"
)

print(
    f"Rows after study-population filter: {len(df):,}"
)


# ============================================================
# 7. MAKE IMPORTANT VARIABLES NUMERIC
# ============================================================

for col in [
    "INF_ALL",
    "RSV",
    "SPEC_PROCESSED_NB",
    "RSV_PROCESSED",
]:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 8. AUDIT EACH COUNTRY-WEEK
# ============================================================

weekly_audit = (
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
            "country",
            "size"
        ),

        number_origin_sources=(
            "ORIGIN_SOURCE",
            lambda x: x.dropna().nunique()
        ),

        origin_sources=(
            "ORIGIN_SOURCE",
            join_unique
        ),

        number_rsv_sources=(
            "PSOURCE_RSV",
            lambda x: x.dropna().nunique()
        ),

        rsv_sources=(
            "PSOURCE_RSV",
            join_unique
        ),

        influenza_values_present=(
            "INF_ALL",
            lambda x: x.notna().sum()
        ),

        rsv_values_present=(
            "RSV",
            lambda x: x.notna().sum()
        ),

        influenza_denominators_present=(
            "SPEC_PROCESSED_NB",
            lambda x: x.notna().sum()
        ),

        rsv_denominators_present=(
            "RSV_PROCESSED",
            lambda x: x.notna().sum()
        ),
    )
)


# ============================================================
# 9. FLAG MULTIPLE-ROW WEEKS
# ============================================================

weekly_audit["multiple_rows"] = (
    weekly_audit["raw_rows"] > 1
)

weekly_audit["multiple_origin_sources"] = (
    weekly_audit["number_origin_sources"] > 1
)


# ============================================================
# 10. COUNTRY-LEVEL SUMMARY
# ============================================================

country_summary = (
    weekly_audit
    .groupby(
        "country",
        as_index=False
    )
    .agg(
        total_country_weeks=(
            "ISO_WEEK",
            "size"
        ),

        weeks_with_multiple_rows=(
            "multiple_rows",
            "sum"
        ),

        weeks_with_multiple_origin_sources=(
            "multiple_origin_sources",
            "sum"
        ),

        maximum_rows_in_one_week=(
            "raw_rows",
            "max"
        ),
    )
)

country_summary[
    "percent_weeks_multiple_rows"
] = (
    country_summary[
        "weeks_with_multiple_rows"
    ]
    / country_summary[
        "total_country_weeks"
    ]
    * 100
).round(2)


# ============================================================
# 11. EXAMPLE MULTI-ROW WEEKS
# ============================================================

multirow_examples = (
    weekly_audit[
        weekly_audit["multiple_rows"]
    ]
    .sort_values(
        [
            "raw_rows",
            "country",
            "ISO_YEAR",
            "ISO_WEEK",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ]
    )
    .head(100)
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

audit_file = (
    OUTPUT_DIR
    / "flunet_weekly_source_audit.csv"
)

summary_file = (
    OUTPUT_DIR
    / "flunet_source_audit_country_summary.csv"
)

examples_file = (
    OUTPUT_DIR
    / "flunet_multirow_week_examples.csv"
)

weekly_audit.to_csv(
    audit_file,
    index=False
)

country_summary.to_csv(
    summary_file,
    index=False
)

multirow_examples.to_csv(
    examples_file,
    index=False
)


# ============================================================
# 13. PRINT COUNTRY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("COUNTRY SOURCE-AUDIT SUMMARY")
print("=" * 75)

print(
    country_summary
    .sort_values(
        "percent_weeks_multiple_rows",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 14. PRINT SOURCE COMBINATIONS
# ============================================================

print("\n" + "=" * 75)
print("MOST COMMON ORIGIN-SOURCE COMBINATIONS")
print("=" * 75)

print(
    weekly_audit[
        "origin_sources"
    ]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# 15. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("FLUNET SOURCE AUDIT COMPLETE")
print("=" * 75)

print("\nSaved:")
print(audit_file)
print(summary_file)
print(examples_file)

print(
    "\nDo not aggregate weekly counts yet. "
    "First determine whether multiple rows represent "
    "independent or overlapping surveillance streams."
)