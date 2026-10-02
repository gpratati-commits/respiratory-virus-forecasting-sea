from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLU_RSV_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "candidate_country_year_eligibility.csv"
)

COVID_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sarscov2_country_year_quality.csv"
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
# 2. LOAD ELIGIBILITY TABLES
# ============================================================

print("\n" + "=" * 75)
print("FINAL STUDY POPULATION DEFINITION")
print("=" * 75)

flu_rsv = pd.read_csv(FLU_RSV_FILE)
covid = pd.read_csv(COVID_FILE)


# ============================================================
# 3. STANDARDISE COUNTRY NAMES
# ============================================================

flu_rsv = flu_rsv.rename(
    columns={
        "COUNTRY_AREA_TERRITORY": "country"
    }
)

covid = covid.rename(
    columns={
        "Country": "country",
        "case_reporting_ge_80pct": "good_covid_year",
    }
)


# ============================================================
# 4. KEEP REQUIRED VARIABLES
# ============================================================

flu_rsv = flu_rsv[
    [
        "country",
        "ISO_YEAR",
        "good_influenza_year",
        "good_rsv_year",
        "good_joint_year",
    ]
].copy()

covid = covid[
    [
        "country",
        "ISO_YEAR",
        "good_covid_year",
        "case_reporting_percent",
    ]
].copy()


# ============================================================
# 5. KEEP COMPLETE YEARS ONLY
# ============================================================

flu_rsv = flu_rsv[
    flu_rsv["ISO_YEAR"] <= 2025
].copy()

covid = covid[
    covid["ISO_YEAR"] <= 2025
].copy()


# ============================================================
# 6. MERGE SURVEILLANCE ELIGIBILITY TABLES
# ============================================================

study = flu_rsv.merge(
    covid,
    on=[
        "country",
        "ISO_YEAR",
    ],
    how="outer",
)


# ============================================================
# 7. CLEAN BOOLEAN VARIABLES
# ============================================================

boolean_columns = [
    "good_influenza_year",
    "good_rsv_year",
    "good_joint_year",
    "good_covid_year",
]

for col in boolean_columns:
    study[col] = (
        study[col]
        .fillna(False)
        .astype(bool)
    )


# ============================================================
# 8. DEFINE ANALYSIS-SPECIFIC ELIGIBILITY
# ============================================================

study["influenza_eligible"] = (
    study["good_influenza_year"]
)

study["rsv_eligible"] = (
    study["good_rsv_year"]
)

study["covid_eligible"] = (
    study["good_covid_year"]
)

study["influenza_rsv_eligible"] = (
    study["good_influenza_year"]
    & study["good_rsv_year"]
)

study["influenza_covid_eligible"] = (
    study["good_influenza_year"]
    & study["good_covid_year"]
)

study["three_pathogen_eligible"] = (
    study["good_influenza_year"]
    & study["good_rsv_year"]
    & study["good_covid_year"]
)


# ============================================================
# 9. SORT FINAL TABLE
# ============================================================

study = study.sort_values(
    [
        "country",
        "ISO_YEAR",
    ]
)


# ============================================================
# 10. SAVE FULL STUDY POPULATION TABLE
# ============================================================

study_file = (
    OUTPUT_DIR
    / "final_study_population.csv"
)

study.to_csv(
    study_file,
    index=False
)


# ============================================================
# 11. THREE-PATHOGEN ELIGIBLE SUBSET
# ============================================================

three = study[
    study["three_pathogen_eligible"]
].copy()

three_file = (
    OUTPUT_DIR
    / "three_pathogen_country_years.csv"
)

three.to_csv(
    three_file,
    index=False
)


# ============================================================
# 12. COUNTRY SUMMARY
# ============================================================

summary = (
    study
    .groupby(
        "country",
        as_index=False
    )
    .agg(
        influenza_years=(
            "influenza_eligible",
            "sum"
        ),
        rsv_years=(
            "rsv_eligible",
            "sum"
        ),
        covid_years=(
            "covid_eligible",
            "sum"
        ),
        influenza_covid_years=(
            "influenza_covid_eligible",
            "sum"
        ),
        three_pathogen_years=(
            "three_pathogen_eligible",
            "sum"
        ),
    )
)

summary_file = (
    OUTPUT_DIR
    / "final_country_analysis_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 13. PRINT THREE-PATHOGEN COUNTRY-YEARS
# ============================================================

print("\n" + "=" * 75)
print("THREE-PATHOGEN ELIGIBLE COUNTRY-YEARS")
print("=" * 75)

if three.empty:

    print("\nNo country-years meet all three criteria.")

else:

    for country, group in three.groupby("country"):

        years = sorted(
            group["ISO_YEAR"]
            .astype(int)
            .tolist()
        )

        print(
            f"{country}: {years}"
        )


# ============================================================
# 14. PRINT SINGAPORE ELIGIBILITY
# ============================================================

print("\n" + "=" * 75)
print("SINGAPORE ELIGIBILITY")
print("=" * 75)

sg = study[
    study["country"].eq("Singapore")
]

if sg.empty:

    print("\nSingapore not found.")

else:

    print(
        sg[
            [
                "ISO_YEAR",
                "influenza_eligible",
                "rsv_eligible",
                "covid_eligible",
                "influenza_covid_eligible",
                "three_pathogen_eligible",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 15. PRINT FINAL COUNTRY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL COUNTRY SUMMARY")
print("=" * 75)

print(
    summary
    .sort_values(
        "three_pathogen_years",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 16. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STUDY POPULATION DEFINITION COMPLETE")
print("=" * 75)

print("\nSaved:")
print(study_file)
print(three_file)
print(summary_file)