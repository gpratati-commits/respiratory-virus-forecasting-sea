"""
Define the forecasting design for the multi-pathogen respiratory-virus
forecasting project.

This script does NOT fit forecasting models.

It records:
- analysis cohorts
- forecast horizons
- target pathogens
- training periods
- validation periods
- test periods
- surveillance coverage within each period

Final train/validation/test labels for individual forecasts will later be
assigned using the TARGET DATE, not simply the forecast-origin date.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "respiratory_virus_master_weekly.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESIGN_FILE = OUTPUT_DIR / "forecast_design.csv"
COVERAGE_FILE = OUTPUT_DIR / "forecast_design_coverage_check.csv"


# ============================================================
# 2. FORECAST HORIZONS
# ============================================================

FORECAST_HORIZONS = [1, 2, 4]


# ============================================================
# 3. PATHOGEN COLUMN DEFINITIONS
# ============================================================

PATHOGEN_COLUMNS = {
    "influenza": "influenza_positive",
    "rsv": "rsv_positive",
    "sarscov2": "sarscov2_cases",
}


# ============================================================
# 4. FROZEN FORECAST DESIGN
# ============================================================

# Dates refer to TARGET periods.
#
# Example:
# If a forecast originates in December 2023 but its target is
# January 2024, that forecast belongs to the 2024 TEST period.

COUNTRY_DESIGNS = [
    {
        "country": "Indonesia",
        "analysis_group": "primary_regional",
        "required_pathogens": ["influenza", "rsv", "sarscov2"],
        "target_pathogens": ["influenza", "rsv", "sarscov2"],
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
        "note": "Primary three-pathogen forecasting cohort",
    },
    {
        "country": "Malaysia",
        "analysis_group": "primary_regional",
        "required_pathogens": ["influenza", "rsv", "sarscov2"],
        "target_pathogens": ["influenza", "rsv", "sarscov2"],
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
        "note": "Primary three-pathogen forecasting cohort",
    },
    {
        "country": "Brunei Darussalam",
        "analysis_group": "primary_regional",
        "required_pathogens": ["influenza", "rsv", "sarscov2"],
        "target_pathogens": ["influenza", "rsv", "sarscov2"],
        "train_start": "2021-01-01",
        "train_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
        "note": "Primary three-pathogen forecasting cohort",
    },
    {
        "country": "Philippines",
        "analysis_group": "secondary_regional",
        "required_pathogens": ["influenza", "rsv", "sarscov2"],
        "target_pathogens": ["influenza", "rsv", "sarscov2"],
        "train_start": "2020-01-01",
        "train_end": "2021-12-31",
        "validation_start": "2022-01-01",
        "validation_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "note": "Secondary analysis because three-pathogen overlap ends in 2023",
    },
    {
        "country": "Thailand",
        "analysis_group": "exploratory_regional",
        "required_pathogens": ["influenza", "rsv", "sarscov2"],
        "target_pathogens": ["influenza", "rsv", "sarscov2"],
        "train_start": "2023-01-01",
        "train_end": "2023-12-31",
        "validation_start": "2024-01-01",
        "validation_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
        "note": "Exploratory only because the training history is short",
    },
    {
        "country": "Singapore",
        "analysis_group": "singapore_case_study",
        "required_pathogens": ["influenza", "sarscov2"],
        "target_pathogens": ["influenza", "sarscov2"],
        "train_start": "2020-01-01",
        "train_end": "2021-12-31",
        "validation_start": "2022-01-01",
        "validation_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
        "note": "Two-pathogen Singapore case study; RSV unavailable",
    },
]


# ============================================================
# 5. LOAD MASTER DATA
# ============================================================

print("\n" + "=" * 80)
print("FORECAST DESIGN")
print("=" * 80)

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found:\n{MASTER_FILE}"
    )

df = pd.read_csv(MASTER_FILE)

required_master_columns = [
    "country",
    "week_start",
    "influenza_positive",
    "rsv_positive",
    "sarscov2_cases",
]

missing_columns = [
    col
    for col in required_master_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The master dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce"
)

if df["week_start"].isna().any():
    raise ValueError(
        "Some week_start values could not be converted to dates."
    )


# ============================================================
# 6. CREATE DESIGN TABLE
# ============================================================

design_rows = []

for design in COUNTRY_DESIGNS:

    for target_pathogen in design["target_pathogens"]:

        for horizon in FORECAST_HORIZONS:

            design_rows.append({
                "country": design["country"],
                "analysis_group": design["analysis_group"],
                "target_pathogen": target_pathogen,
                "forecast_horizon_weeks": horizon,
                "required_pathogens": "+".join(
                    design["required_pathogens"]
                ),
                "train_start": design["train_start"],
                "train_end": design["train_end"],
                "validation_start": design["validation_start"],
                "validation_end": design["validation_end"],
                "test_start": design["test_start"],
                "test_end": design["test_end"],
                "split_basis": "target_date",
                "note": design["note"],
            })

design_df = pd.DataFrame(design_rows)

design_df.to_csv(
    DESIGN_FILE,
    index=False
)


# ============================================================
# 7. CHECK SURVEILLANCE COVERAGE
# ============================================================

coverage_rows = []

for design in COUNTRY_DESIGNS:

    country = design["country"]

    country_df = df[
        df["country"] == country
    ].copy()

    required_columns = [
        PATHOGEN_COLUMNS[pathogen]
        for pathogen in design["required_pathogens"]
    ]

    periods = {
        "train": (
            design["train_start"],
            design["train_end"],
        ),
        "validation": (
            design["validation_start"],
            design["validation_end"],
        ),
        "test": (
            design["test_start"],
            design["test_end"],
        ),
    }

    for split, (start, end) in periods.items():

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        period_df = country_df[
            (country_df["week_start"] >= start)
            & (country_df["week_start"] <= end)
        ].copy()

        if period_df.empty:
            complete_weeks = 0
        else:
            complete_weeks = (
                period_df[required_columns]
                .notna()
                .all(axis=1)
                .sum()
            )

        coverage_rows.append({
            "country": country,
            "analysis_group": design["analysis_group"],
            "split": split,
            "start": start.date(),
            "end": end.date(),
            "rows_in_period": len(period_df),
            "complete_required_pathogen_weeks": int(
                complete_weeks
            ),
            "required_pathogens": "+".join(
                design["required_pathogens"]
            ),
        })

coverage_df = pd.DataFrame(coverage_rows)

coverage_df.to_csv(
    COVERAGE_FILE,
    index=False
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\nForecast horizons:")
print(FORECAST_HORIZONS)

print("\nPrimary regional cohort:")
print(" - Indonesia")
print(" - Malaysia")
print(" - Brunei Darussalam")

print("\nSecondary regional analysis:")
print(" - Philippines")

print("\nExploratory regional analysis:")
print(" - Thailand")

print("\nSingapore case study:")
print(" - Influenza")
print(" - SARS-CoV-2")

print("\n" + "=" * 80)
print("SURVEILLANCE COVERAGE BY FORECAST SPLIT")
print("=" * 80)

print(
    coverage_df.to_string(
        index=False
    )
)

print("\nSaved forecast design:")
print(DESIGN_FILE)

print("\nSaved coverage check:")
print(COVERAGE_FILE)

print("\nIMPORTANT:")
print(
    "Individual forecasts will later be assigned to "
    "train/validation/test according to TARGET DATE."
)

print(
    "The final test period must not be used for "
    "feature selection, model tuning, or hyperparameter tuning."
)

print("\nFORECAST DESIGN COMPLETE")