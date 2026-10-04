"""
Create leakage-safe weekly forecasting features for the respiratory-virus
forecasting project.

Main principles
---------------
1. Build a complete 7-day calendar for every country before creating lags.
2. Missing surveillance observations remain missing (NaN), never zero-filled.
3. Predictor lags use only observations before the forecast origin.
4. Forecast targets are created at 1-, 2-, and 4-week horizons.
5. Train/validation/test membership is determined using TARGET DATE.
6. M2 and M3 can later be compared on an identical common evaluation set.

This script creates features only. It does NOT fit forecasting models.
"""

from pathlib import Path

import numpy as np
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

DESIGN_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "forecast_design.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = (
    PROCESSED_DIR
    / "respiratory_forecasting_features.csv"
)

FEATURE_QC_FILE = (
    TABLE_DIR
    / "forecast_feature_qc.csv"
)

CONTINUITY_QC_FILE = (
    TABLE_DIR
    / "forecast_weekly_calendar_qc.csv"
)


# ============================================================
# 2. FORECAST SETTINGS
# ============================================================

HORIZONS = [1, 2, 4]

OWN_LAGS = [1, 2, 3, 4, 8]

OTHER_PATHOGEN_LAGS = [1, 2, 4]

PATHOGEN_COLUMNS = {
    "influenza": "influenza_positive",
    "rsv": "rsv_positive",
    "sarscov2": "sarscov2_cases",
}


# ============================================================
# 3. LOAD INPUT DATA
# ============================================================

print("\n" + "=" * 85)
print("CREATE LEAKAGE-SAFE FORECASTING FEATURES")
print("=" * 85)

if not MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Master dataset not found:\n{MASTER_FILE}"
    )

if not DESIGN_FILE.exists():
    raise FileNotFoundError(
        f"Forecast design not found:\n{DESIGN_FILE}"
    )

master = pd.read_csv(MASTER_FILE)

design = pd.read_csv(DESIGN_FILE)

required_columns = [
    "country",
    "week_start",
    "influenza_positive",
    "rsv_positive",
    "sarscov2_cases",
]

missing_columns = [
    column
    for column in required_columns
    if column not in master.columns
]

if missing_columns:
    raise ValueError(
        "Master dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

master["week_start"] = pd.to_datetime(
    master["week_start"],
    errors="coerce",
)

if master["week_start"].isna().any():
    raise ValueError(
        "Some week_start values could not be converted to dates."
    )


# ============================================================
# 4. VERIFY UNIQUE COUNTRY-WEEK RECORDS
# ============================================================

duplicate_mask = master.duplicated(
    subset=["country", "week_start"],
    keep=False,
)

if duplicate_mask.any():

    duplicates = master.loc[
        duplicate_mask,
        ["country", "week_start"],
    ].sort_values(
        ["country", "week_start"]
    )

    print("\nDuplicate country-week rows found:")
    print(duplicates.to_string(index=False))

    raise ValueError(
        "Forecast features cannot safely be created until "
        "duplicate country-week records are resolved."
    )


# ============================================================
# 5. LIMIT TO COUNTRIES IN THE FROZEN FORECAST DESIGN
# ============================================================

design_countries = sorted(
    design["country"]
    .dropna()
    .unique()
)

master = master[
    master["country"].isin(design_countries)
].copy()

print("\nCountries in forecast design:")

for country in design_countries:
    print(" -", country)


# ============================================================
# 6. BUILD A COMPLETE WEEKLY CALENDAR
# ============================================================

calendar_frames = []
continuity_rows = []

for country in design_countries:

    country_data = (
        master[
            master["country"] == country
        ]
        .sort_values("week_start")
        .copy()
    )

    if country_data.empty:
        raise ValueError(
            f"No master-data records found for {country}."
        )

    observed_dates = (
        country_data["week_start"]
        .drop_duplicates()
        .sort_values()
    )

    before_gaps = (
        observed_dates
        .diff()
        .dt.days
        .dropna()
    )

    nonweekly_before = int(
        (before_gaps != 7).sum()
    )

    start_date = observed_dates.min()
    end_date = observed_dates.max()

    # All source week_start dates are expected to represent
    # the same weekly weekday. A full 7-day sequence is created
    # from the country's first observed date.
    complete_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="7D",
    )

    country_data = (
        country_data
        .set_index("week_start")
        .reindex(complete_dates)
    )

    country_data.index.name = "week_start"

    country_data["country"] = country

    country_data = (
        country_data
        .reset_index()
        .sort_values("week_start")
    )

    after_gaps = (
        country_data["week_start"]
        .diff()
        .dt.days
        .dropna()
    )

    nonweekly_after = int(
        (after_gaps != 7).sum()
    )

    inserted_weeks = (
        len(country_data)
        - len(observed_dates)
    )

    continuity_rows.append({
        "country": country,
        "observed_weeks_before": len(observed_dates),
        "non_7_day_gaps_before": nonweekly_before,
        "calendar_weeks_after": len(country_data),
        "inserted_calendar_weeks": inserted_weeks,
        "non_7_day_gaps_after": nonweekly_after,
    })

    calendar_frames.append(country_data)


weekly = pd.concat(
    calendar_frames,
    ignore_index=True,
)

weekly = weekly.sort_values(
    ["country", "week_start"]
).reset_index(drop=True)


# ============================================================
# 7. CHECK THE REBUILT WEEKLY CALENDAR
# ============================================================

continuity_qc = pd.DataFrame(
    continuity_rows
)

continuity_qc.to_csv(
    CONTINUITY_QC_FILE,
    index=False,
)

if (
    continuity_qc["non_7_day_gaps_after"] != 0
).any():
    raise ValueError(
        "Weekly calendar reconstruction failed: "
        "non-7-day gaps remain."
    )

print("\nWeekly calendar reconstruction:")
print(
    continuity_qc.to_string(
        index=False
    )
)


# ============================================================
# 8. CREATE LAGGED PATHOGEN FEATURES
# ============================================================

for pathogen, column in PATHOGEN_COLUMNS.items():

    grouped = weekly.groupby(
        "country",
        sort=False,
    )[column]

    for lag in OWN_LAGS:

        weekly[
            f"{pathogen}_lag{lag}"
        ] = grouped.shift(lag)

    # Four-week rolling mean uses ONLY previous weeks.
    weekly[
        f"{pathogen}_rolling4"
    ] = (
        weekly.groupby(
            "country",
            sort=False,
        )[column]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=4,
                min_periods=4,
            )
            .mean()
        )
    )


# ============================================================
# 9. CREATE ONE ROW PER TARGET PATHOGEN AND HORIZON
# ============================================================

forecast_rows = []

for _, design_row in design.iterrows():

    country = design_row["country"]
    target_pathogen = design_row["target_pathogen"]
    horizon = int(
        design_row["forecast_horizon_weeks"]
    )

    target_column = PATHOGEN_COLUMNS[
        target_pathogen
    ]

    country_weekly = (
        weekly[
            weekly["country"] == country
        ]
        .copy()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    # Forecast origin is the current row.
    country_weekly["forecast_origin"] = (
        country_weekly["week_start"]
    )

    # Target date is exactly h weeks after the origin.
    country_weekly["target_date"] = (
        country_weekly["forecast_origin"]
        + pd.to_timedelta(
            horizon * 7,
            unit="D",
        )
    )

    # Because calendar is now complete and exactly weekly,
    # shift(-h) corresponds exactly to t+h.
    country_weekly["target_value"] = (
        country_weekly[target_column]
        .shift(-horizon)
    )

    country_weekly["target_pathogen"] = (
        target_pathogen
    )

    country_weekly["forecast_horizon_weeks"] = (
        horizon
    )

    country_weekly["analysis_group"] = (
        design_row["analysis_group"]
    )

    country_weekly["required_pathogens"] = (
        design_row["required_pathogens"]
    )


    # ========================================================
    # 10. ASSIGN SPLIT BY TARGET DATE
    # ========================================================

    train_start = pd.Timestamp(
        design_row["train_start"]
    )
    train_end = pd.Timestamp(
        design_row["train_end"]
    )

    validation_start = pd.Timestamp(
        design_row["validation_start"]
    )
    validation_end = pd.Timestamp(
        design_row["validation_end"]
    )

    test_start = pd.Timestamp(
        design_row["test_start"]
    )
    test_end = pd.Timestamp(
        design_row["test_end"]
    )

    conditions = [
        country_weekly["target_date"].between(
            train_start,
            train_end,
            inclusive="both",
        ),
        country_weekly["target_date"].between(
            validation_start,
            validation_end,
            inclusive="both",
        ),
        country_weekly["target_date"].between(
            test_start,
            test_end,
            inclusive="both",
        ),
    ]

    choices = [
        "train",
        "validation",
        "test",
    ]

    country_weekly["split"] = np.select(
        conditions,
        choices,
        default="outside_design",
    )


    # ========================================================
    # 11. CREATE SEASONAL FEATURES
    # ========================================================

    # The target calendar date is known when the forecast is made.
    # These are calendar variables, not future disease observations.

    target_day = (
        country_weekly["target_date"]
        .dt.dayofyear
    )

    country_weekly["season_sin"] = np.sin(
        2.0
        * np.pi
        * target_day
        / 365.25
    )

    country_weekly["season_cos"] = np.cos(
        2.0
        * np.pi
        * target_day
        / 365.25
    )


    # ========================================================
    # 12. DEFINE M1 / M2 / M3 FEATURE SETS
    # ========================================================

    own_features = [
        f"{target_pathogen}_lag1",
        f"{target_pathogen}_lag2",
        f"{target_pathogen}_lag3",
        f"{target_pathogen}_lag4",
        f"{target_pathogen}_lag8",
        f"{target_pathogen}_rolling4",
    ]

    m1_features = own_features.copy()

    m2_features = (
        own_features
        + [
            "season_sin",
            "season_cos",
        ]
    )

    required_pathogens = (
        str(
            design_row["required_pathogens"]
        )
        .split("+")
    )

    other_pathogens = [
        pathogen
        for pathogen in required_pathogens
        if pathogen != target_pathogen
    ]

    cross_pathogen_features = []

    for other_pathogen in other_pathogens:

        for lag in OTHER_PATHOGEN_LAGS:

            cross_pathogen_features.append(
                f"{other_pathogen}_lag{lag}"
            )

        cross_pathogen_features.append(
            f"{other_pathogen}_rolling4"
        )

    m3_features = (
        m2_features
        + cross_pathogen_features
    )

    country_weekly["m1_features_complete"] = (
        country_weekly[m1_features]
        .notna()
        .all(axis=1)
    )

    country_weekly["m2_features_complete"] = (
        country_weekly[m2_features]
        .notna()
        .all(axis=1)
    )

    country_weekly["m3_features_complete"] = (
        country_weekly[m3_features]
        .notna()
        .all(axis=1)
    )

    country_weekly["target_observed"] = (
        country_weekly["target_value"]
        .notna()
    )

    # Common evaluation set for the central M2 vs M3 comparison.
    country_weekly[
        "common_m2_m3_eligible"
    ] = (
        country_weekly["target_observed"]
        & country_weekly["m2_features_complete"]
        & country_weekly["m3_features_complete"]
        & country_weekly["split"].isin(
            [
                "train",
                "validation",
                "test",
            ]
        )
    )

    # Store feature names for reproducibility.
    country_weekly["m1_feature_names"] = (
        "|".join(m1_features)
    )

    country_weekly["m2_feature_names"] = (
        "|".join(m2_features)
    )

    country_weekly["m3_feature_names"] = (
        "|".join(m3_features)
    )

    forecast_rows.append(
        country_weekly
    )


# ============================================================
# 13. COMBINE ALL FORECASTING ROWS
# ============================================================

features = pd.concat(
    forecast_rows,
    ignore_index=True,
)

features = features[
    features["split"].isin(
        [
            "train",
            "validation",
            "test",
        ]
    )
].copy()

features = features.sort_values(
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "forecast_origin",
    ]
).reset_index(drop=True)


# ============================================================
# 14. AUTOMATED LEAKAGE CHECKS
# ============================================================

expected_target_dates = (
    features["forecast_origin"]
    + pd.to_timedelta(
        features["forecast_horizon_weeks"] * 7,
        unit="D",
    )
)

if not (
    expected_target_dates
    == features["target_date"]
).all():
    raise ValueError(
        "Target-date leakage check failed."
    )

if not (
    features["target_date"]
    > features["forecast_origin"]
).all():
    raise ValueError(
        "Every target date must occur after its forecast origin."
    )


# ============================================================
# 15. CREATE FEATURE QC TABLE
# ============================================================

qc = (
    features
    .groupby(
        [
            "analysis_group",
            "country",
            "target_pathogen",
            "forecast_horizon_weeks",
            "split",
        ],
        dropna=False,
    )
    .agg(
        total_forecast_rows=(
            "target_value",
            "size",
        ),
        observed_targets=(
            "target_observed",
            "sum",
        ),
        m1_complete=(
            "m1_features_complete",
            "sum",
        ),
        m2_complete=(
            "m2_features_complete",
            "sum",
        ),
        m3_complete=(
            "m3_features_complete",
            "sum",
        ),
        common_m2_m3_rows=(
            "common_m2_m3_eligible",
            "sum",
        ),
    )
    .reset_index()
)

qc.to_csv(
    FEATURE_QC_FILE,
    index=False,
)


# ============================================================
# 16. SAVE FORECAST FEATURES
# ============================================================

features.to_csv(
    FEATURE_FILE,
    index=False,
)


# ============================================================
# 17. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 85)
print("FORECAST FEATURE QC")
print("=" * 85)

print(
    qc.to_string(
        index=False
    )
)

print("\nSaved forecasting features:")
print(FEATURE_FILE)

print("\nSaved feature QC:")
print(FEATURE_QC_FILE)

print("\nSaved weekly-calendar QC:")
print(CONTINUITY_QC_FILE)

print("\nImportant:")
print(
    "- Missing surveillance observations were NOT converted to zero."
)
print(
    "- Forecast split was assigned using TARGET DATE."
)
print(
    "- Predictor lags contain only earlier observations."
)
print(
    "- M2 and M3 have an explicit common evaluation-set flag."
)

print("\nFORECAST FEATURE CREATION COMPLETE")