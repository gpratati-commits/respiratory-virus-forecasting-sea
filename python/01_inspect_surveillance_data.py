from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "VIW_FNT.csv"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "VIW_FLU_METADATA.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CHECK THAT FILES EXIST
# ============================================================

print("\n" + "=" * 70)
print("WHO RESPIRATORY SURVEILLANCE DATA INSPECTION")
print("=" * 70)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Main WHO data file not found: {DATA_FILE}")

if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

print("\nFiles found successfully:")
print(f"Main dataset: {DATA_FILE}")
print(f"Metadata:     {METADATA_FILE}")


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading WHO surveillance dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Dataset loaded successfully.")


# ============================================================
# 4. BASIC DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print(f"\nNumber of rows:    {len(df):,}")
print(f"Number of columns: {len(df.columns):,}")


# ============================================================
# 5. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# 6. DATE / YEAR COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL COVERAGE")
print("=" * 70)

if "ISO_YEAR" in df.columns:
    print(f"\nEarliest ISO year: {df['ISO_YEAR'].min()}")
    print(f"Latest ISO year:   {df['ISO_YEAR'].max()}")

if "ISO_WEEK" in df.columns:
    print(f"Earliest week number: {df['ISO_WEEK'].min()}")
    print(f"Latest week number:   {df['ISO_WEEK'].max()}")

if "ISO_WEEKSTARTDATE" in df.columns:
    dates = pd.to_datetime(df["ISO_WEEKSTARTDATE"], errors="coerce")

    print(f"Earliest date: {dates.min()}")
    print(f"Latest date:   {dates.max()}")


# ============================================================
# 7. COUNTRIES
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY COVERAGE")
print("=" * 70)

if "COUNTRY_AREA_TERRITORY" in df.columns:

    countries = (
        df["COUNTRY_AREA_TERRITORY"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    print(f"\nNumber of countries/areas: {len(countries)}")

    print("\nFirst 30 countries/areas:")
    for country in countries[:30]:
        print(country)


# ============================================================
# 8. POTENTIAL RESPIRATORY VIRUS VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("RESPIRATORY VIRUS VARIABLES")
print("=" * 70)

keywords = [
    "RSV",
    "RESP",
    "COVID",
    "COV",
    "SARS",
    "INF_A",
    "INF_B",
    "AH1",
    "AH3",
]

virus_columns = []

for column in df.columns:

    upper_col = column.upper()

    if any(keyword in upper_col for keyword in keywords):
        virus_columns.append(column)

print("\nPotential respiratory-virus columns:")

for column in virus_columns:
    print(column)


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSINGNESS")
print("=" * 70)

missing_summary = pd.DataFrame({
    "column": df.columns,
    "missing_n": df.isna().sum().values,
    "missing_percent": (
        df.isna().mean().values * 100
    ),
})

missing_summary = missing_summary.sort_values(
    "missing_percent",
    ascending=False
)

print("\nTop 20 columns with highest missingness:")
print(
    missing_summary.head(20).to_string(
        index=False
    )
)

missing_summary.to_csv(
    OUTPUT_DIR / "who_missingness_summary.csv",
    index=False,
)


# ============================================================
# 10. POSSIBLE SOUTHEAST ASIAN COUNTRIES
# ============================================================

print("\n" + "=" * 70)
print("SOUTHEAST ASIAN COUNTRY CHECK")
print("=" * 70)

candidate_countries = [
    "Singapore",
    "Malaysia",
    "Thailand",
    "Philippines",
    "Indonesia",
    "Viet Nam",
    "Vietnam",
    "Cambodia",
    "Lao People's Democratic Republic",
    "Myanmar",
    "Brunei Darussalam",
]

if "COUNTRY_AREA_TERRITORY" in df.columns:

    available_countries = set(
        df["COUNTRY_AREA_TERRITORY"]
        .dropna()
        .astype(str)
    )

    for country in candidate_countries:

        if country in available_countries:
            print(f"FOUND:     {country}")
        else:
            print(f"NOT FOUND: {country}")


# ============================================================
# 11. COUNTRY-YEAR-WEEK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

possible_keys = [
    "COUNTRY_CODE",
    "ISO_YEAR",
    "ISO_WEEK",
]

if all(column in df.columns for column in possible_keys):

    duplicate_count = df.duplicated(
        subset=possible_keys,
        keep=False
    ).sum()

    print(
        "\nRows involved in duplicated "
        "country-year-week combinations:"
    )

    print(f"{duplicate_count:,}")

    print(
        "\nNOTE: Duplicates do not automatically mean errors."
    )

    print(
        "WHO data may contain multiple surveillance streams "
        "or source categories for the same country/week."
    )


# ============================================================
# 12. WEEKLY REPORTING COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("REPORTING COVERAGE")
print("=" * 70)

required_columns = [
    "COUNTRY_AREA_TERRITORY",
    "ISO_YEAR",
    "ISO_WEEK",
]

if all(column in df.columns for column in required_columns):

    weekly = (
        df[
            [
                "COUNTRY_AREA_TERRITORY",
                "ISO_YEAR",
                "ISO_WEEK",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    coverage = (
        weekly
        .groupby(
            [
                "COUNTRY_AREA_TERRITORY",
                "ISO_YEAR",
            ]
        )
        .size()
        .reset_index(name="reported_weeks")
    )

    coverage["expected_weeks"] = 52

    coverage["reporting_percent"] = (
        coverage["reported_weeks"]
        / coverage["expected_weeks"]
        * 100
    )

    coverage.to_csv(
        OUTPUT_DIR / "who_country_year_reporting_coverage.csv",
        index=False,
    )

    print(
        "\nCountry-year reporting table saved to:"
    )

    print(
        OUTPUT_DIR
        / "who_country_year_reporting_coverage.csv"
    )


# ============================================================
# 13. SAVE COLUMN INFORMATION
# ============================================================

column_summary = pd.DataFrame({
    "column_number": range(1, len(df.columns) + 1),
    "column_name": df.columns,
    "dtype": df.dtypes.astype(str).values,
})

column_summary.to_csv(
    OUTPUT_DIR / "who_column_summary.csv",
    index=False,
)


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print("\nCreated files:")

print(
    OUTPUT_DIR
    / "who_column_summary.csv"
)

print(
    OUTPUT_DIR
    / "who_missingness_summary.csv"
)

print(
    OUTPUT_DIR
    / "who_country_year_reporting_coverage.csv"
)

print(
    "\nNext step: evaluate surveillance completeness "
    "before selecting countries or study years."
)