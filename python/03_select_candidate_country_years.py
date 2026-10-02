from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "candidate_country_quality.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Keep only complete years
df = df[df["ISO_YEAR"] <= 2025].copy()

df["good_influenza_year"] = (
    df["influenza_reporting_percent"] >= 80
)

df["good_rsv_year"] = (
    df["rsv_reporting_percent"] >= 80
)

df["good_joint_year"] = (
    df["good_influenza_year"]
    & df["good_rsv_year"]
)

output_file = (
    OUTPUT_DIR
    / "candidate_country_year_eligibility.csv"
)

df.to_csv(output_file, index=False)

joint = df[df["good_joint_year"]].copy()

summary = (
    joint
    .groupby("COUNTRY_AREA_TERRITORY")
    .agg(
        first_joint_year=("ISO_YEAR", "min"),
        last_joint_year=("ISO_YEAR", "max"),
        number_joint_years=("ISO_YEAR", "nunique"),
    )
    .reset_index()
    .sort_values(
        "number_joint_years",
        ascending=False
    )
)

summary_file = (
    OUTPUT_DIR
    / "joint_influenza_rsv_country_summary.csv"
)

summary.to_csv(summary_file, index=False)

print("\nJoint influenza + RSV eligible years:\n")

for country, group in joint.groupby(
    "COUNTRY_AREA_TERRITORY"
):
    years = sorted(
        group["ISO_YEAR"]
        .astype(int)
        .tolist()
    )
    print(f"{country}: {years}")

print("\nCountry summary:\n")
print(summary.to_string(index=False))

print("\nSaved:")
print(output_file)
print(summary_file)