"""
Step 28: Freeze manuscript result evidence.

This script does NOT refit, tune, or alter any forecasting model.
It collects the final analysis tables into one reproducible
manuscript evidence document.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "tables"
MANUSCRIPT_DIR = ROOT / "manuscript"

MANUSCRIPT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUTS = {
    "Overall model summary":
        TABLE_DIR / "overall_model_summary.csv",

    "Overall model ranking":
        TABLE_DIR / "overall_model_ranking.csv",

    "Cross-pathogen information summary":
        TABLE_DIR / "cross_pathogen_information_summary.csv",

    "Robustness synthesis":
        TABLE_DIR / "robustness_overall_synthesis.csv",

    "Uncertainty calibration":
        TABLE_DIR / "robustness_uncertainty_summary.csv",

    "Singapore case-study summary":
        TABLE_DIR / "singapore_case_study_summary.csv",

    "Seasonality peak summary":
        TABLE_DIR / "final_seasonality_peak_summary.csv",

    "Grouped SHAP summary":
        TABLE_DIR / "final_shap_group_summary.csv",
}


OUTPUT_MD = (
    MANUSCRIPT_DIR
    / "manuscript_results_evidence.md"
)

OUTPUT_INVENTORY = (
    TABLE_DIR
    / "manuscript_results_inventory.csv"
)


print("\n" + "=" * 95)
print("STEP 28: MANUSCRIPT RESULTS EVIDENCE")
print("=" * 95)


# ------------------------------------------------------------
# CHECK INPUT FILES
# ------------------------------------------------------------

missing = [
    path
    for path in INPUTS.values()
    if not path.exists()
]


if missing:

    print("\nMissing required files:")

    for path in missing:
        print("-", path)

    raise FileNotFoundError(
        "One or more final manuscript result tables are missing."
    )


# ------------------------------------------------------------
# LOAD TABLES
# ------------------------------------------------------------

tables = {}

inventory_rows = []


for label, path in INPUTS.items():

    df = pd.read_csv(path)

    tables[label] = df

    inventory_rows.append({
        "result_section": label,
        "file":
            str(
                path.relative_to(ROOT)
            ),
        "rows":
            len(df),
        "columns":
            len(df.columns),
        "column_names":
            " | ".join(
                df.columns.astype(str)
            ),
    })


inventory = pd.DataFrame(
    inventory_rows
)


inventory.to_csv(
    OUTPUT_INVENTORY,
    index=False,
)


# ------------------------------------------------------------
# BUILD HUMAN-READABLE EVIDENCE DOCUMENT
# ------------------------------------------------------------

lines = []

lines.append(
    "# Respiratory Virus Forecasting Project"
)

lines.append(
    "## Frozen manuscript result evidence"
)

lines.append("")

lines.append(
    "This document contains the final numerical tables used "
    "for manuscript preparation."
)

lines.append(
    "No model fitting, tuning, or test-set optimization was "
    "performed in Step 28."
)

lines.append("")


for label, df in tables.items():

    lines.append(
        f"## {label}"
    )

    lines.append("")

    lines.append(
        f"Rows: {len(df)}; "
        f"Columns: {len(df.columns)}"
    )

    lines.append("")

    lines.append("```text")

    lines.append(
        df.to_string(
            index=False
        )
    )

    lines.append("```")

    lines.append("")


OUTPUT_MD.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ------------------------------------------------------------
# TERMINAL SUMMARY
# ------------------------------------------------------------

print("\nMANUSCRIPT INPUT TABLES\n")


for label, df in tables.items():

    print("-" * 80)

    print(label)

    print(
        "Rows:",
        len(df),
        "| Columns:",
        len(df.columns),
    )

    print(
        "Columns:",
        df.columns.tolist(),
    )


print("\nSaved evidence document:")
print(OUTPUT_MD)


print("\nSaved result inventory:")
print(OUTPUT_INVENTORY)


# ------------------------------------------------------------
# FINAL AUDIT
# ------------------------------------------------------------

audit_pass = (
    OUTPUT_MD.exists()
    and OUTPUT_MD.stat().st_size > 1000
    and OUTPUT_INVENTORY.exists()
    and len(inventory) == len(INPUTS)
)


if audit_pass:

    print(
        "\nMANUSCRIPT RESULTS EVIDENCE AUDIT: PASS"
    )

else:

    print(
        "\nMANUSCRIPT RESULTS EVIDENCE AUDIT: REVIEW REQUIRED"
    )


print(
    "\nImportant:"
)

print(
    "- No forecasting model was retrained."
)

print(
    "- Existing final result tables were not modified."
)

print(
    "- TEST results remain evaluation-only."
)

print(
    "- These frozen values should be used when writing "
    "the manuscript."
)


print(
    "\nSTEP 28 COMPLETE"
)