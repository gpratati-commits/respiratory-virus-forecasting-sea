"""
Step 27: Final manuscript figure assembly.

This script does NOT retrain or retune any forecasting model.

It:
1. Creates a regional seasonality summary figure.
2. Creates a grouped SHAP importance summary figure.
3. Reuses the frozen Step-26 forecasting figures.
4. Places the final Figure 1–6 set in one manuscript folder.
5. Places uncertainty calibration in supplementary material.
6. Creates compact summary tables and a figure manifest.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "tables"

STEP26_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "final_manuscript"
)

FINAL_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "manuscript_final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEASONALITY_FILE = (
    TABLE_DIR
    / "seasonality_weekly_profiles.csv"
)

SHAP_FILE = (
    TABLE_DIR
    / "xgboost_shap_grouped_importance.csv"
)

SEASONALITY_SUMMARY_FILE = (
    TABLE_DIR
    / "final_seasonality_peak_summary.csv"
)

SHAP_SUMMARY_FILE = (
    TABLE_DIR
    / "final_shap_group_summary.csv"
)

MANIFEST_FILE = (
    TABLE_DIR
    / "final_manuscript_figure_manifest.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def require_columns(df, columns, table_name):
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{table_name} is missing columns:\n{missing}"
        )


def save_figure(fig, stem):
    png = FINAL_DIR / f"{stem}.png"
    pdf = FINAL_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png, pdf


def copy_figure_pair(
    source_stem,
    destination_stem,
):
    source_png = (
        STEP26_DIR
        / f"{source_stem}.png"
    )

    source_pdf = (
        STEP26_DIR
        / f"{source_stem}.pdf"
    )

    require_file(source_png)
    require_file(source_pdf)

    destination_png = (
        FINAL_DIR
        / f"{destination_stem}.png"
    )

    destination_pdf = (
        FINAL_DIR
        / f"{destination_stem}.pdf"
    )

    shutil.copy2(
        source_png,
        destination_png,
    )

    shutil.copy2(
        source_pdf,
        destination_pdf,
    )

    return (
        destination_png,
        destination_pdf,
    )


def clean_pathogen(value):
    x = str(value).strip().lower()

    mapping = {
        "influenza": "Influenza",
        "rsv": "RSV",
        "sarscov2": "SARS-CoV-2",
        "sars-cov-2": "SARS-CoV-2",
        "sars_cov_2": "SARS-CoV-2",
    }

    return mapping.get(
        x,
        str(value),
    )


def clean_feature_group(value):
    mapping = {
        "own_pathogen_history":
            "Own-pathogen history",

        "other_pathogen_activity":
            "Other-pathogen activity",

        "seasonality":
            "Seasonality",
    }

    return mapping.get(
        value,
        value,
    )


print("\n" + "=" * 95)
print("STEP 27: FINAL MANUSCRIPT FIGURE ASSEMBLY")
print("=" * 95)


# ============================================================
# INPUT CHECK
# ============================================================

require_file(
    SEASONALITY_FILE
)

require_file(
    SHAP_FILE
)


seasonality = pd.read_csv(
    SEASONALITY_FILE
)

shap = pd.read_csv(
    SHAP_FILE
)


require_columns(
    seasonality,
    [
        "country",
        "pathogen",
        "ISO_WEEK",
        "mean_relative_activity",
        "median_relative_activity",
        "sd_relative_activity",
    ],
    "seasonality_weekly_profiles.csv",
)


require_columns(
    shap,
    [
        "country",
        "target_pathogen",
        "forecast_horizon_weeks",
        "feature_group",
        "mean_absolute_shap",
        "relative_importance_pct",
    ],
    "xgboost_shap_grouped_importance.csv",
)


# ============================================================
# FIGURE 1
# REGIONAL SEASONALITY
# ============================================================

PRIMARY_COUNTRIES = [
    "Brunei Darussalam",
    "Indonesia",
    "Malaysia",
]


season = seasonality[
    seasonality["country"].isin(
        PRIMARY_COUNTRIES
    )
].copy()


season[
    "pathogen_display"
] = (
    season["pathogen"]
    .map(clean_pathogen)
)


season[
    "ISO_WEEK"
] = pd.to_numeric(
    season["ISO_WEEK"],
    errors="coerce",
)


season[
    "mean_relative_activity"
] = pd.to_numeric(
    season["mean_relative_activity"],
    errors="coerce",
)


season = season.dropna(
    subset=[
        "ISO_WEEK",
        "mean_relative_activity",
    ]
)


fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10, 10),
    sharex=True,
)


for ax, country in zip(
    axes,
    PRIMARY_COUNTRIES,
):

    country_data = season[
        season["country"]
        == country
    ]

    pathogens = [
        "Influenza",
        "RSV",
        "SARS-CoV-2",
    ]

    for pathogen in pathogens:

        subset = country_data[
            country_data[
                "pathogen_display"
            ]
            == pathogen
        ].sort_values(
            "ISO_WEEK"
        )

        if subset.empty:
            continue

        ax.plot(
            subset["ISO_WEEK"],
            subset[
                "mean_relative_activity"
            ],
            linewidth=1.7,
            label=pathogen,
        )

    ax.set_title(country)

    ax.set_ylabel(
        "Mean relative activity"
    )

    ax.grid(
        alpha=0.2
    )


axes[-1].set_xlabel(
    "ISO epidemiological week"
)


handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


fig.suptitle(
    "Seasonal respiratory-virus activity in the primary regional forecasting settings",
    y=0.985,
    fontsize=14,
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=3,
    frameon=False,
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.90,
    ]
)



figure1_png, figure1_pdf = (
    save_figure(
        fig,
        "figure_01_regional_seasonality",
    )
)


# ============================================================
# SEASONALITY PEAK SUMMARY
# ============================================================

peak_rows = []


for (
    country,
    pathogen
), group in season.groupby(
    [
        "country",
        "pathogen_display",
    ]
):

    group = group.dropna(
        subset=[
            "mean_relative_activity"
        ]
    )

    if group.empty:
        continue

    peak_index = (
        group[
            "mean_relative_activity"
        ]
        .idxmax()
    )

    peak_row = group.loc[
        peak_index
    ]

    peak_rows.append({
        "country":
            country,

        "pathogen":
            pathogen,

        "peak_iso_week":
            int(
                peak_row[
                    "ISO_WEEK"
                ]
            ),

        "peak_mean_relative_activity":
            peak_row[
                "mean_relative_activity"
            ],
    })


seasonality_summary = (
    pd.DataFrame(
        peak_rows
    )
)


seasonality_summary.to_csv(
    SEASONALITY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# FIGURE 2
# OVERALL MODEL RANKING
# Reuse frozen Step-26 figure
# ============================================================

figure2_png, figure2_pdf = (
    copy_figure_pair(
        "figure_01_overall_model_ranking",
        "figure_02_overall_model_ranking",
    )
)


# ============================================================
# FIGURE 3
# CROSS-PATHOGEN EFFECT BY PATHOGEN
# ============================================================

figure3_png, figure3_pdf = (
    copy_figure_pair(
        "figure_02_cross_pathogen_effect_by_pathogen",
        "figure_03_cross_pathogen_effect_by_pathogen",
    )
)


# ============================================================
# FIGURE 4
# CROSS-PATHOGEN EFFECT BY HORIZON
# ============================================================

figure4_png, figure4_pdf = (
    copy_figure_pair(
        "figure_03_cross_pathogen_effect_by_horizon",
        "figure_04_cross_pathogen_effect_by_horizon",
    )
)


# ============================================================
# FIGURE 5
# GROUPED SHAP IMPORTANCE
# ============================================================

shap_clean = shap.copy()


shap_clean[
    "relative_importance_pct"
] = pd.to_numeric(
    shap_clean[
        "relative_importance_pct"
    ],
    errors="coerce",
)


shap_clean = shap_clean.dropna(
    subset=[
        "relative_importance_pct"
    ]
)


shap_summary = (
    shap_clean
    .groupby(
        "feature_group"
    )
    .agg(
        n=(
            "relative_importance_pct",
            "size",
        ),

        mean_relative_importance_pct=(
            "relative_importance_pct",
            "mean",
        ),

        median_relative_importance_pct=(
            "relative_importance_pct",
            "median",
        ),

        q1_relative_importance_pct=(
            "relative_importance_pct",
            lambda x:
                x.quantile(0.25),
        ),

        q3_relative_importance_pct=(
            "relative_importance_pct",
            lambda x:
                x.quantile(0.75),
        ),
    )
    .reset_index()
)


shap_summary[
    "feature_group_display"
] = (
    shap_summary[
        "feature_group"
    ]
    .map(
        clean_feature_group
    )
)


shap_summary = (
    shap_summary
    .sort_values(
        "median_relative_importance_pct",
        ascending=True,
    )
)


shap_summary.to_csv(
    SHAP_SUMMARY_FILE,
    index=False,
)


median_values = (
    shap_summary[
        "median_relative_importance_pct"
    ]
    .to_numpy()
)


q1_values = (
    shap_summary[
        "q1_relative_importance_pct"
    ]
    .to_numpy()
)


q3_values = (
    shap_summary[
        "q3_relative_importance_pct"
    ]
    .to_numpy()
)


lower_error = (
    median_values
    - q1_values
)


upper_error = (
    q3_values
    - median_values
)


y = np.arange(
    len(shap_summary)
)


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


bars = ax.barh(
    y,
    median_values,
    xerr=np.vstack(
        [
            lower_error,
            upper_error,
        ]
    ),
    capsize=4,
)


ax.set_yticks(y)


ax.set_yticklabels(
    shap_summary[
        "feature_group_display"
    ]
)


ax.set_xlabel(
    "Median relative SHAP importance (%)"
)


ax.set_title(
    "Relative contribution of predictor groups in XGBoost forecasts"
)


ax.grid(
    axis="x",
    alpha=0.2,
)


for bar, value in zip(
    bars,
    median_values,
):

    ax.text(
        value,
        bar.get_y()
        + bar.get_height() / 2,
        f"  {value:.1f}%",
        va="center",
    )


fig.tight_layout()


figure5_png, figure5_pdf = (
    save_figure(
        fig,
        "figure_05_grouped_shap_importance",
    )
)


# ============================================================
# FIGURE 6
# SINGAPORE CASE STUDY
# ============================================================

figure6_png, figure6_pdf = (
    copy_figure_pair(
        "figure_05_singapore_cross_pathogen_effect",
        "figure_06_singapore_cross_pathogen_effect",
    )
)


# ============================================================
# SUPPLEMENTARY FIGURE S1
# UNCERTAINTY CALIBRATION
# ============================================================

supp1_png, supp1_pdf = (
    copy_figure_pair(
        "figure_04_uncertainty_calibration",
        "supplementary_figure_s1_uncertainty_calibration",
    )
)


# ============================================================
# FINAL MANIFEST
# ============================================================

manifest = pd.DataFrame([
    {
        "figure":
            "Figure 1",

        "png":
            figure1_png.name,

        "pdf":
            figure1_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Regional respiratory-virus seasonality",
    },

    {
        "figure":
            "Figure 2",

        "png":
            figure2_png.name,

        "pdf":
            figure2_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Overall forecasting-model ranking",
    },

    {
        "figure":
            "Figure 3",

        "png":
            figure3_png.name,

        "pdf":
            figure3_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Cross-pathogen effect by target pathogen",
    },

    {
        "figure":
            "Figure 4",

        "png":
            figure4_png.name,

        "pdf":
            figure4_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Cross-pathogen effect by forecast horizon",
    },

    {
        "figure":
            "Figure 5",

        "png":
            figure5_png.name,

        "pdf":
            figure5_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Grouped XGBoost SHAP importance",
    },

    {
        "figure":
            "Figure 6",

        "png":
            figure6_png.name,

        "pdf":
            figure6_pdf.name,

        "role":
            "Main manuscript",

        "purpose":
            "Singapore influenza–SARS-CoV-2 case study",
    },

    {
        "figure":
            "Supplementary Figure S1",

        "png":
            supp1_png.name,

        "pdf":
            supp1_pdf.name,

        "role":
            "Supplementary",

        "purpose":
            "Probabilistic uncertainty calibration sensitivity",
    },
])


manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ============================================================
# FINAL AUDIT
# ============================================================

expected = [
    figure1_png,
    figure1_pdf,
    figure2_png,
    figure2_pdf,
    figure3_png,
    figure3_pdf,
    figure4_png,
    figure4_pdf,
    figure5_png,
    figure5_pdf,
    figure6_png,
    figure6_pdf,
    supp1_png,
    supp1_pdf,
    SEASONALITY_SUMMARY_FILE,
    SHAP_SUMMARY_FILE,
    MANIFEST_FILE,
]


exists_check = all(
    path.exists()
    for path in expected
)


nonempty_figures = all(
    path.stat().st_size > 1000
    for path in expected
    if path.suffix
    in {
        ".png",
        ".pdf",
    }
)


main_pngs = list(
    FINAL_DIR.glob(
        "figure_*.png"
    )
)


main_pdfs = list(
    FINAL_DIR.glob(
        "figure_*.pdf"
    )
)


count_check = (
    len(main_pngs) == 6
    and len(main_pdfs) == 6
)


print("\n" + "=" * 95)
print("FINAL MANUSCRIPT FIGURE QC")
print("=" * 95)

print(
    "All expected outputs exist:",
    exists_check,
)

print(
    "All figure files non-empty:",
    nonempty_figures,
)

print(
    "Six main PNG figures present:",
    len(main_pngs) == 6,
)

print(
    "Six main PDF figures present:",
    len(main_pdfs) == 6,
)


if (
    exists_check
    and nonempty_figures
    and count_check
):

    print(
        "\nFINAL MANUSCRIPT FIGURE ASSEMBLY AUDIT: PASS"
    )

else:

    print(
        "\nFINAL MANUSCRIPT FIGURE ASSEMBLY AUDIT: "
        "REVIEW REQUIRED"
    )


print("\nMain manuscript figures:")

for file in sorted(main_pngs):
    print("-", file.name)


print("\nSupplementary figure:")

print(
    "-",
    supp1_png.name,
)


print("\nSummary tables:")

print(
    "-",
    SEASONALITY_SUMMARY_FILE.name,
)

print(
    "-",
    SHAP_SUMMARY_FILE.name,
)

print(
    "-",
    MANIFEST_FILE.name,
)


print(
    "\nImportant:"
)

print(
    "- Steps 1-26 were not modified."
)

print(
    "- No forecasting model was retrained."
)

print(
    "- Figure 1 summarizes relative weekly activity."
)

print(
    "- Figure 5 summarizes SHAP associations, not causal effects."
)

print(
    "- Singapore remains a two-pathogen secondary case study."
)


print(
    "\nSTEP 27 COMPLETE"
)