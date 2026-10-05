"""
Step 26: Final publication-quality figures.

This script uses frozen outputs from the completed forecasting analyses.
It does NOT train, tune, or refit any model.

Figures created
---------------
1. Overall forecasting model ranking
2. Cross-pathogen information effect by pathogen
3. Cross-pathogen information effect by forecast horizon
4. Probabilistic uncertainty calibration sensitivity
5. Singapore two-pathogen cross-pathogen forecasting effect

PNG versions are useful for GitHub / presentations.
PDF versions are useful for the manuscript.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = ROOT / "outputs" / "tables"

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
    / "final_manuscript"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_RANK_FILE = (
    TABLE_DIR
    / "robustness_model_ranks.csv"
)

PATHOGEN_FILE = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_pathogen.csv"
)

HORIZON_FILE = (
    TABLE_DIR
    / "robustness_cross_pathogen_by_horizon.csv"
)

UNCERTAINTY_FILE = (
    TABLE_DIR
    / "robustness_uncertainty_summary.csv"
)

SINGAPORE_FILE = (
    TABLE_DIR
    / "singapore_cross_pathogen_effect.csv"
)

MANIFEST_FILE = (
    TABLE_DIR
    / "final_figure_manifest.csv"
)


# ============================================================
# 2. GLOBAL FIGURE SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})


# ============================================================
# 3. HELPERS
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required Step-26 input missing:\n{path}"
        )


def require_columns(
    df,
    columns,
    name,
):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing required columns:\n{missing}"
        )


def clean_method_name(value):

    replacements = {
        "Baseline: Persistence":
            "Persistence",

        "Baseline: Recent mean (4 weeks)":
            "Recent mean (4 weeks)",

        "Baseline: Seasonal naive (52 weeks)":
            "Seasonal naive (52 weeks)",

        "Negative Binomial":
            "Negative Binomial",

        "XGBoost":
            "XGBoost",

        "Ensemble":
            "Ensemble",
    }

    return replacements.get(
        value,
        value,
    )


def clean_pathogen_name(value):

    mapping = {
        "influenza":
            "Influenza",

        "rsv":
            "RSV",

        "sarscov2":
            "SARS-CoV-2",
    }

    return mapping.get(
        value,
        value,
    )


def save_figure(
    figure,
    stem,
):

    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"

    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(figure)

    return png, pdf


def grouped_bar(
    data,
    category_column,
    method_column,
    value_column,
    ylabel,
    title,
    zero_line=False,
):

    categories = list(
        dict.fromkeys(
            data[category_column].tolist()
        )
    )

    methods = list(
        dict.fromkeys(
            data[method_column].tolist()
        )
    )

    x = np.arange(
        len(categories)
    )

    width = (
        0.8
        / max(
            len(methods),
            1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    for index, method in enumerate(methods):

        subset = (
            data[
                data[method_column]
                == method
            ]
            .set_index(
                category_column
            )
        )

        values = [
            subset.loc[
                category,
                value_column,
            ]
            if category in subset.index
            else np.nan
            for category in categories
        ]

        offset = (
            index
            - (
                len(methods) - 1
            ) / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=method,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if zero_line:
        ax.axhline(
            0,
            linewidth=1,
        )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    return fig


# ============================================================
# 4. CHECK INPUTS
# ============================================================

required_files = [
    MODEL_RANK_FILE,
    PATHOGEN_FILE,
    HORIZON_FILE,
    UNCERTAINTY_FILE,
    SINGAPORE_FILE,
]

for file in required_files:
    require_file(file)


print("\n" + "=" * 95)
print("STEP 26: FINAL MANUSCRIPT FIGURES")
print("=" * 95)


# ============================================================
# 5. LOAD TABLES
# ============================================================

model_rank = pd.read_csv(
    MODEL_RANK_FILE
)

by_pathogen = pd.read_csv(
    PATHOGEN_FILE
)

by_horizon = pd.read_csv(
    HORIZON_FILE
)

uncertainty = pd.read_csv(
    UNCERTAINTY_FILE
)

singapore = pd.read_csv(
    SINGAPORE_FILE
)


# ============================================================
# 6. VALIDATE TABLE STRUCTURES
# ============================================================

require_columns(
    model_rank,
    [
        "information_set",
        "forecast_method",
        "mean_mae_rank",
    ],
    "robustness_model_ranks.csv",
)


require_columns(
    by_pathogen,
    [
        "target_pathogen",
        "forecast_method",
        "median_m3_vs_m2_mae_improvement_pct",
    ],
    "robustness_cross_pathogen_by_pathogen.csv",
)


require_columns(
    by_horizon,
    [
        "forecast_horizon_weeks",
        "forecast_method",
        "median_m3_vs_m2_mae_improvement_pct",
    ],
    "robustness_cross_pathogen_by_horizon.csv",
)


require_columns(
    uncertainty,
    [
        "model",
        "primary_mean_coverage_80",
        "calibrated_mean_coverage_80",
    ],
    "robustness_uncertainty_summary.csv",
)


require_columns(
    singapore,
    [
        "forecast_method",
        "target_pathogen",
        "forecast_horizon_weeks",
        "m3_vs_m2_mae_improvement_pct",
    ],
    "singapore_cross_pathogen_effect.csv",
)


# ============================================================
# 7. FIGURE 1:
# OVERALL MODEL RANKING
# ============================================================

ranking = model_rank[
    model_rank[
        "information_set"
    ].isin(
        [
            "M2",
            "M3",
        ]
    )
].copy()


ranking[
    "display_method"
] = (
    ranking[
        "forecast_method"
    ]
    .map(
        clean_method_name
    )
)


ranking[
    "label"
] = (
    ranking[
        "information_set"
    ].astype(str)
    + " | "
    + ranking[
        "display_method"
    ]
)


ranking = ranking.sort_values(
    "mean_mae_rank",
    ascending=True,
)


fig, ax = plt.subplots(
    figsize=(9, 6.5)
)


y = np.arange(
    len(ranking)
)


bars = ax.barh(
    y,
    ranking[
        "mean_mae_rank"
    ],
)


ax.set_yticks(y)

ax.set_yticklabels(
    ranking["label"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "Mean MAE rank (lower is better)"
)

ax.set_title(
    "Overall forecasting model ranking"
)


for bar, value in zip(
    bars,
    ranking[
        "mean_mae_rank"
    ],
):

    ax.text(
        value,
        bar.get_y()
        + bar.get_height() / 2,
        f"  {value:.2f}",
        va="center",
    )


fig.tight_layout()


figure1_png, figure1_pdf = (
    save_figure(
        fig,
        "figure_01_overall_model_ranking",
    )
)


# ============================================================
# 8. FIGURE 2:
# CROSS-PATHOGEN EFFECT BY TARGET PATHOGEN
# ============================================================

pathogen_plot = (
    by_pathogen.copy()
)


pathogen_plot[
    "target_pathogen"
] = (
    pathogen_plot[
        "target_pathogen"
    ]
    .map(
        clean_pathogen_name
    )
)


pathogen_plot[
    "forecast_method"
] = (
    pathogen_plot[
        "forecast_method"
    ]
    .map(
        clean_method_name
    )
)


pathogen_order = [
    "Influenza",
    "RSV",
    "SARS-CoV-2",
]


pathogen_plot[
    "target_pathogen"
] = pd.Categorical(
    pathogen_plot[
        "target_pathogen"
    ],
    categories=pathogen_order,
    ordered=True,
)


pathogen_plot = (
    pathogen_plot
    .sort_values(
        [
            "target_pathogen",
            "forecast_method",
        ]
    )
)


fig = grouped_bar(
    pathogen_plot,
    category_column=
        "target_pathogen",

    method_column=
        "forecast_method",

    value_column=
        "median_m3_vs_m2_mae_improvement_pct",

    ylabel=
        "Median M3 vs M2 MAE improvement (%)",

    title=
        "Effect of cross-pathogen information by target pathogen",

    zero_line=True,
)


figure2_png, figure2_pdf = (
    save_figure(
        fig,
        "figure_02_cross_pathogen_effect_by_pathogen",
    )
)


# ============================================================
# 9. FIGURE 3:
# CROSS-PATHOGEN EFFECT BY FORECAST HORIZON
# ============================================================

horizon_plot = (
    by_horizon.copy()
)


horizon_plot[
    "forecast_method"
] = (
    horizon_plot[
        "forecast_method"
    ]
    .map(
        clean_method_name
    )
)


horizon_plot[
    "horizon_label"
] = (
    horizon_plot[
        "forecast_horizon_weeks"
    ]
    .astype(int)
    .astype(str)
    + "-week"
)


horizon_order = [
    "1-week",
    "2-week",
    "4-week",
]


horizon_plot[
    "horizon_label"
] = pd.Categorical(
    horizon_plot[
        "horizon_label"
    ],
    categories=horizon_order,
    ordered=True,
)


horizon_plot = (
    horizon_plot
    .sort_values(
        [
            "horizon_label",
            "forecast_method",
        ]
    )
)


fig = grouped_bar(
    horizon_plot,
    category_column=
        "horizon_label",

    method_column=
        "forecast_method",

    value_column=
        "median_m3_vs_m2_mae_improvement_pct",

    ylabel=
        "Median M3 vs M2 MAE improvement (%)",

    title=
        "Effect of cross-pathogen information by forecast horizon",

    zero_line=True,
)


figure3_png, figure3_pdf = (
    save_figure(
        fig,
        "figure_03_cross_pathogen_effect_by_horizon",
    )
)


# ============================================================
# 10. FIGURE 4:
# UNCERTAINTY CALIBRATION
# ============================================================

uncertainty_plot = (
    uncertainty.copy()
)


uncertainty_plot[
    "model"
] = (
    uncertainty_plot[
        "model"
    ]
    .astype(str)
)


models = (
    uncertainty_plot[
        "model"
    ].tolist()
)


x = np.arange(
    len(models)
)


width = 0.34


fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)


ax.bar(
    x - width / 2,
    uncertainty_plot[
        "primary_mean_coverage_80"
    ],
    width,
    label="Primary intervals",
)


ax.bar(
    x + width / 2,
    uncertainty_plot[
        "calibrated_mean_coverage_80"
    ],
    width,
    label="Validation-calibrated intervals",
)


ax.axhline(
    0.80,
    linestyle="--",
    linewidth=1,
    label="Nominal 80% coverage",
)


ax.set_xticks(x)

ax.set_xticklabels(
    models
)


ax.set_ylim(
    0,
    1.05,
)


ax.set_ylabel(
    "Observed test-set coverage"
)

ax.set_title(
    "Sensitivity of probabilistic forecast calibration"
)

ax.legend(
    frameon=False
)


fig.tight_layout()


figure4_png, figure4_pdf = (
    save_figure(
        fig,
        "figure_04_uncertainty_calibration",
    )
)


# ============================================================
# 11. FIGURE 5:
# SINGAPORE TWO-PATHOGEN CASE STUDY
# ============================================================

singapore_plot = (
    singapore.copy()
)


singapore_plot[
    "forecast_method"
] = (
    singapore_plot[
        "forecast_method"
    ]
    .map(
        clean_method_name
    )
)


singapore_plot[
    "pathogen_label"
] = (
    singapore_plot[
        "target_pathogen"
    ]
    .map(
        clean_pathogen_name
    )
)


singapore_plot[
    "category"
] = (
    singapore_plot[
        "pathogen_label"
    ]
    + "\n"
    + singapore_plot[
        "forecast_horizon_weeks"
    ]
    .astype(int)
    .astype(str)
    + "-week"
)


singapore_order = [
    "Influenza\n1-week",
    "Influenza\n2-week",
    "Influenza\n4-week",
    "SARS-CoV-2\n1-week",
    "SARS-CoV-2\n2-week",
    "SARS-CoV-2\n4-week",
]


singapore_plot[
    "category"
] = pd.Categorical(
    singapore_plot[
        "category"
    ],
    categories=singapore_order,
    ordered=True,
)


singapore_plot = (
    singapore_plot
    .sort_values(
        [
            "category",
            "forecast_method",
        ]
    )
)


fig = grouped_bar(
    singapore_plot,
    category_column=
        "category",

    method_column=
        "forecast_method",

    value_column=
        "m3_vs_m2_mae_improvement_pct",

    ylabel=
        "M3 vs M2 MAE improvement (%)",

    title=
        "Singapore influenza–SARS-CoV-2 case study",

    zero_line=True,
)


figure5_png, figure5_pdf = (
    save_figure(
        fig,
        "figure_05_singapore_cross_pathogen_effect",
    )
)


# ============================================================
# 12. FIGURE MANIFEST
# ============================================================

manifest = pd.DataFrame([
    {
        "figure":
            "Figure 1",

        "png":
            figure1_png.name,

        "pdf":
            figure1_pdf.name,

        "purpose":
            "Overall forecasting model ranking",
    },

    {
        "figure":
            "Figure 2",

        "png":
            figure2_png.name,

        "pdf":
            figure2_pdf.name,

        "purpose":
            "Cross-pathogen effect by pathogen",
    },

    {
        "figure":
            "Figure 3",

        "png":
            figure3_png.name,

        "pdf":
            figure3_pdf.name,

        "purpose":
            "Cross-pathogen effect by forecast horizon",
    },

    {
        "figure":
            "Figure 4",

        "png":
            figure4_png.name,

        "pdf":
            figure4_pdf.name,

        "purpose":
            "Probabilistic calibration sensitivity",
    },

    {
        "figure":
            "Figure 5",

        "png":
            figure5_png.name,

        "pdf":
            figure5_pdf.name,

        "purpose":
            "Singapore two-pathogen case study",
    },
])


manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


# ============================================================
# 13. FINAL QC
# ============================================================

expected_outputs = [
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
    MANIFEST_FILE,
]


existence_check = all(
    path.exists()
    for path in expected_outputs
)


nonempty_check = all(
    path.stat().st_size > 1000
    for path in expected_outputs
    if path.suffix
    in {
        ".png",
        ".pdf",
    }
)


if (
    existence_check
    and nonempty_check
):

    print(
        "\nFINAL FIGURE GENERATION AUDIT: PASS"
    )

else:

    print(
        "\nFINAL FIGURE GENERATION AUDIT: "
        "REVIEW REQUIRED"
    )


print("\nSaved final figures:")

for file in expected_outputs:

    if file.suffix in {
        ".png",
        ".pdf",
    }:
        print(file)


print(
    "\nSaved figure manifest:"
)

print(
    MANIFEST_FILE
)


print(
    "\nImportant:"
)

print(
    "- These figures summarize frozen analysis outputs."
)

print(
    "- No forecasting models were retrained."
)

print(
    "- Positive M3-vs-M2 improvement means "
    "cross-pathogen information reduced MAE."
)

print(
    "- Lower mean MAE rank is better."
)

print(
    "- The Singapore analysis remains a "
    "two-pathogen secondary case study."
)


print(
    "\nFINAL MANUSCRIPT FIGURES COMPLETE"
)