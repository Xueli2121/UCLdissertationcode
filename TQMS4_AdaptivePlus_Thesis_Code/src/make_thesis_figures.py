#!/usr/bin/env python3

"""
Create the final thesis figures from the confirmation-run outputs.

Expected project structure
--------------------------
project_root/
├── src/
│   └── make_thesis_figures.py
└── results/
    └── confirmation/
    ├── structure_only_raw.csv
    └── worst_class_results/
        ├── main_results_summary.csv
        ├── appendix_marginal_coverage.csv
        └── primary_diagnostic_validation.csv

Run from the project root:

python src/make_thesis_figures.py \
    --results-dir results/confirmation \
    --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Plotting style
# =========================================================

BLUE = "#2A62AD"
BLUE_LIGHT = "#AFC6E5"

GREEN = "#2A7F62"
GREEN_LIGHT = "#B7D5CA"

ORANGE = "#C75B45"
ORANGE_LIGHT = "#E4B1A5"

GREY = "#5C5C5C"
GRID = "#D9D9D9"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.7,
        "figure.dpi": 150,
        "savefig.dpi": 400,
    }
)


# =========================================================
# Helper functions
# =========================================================

def require_columns(
    data: pd.DataFrame,
    required: list[str],
    file_name: str,
) -> None:
    """Check that all required columns are present."""

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}"
        )


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    file_stem: str,
) -> None:
    """Save both vector PDF and high-resolution PNG versions."""

    output_dir.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_dir / f"{file_stem}.pdf",
        bbox_inches="tight",
    )

    figure.savefig(
        output_dir / f"{file_stem}.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_summary_line(
    axis: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    colour: str,
    fill_colour: str,
    label: str | None = None,
) -> None:
    """Plot a mean line with a confidence band."""

    axis.fill_between(
        x,
        lower,
        upper,
        color=fill_colour,
        alpha=0.45,
        linewidth=0,
    )

    axis.plot(
        x,
        mean,
        color=colour,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        label=label,
        zorder=3,
    )


# =========================================================
# Figure 3.1: validity
# =========================================================

def make_validity_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:

    required = [
        "structure_lambda",
        "worst_class_coverage_mean",
        "worst_class_coverage_ci_low",
        "worst_class_coverage_ci_high",
        "delta_worst_class_coverage_mean",
        "delta_worst_class_coverage_ci_low",
        "delta_worst_class_coverage_ci_high",
    ]

    require_columns(
        summary,
        required,
        "main_results_summary.csv",
    )

    summary = summary.sort_values("structure_lambda")

    lam = summary["structure_lambda"].to_numpy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.25),
        sharex=True,
    )

    # Panel (a): absolute worst-class coverage
    plot_summary_line(
        axes[0],
        lam,
        summary["worst_class_coverage_mean"].to_numpy(),
        summary["worst_class_coverage_ci_low"].to_numpy(),
        summary["worst_class_coverage_ci_high"].to_numpy(),
        BLUE,
        BLUE_LIGHT,
        label="Misspecified Adaptive+",
    )

    axes[0].axhline(
        0.90,
        color=GREY,
        linestyle="--",
        linewidth=1.3,
        label="Nominal target",
    )

    axes[0].set_title("(a) Absolute validity")
    axes[0].set_ylabel("Worst-class clean coverage")
    axes[0].set_xlabel(
        r"Relative structural path position, $\lambda$"
    )
    axes[0].set_xticks(lam)
    axes[0].legend(frameon=False, loc="lower left")

    # Panel (b): paired difference from Oracle
    plot_summary_line(
        axes[1],
        lam,
        summary["delta_worst_class_coverage_mean"].to_numpy(),
        summary["delta_worst_class_coverage_ci_low"].to_numpy(),
        summary["delta_worst_class_coverage_ci_high"].to_numpy(),
        ORANGE,
        ORANGE_LIGHT,
    )

    axes[1].axhline(
        0,
        color=GREY,
        linestyle="--",
        linewidth=1.3,
    )

    axes[1].set_title("(b) Paired effect")
    axes[1].set_ylabel(
        r"Paired coverage difference, $\Delta C_{\min}$"
    )
    axes[1].set_xlabel(
        r"Relative structural path position, $\lambda$"
    )
    axes[1].set_xticks(lam)

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_validity_two_panel",
    )


# =========================================================
# Figure 3.2: efficiency
# =========================================================

def make_efficiency_figure(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:

    required = [
        "structure_lambda",
        "delta_set_size_mean",
        "delta_set_size_ci_low",
        "delta_set_size_ci_high",
    ]

    require_columns(
        summary,
        required,
        "main_results_summary.csv",
    )

    summary = summary.sort_values("structure_lambda")

    lam = summary["structure_lambda"].to_numpy()

    figure, axis = plt.subplots(
        figsize=(6.4, 3.7),
    )

    plot_summary_line(
        axis,
        lam,
        summary["delta_set_size_mean"].to_numpy(),
        summary["delta_set_size_ci_low"].to_numpy(),
        summary["delta_set_size_ci_high"].to_numpy(),
        GREEN,
        GREEN_LIGHT,
    )

    axis.axhline(
        0,
        color=GREY,
        linestyle="--",
        linewidth=1.3,
    )

    axis.set_xlabel(
        r"Relative structural path position, $\lambda$"
    )

    axis.set_ylabel(
        r"Paired mean set-size difference, $\Delta S$"
    )

    axis.set_xticks(lam)

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_efficiency_paired",
    )


# =========================================================
# Figure 3.3: diagnostic comparison
# =========================================================

def make_diagnostic_figure(
    raw: pd.DataFrame,
    diagnostic_results: pd.DataFrame,
    output_dir: Path,
) -> None:

    required_raw = [
        "structure_lambda",
        "delta_worst_class_coverage",
        "d_M_fro",
        "G_Aplus",
    ]

    require_columns(
        raw,
        required_raw,
        "structure_only_raw.csv",
    )

    required_diagnostics = [
        "predictor",
        "cv_R2",
        "cv_MAE",
    ]

    require_columns(
        diagnostic_results,
        required_diagnostics,
        "primary_diagnostic_validation.csv",
    )

    # Lambda = 0 is excluded because all quantities are zero
    diagnostic_data = raw.loc[
        raw["structure_lambda"] > 0
    ].copy()

    metrics = diagnostic_results.set_index("predictor")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.25),
        sharey=True,
    )

    specifications = [
        (
            "d_M_fro",
            r"$d_M$ (Frobenius)",
            "Unsigned reverse-matrix distance",
            BLUE,
        ),
        (
            "G_Aplus",
            r"$G_{A+}$",
            "Algorithm-aware signed diagnostic",
            ORANGE,
        ),
    ]

    y = diagnostic_data[
        "delta_worst_class_coverage"
    ].to_numpy()

    for axis, (
        predictor,
        x_label,
        title,
        colour,
    ) in zip(axes, specifications):

        x = diagnostic_data[predictor].to_numpy()

        axis.scatter(
            x,
            y,
            s=12,
            alpha=0.20,
            color=colour,
            edgecolors="none",
            rasterized=True,
        )

        # Descriptive full-sample linear fit
        slope, intercept = np.polyfit(x, y, deg=1)

        x_grid = np.linspace(
            np.min(x),
            np.max(x),
            200,
        )

        axis.plot(
            x_grid,
            intercept + slope * x_grid,
            color=colour,
            linewidth=2.0,
        )

        axis.axhline(
            0,
            color=GREY,
            linestyle="--",
            linewidth=1.2,
        )

        axis.set_title(title)
        axis.set_xlabel(x_label)

        if predictor in metrics.index:
            cv_r2 = metrics.loc[predictor, "cv_R2"]
            cv_mae = metrics.loc[predictor, "cv_MAE"]

            metric_text = (
                rf"CV $R^2$ = {cv_r2:.3f}"
                "\n"
                rf"CV MAE = {cv_mae:.4f}"
            )

            axis.text(
                0.96,
                0.94,
                metric_text,
                transform=axis.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": GRID,
                    "alpha": 0.90,
                },
            )

    axes[0].set_ylabel(
        r"Paired coverage difference, $\Delta C_{\min}$"
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_diagnostic_comparison",
    )


# =========================================================
# Appendix figure: marginal versus worst-class coverage
# =========================================================

def make_appendix_coverage_figure(
    summary: pd.DataFrame,
    marginal: pd.DataFrame,
    output_dir: Path,
) -> None:

    required_main = [
        "structure_lambda",
        "worst_class_coverage_mean",
        "worst_class_coverage_ci_low",
        "worst_class_coverage_ci_high",
    ]

    require_columns(
        summary,
        required_main,
        "main_results_summary.csv",
    )

    required_marginal = [
        "structure_lambda",
        "coverage_mean",
        "coverage_ci_low",
        "coverage_ci_high",
    ]

    require_columns(
        marginal,
        required_marginal,
        "appendix_marginal_coverage.csv",
    )

    summary = summary.sort_values("structure_lambda")
    marginal = marginal.sort_values("structure_lambda")

    lam = summary["structure_lambda"].to_numpy()

    figure, axis = plt.subplots(
        figsize=(6.4, 3.7),
    )

    plot_summary_line(
        axis,
        lam,
        marginal["coverage_mean"].to_numpy(),
        marginal["coverage_ci_low"].to_numpy(),
        marginal["coverage_ci_high"].to_numpy(),
        GREEN,
        GREEN_LIGHT,
        label="Marginal clean coverage",
    )

    plot_summary_line(
        axis,
        lam,
        summary["worst_class_coverage_mean"].to_numpy(),
        summary["worst_class_coverage_ci_low"].to_numpy(),
        summary["worst_class_coverage_ci_high"].to_numpy(),
        BLUE,
        BLUE_LIGHT,
        label="Worst-class clean coverage",
    )

    axis.axhline(
        0.90,
        color=GREY,
        linestyle="--",
        linewidth=1.3,
        label="Nominal target",
    )

    axis.set_xlabel(
        r"Relative structural path position, $\lambda$"
    )

    axis.set_ylabel("Clean coverage")
    axis.set_xticks(lam)

    axis.legend(
        frameon=False,
        loc="lower left",
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_appendix_marginal_vs_worst",
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Create the final thesis figures."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/confirmatory"),
        help="Confirmation-run results directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory in which the figures are saved.",
    )

    arguments = parser.parse_args()

    results_dir = arguments.results_dir.resolve()

    if arguments.output_dir is None:
        output_dir = results_dir / "thesis_figures"
    else:
        output_dir = arguments.output_dir.resolve()

    result_summary_file = (
        results_dir
        / "worst_class_results"
        / "main_results_summary.csv"
    )

    marginal_file = (
        results_dir
        / "worst_class_results"
        / "appendix_marginal_coverage.csv"
    )

    diagnostic_file = (
        results_dir
        / "worst_class_results"
        / "primary_diagnostic_validation.csv"
    )

    raw_file = (
        results_dir
        / "structure_only_raw.csv"
    )

    required_files = [
        result_summary_file,
        marginal_file,
        diagnostic_file,
        raw_file,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file was not found: {file_path}"
            )

    summary = pd.read_csv(result_summary_file)
    marginal = pd.read_csv(marginal_file)
    diagnostics = pd.read_csv(diagnostic_file)
    raw = pd.read_csv(raw_file)

    make_validity_figure(
        summary,
        output_dir,
    )

    make_efficiency_figure(
        summary,
        output_dir,
    )

    make_diagnostic_figure(
        raw,
        diagnostics,
        output_dir,
    )

    make_appendix_coverage_figure(
        summary,
        marginal,
        output_dir,
    )

    print(f"Figures written to: {output_dir}")


if __name__ == "__main__":
    main()