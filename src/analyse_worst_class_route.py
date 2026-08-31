#!/usr/bin/env python3
"""Reanalyse an existing Structure-only Adaptive+ run for the final thesis route.

This script does NOT rerun the 50 x 10 simulation.  It reads the existing
``structure_only_raw.csv`` and reorganises the analysis around:

1. primary validity outcome: worst-class clean coverage;
2. efficiency outcome: mean prediction-set size;
3. primary diagnostic: G_Aplus = max_k g_k;
4. marginal clean coverage: retained only as an appendix transparency check.

Expected usage
--------------

    python src/analyse_worst_class_route.py \
    --raw results/confirmation/structure_only_raw.csv \
    --output-dir results/confirmation/worst_class_results

Outputs
-------

``main_results_summary.csv``
    Cluster-bootstrap estimates for worst-class coverage and set size.

``primary_diagnostic_validation.csv``
    Scenario-grouped cross-validation comparing d_T, d_M, d_V and G_Aplus
    as predictors of the worst-class coverage difference.

``appendix_marginal_coverage.csv``
    Marginal coverage retained outside the main analysis.

``figure_primary_worst_class_coverage.png``
    Main validity figure with the nominal target marked.

``figure_primary_diagnostic.png``
    d_M versus G_Aplus as explanations of worst-class coverage loss.

``figure_efficiency_set_size.png``
    Paired set-size difference relative to Oracle Adaptive+.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


PRIMARY_REQUIRED_COLUMNS = {
    "scenario_id",
    "repetition_id",
    "structure_lambda",
    "worst_class_coverage",
    "oracle_worst_class_coverage",
    "delta_worst_class_coverage",
    "mean_set_size",
    "oracle_mean_set_size",
    "delta_set_size",
    "d_T_fro",
    "d_M_fro",
    "G_Aplus",
}

APPENDIX_COLUMNS = {
    "coverage",
    "oracle_coverage",
    "delta_coverage",
}


def validate_raw_results(raw: pd.DataFrame) -> None:
    """Check that the file is complete and compatible with this analysis."""
    missing = sorted(PRIMARY_REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.empty:
        raise ValueError("The raw results file is empty.")
    if raw[list(PRIMARY_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Primary analysis columns contain missing values.")
    if raw.duplicated(["scenario_id", "repetition_id", "structure_lambda"]).any():
        raise ValueError("Duplicate scenario/repetition/lambda rows detected.")
    lambdas = np.sort(raw["structure_lambda"].unique())
    if not np.any(np.isclose(lambdas, 0.0)):
        raise ValueError("The results must contain lambda=0 as the Oracle check.")
    zero = raw[np.isclose(raw["structure_lambda"], 0.0)]
    paired_columns = ["delta_worst_class_coverage", "delta_set_size"]
    max_zero_error = float(zero[paired_columns].abs().to_numpy().max())
    if max_zero_error > 1e-12:
        raise ValueError(
            "The lambda=0 correct-specification check failed: "
            f"maximum paired difference is {max_zero_error:.3g}."
        )
    if not APPENDIX_COLUMNS.issubset(raw.columns):
        print(
            "Warning: marginal coverage columns are absent; the appendix table "
            "will not be created.",
            file=sys.stderr,
        )


def cluster_bootstrap_mean(
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap scenario-level values, preserving within-scenario dependence."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    if len(values) == 1 or draws < 2:
        return mean, math.nan, math.nan
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return (
        mean,
        float(np.quantile(bootstrap_means, tail)),
        float(np.quantile(bootstrap_means, 1.0 - tail)),
    )


def add_bootstrap_triplet(
    row: dict[str, float],
    name: str,
    values: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> None:
    mean, low, high = cluster_bootstrap_mean(values, draws, rng)
    row[f"{name}_mean"] = mean
    row[f"{name}_ci_low"] = low
    row[f"{name}_ci_high"] = high


def build_main_summary(
    raw: pd.DataFrame,
    nominal_coverage: float,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Summarise primary validity and efficiency outcomes by lambda."""
    work = raw.copy()
    work["worst_class_failure"] = (
        work["worst_class_coverage"] < nominal_coverage
    ).astype(float)
    metrics = [
        "worst_class_coverage",
        "delta_worst_class_coverage",
        "worst_class_failure",
        "mean_set_size",
        "delta_set_size",
        "G_Aplus",
        "d_T_fro",
        "d_M_fro",
    ]
    if "d_V_fro" in work.columns:
        metrics.append("d_V_fro")

    # Repetitions share the same matrix scenario, so average repetitions first
    # and bootstrap the 50 independent scenarios rather than 500 raw rows.
    scenario_means = (
        work.groupby(["structure_lambda", "scenario_id"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for lam, group in scenario_means.groupby("structure_lambda", sort=True):
        row: dict[str, float] = {
            "structure_lambda": float(lam),
            "n_scenarios": float(group["scenario_id"].nunique()),
            "nominal_coverage": float(nominal_coverage),
        }
        for metric in metrics:
            add_bootstrap_triplet(
                row,
                metric,
                group[metric].to_numpy(dtype=float),
                bootstrap_draws,
                rng,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_marginal_appendix(
    raw: pd.DataFrame,
    nominal_coverage: float,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame | None:
    """Keep marginal coverage available without making it the main outcome."""
    if not APPENDIX_COLUMNS.issubset(raw.columns):
        return None
    work = raw.copy()
    work["marginal_failure"] = (work["coverage"] < nominal_coverage).astype(float)
    metrics = ["coverage", "delta_coverage", "marginal_failure"]
    scenario_means = (
        work.groupby(["structure_lambda", "scenario_id"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for lam, group in scenario_means.groupby("structure_lambda", sort=True):
        row: dict[str, float] = {
            "structure_lambda": float(lam),
            "n_scenarios": float(group["scenario_id"].nunique()),
            "nominal_coverage": float(nominal_coverage),
        }
        for metric in metrics:
            add_bootstrap_triplet(
                row,
                metric,
                group[metric].to_numpy(dtype=float),
                bootstrap_draws,
                rng,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_cv_diagnostic(
    data: pd.DataFrame,
    predictor: str,
    outcome: str,
) -> dict[str, float | str]:
    """One-predictor linear validation with scenarios kept in the same fold."""
    columns = [predictor, outcome, "scenario_id"]
    subset = (
        data[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    groups = subset["scenario_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two independent scenarios are needed for CV.")
    X = subset[[predictor]].to_numpy(dtype=float)
    y = subset[outcome].to_numpy(dtype=float)
    predictions = np.full(len(y), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    rho = spearmanr(X[:, 0], y, nan_policy="omit").statistic
    return {
        "predictor": predictor,
        "outcome": outcome,
        "spearman_rho": float(rho),
        "cv_R2": float(r2_score(y, predictions)),
        "cv_MAE": float(mean_absolute_error(y, predictions)),
        "n_rows": float(len(subset)),
        "n_scenarios": float(len(unique_groups)),
    }


def build_primary_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate diagnostics only against the selected primary coverage outcome."""
    data = raw[raw["structure_lambda"] > 0.0].copy()
    predictors = ["d_T_fro", "d_M_fro"]
    if "d_V_fro" in data.columns:
        predictors.append("d_V_fro")
    predictors.append("G_Aplus")
    rows = [
        grouped_cv_diagnostic(
            data,
            predictor,
            "delta_worst_class_coverage",
        )
        for predictor in predictors
    ]
    return pd.DataFrame(rows).sort_values("cv_R2", ascending=False)


def prepare_matplotlib(output_dir: Path) -> None:
    cache = output_dir / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def plot_primary_coverage(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["structure_lambda"].to_numpy(dtype=float)
    y = summary["worst_class_coverage_mean"].to_numpy(dtype=float)
    low = summary["worst_class_coverage_ci_low"].to_numpy(dtype=float)
    high = summary["worst_class_coverage_ci_high"].to_numpy(dtype=float)
    nominal = float(summary["nominal_coverage"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(
        nominal,
        color="#555555",
        linewidth=1.4,
        linestyle="--",
        label=f"Nominal target = {nominal:.2f}",
    )
    ax.plot(x, y, marker="o", markersize=7, linewidth=2.4, color="#2457A6")
    ax.fill_between(x, low, high, color="#2457A6", alpha=0.18)
    ax.set(
        xlabel="Structure misspecification strength (lambda)",
        ylabel="Worst-class clean coverage",
        title="T-structure misspecification and worst-class coverage",
    )
    ax.set_xticks(x)
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_primary_worst_class_coverage.png", dpi=240)
    plt.close(fig)


def plot_primary_diagnostic(raw: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = raw[raw["structure_lambda"] > 0.0]
    outcome = data["delta_worst_class_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    axes[0].scatter(
        data["d_M_fro"],
        outcome,
        s=17,
        alpha=0.42,
        color="#3A88C3",
        edgecolors="none",
    )
    axes[0].set(
        xlabel="d_M (Frobenius)",
        ylabel="Worst-class coverage difference\n(misspecified - Oracle)",
        title="Unsigned matrix distance",
    )
    axes[1].scatter(
        data["G_Aplus"],
        outcome,
        s=17,
        alpha=0.42,
        color="#C55A42",
        edgecolors="none",
    )
    axes[1].set(
        xlabel="G_Aplus = max classwise boundary distortion",
        title="Algorithm-aware worst-class diagnostic",
    )
    for ax in axes:
        ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--")
        ax.grid(alpha=0.14)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_primary_diagnostic.png", dpi=240)
    plt.close(fig)


def plot_efficiency(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["structure_lambda"].to_numpy(dtype=float)
    y = summary["delta_set_size_mean"].to_numpy(dtype=float)
    low = summary["delta_set_size_ci_low"].to_numpy(dtype=float)
    high = summary["delta_set_size_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(0.0, color="#555555", linewidth=1.3, linestyle="--")
    ax.plot(x, y, marker="o", markersize=7, linewidth=2.4, color="#2F7D5B")
    ax.fill_between(x, low, high, color="#2F7D5B", alpha=0.18)
    ax.set(
        xlabel="Structure misspecification strength (lambda)",
        ylabel="Mean set-size difference (misspecified - Oracle)",
        title="Efficiency cost of T-structure misspecification",
    )
    ax.set_xticks(x)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_efficiency_set_size.png", dpi=240)
    plt.close(fig)


def write_run_metadata(
    raw: pd.DataFrame,
    output_dir: Path,
    raw_path: Path,
    nominal_coverage: float,
    bootstrap_draws: int,
    seed: int,
) -> None:
    metadata = {
        "source_file": os.path.relpath(
    raw_path.resolve(),
    start=output_dir.resolve(),
),
        "n_rows": int(len(raw)),
        "n_scenarios": int(raw["scenario_id"].nunique()),
        "n_repetitions": int(raw["repetition_id"].nunique()),
        "lambda_grid": [float(x) for x in sorted(raw["structure_lambda"].unique())],
        "primary_validity_outcome": "worst_class_coverage",
        "efficiency_outcome": "mean_set_size",
        "primary_diagnostic": "G_Aplus",
        "appendix_outcome": "marginal coverage",
        "nominal_coverage": float(nominal_coverage),
        "bootstrap_unit": "scenario_id",
        "bootstrap_draws": int(bootstrap_draws),
        "analysis_seed": int(seed),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_analysis(
    raw_path: Path,
    output_dir: Path,
    alpha: float,
    bootstrap_draws: int,
    seed: int,
) -> None:
    raw = pd.read_csv(raw_path)
    validate_raw_results(raw)
    nominal_coverage = 1.0 - alpha
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_matplotlib(output_dir)

    main_summary = build_main_summary(
        raw,
        nominal_coverage,
        bootstrap_draws,
        seed,
    )
    diagnostics = build_primary_diagnostics(raw)
    marginal_appendix = build_marginal_appendix(
        raw,
        nominal_coverage,
        bootstrap_draws,
        seed + 1,
    )

    main_summary.to_csv(output_dir / "main_results_summary.csv", index=False)
    diagnostics.to_csv(
        output_dir / "primary_diagnostic_validation.csv",
        index=False,
    )
    if marginal_appendix is not None:
        marginal_appendix.to_csv(
            output_dir / "appendix_marginal_coverage.csv",
            index=False,
        )

    plot_primary_coverage(main_summary, output_dir)
    plot_primary_diagnostic(raw, output_dir)
    plot_efficiency(main_summary, output_dir)
    write_run_metadata(
        raw,
        output_dir,
        raw_path,
        nominal_coverage,
        bootstrap_draws,
        seed,
    )

    lambda_max = main_summary.loc[main_summary["structure_lambda"].idxmax()]
    print(f"Analysed {len(raw):,} rows from {raw_path}")
    print(
        "At maximum lambda: worst-class coverage "
        f"{lambda_max['worst_class_coverage_mean']:.4f}; "
        "paired difference "
        f"{lambda_max['delta_worst_class_coverage_mean']:.4f}; "
        "set-size difference "
        f"{lambda_max['delta_set_size_mean']:.4f}."
    )
    print(f"Outputs written to {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Path to structure_only_raw.csv from the completed simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("worst_class_results"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Miscoverage level; nominal coverage is 1-alpha.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1).")
    if args.bootstrap_draws < 2:
        parser.error("--bootstrap-draws must be at least 2.")
    if not args.raw.exists():
        parser.error(f"Raw results file does not exist: {args.raw}")
    run_analysis(
        args.raw,
        args.output_dir,
        args.alpha,
        args.bootstrap_draws,
        args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
