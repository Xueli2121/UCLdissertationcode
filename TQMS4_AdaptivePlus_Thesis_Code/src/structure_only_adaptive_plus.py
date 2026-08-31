#!/usr/bin/env python3
"""Structure-only misspecification experiment for Adaptive+.

This script implements the revised one-factor route:

    T_true --(redistribute off-diagonal error flow; fixed diagonals)--> T_assumed
       |                                                              |
       +---------------- Bayes: T, rho -> M ---------------------------+
                                      |
                           Adaptive+ calibration
                                      |
                         clean coverage / set size

The experimental factor is ``structure_lambda``.  For every clean class k,
``T[k, k]`` and therefore the class-specific corruption rate are held fixed.
Only the destinations of corrupted labels are redistributed.  Oracle
Adaptive+ uses ``M_true``; misspecified Adaptive+ uses ``M_assumed``.  They
share the same data, classifier, noisy calibration labels and APS
randomisation, so all reported differences are paired.

After estimating the T -> response relationship, the script computes the
algorithm-aware signed diagnostic G_Aplus at the Oracle decision boundary.
This diagnostic is not an additional experimental factor.

Matrix convention (the same as Sesia, Wang and Tong's reference code):

    T[j, k] = P(Y_tilde=j | Y=k)        (columns sum to one)
    M[j, k] = P(Y=k | Y_tilde=j)        (rows sum to one)

Adaptive+ implementation notes
------------------------------
The calibration equations follow ``cln/classification.py`` in the authors'
official MIT-licensed repository:
https://github.com/msesia/conformal-label-noise

This file is self-contained; the repository is not required at run time.
APS conformity scores are implemented directly, and ``optimistic=True`` is
used, corresponding to the Adaptive+ decision rule in the project.

Examples
--------
Quick validation:

    python src/structure_only_adaptive_plus.py self-test

    python src/structure_only_adaptive_plus.py run \
        --quick \
        --n-jobs 1 \
        --output-dir results/quick_test

Development run:

    python src/structure_only_adaptive_plus.py run \
        --seed 2026 \
        --n-scenarios 50 \
        --n-repetitions 10 \
        --lambda-grid 0,0.25,0.5,0.75,1 \
        --true-flow-concentration 1.0 \
        --endpoint-flow-concentration 0.30 \
        --output-dir results/development

Independent confirmation run:

    python src/structure_only_adaptive_plus.py run \
        --seed 2027 \
        --n-scenarios 50 \
        --n-repetitions 10 \
        --lambda-grid 0,0.25,0.5,0.75,1 \
        --true-flow-concentration 1.0 \
        --endpoint-flow-concentration 0.30 \
        --output-dir results/confirmation

The run writes raw paired results, configuration metadata, intermediate
summaries and diagnostic outputs to the selected output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExperimentConfig:
    """All quantities fixed before running the experiment."""

    seed: int = 2026
    n_scenarios: int = 50
    n_repetitions: int = 10
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    n_classes: int = 4
    n_features: int = 20
    n_informative: int = 12
    n_redundant: int = 4
    n_train: int = 1_000
    n_cal: int = 5_000
    n_test: int = 5_000
    class_sep: float = 1.0

    alpha: float = 0.10
    noise_rates: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20)
    true_flow_concentration: float = 1.0
    endpoint_flow_concentration: float = 0.30
    max_condition_number: float = 100.0
    max_endpoint_draws: int = 10_000

    n_estimators: int = 300
    min_samples_leaf: int = 2
    n_jobs: int = -1
    noisy_training_labels: bool = False

    n_mc_c: int = 1_000
    bootstrap_draws: int = 5_000

    def validate(self) -> None:
        if self.n_classes < 3:
            raise ValueError("Structure redistribution requires at least 3 classes.")
        if len(self.noise_rates) != self.n_classes:
            raise ValueError("noise_rates must contain one value per class.")
        if any(not (0.0 < x < 1.0) for x in self.noise_rates):
            raise ValueError("Every class-specific noise rate must be in (0, 1).")
        if any(not (0.0 <= x <= 1.0) for x in self.lambda_grid):
            raise ValueError("Every structure_lambda must be in [0, 1].")
        if 0.0 not in self.lambda_grid:
            raise ValueError("lambda_grid must include 0 for the correct-specification check.")
        if self.n_informative + self.n_redundant > self.n_features:
            raise ValueError("n_informative + n_redundant exceeds n_features.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if self.n_scenarios < 1 or self.n_repetitions < 1:
            raise ValueError("n_scenarios and n_repetitions must be positive.")
        if self.n_mc_c < 1:
            raise ValueError("n_mc_c must be positive.")


@dataclass
class CalibrationResult:
    tau: FloatArray
    i_star: IntArray
    delta_hat: Dict[int, FloatArray]
    delta_const: FloatArray
    a_plus: Dict[int, FloatArray]
    decision_curve: Dict[int, FloatArray]


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def matrix_to_json(matrix: FloatArray) -> str:
    return json.dumps(np.asarray(matrix).round(12).tolist(), separators=(",", ":"))


def assert_probability_matrix_t(T: FloatArray, atol: float = 1e-10) -> None:
    """Validate T[j,k] = P(observed j | clean k)."""
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square.")
    if np.any(T < -atol) or np.any(T > 1.0 + atol):
        raise ValueError("T contains invalid probabilities.")
    if not np.allclose(T.sum(axis=0), 1.0, atol=atol):
        raise ValueError("Columns of T must sum to one.")


def assert_probability_matrix_m(M: FloatArray, atol: float = 1e-10) -> None:
    """Validate M[j,k] = P(clean k | observed j)."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square.")
    if np.any(M < -atol) or np.any(M > 1.0 + atol):
        raise ValueError("M contains invalid probabilities.")
    if not np.allclose(M.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Rows of M must sum to one.")


def draw_structure_matrix(
    noise_rates: Sequence[float],
    concentration: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a legal column-stochastic T with prescribed diagonals.

    For clean class k, the total off-diagonal mass is exactly noise_rates[k].
    A Dirichlet draw determines only where that mass flows.
    """
    eps = np.asarray(noise_rates, dtype=float)
    K = len(eps)
    T = np.zeros((K, K), dtype=float)
    for k in range(K):
        other = np.delete(np.arange(K), k)
        flow = rng.dirichlet(np.full(K - 1, concentration, dtype=float))
        T[k, k] = 1.0 - eps[k]
        T[other, k] = eps[k] * flow
    assert_probability_matrix_t(T)
    return T


def mix_error_flow(
    T_true: FloatArray,
    T_endpoint: FloatArray,
    structure_lambda: float,
) -> FloatArray:
    """Interpolate off-diagonal flow while preserving every diagonal exactly."""
    if not 0.0 <= structure_lambda <= 1.0:
        raise ValueError("structure_lambda must be in [0, 1].")
    if T_true.shape != T_endpoint.shape:
        raise ValueError("T matrices must have the same shape.")
    if not np.allclose(np.diag(T_true), np.diag(T_endpoint), atol=1e-12):
        raise ValueError("Endpoint must preserve all class-specific noise rates.")
    T = (1.0 - structure_lambda) * T_true + structure_lambda * T_endpoint
    np.fill_diagonal(T, np.diag(T_true))
    assert_probability_matrix_t(T)
    return T


def convert_t_to_m(T: FloatArray, rho: Sequence[float]) -> FloatArray:
    """Bayes conversion from forward contamination T to Adaptive+ input M."""
    assert_probability_matrix_t(T)
    rho_array = np.asarray(rho, dtype=float)
    if rho_array.shape != (T.shape[0],):
        raise ValueError("rho must contain one clean class probability per class.")
    if np.any(rho_array <= 0.0):
        raise ValueError("rho must be strictly positive.")
    rho_array = rho_array / rho_array.sum()
    rho_tilde = T @ rho_array
    if np.any(rho_tilde <= 0.0):
        raise ValueError("Observed-label marginal contains a zero-probability class.")
    M = T * rho_array[np.newaxis, :] / rho_tilde[:, np.newaxis]
    assert_probability_matrix_m(M)
    return M


def sample_noisy_labels(
    y_clean: IntArray,
    T: FloatArray,
    rng: np.random.Generator,
) -> IntArray:
    """Sample Y_tilde from the columns of T."""
    assert_probability_matrix_t(T)
    y_noisy = np.empty_like(y_clean, dtype=np.int64)
    for k in range(T.shape[0]):
        idx = np.flatnonzero(y_clean == k)
        if len(idx):
            y_noisy[idx] = rng.choice(T.shape[0], size=len(idx), p=T[:, k])
    return y_noisy


def make_split_data(
    cfg: ExperimentConfig,
    seed: int,
) -> tuple[FloatArray, IntArray, FloatArray, IntArray, FloatArray, IntArray]:
    """Generate one clean train/calibration/test split."""
    n_total = cfg.n_train + cfg.n_cal + cfg.n_test
    X, y = make_classification(
        n_samples=n_total,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=0,
        n_classes=cfg.n_classes,
        n_clusters_per_class=1,
        weights=np.full(cfg.n_classes, 1.0 / cfg.n_classes),
        class_sep=cfg.class_sep,
        flip_y=0.0,
        random_state=seed,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        train_size=cfg.n_train,
        stratify=y,
        random_state=seed + 1,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest,
        y_rest,
        train_size=cfg.n_cal,
        test_size=cfg.n_test,
        stratify=y_rest,
        random_state=seed + 2,
    )
    return (
        np.asarray(X_train, dtype=float),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_cal, dtype=float),
        np.asarray(y_cal, dtype=np.int64),
        np.asarray(X_test, dtype=float),
        np.asarray(y_test, dtype=np.int64),
    )


def aligned_predict_proba(
    classifier: RandomForestClassifier,
    X: FloatArray,
    n_classes: int,
) -> FloatArray:
    """Return probability columns 0,...,K-1 even if an estimator omits a class."""
    raw = np.asarray(classifier.predict_proba(X), dtype=float)
    out = np.zeros((len(X), n_classes), dtype=float)
    out[:, np.asarray(classifier.classes_, dtype=int)] = raw
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Classifier produced a zero probability row.")
    return out / row_sum


def aps_scores(probabilities: FloatArray, uniforms: FloatArray) -> FloatArray:
    """Randomised APS score for every observation/candidate class.

    For candidate k, the score is the probability mass of classes ranked
    before k plus U times p_k.  Smaller scores are more conforming.
    """
    p = np.asarray(probabilities, dtype=float)
    u = np.asarray(uniforms, dtype=float)
    if p.ndim != 2 or u.shape != (p.shape[0],):
        raise ValueError("Invalid shapes for probabilities or uniforms.")
    order = np.argsort(-p, axis=1, kind="stable")
    p_sorted = np.take_along_axis(p, order, axis=1)
    before_sorted = np.cumsum(p_sorted, axis=1) - p_sorted
    score_sorted = before_sorted + u[:, np.newaxis] * p_sorted
    scores = np.empty_like(score_sorted)
    np.put_along_axis(scores, order, score_sorted, axis=1)
    return np.clip(scores, 0.0, 1.0)


def estimate_c_const(
    n_k: int,
    n_mc: int,
    base_seed: int,
    cache: Dict[int, float],
) -> float:
    """Deterministic cached version of the Monte Carlo constant in Adaptive+."""
    if n_k not in cache:
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, n_k, n_mc]))
        U = np.sort(rng.uniform(size=(n_mc, n_k)), axis=1)
        ranks = np.arange(1, n_k + 1, dtype=float) / n_k
        cache[n_k] = float(np.mean(np.max(ranks[np.newaxis, :] - U, axis=1)))
    return cache[n_k]


def ecdf_on_grid(values: FloatArray, grid: FloatArray) -> FloatArray:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        raise ValueError("Each observed calibration class must be represented.")
    return np.searchsorted(sorted_values, grid, side="right") / len(sorted_values)


def calibrate_adaptive_plus(
    scores_cal: FloatArray,
    y_tilde_cal: IntArray,
    M: FloatArray,
    alpha: float,
    n_mc_c: int,
    c_seed: int,
    c_cache: Dict[int, float],
) -> CalibrationResult:
    """Label-conditional Adaptive+ calibration with a supplied M matrix."""
    assert_probability_matrix_m(M)
    K = M.shape[0]
    if scores_cal.shape != (len(y_tilde_cal), K):
        raise ValueError("scores_cal has incompatible shape.")
    counts = np.bincount(y_tilde_cal, minlength=K)
    if np.any(counts == 0):
        raise ValueError("Every observed class needs at least one calibration point.")

    V = np.linalg.inv(M)
    n_min = int(np.min(counts))
    tau = np.ones(K, dtype=float)
    i_star = np.full(K, -1, dtype=np.int64)
    delta_hat: Dict[int, FloatArray] = {}
    a_plus: Dict[int, FloatArray] = {}
    decision_curve: Dict[int, FloatArray] = {}
    delta_const = np.empty(K, dtype=float)

    for k in range(K):
        idx_k = y_tilde_cal == k
        grid = np.sort(scores_cal[idx_k, k])
        F = np.vstack(
            [ecdf_on_grid(scores_cal[y_tilde_cal == ell, k], grid) for ell in range(K)]
        )
        delta_k_hat = V[k, :] @ F - F[k, :]
        delta_hat[k] = np.asarray(delta_k_hat, dtype=float)

        n_k = int(counts[k])
        c_nk = estimate_c_const(n_k, n_mc_c, c_seed, c_cache)
        offdiag_weight = float(np.sum(np.abs(V[k, :])) - abs(V[k, k]))
        coeff = 2.0 * offdiag_weight / math.sqrt(n_min)
        concentration_term = min(
            K * math.sqrt(math.pi / 2.0),
            1.0 / math.sqrt(n_min)
            + math.sqrt((math.log(2.0 * K) + math.log(n_min)) / 2.0),
        )
        delta_const[k] = c_nk + coeff * concentration_term

        # Adaptive+ (the optimistic rule in the reference implementation).
        correction = np.maximum(
            -(1.0 - alpha) / n_k,
            delta_hat[k] - delta_const[k],
        )
        a_plus[k] = correction
        curve = (
            np.arange(1, n_k + 1, dtype=float) / n_k
            - (1.0 - alpha)
            + correction
        )
        decision_curve[k] = curve
        admissible = np.flatnonzero(curve >= 0.0)
        if len(admissible):
            i_star[k] = int(admissible[0])
            tau[k] = float(grid[i_star[k]])

    return CalibrationResult(
        tau=tau,
        i_star=i_star,
        delta_hat=delta_hat,
        delta_const=delta_const,
        a_plus=a_plus,
        decision_curve=decision_curve,
    )


def calibrate_standard(
    scores_cal: FloatArray,
    y_tilde_cal: IntArray,
    alpha: float,
) -> FloatArray:
    """Label-conditional standard split-conformal APS benchmark."""
    K = scores_cal.shape[1]
    tau = np.ones(K, dtype=float)
    for k in range(K):
        values = np.sort(scores_cal[y_tilde_cal == k, k])
        if len(values) == 0:
            continue
        rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
        rank = min(max(rank, 1), len(values))
        tau[k] = float(values[rank - 1])
    return tau


def evaluate_prediction_sets(
    scores_test: FloatArray,
    y_test: IntArray,
    tau: FloatArray,
) -> dict[str, object]:
    """Evaluate clean-label marginal and class-conditional performance."""
    prediction_sets = scores_test <= tau[np.newaxis, :]
    contains = prediction_sets[np.arange(len(y_test)), y_test]
    class_coverage = np.array(
        [np.mean(contains[y_test == k]) for k in range(scores_test.shape[1])],
        dtype=float,
    )
    class_size = np.array(
        [np.mean(prediction_sets[y_test == k].sum(axis=1)) for k in range(scores_test.shape[1])],
        dtype=float,
    )
    return {
        "coverage": float(np.mean(contains)),
        "worst_class_coverage": float(np.min(class_coverage)),
        "mean_set_size": float(np.mean(prediction_sets.sum(axis=1))),
        "class_coverage": class_coverage,
        "class_set_size": class_size,
    }


def compute_g_aplus(
    oracle: CalibrationResult,
    assumed: CalibrationResult,
) -> tuple[float, FloatArray]:
    """Signed correction error at Oracle's classwise decision boundaries.

    Positive G_Aplus means that at least one assumed correction is larger at
    the Oracle boundary.  This tends to move the selected index earlier,
    lower the threshold, shrink sets and reduce coverage.
    """
    K = len(oracle.tau)
    by_class = np.full(K, np.nan, dtype=float)
    for k in range(K):
        idx = int(oracle.i_star[k])
        if idx >= 0:
            by_class[k] = assumed.a_plus[k][idx] - oracle.a_plus[k][idx]
    if np.all(np.isnan(by_class)):
        return float("nan"), by_class
    return float(np.nanmax(by_class)), by_class


def validate_structure_pair(T_true: FloatArray, T_assumed: FloatArray) -> None:
    """Fail loudly if a supposed structure-only perturbation changes rates."""
    assert_probability_matrix_t(T_true)
    assert_probability_matrix_t(T_assumed)
    if not np.allclose(np.diag(T_true), np.diag(T_assumed), atol=1e-12):
        raise AssertionError("A structure-only perturbation changed T's diagonal.")
    true_offdiag = 1.0 - np.diag(T_true)
    assumed_offdiag = 1.0 - np.diag(T_assumed)
    if not np.allclose(true_offdiag, assumed_offdiag, atol=1e-12):
        raise AssertionError("A structure-only perturbation changed noise rates.")


def draw_valid_matrix_pair(
    cfg: ExperimentConfig,
    scenario_seed: int,
    rho: FloatArray,
) -> tuple[FloatArray, FloatArray, int]:
    """Draw T_true and one endpoint satisfying the pre-specified stability rule."""
    rng = np.random.default_rng(scenario_seed)
    T_true = draw_structure_matrix(
        cfg.noise_rates,
        cfg.true_flow_concentration,
        rng,
    )
    M_true = convert_t_to_m(T_true, rho)
    if np.linalg.cond(M_true) > cfg.max_condition_number:
        raise RuntimeError(
            "The drawn true matrix is ill-conditioned. Change the scenario seed "
            "or lower the noise rates."
        )
    for attempt in range(1, cfg.max_endpoint_draws + 1):
        endpoint = draw_structure_matrix(
            cfg.noise_rates,
            cfg.endpoint_flow_concentration,
            rng,
        )
        valid = True
        for lam in cfg.lambda_grid:
            T_assumed = mix_error_flow(T_true, endpoint, lam)
            M_assumed = convert_t_to_m(T_assumed, rho)
            if np.linalg.cond(M_assumed) > cfg.max_condition_number:
                valid = False
                break
        if valid:
            return T_true, endpoint, attempt
    raise RuntimeError("Could not draw a stable structure endpoint.")


def run_one_repetition(
    cfg: ExperimentConfig,
    scenario_id: int,
    repetition_id: int,
    T_true: FloatArray,
    T_endpoint: FloatArray,
    endpoint_draws: int,
    rho: FloatArray,
    repetition_seed: int,
) -> list[dict[str, object]]:
    """Run all lambda levels on shared data for one paired repetition."""
    seed_sequence = np.random.SeedSequence(repetition_seed)
    child = seed_sequence.spawn(8)
    data_seed = int(child[0].generate_state(1)[0])
    train_noise_rng = np.random.default_rng(child[1])
    cal_noise_rng = np.random.default_rng(child[2])
    cal_u_rng = np.random.default_rng(child[3])
    test_u_rng = np.random.default_rng(child[4])
    classifier_seed = int(child[5].generate_state(1)[0])
    c_seed = int(child[6].generate_state(1)[0])

    X_train, y_train, X_cal, y_cal, X_test, y_test = make_split_data(cfg, data_seed)
    y_train_fit = (
        sample_noisy_labels(y_train, T_true, train_noise_rng)
        if cfg.noisy_training_labels
        else y_train
    )
    y_tilde_cal = sample_noisy_labels(y_cal, T_true, cal_noise_rng)

    classifier = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=classifier_seed,
        n_jobs=cfg.n_jobs,
        class_weight=None,
    )
    classifier.fit(X_train, y_train_fit)
    p_cal = aligned_predict_proba(classifier, X_cal, cfg.n_classes)
    p_test = aligned_predict_proba(classifier, X_test, cfg.n_classes)
    scores_cal = aps_scores(p_cal, cal_u_rng.uniform(size=cfg.n_cal))
    scores_test = aps_scores(p_test, test_u_rng.uniform(size=cfg.n_test))

    M_true = convert_t_to_m(T_true, rho)
    c_cache: Dict[int, float] = {}
    oracle_cal = calibrate_adaptive_plus(
        scores_cal,
        y_tilde_cal,
        M_true,
        cfg.alpha,
        cfg.n_mc_c,
        c_seed,
        c_cache,
    )
    oracle_metrics = evaluate_prediction_sets(scores_test, y_test, oracle_cal.tau)

    tau_standard = calibrate_standard(scores_cal, y_tilde_cal, cfg.alpha)
    standard_metrics = evaluate_prediction_sets(scores_test, y_test, tau_standard)

    rows: list[dict[str, object]] = []
    for lam in cfg.lambda_grid:
        T_assumed = mix_error_flow(T_true, T_endpoint, lam)
        validate_structure_pair(T_true, T_assumed)
        M_assumed = convert_t_to_m(T_assumed, rho)
        assumed_cal = calibrate_adaptive_plus(
            scores_cal,
            y_tilde_cal,
            M_assumed,
            cfg.alpha,
            cfg.n_mc_c,
            c_seed,
            c_cache,
        )
        assumed_metrics = evaluate_prediction_sets(scores_test, y_test, assumed_cal.tau)
        g_aplus, g_by_class = compute_g_aplus(oracle_cal, assumed_cal)

        V_true = np.linalg.inv(M_true)
        V_assumed = np.linalg.inv(M_assumed)
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "repetition_id": repetition_id,
            "repetition_seed": repetition_seed,
            "structure_lambda": float(lam),
            "endpoint_draws": endpoint_draws,
            "coverage": assumed_metrics["coverage"],
            "worst_class_coverage": assumed_metrics["worst_class_coverage"],
            "mean_set_size": assumed_metrics["mean_set_size"],
            "oracle_coverage": oracle_metrics["coverage"],
            "oracle_worst_class_coverage": oracle_metrics["worst_class_coverage"],
            "oracle_mean_set_size": oracle_metrics["mean_set_size"],
            "standard_coverage": standard_metrics["coverage"],
            "standard_worst_class_coverage": standard_metrics["worst_class_coverage"],
            "standard_mean_set_size": standard_metrics["mean_set_size"],
            "delta_coverage": float(assumed_metrics["coverage"])
            - float(oracle_metrics["coverage"]),
            "delta_worst_class_coverage": float(assumed_metrics["worst_class_coverage"])
            - float(oracle_metrics["worst_class_coverage"]),
            "delta_set_size": float(assumed_metrics["mean_set_size"])
            - float(oracle_metrics["mean_set_size"]),
            "d_T_fro": float(np.linalg.norm(T_assumed - T_true, ord="fro")),
            "d_M_fro": float(np.linalg.norm(M_assumed - M_true, ord="fro")),
            "d_V_fro": float(np.linalg.norm(V_assumed - V_true, ord="fro")),
            "G_Aplus": g_aplus,
            "cond_M_true": float(np.linalg.cond(M_true)),
            "cond_M_assumed": float(np.linalg.cond(M_assumed)),
            "tau_oracle_mean": float(np.mean(oracle_cal.tau)),
            "tau_assumed_mean": float(np.mean(assumed_cal.tau)),
            "T_true": matrix_to_json(T_true),
            "T_assumed": matrix_to_json(T_assumed),
            "M_true": matrix_to_json(M_true),
            "M_assumed": matrix_to_json(M_assumed),
        }
        for k in range(cfg.n_classes):
            row[f"coverage_class_{k}"] = float(assumed_metrics["class_coverage"][k])
            row[f"oracle_coverage_class_{k}"] = float(
                oracle_metrics["class_coverage"][k]
            )
            row[f"delta_coverage_class_{k}"] = (
                float(assumed_metrics["class_coverage"][k])
                - float(oracle_metrics["class_coverage"][k])
            )
            row[f"G_Aplus_class_{k}"] = float(g_by_class[k])
            row[f"i_star_oracle_class_{k}"] = int(oracle_cal.i_star[k])
            row[f"i_star_assumed_class_{k}"] = int(assumed_cal.i_star[k])
            row[f"tau_oracle_class_{k}"] = float(oracle_cal.tau[k])
            row[f"tau_assumed_class_{k}"] = float(assumed_cal.tau[k])
        rows.append(row)
    return rows


def bootstrap_mean_ci(
    values_by_cluster: FloatArray,
    draws: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values_by_cluster, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1 or draws < 2:
        return mean, float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    boot = np.mean(values[idx], axis=1)
    tail = (1.0 - level) / 2.0
    return mean, float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))


def summarise_results(
    raw: pd.DataFrame,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Cluster bootstrap by scenario after averaging repetitions."""
    metrics = [
        "coverage",
        "worst_class_coverage",
        "mean_set_size",
        "delta_coverage",
        "delta_worst_class_coverage",
        "delta_set_size",
        "d_T_fro",
        "d_M_fro",
        "d_V_fro",
        "G_Aplus",
    ]
    scenario_means = (
        raw.groupby(["structure_lambda", "scenario_id"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for lam, group in scenario_means.groupby("structure_lambda", sort=True):
        row: dict[str, float] = {"structure_lambda": float(lam)}
        for metric in metrics:
            mean, low, high = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float),
                bootstrap_draws,
                rng,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def cross_validated_single_predictor(
    data: pd.DataFrame,
    predictor: str,
    outcome: str,
    group_column: str,
) -> dict[str, float | str]:
    subset = data[[predictor, outcome, group_column]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = subset[group_column].to_numpy()
    unique_groups = np.unique(groups)
    if len(subset) < 3 or len(unique_groups) < 2:
        return {
            "predictor": predictor,
            "outcome": outcome,
            "spearman_rho": float("nan"),
            "cv_R2": float("nan"),
            "cv_MAE": float("nan"),
            "n": float(len(subset)),
        }
    X = subset[[predictor]].to_numpy(dtype=float)
    y = subset[outcome].to_numpy(dtype=float)
    folds = GroupKFold(n_splits=min(5, len(unique_groups)))
    pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in folds.split(X, y, groups=groups):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    rho_result = spearmanr(X[:, 0], y, nan_policy="omit")
    return {
        "predictor": predictor,
        "outcome": outcome,
        "spearman_rho": float(rho_result.statistic),
        "cv_R2": float(r2_score(y, pred)),
        "cv_MAE": float(mean_absolute_error(y, pred)),
        "n": float(len(subset)),
    }


def validate_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare unsigned matrix distances with the signed G_Aplus diagnostic."""
    # Correct-specification rows are a deterministic all-zero check, not a
    # diagnostic challenge; exclude them from predictive validation.
    data = raw.loc[raw["structure_lambda"] > 0.0].copy()
    rows: list[dict[str, float | str]] = []
    for outcome in ("delta_coverage", "delta_worst_class_coverage", "delta_set_size"):
        for predictor in ("d_T_fro", "d_M_fro", "d_V_fro", "G_Aplus"):
            rows.append(
                cross_validated_single_predictor(
                    data,
                    predictor,
                    outcome,
                    "scenario_id",
                )
            )
    return pd.DataFrame(rows)


def make_plots(raw: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    matplotlib_cache = output_dir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = summary["structure_lambda"].to_numpy(dtype=float)
    y = summary["delta_coverage_mean"].to_numpy(dtype=float)
    low = summary["delta_coverage_ci_low"].to_numpy(dtype=float)
    high = summary["delta_coverage_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.plot(x, y, marker="o", linewidth=2.0, color="#2457A6")
    ax.fill_between(x, low, high, color="#2457A6", alpha=0.18)
    ax.set(
        xlabel="Structure misspecification strength (lambda)",
        ylabel="Coverage difference: misspecified - Oracle",
        title="T-structure misspecification and clean coverage",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure_T_to_coverage.png", dpi=220)
    plt.close(fig)

    data = raw.loc[raw["structure_lambda"] > 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    axes[0].scatter(data["d_M_fro"], data["delta_coverage"], s=16, alpha=0.45)
    axes[0].set(xlabel="d_M (Frobenius)", ylabel="Delta coverage", title="Unsigned distance")
    axes[1].scatter(data["G_Aplus"], data["delta_coverage"], s=16, alpha=0.45, color="#B44C3A")
    axes[1].set(xlabel="G_Aplus", title="Algorithm-aware signed diagnostic")
    for ax in axes:
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_diagnostics.png", dpi=220)
    plt.close(fig)


def run_experiment(cfg: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    cfg.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    rho = np.full(cfg.n_classes, 1.0 / cfg.n_classes, dtype=float)
    master = np.random.SeedSequence(cfg.seed)
    scenario_sequences = master.spawn(cfg.n_scenarios)
    all_rows: list[dict[str, object]] = []

    for scenario_id, scenario_sequence in enumerate(scenario_sequences):
        scenario_children = scenario_sequence.spawn(cfg.n_repetitions + 1)
        matrix_seed = int(scenario_children[0].generate_state(1)[0])
        T_true, endpoint, endpoint_draws = draw_valid_matrix_pair(
            cfg,
            matrix_seed,
            rho,
        )
        for repetition_id in range(cfg.n_repetitions):
            repetition_seed = int(scenario_children[repetition_id + 1].generate_state(1)[0])
            rows = run_one_repetition(
                cfg,
                scenario_id,
                repetition_id,
                T_true,
                endpoint,
                endpoint_draws,
                rho,
                repetition_seed,
            )
            all_rows.extend(rows)
        print(
            f"completed scenario {scenario_id + 1}/{cfg.n_scenarios}",
            flush=True,
        )

    raw = pd.DataFrame(all_rows).sort_values(
        ["scenario_id", "repetition_id", "structure_lambda"]
    )
    raw_path = output_dir / "structure_only_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarise_results(raw, cfg.bootstrap_draws, cfg.seed + 101)
    summary.to_csv(output_dir / "structure_only_summary.csv", index=False)
    diagnostics = validate_diagnostics(raw)
    diagnostics.to_csv(output_dir / "diagnostic_validation.csv", index=False)
    make_plots(raw, summary, output_dir)

    zero = raw.loc[raw["structure_lambda"] == 0.0]
    max_zero_error = float(
        zero[["delta_coverage", "delta_worst_class_coverage", "delta_set_size"]]
        .abs()
        .to_numpy()
        .max()
    )
    if max_zero_error > 1e-12:
        raise AssertionError(
            f"Correct-specification check failed: maximum paired error={max_zero_error}."
        )
    print(f"wrote {len(raw):,} rows to {raw_path}")
    print("correct-specification check passed (all paired differences at lambda=0 are zero)")
    return raw


def self_test() -> None:
    rng = np.random.default_rng(17)
    eps = (0.10, 0.20, 0.25, 0.15)
    T_true = draw_structure_matrix(eps, 1.0, rng)
    endpoint = draw_structure_matrix(eps, 0.3, rng)
    T_assumed = mix_error_flow(T_true, endpoint, 0.7)
    validate_structure_pair(T_true, T_assumed)
    assert np.allclose(np.diag(T_true), 1.0 - np.asarray(eps))
    assert np.allclose(np.diag(T_assumed), np.diag(T_true))

    rho = np.full(4, 0.25)
    M_true = convert_t_to_m(T_true, rho)
    assert np.allclose(M_true.sum(axis=1), 1.0)
    assert not np.shares_memory(T_true, M_true)

    n = 800
    p = rng.dirichlet(np.ones(4), size=n)
    scores = aps_scores(p, rng.uniform(size=n))
    y_clean = rng.integers(0, 4, size=n, dtype=np.int64)
    y_tilde = sample_noisy_labels(y_clean, T_true, rng)
    cache: Dict[int, float] = {}
    result_a = calibrate_adaptive_plus(scores, y_tilde, M_true, 0.1, 100, 33, cache)
    result_b = calibrate_adaptive_plus(scores, y_tilde, M_true, 0.1, 100, 33, cache)
    g, by_class = compute_g_aplus(result_a, result_b)
    assert np.allclose(result_a.tau, result_b.tau)
    assert abs(g) < 1e-12
    assert np.nanmax(np.abs(by_class)) < 1e-12
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test", help="Run matrix and calibration invariance checks.")

    run = sub.add_parser("run", help="Run the structure-only experiment.")
    run.add_argument("--output-dir", type=Path, default=Path("results/run_output"))
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--n-scenarios", type=int, default=50)
    run.add_argument("--n-repetitions", type=int, default=10)
    run.add_argument("--lambda-grid", type=parse_float_tuple, default=(0.0, 0.25, 0.5, 0.75, 1.0))
    run.add_argument("--noise-rates", type=parse_float_tuple, default=(0.20, 0.20, 0.20, 0.20))
    run.add_argument("--alpha", type=float, default=0.10)
    run.add_argument("--n-train", type=int, default=1_000)
    run.add_argument("--n-cal", type=int, default=5_000)
    run.add_argument("--n-test", type=int, default=5_000)
    run.add_argument("--n-estimators", type=int, default=300)
    run.add_argument("--min-samples-leaf", type=int, default=2)
    run.add_argument("--n-jobs", type=int, default=-1)
    run.add_argument("--n-mc-c", type=int, default=1_000)
    run.add_argument("--bootstrap-draws", type=int, default=5_000)
    run.add_argument("--max-condition-number", type=float, default=100.0)
    run.add_argument("--true-flow-concentration", type=float, default=1.0)
    run.add_argument("--endpoint-flow-concentration", type=float, default=0.30)
    run.add_argument(
        "--noisy-training-labels",
        action="store_true",
        help="Contaminate training labels with T_true; the classifier remains paired/fixed.",
    )
    run.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run: 2 scenarios x 2 repetitions with smaller samples.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    values = dict(
        seed=args.seed,
        n_scenarios=args.n_scenarios,
        n_repetitions=args.n_repetitions,
        lambda_grid=tuple(args.lambda_grid),
        alpha=args.alpha,
        noise_rates=tuple(args.noise_rates),
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        n_mc_c=args.n_mc_c,
        bootstrap_draws=args.bootstrap_draws,
        max_condition_number=args.max_condition_number,
        true_flow_concentration=args.true_flow_concentration,
        endpoint_flow_concentration=args.endpoint_flow_concentration,
        noisy_training_labels=args.noisy_training_labels,
    )
    if args.quick:
        values.update(
            n_scenarios=2,
            n_repetitions=2,
            lambda_grid=(0.0, 0.5, 1.0),
            n_train=500,
            n_cal=1_000,
            n_test=1_000,
            n_estimators=80,
            n_mc_c=200,
            bootstrap_draws=500,
        )
    return ExperimentConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test()
        return 0
    if args.command == "run":
        cfg = config_from_args(args)
        run_experiment(cfg, args.output_dir)
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
