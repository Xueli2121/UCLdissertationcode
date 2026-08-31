# Structural Misspecification in Adaptive+ Conformal Classification

Code and reproducibility materials for the MSc dissertation:

*Structural Misspecification in Adaptive+ Conformal Classification:
Validity, Efficiency, and an Algorithm-Aware Diagnostic*

Candidate code: TQMS4

## Research overview

This project studies how structural misspecification of a label-contamination matrix affects the original Adaptive+ conformal classification procedure.

The diagonal entries of the forward transition matrix are held fixed at 0.80, so the class-specific label-noise rate remains 0.20. Only the off-diagonal allocation of error flow is changed. The experiment compares:

1. Standard label-conditional conformal calibration;
2. Oracle Adaptive+, using the true contamination matrix;
3. misspecified Adaptive+, using a perturbed contamination matrix.

The primary outcomes are worst-class clean coverage and mean prediction-set size. The analysis also compares three unsigned Frobenius distances with the signed algorithm-aware diagnostic G_Aplus.

## Repository structure

```text
src/
    structure_only_adaptive_plus.py
    analyse_worst_class_route.py
    make_thesis_figures.py

results/
    development/
        config.json
        structure_only_raw.csv
        worst_class_results/

    confirmation/
        config.json
        structure_only_raw.csv
        worst_class_results/

figures/
    final thesis figures

requirements.txt
LICENSE
NOTICE.md
README.md
```

## Installation

The final analysis used Python 3.12.7.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick validation

```bash
python src/structure_only_adaptive_plus.py self-test
```

## Reproducing the retained analysis

```bash
python src/analyse_worst_class_route.py \
    --raw results/confirmation/structure_only_raw.csv \
    --output-dir results/confirmation/worst_class_results
```

## Regenerating the thesis figures

```bash
python src/make_thesis_figures.py \
    --results-dir results/confirmation \
    --output-dir figures
```

The retained raw outputs allow the analysis to be reproduced without
rerunning the complete simulation.
