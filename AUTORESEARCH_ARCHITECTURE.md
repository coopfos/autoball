# Autoball Autoresearch Architecture

This repository should follow the same core idea as Karpathy's `autoresearch`: keep the search surface small, keep evaluation fixed, and make "keep vs discard" automatic. The direct analogue is not a single mutable `train.py`, because MLB modeling needs separate data-building and modeling steps. The right adaptation is a small fixed set of contracts.

## Core Pieces

`program.md`

- Human-authored research instructions.
- Defines allowed mutation scope, fixed split, primary metric, and promotion rules.
- This is the main steering interface for the autonomous agent.

`scripts/`

- Mutable research surface.
- Contains dataset builders, model trainers, and evaluators.
- Should stay modular and shallow rather than becoming an opaque framework.

`data/raw/`

- Fixed source-of-truth data.
- Raw scraped season masters and their parquet equivalents.
- Not part of the mutation loop.

`data/training sets/`

- Derived artifacts.
- Feature matrices, validation predictions, metrics files, and lightweight model artifacts.
- Can be regenerated.

## Recommended Loop

1. Load `program.md`.
2. Inspect the current champion metrics.
3. Choose one scoped hypothesis.
4. Modify or add one small script.
5. Rebuild only the necessary derived data.
6. Train on 2024.
7. Validate on 2025.
8. Compare against the champion using the primary metric.
9. If better, promote and checkpoint. If worse, discard code changes and keep only the experiment log.

## Why The Agent Should Not Commit Everything

Karpathy's repo can commit nearly every successful mutation because the mutable surface is one compact training file and one scalar metric. This project has more failure modes:

- hidden leakage in dataset construction
- noisy wins from feature churn
- larger diffs spanning both data prep and modeling
- generated artifacts that do not belong in source control

So the agent should commit accepted improvements, but only to a research branch and only after passing the evaluation gate. Attempted runs belong in an experiment ledger, not in git history.

Recommended policy:

- no commit for failed runs
- no commit for metric regressions
- no commit for tiny non-robust wins without a reproducible artifact trail
- commit accepted improvements to `research/<date-or-theme>`
- promote to `main` manually after review or after a stronger promotion threshold

## Evaluation Metric

The primary metric should be validation log loss.

Reasoning:

- it works across logistic regression, boosted trees, neural nets, and ensembles
- it rewards calibrated probabilities, not just correct labels
- it is threshold-free, unlike accuracy
- it aligns with the eventual need to rank confidence, not only classify

Secondary metrics should remain:

- Brier score: calibration and probability quality
- ROC AUC: ranking quality
- accuracy: sanity check only

If later the project moves from game-win classification to market decisioning, the research loop can add a separate business metric, but the base model-selection metric should remain a stable probabilistic score.

## Champion / Challenger Contract

Each candidate run should emit:

- model name
- feature set name
- train split definition
- validation split definition
- validation log loss
- validation Brier score
- validation ROC AUC
- validation accuracy
- path to validation predictions
- short textual hypothesis

Promotion rule for now:

- challenger replaces champion only if validation log loss is lower by a non-trivial margin and there is no evidence of leakage or broken reproducibility

A practical initial margin is:

- absolute log-loss improvement of at least `0.002`

That avoids promoting noise from tiny code changes.

## Recommended Near-Term Repo Shape

Keep the structure simple:

- `program.md`: agent instructions
- `scripts/build_baseline_logreg.py`: existing baseline
- `scripts/build_features_v*.py`: explicit feature builders
- `scripts/train_<model>.py`: model training entrypoints
- `scripts/evaluate_predictions.py`: shared metric computation
- `data/training sets/experiments/`: per-run metrics and notes
- `data/training sets/champion.json`: current best run metadata

This preserves the `autoresearch` spirit: a small number of files, a fixed eval contract, and one obvious mechanism for improvement.
