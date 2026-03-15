# Autoball Research Program

This repository adapts the minimal `autoresearch` pattern to MLB game prediction.

## Objective

Improve out-of-sample probabilistic prediction of MLB game outcomes through an autonomous loop that can:

- modify feature engineering
- modify training-set construction
- modify model families and hyperparameters
- compare experiments on a fixed evaluation contract
- keep only changes that improve the benchmark

The initial target is binary classification of `home_team_win`.

## Trust Boundary

Treat these as fixed unless explicitly permitted in this file:

- `data/raw/**`: scraped source data and season master files
- `data/br_game_ids_*.csv`: season manifests
- historical experiment outputs already written under `data/training sets/`

The agent may create new derived datasets, experiment outputs, and scripts, but should avoid mutating raw scraped data.

## Mutable Surface

The agent is allowed to modify:

- `scripts/build_baseline_logreg.py`
- new modeling scripts under `scripts/`
- dataset-building logic under `scripts/`
- small config or documentation files related to experiments

The agent should prefer adding small, composable scripts instead of creating a large framework early.

## Loop

For each iteration:

1. Read this file and the current modeling/evaluation scripts.
2. Propose one bounded change.
3. Build the derived training/validation data if needed.
4. Train the candidate model.
5. Evaluate on the fixed validation slice.
6. Record metrics and a short experiment note.
7. Keep the change only if it beats the current champion on the primary metric without violating constraints.

## Fixed Evaluation Contract

The evaluation split is:

- train: 2024 season
- validation: 2025 season

Use only features available before first pitch. No same-game leakage.

Primary metric:

- validation log loss on `home_team_win`

Secondary metrics:

- validation Brier score
- validation ROC AUC
- validation accuracy

Champion selection is based on lowest validation log loss. Secondary metrics are tie-breakers and diagnostics only.

## Constraints

- Prefer probability-producing models.
- Do not optimize for accuracy at the expense of calibration.
- Do not use the 2025 labels in feature construction for earlier 2025 games.
- Do not overwrite prior experiment outputs unless they are reproducible regenerations of the same artifact.
- Do not commit large generated files such as full parquet datasets or trained model binaries unless explicitly requested.

## Commit Policy

Use git as the mutation memory, but do not commit every attempted change.

Commit only when all of the following are true:

- the run completed successfully
- the code is reproducible
- the candidate beats the current champion on validation log loss
- the diff is coherent enough to preserve as a checkpoint

Failed or non-improving runs should be logged in an experiment registry, not committed.

Prefer committing to a dedicated research branch, not directly to `main`.

## Early Search Space

Search these first:

- rolling-window features instead of full season-to-date averages
- home/away splits
- bullpen workload proxies from recent pitching totals
- lineup-strength proxies from recent batting totals
- regularization and calibration
- tree-based baselines and linear feature expansions

Avoid these early:

- giant framework rewrites
- changing the raw scrape format
- adding external paid data
- optimizing on the validation set through repeated ad hoc manual cherry-picking

## Reporting

Every accepted run should leave behind:

- a metrics file
- validation predictions
- a short experiment summary with hypothesis, change, and result

Keep the loop simple enough that an agent can understand the full state quickly.
