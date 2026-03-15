from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


TEAM_FEATURE_COLUMNS = [
    "games_played_before",
    "win_pct_before",
    "rest_days",
    "bat_R_mean_before",
    "bat_H_mean_before",
    "bat_BB_mean_before",
    "bat_SO_mean_before",
    "bat_OBP_mean_before",
    "bat_SLG_mean_before",
    "bat_RE24_mean_before",
    "pitch_R_mean_before",
    "pitch_H_mean_before",
    "pitch_BB_mean_before",
    "pitch_SO_mean_before",
    "pitch_HR_mean_before",
    "pitch_ERA_mean_before",
    "pitch_RE24_mean_before",
]

DIFF_FEATURE_COLUMNS = [
    "games_played_before_diff",
    "win_pct_before_diff",
    "rest_days_diff",
    "bat_R_mean_before_diff",
    "bat_H_mean_before_diff",
    "bat_BB_mean_before_diff",
    "bat_SO_mean_before_diff",
    "bat_OBP_mean_before_diff",
    "bat_SLG_mean_before_diff",
    "bat_RE24_mean_before_diff",
    "pitch_R_mean_before_diff",
    "pitch_H_mean_before_diff",
    "pitch_BB_mean_before_diff",
    "pitch_SO_mean_before_diff",
    "pitch_HR_mean_before_diff",
    "pitch_ERA_mean_before_diff",
    "pitch_RE24_mean_before_diff",
    "home_games_played_before",
    "away_games_played_before",
    "home_rest_days",
    "away_rest_days",
]


def load_team_game_totals(data_root: Path, season: int) -> pd.DataFrame:
    batting = pd.read_parquet(data_root / "raw" / str(season) / "batting_master.parquet")
    pitching = pd.read_parquet(data_root / "raw" / str(season) / "pitching_master.parquet")

    batting = batting[batting["Batting"].eq("Team Totals")].copy()
    pitching = pitching[pitching["Pitching"].eq("Team Totals")].copy()

    batting = batting.rename(
        columns={
            "R": "bat_R",
            "H": "bat_H",
            "BB": "bat_BB",
            "SO": "bat_SO",
            "OBP": "bat_OBP",
            "SLG": "bat_SLG",
            "RE24": "bat_RE24",
        }
    )[["game_id", "team_code", "bat_R", "bat_H", "bat_BB", "bat_SO", "bat_OBP", "bat_SLG", "bat_RE24"]]

    pitching = pitching.rename(
        columns={
            "R": "pitch_R",
            "H": "pitch_H",
            "BB": "pitch_BB",
            "SO": "pitch_SO",
            "HR": "pitch_HR",
            "ERA": "pitch_ERA",
            "RE24": "pitch_RE24",
        }
    )[["game_id", "team_code", "pitch_R", "pitch_H", "pitch_BB", "pitch_SO", "pitch_HR", "pitch_ERA", "pitch_RE24"]]

    team_games = batting.merge(pitching, on=["game_id", "team_code"], how="inner", validate="one_to_one")

    for column in team_games.columns.difference(["game_id", "team_code"]):
        team_games[column] = pd.to_numeric(team_games[column], errors="coerce")

    return team_games


def build_preseason_defaults(team_games: pd.DataFrame) -> dict[str, float]:
    defaults = {
        "games_played_before": 0.0,
        "win_pct_before": 0.5,
        "rest_days": 1.0,
    }
    stat_columns = [
        "bat_R",
        "bat_H",
        "bat_BB",
        "bat_SO",
        "bat_OBP",
        "bat_SLG",
        "bat_RE24",
        "pitch_R",
        "pitch_H",
        "pitch_BB",
        "pitch_SO",
        "pitch_HR",
        "pitch_ERA",
        "pitch_RE24",
    ]
    for column in stat_columns:
        defaults[f"{column}_mean_before"] = float(team_games[column].mean())
    return defaults


def build_schedule_frame(data_root: Path, season: int) -> pd.DataFrame:
    schedule = pd.read_csv(data_root / f"br_game_ids_{season}.csv", parse_dates=["game_date"])
    schedule["home_team_code"] = np.where(schedule["dest"].fillna("") == "@", schedule["Opp"], schedule["Tm"])
    schedule["away_team_code"] = np.where(schedule["dest"].fillna("") == "@", schedule["Tm"], schedule["Opp"])
    schedule["season"] = season
    return schedule[["season", "game_id", "game_date", "game_number", "home_team_code", "away_team_code"]]


def add_team_history_features(
    schedule: pd.DataFrame,
    team_games: pd.DataFrame,
    preseason_defaults: dict[str, float],
) -> pd.DataFrame:
    team_schedule = pd.concat(
        [
            schedule[["season", "game_id", "game_date", "game_number", "home_team_code"]]
            .rename(columns={"home_team_code": "team_code"})
            .assign(is_home=1),
            schedule[["season", "game_id", "game_date", "game_number", "away_team_code"]]
            .rename(columns={"away_team_code": "team_code"})
            .assign(is_home=0),
        ],
        ignore_index=True,
    ).sort_values(["team_code", "game_date", "game_number", "game_id", "is_home"]).reset_index(drop=True)

    team_schedule = team_schedule.merge(team_games, on=["game_id", "team_code"], how="left", validate="one_to_one")

    stat_columns = [
        "bat_R",
        "bat_H",
        "bat_BB",
        "bat_SO",
        "bat_OBP",
        "bat_SLG",
        "bat_RE24",
        "pitch_R",
        "pitch_H",
        "pitch_BB",
        "pitch_SO",
        "pitch_HR",
        "pitch_ERA",
        "pitch_RE24",
    ]

    team_schedule["team_win"] = (team_schedule["bat_R"] > team_schedule["pitch_R"]).astype(float)
    team_schedule["games_played_before"] = team_schedule.groupby("team_code").cumcount()

    previous_date = team_schedule.groupby("team_code")["game_date"].shift(1)
    team_schedule["rest_days"] = (team_schedule["game_date"] - previous_date).dt.days - 1
    team_schedule["rest_days"] = team_schedule["rest_days"].clip(lower=0)

    for column in stat_columns + ["team_win"]:
        prior_sum = team_schedule.groupby("team_code")[column].cumsum() - team_schedule[column]
        prior_mean = np.where(
            team_schedule["games_played_before"] > 0,
            prior_sum / team_schedule["games_played_before"],
            np.nan,
        )
        feature_name = "win_pct_before" if column == "team_win" else f"{column}_mean_before"
        team_schedule[feature_name] = prior_mean

    for column, default in preseason_defaults.items():
        team_schedule[column] = team_schedule[column].fillna(default)

    return team_schedule[["season", "game_id", "team_code", *TEAM_FEATURE_COLUMNS, "bat_R", "pitch_R"]]


def build_game_level_dataset(schedule: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    home = team_features.rename(
        columns={
            "team_code": "home_team_code",
            **{column: f"home_{column}" for column in TEAM_FEATURE_COLUMNS},
            "bat_R": "home_score",
            "pitch_R": "home_runs_allowed",
        }
    )
    away = team_features.rename(
        columns={
            "team_code": "away_team_code",
            **{column: f"away_{column}" for column in TEAM_FEATURE_COLUMNS},
            "bat_R": "away_score",
            "pitch_R": "away_runs_allowed",
        }
    )

    games = schedule.merge(home, on=["season", "game_id", "home_team_code"], how="inner", validate="one_to_one")
    games = games.merge(away, on=["season", "game_id", "away_team_code"], how="inner", validate="one_to_one")

    for column in TEAM_FEATURE_COLUMNS:
        games[f"{column}_diff"] = games[f"home_{column}"] - games[f"away_{column}"]

    games["target_home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    return games[
        [
            "season",
            "game_id",
            "game_date",
            "game_number",
            "home_team_code",
            "away_team_code",
            "home_score",
            "away_score",
            "target_home_win",
            *DIFF_FEATURE_COLUMNS,
        ]
    ].copy()


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float]:
    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        weights = params[:-1]
        intercept = params[-1]
        logits = x @ weights + intercept
        probs = expit(logits)
        probs = np.clip(probs, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)) + 0.5 * l2 * np.sum(weights ** 2)
        error = probs - y
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = np.mean(error)
        gradient = np.concatenate([grad_w, [grad_b]])
        return loss, gradient

    initial = np.zeros(x.shape[1] + 1, dtype=float)
    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=initial,
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Logistic regression failed: {result.message}")
    return result.x[:-1], float(result.x[-1])


def predict_proba(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return expit(x @ weights + intercept)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(int) == y_true))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    positive = y_true == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    output_root = data_root / "training sets"
    output_root.mkdir(parents=True, exist_ok=True)

    season_frames = []
    default_source_games = load_team_game_totals(data_root, 2024)
    preseason_defaults = build_preseason_defaults(default_source_games)
    for season in (2024, 2025):
        schedule = build_schedule_frame(data_root, season)
        team_games = default_source_games if season == 2024 else load_team_game_totals(data_root, season)
        team_features = add_team_history_features(schedule, team_games, preseason_defaults)
        season_frames.append(build_game_level_dataset(schedule, team_features))

    full_dataset = pd.concat(season_frames, ignore_index=True).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    train_df = full_dataset[full_dataset["season"] == 2024].reset_index(drop=True)
    validation_df = full_dataset[full_dataset["season"] == 2025].reset_index(drop=True)

    x_train = train_df[DIFF_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["target_home_win"].to_numpy(dtype=float)
    x_validation = validation_df[DIFF_FEATURE_COLUMNS].to_numpy(dtype=float)
    y_validation = validation_df["target_home_win"].to_numpy(dtype=float)

    mean, std = standardize_fit(x_train)
    x_train_std = (x_train - mean) / std
    x_validation_std = (x_validation - mean) / std

    weights, intercept = fit_logistic_regression(x_train_std, y_train, l2=0.01)

    train_pred = predict_proba(x_train_std, weights, intercept)
    validation_pred = predict_proba(x_validation_std, weights, intercept)

    train_with_pred = train_df.copy()
    validation_with_pred = validation_df.copy()
    train_with_pred["pred_home_win_prob"] = train_pred
    validation_with_pred["pred_home_win_prob"] = validation_pred

    metrics = {
        "train": {
            "games": int(len(train_df)),
            "home_win_rate": float(y_train.mean()),
            "accuracy": accuracy(y_train, train_pred),
            "log_loss": log_loss(y_train, train_pred),
            "brier_score": brier_score(y_train, train_pred),
            "roc_auc": roc_auc(y_train.astype(int), train_pred),
        },
        "validation": {
            "games": int(len(validation_df)),
            "home_win_rate": float(y_validation.mean()),
            "accuracy": accuracy(y_validation, validation_pred),
            "log_loss": log_loss(y_validation, validation_pred),
            "brier_score": brier_score(y_validation, validation_pred),
            "roc_auc": roc_auc(y_validation.astype(int), validation_pred),
        },
    }

    coefficient_frame = pd.DataFrame(
        {
            "feature": DIFF_FEATURE_COLUMNS,
            "coefficient_standardized": weights,
            "train_mean": mean,
            "train_std": std,
        }
    ).sort_values("coefficient_standardized", key=lambda s: s.abs(), ascending=False)

    full_dataset.to_parquet(output_root / "game_level_features_all.parquet", index=False)
    train_with_pred.to_parquet(output_root / "train_2024_logreg_features.parquet", index=False)
    validation_with_pred.to_parquet(output_root / "validation_2025_logreg_features.parquet", index=False)
    validation_with_pred[
        ["game_id", "game_date", "home_team_code", "away_team_code", "target_home_win", "pred_home_win_prob"]
    ].to_csv(output_root / "validation_2025_predictions.csv", index=False)
    coefficient_frame.to_csv(output_root / "baseline_logreg_coefficients.csv", index=False)

    with (output_root / "baseline_logreg_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    joblib.dump(
        {
            "feature_columns": DIFF_FEATURE_COLUMNS,
            "weights": weights,
            "intercept": intercept,
            "train_mean": mean,
            "train_std": std,
            "metrics": metrics,
        },
        output_root / "baseline_logreg_model.joblib",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
