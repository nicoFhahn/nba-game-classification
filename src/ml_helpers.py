from tqdm import tqdm
from sklearn.metrics import accuracy_score
from typing import Callable
from pathlib import Path

from supabase_helper import fetch_entire_table

import polars as pl
import datetime
import json
import glob

import features
import stepwise_feature_selection
import ml_pipeline
import importlib
importlib.reload(ml_pipeline)

def load_data(supabase) -> (pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame):
    schedule = fetch_entire_table(supabase, "schedule")
    games = fetch_entire_table(supabase, "boxscore")
    player_boxscore = fetch_entire_table(supabase, "player-boxscore")
    elo = fetch_entire_table(supabase, "elo")
    playoffs = fetch_entire_table(supabase, "playoffs")
    locations = fetch_entire_table(supabase, "location")
    schedule = schedule.join(
        games,
        on="game_id"
    ).join(
        locations,
        on="arena"
    ).join(
        playoffs,
        on="season_id"
    ).with_columns(
        pl.col("date").str.to_date(),
        pl.col("playoff_start").str.to_date()
    ).with_columns(
        (pl.col("date") >= pl.col("playoff_start")).alias("is_playoff_game")
    )
    return games, schedule, player_boxscore, elo

def preprocess_data(
        games: pl.DataFrame,
        schedule: pl.DataFrame,
        player_boxscore: pl.DataFrame,
        elo: pl.DataFrame
):
    final_data = []
    for s in tqdm(set(schedule["season_id"])):
        season_schedule = schedule.filter(
            pl.col("season_id") == s
        ).sort("date").select([
            "game_id", "date", "home_team", "guest_team", "pts_home", "pts_guest", "is_playoff_game",
            "latitude", "longitude"
        ]).with_columns([
            (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
        ])
        season_games = season_schedule.select(
            "game_id", "date", "home_team", "guest_team"
        ).join(
            games,
            on="game_id"
        )
        teams = set(season_games["home_team"])
        boxscore_dfs = [features.team_boxscores(season_games, team) for team in teams]
        team_stats = [features.team_season_stats(boxscore_df) for boxscore_df in boxscore_dfs]
        season_schedule = features.add_overall_winning_pct(season_schedule)
        season_schedule = features.add_location_winning_pct(season_schedule)
        season_schedule = features.h2h(season_schedule)
        season_schedule = features.compute_travel(season_schedule)
        team_stats = pl.concat(team_stats)
        joined = season_schedule.drop([
            "date", "pts_home", "pts_guest",
            "home_win", "guest_win", "latitude", "longitude"
        ]).join(
            team_stats.drop(["date", "is_win"]).rename(
                lambda c: f"{c}_home"
            ),
            left_on=["game_id", "home_team"],
            right_on=["game_id_home", "team_home"]
        ).join(
            team_stats.drop(["date", "is_win"]).rename(
                lambda c: f"{c}_guest"
            ),
            left_on=["game_id", "guest_team"],
            right_on=["game_id_guest", "team_guest"]
        )
        season_boxscore = schedule[["game_id", "season_id"]].join(
            player_boxscore,
            on="game_id"
        ).filter(
            pl.col("season_id") == s
        )
        season_boxscores = season_boxscore.partition_by("team_id")
        season_player_stats = [features.player_season_stats(boxscore) for boxscore in season_boxscores]
        season_expected_stats = [features.calculate_expected_team_stats(sps) for sps in season_player_stats]
        joined = joined.join(
            pl.concat(season_expected_stats).drop("num_players").rename(
                lambda c: f"{c}_home_team"
            ),
            left_on=["game_id", "home_team"],
            right_on=["game_id_home_team", "team_id_home_team"],
            how="left"
        ).join(
            pl.concat(season_expected_stats).drop("num_players").rename(
                lambda c: f"{c}_guest_team"
            ),
            left_on=["game_id", "guest_team"],
            right_on=["game_id_guest_team", "team_id_guest_team"],
            how="left"
        )
        final_data.append(joined)
    elo_w_change = elo.with_columns(
        pl.col("elo_before").shift(7).over(pl.col("team_id")).alias("elo_7_games_before")
    ).with_columns(
        (pl.col("elo_before") - pl.col("elo_7_games_before")).alias("elo_change_7_absolute"),
        (pl.col("elo_before") / pl.col("elo_7_games_before") - 1).alias("elo_change_7_relative")
    ).drop("elo_7_games_before")
    df = pl.concat(final_data).join(
        elo_w_change.drop("id", "elo_after").rename({
            "elo_before": "elo_home",
            "elo_change_7_absolute": "elo_change_7_absolute_home",
            "elo_change_7_relative": "elo_change_7_relative_home"
        }),
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team_id"]
    ).join(
        elo_w_change.drop("id", "elo_after").rename({
            "elo_before": "elo_guest",
            "elo_change_7_absolute": "elo_change_7_absolute_guest",
            "elo_change_7_relative": "elo_change_7_relative_guest"
        }),
        left_on=["game_id", "guest_team"],
        right_on=["game_id", "team_id"]
    )
    df = schedule.select(
        "game_id", "date"
    ).join(
        df,
        on="game_id"
    ).sort("date")
    return df

def train_test(
        df: pl.DataFrame,
        cutoff_date: datetime.date,
        train_size: float = 0.8
) -> (pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame):
    filtered_df = df.filter(
        pl.col("date") < cutoff_date
    )
    train = filtered_df[:round(filtered_df.shape[0] * train_size)]
    test = filtered_df.filter(
        ~pl.col("game_id").is_in(train["game_id"].to_list())
    )
    X_train = train.drop(["game_id", "home_team", "guest_team", "is_home_win"])
    X_test = test.drop(["game_id", "home_team", "guest_team", "is_home_win"])
    y_train = train.select("is_home_win")
    y_test = test.select("is_home_win")
    return X_train, y_train, X_test, y_test

def run_pipeline(
        supabase,
        cutoff_date: datetime.date,
        output_dir: str,
        train_size: float = 0.8,
        random_state: int = 87654323,
        n_folds: int = 5,
        importance_type: str = 'shap',
        metric: Callable = accuracy_score,
        run_fs: bool = True,
        n_trials: int = 500,
        max_estimators: int = 2000,
        weight_decay: float = 0.99,
        use_weights: bool = False,
        n_jobs_optuna: int = 1,
        threshold_metric: str = 'accuracy',
        save_frequency: str = 'model'
):
    games, schedule, player_boxscore, elo = load_data(supabase)
    print("Data loaded")
    preprocessed_data = preprocess_data(games, schedule, player_boxscore, elo)
    print("Data preprocessed")
    X_train, y_train, X_test, y_test = train_test(
        preprocessed_data, cutoff_date, train_size
    )
    print("Data split")
    filename = f"best_features_{output_dir}.json"
    if run_fs:
        selector = stepwise_feature_selection.LGBMStepwiseFeatureSelector(
            importance_type=importance_type,
            n_folds=n_folds,
            random_state=random_state,
            verbose=True,
            metric=metric
        )
        selector.fit(
            X_train.drop("date").to_numpy(),
            y_train.to_numpy().ravel()
        )
        selected_indices = [
            int(feat.replace('feature_', ''))
            for feat in selector.best_features_
        ]
        original_feature_names = X_train.columns
        best_features_original = [
            original_feature_names[idx]
            for idx in selected_indices
        ]
        bf = {
            "features": best_features_original
        }
        with open(filename, "w") as f:
            json.dump(bf, f)
    else:
        with open(filename, "r") as f:
            bf = json.load(f)
    X_train_best = X_train.select(bf["features"] + ["date"])
    X_test_best = X_test.select(bf["features"] + ["date"])
    checkpoint_exists = Path(output_dir).exists()
    results = ml_pipeline.main_pipeline(
        X_train=X_train_best,
        y_train=y_train,
        X_test=X_test_best,
        y_test=y_test,
        n_trials=n_trials,
        max_estimators=max_estimators,
        weight_decay=weight_decay,
        use_weights=use_weights,
        threshold_metric=threshold_metric,
        n_jobs_optuna=n_jobs_optuna,
        random_state=random_state,
        optuna_verbosity=1,
        output_dir=output_dir,
        load_checkpoint=checkpoint_exists,
        save_frequency=save_frequency,
        date_column='date'
    )
    saved = ml_pipeline.load_pipeline(output_dir)
    return saved

