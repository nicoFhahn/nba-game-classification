import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from supabase import create_client, Client
    from supabase_helper import fetch_entire_table
    from ml_pipeline import main_pipeline, load_pipeline
    import features
    from tqdm import tqdm
    from datetime import date
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    import polars as pl
    import numpy as np
    import polars.selectors as cs
    import lightgbm as lgbm
    import json
    import stepwise_feature_selection
    from google.cloud import secretmanager
    return (
        create_client,
        features,
        fetch_entire_table,
        json,
        load_pipeline,
        log_loss,
        pl,
        secretmanager,
        stepwise_feature_selection,
        tqdm,
    )


@app.cell
def _(create_client, fetch_entire_table, json, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    schedule = fetch_entire_table(supabase, "schedule")
    games = fetch_entire_table(supabase, "boxscore")
    elo = fetch_entire_table(supabase, "elo")
    schedule = schedule.join(
        games,
        on="game_id"
    )
    return elo, games, schedule, supabase


@app.cell
def _(features, games, pl, schedule, tqdm):
    final_data = []
    for s in tqdm(set(schedule["season_id"])):
        season_schedule = schedule.filter(
            pl.col("season_id") == s
        ).sort("date").select([
            "game_id", "date", "home_team", "guest_team", "pts_home", "pts_guest"
        ]).with_columns([
            (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
        ])
        season_games = season_schedule.select(
            "game_id", "date", "home_team", "guest_team"
        ).join(
            games,
            on="game_id"
        ).with_columns(
            pl.col("date").str.to_date()
        )
        teams = set(season_games["home_team"])
        boxscore_dfs = [features.team_boxscores(season_games, team) for team in teams]
        team_stats = [features.team_season_stats(boxscore_df) for boxscore_df in boxscore_dfs]
        season_schedule = features.add_overall_winning_pct(season_schedule)
        season_schedule = features.add_location_winning_pct(season_schedule)
        season_schedule = features.h2h(season_schedule)
        team_stats = pl.concat(team_stats)
        joined = season_schedule.drop([
            "date", "pts_home", "pts_guest",
            "home_win", "guest_win"
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
        final_data.append(joined)
    return (final_data,)


@app.cell
def _(elo, final_data, pl, schedule):
    df = pl.concat(final_data).join(
        elo.select("game_id", "team_id", "elo_before").rename({
            "elo_before": "elo_home"
        }),
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team_id"]
    ).join(
        elo.select("game_id", "team_id", "elo_before").rename({
            "elo_before": "elo_guest"
        }),
        left_on=["game_id", "guest_team"],
        right_on=["game_id", "team_id"]
    )
    df = schedule.select(
        "game_id", "date"
    ).join(
        df,
        on="game_id"
    ).with_columns(
        pl.col("date").str.to_date()
    ).sort("date")
    return (df,)


@app.cell
def _(df, pl, schedule):
    train_ids = schedule.filter(pl.col("season_id") <= 2024)["game_id"].to_list()
    test_ids = schedule.filter(pl.col("season_id") == 2025)["game_id"].to_list()
    val_ids = schedule.filter(pl.col("season_id") == 2026)["game_id"].to_list()
    train = df.filter(
        pl.col("game_id").is_in(train_ids)
    )
    test = df.filter(
        pl.col("game_id").is_in(test_ids)
    )
    val = df.filter(
        pl.col("game_id").is_in(val_ids)
    )
    X_train = train.drop(["game_id", "date", "home_team", "guest_team", "is_home_win"])
    X_test = test.drop(["game_id", "date", "home_team", "guest_team", "is_home_win"])
    X_val = val.drop(["game_id", "date", "home_team", "guest_team", "is_home_win"])
    y_train = train.select("is_home_win")
    y_test = test.select("is_home_win")
    y_val = val.select("is_home_win")
    return X_test, X_train, X_val, val, y_test, y_train


@app.cell
def _(X_train, json, log_loss, stepwise_feature_selection, y_train):
    run_fs = False
    if run_fs:
        selector = stepwise_feature_selection.LGBMStepwiseFeatureSelector(
            importance_type='shap',
            n_folds=5,
            random_state=87654323,
            verbose=True,
            metric=log_loss
        )
        selector.fit(
            X_train.to_numpy(),
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
        with open("bf.json", "w") as f:
            json.dump(bf, f)
    else:
        with open("bf.json", "r") as f:
            bf = json.load(f)
    return (bf,)


@app.cell
def _(X_test, X_train, X_val, bf):
    X_train_best = X_train.select(bf["features"])
    X_test_best = X_test.select(bf["features"])
    X_val_best = X_val.select(bf["features"])
    return X_test_best, X_train_best, X_val_best


@app.cell
def _(X_test_best, X_train_best, y_test, y_train):
    from pathlib import Path
    import importlib
    import ml_pipeline
    importlib.reload(ml_pipeline)
    output_dir = 'ensemble_model_v2'
    n_trials = 500
    max_estimators = 2000
    threshold_metric = 'accuracy'
    n_jobs_optuna = 1
    random_state = 2343242
    optuna_verbosity = 1
    save_frequency = 'model'

    # Check if checkpoint exists
    checkpoint_exists = Path(output_dir).exists()
    results = ml_pipeline.main_pipeline(
        X_train=X_train_best,
        y_train=y_train,
        X_test=X_test_best,
        y_test=y_test,
        n_trials=n_trials,
        max_estimators=max_estimators,
        threshold_metric=threshold_metric,
        n_jobs_optuna=n_jobs_optuna,
        random_state=random_state,
        optuna_verbosity=optuna_verbosity,
        output_dir=output_dir,
        load_checkpoint=checkpoint_exists,
        save_frequency=save_frequency
    )
    return


@app.cell
def _(X_val_best, load_pipeline):
    saved = load_pipeline('ensemble_model_v2')
    predictions = saved['ensemble'].predict_proba(X_val_best.to_numpy())
    threshold=saved['threshold']
    return predictions, threshold


@app.cell
def _(pl, predictions, threshold, val):
    pred_df = val.select(["game_id", "date", "home_team", "guest_team", "is_home_win"]).with_columns([
        pl.Series(
            "proba",
            predictions
        )
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ])
    return (pred_df,)


@app.cell
def _(pred_df_2):
    pred_df_2.sort("proba")
    return


@app.cell
def _():
    # below 55% confidence -> low
    # below 68% confidence -> medium
    # above -> high
    # todo: change confidence to classes
    # add one tab for future predictions, one tab for past predictions
    # add rolling performance metrics
    # check whether the data is live
    # automate data uploads
    # build table for future games
    return


@app.cell
def _(pl, pred_df, supabase):
    supabase.table(
        "predictions"
    ).insert(pred_df.with_columns([
        pl.col("date").cast(pl.String)
    ]).to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
