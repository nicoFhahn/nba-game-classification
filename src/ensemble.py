import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Standard library
    import calendar
    import json
    import time
    from datetime import date, timedelta
    import importlib
    import os

    # Third-party
    import tabpfn_client
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from catboost import CatBoostClassifier
    from google.cloud import secretmanager
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    from supabase import Client, create_client
    from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor
    from tqdm import tqdm
    import joblib
    import optuna

    # Local modules
    import features
    import ml_helpers
    from supabase_helper import (
        fetch_entire_table,
        fetch_filtered_table,
        fetch_distinct_column,
    )

    return (
        create_client,
        cs,
        date,
        fetch_entire_table,
        importlib,
        joblib,
        json,
        ml_helpers,
        pl,
        secretmanager,
        tabpfn_client,
    )


@app.cell
def _(create_client, json, secretmanager, tabpfn_client):
    name = "projects/898760610238/secrets/supabase/versions/7"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    tabpfn_client.set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGYyM2VmYzktYzNlNy00YWNjLTljOGMtMGYyNWVlZGU1YTFiIiwiZXhwIjoxODAzNzQzMTQ3fQ.SrxoILOo8WwLT9d1rF1XXMqQP_IhZrd7DG5z7gf1MO8")
    return (supabase,)


@app.cell
def _(cs, fetch_entire_table, ml_helpers, pl, supabase):
    games, schedule, player_boxscore, elo = pl.read_parquet("../data/games.parquet"), pl.read_parquet("../data/schedule.parquet"), pl.read_parquet("../data/player_boxscore.parquet"), pl.read_parquet("../data/elo.parquet")
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
        (pl.col("date") >= pl.col("playoff_start")).alias("is_playoff_game")
    )
    print("Data loaded")
    preprocessed_data = ml_helpers.preprocess_data(games, schedule, player_boxscore, elo, impute=False).with_columns(
        pl.all().map_elements(
            lambda x: None if x in (float("inf"), float("-inf")) else x
        )
    )
    numeric_cols = [c for c in preprocessed_data.select(cs.numeric()).columns if c != "is_home_win"]
    correlations = {
        col: abs(preprocessed_data.select(pl.corr(col, "is_home_win")).item())
        for col in numeric_cols
        if col != "is_home_win"
    }
    low_corr_cols = {k: v for k, v in correlations.items() if v < 0.01}
    low_corr_cols = list(low_corr_cols.keys())
    preprocessed_data = preprocessed_data.drop(
        low_corr_cols
    ).drop(["id", "id_right"])
    return (preprocessed_data,)


@app.cell
def _(preprocessed_data):
    preprocessed_data.write_parquet("../data/preprocessed_imputed.parquet")
    return


@app.cell
def _(date, importlib, joblib, json, pl, preprocessed_data):
    import ensemble_pipeline
    importlib.reload(ensemble_pipeline)
    date_pairs = [
        (date(2025, 10, 1), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 31))
    ]
    configs = [
        {"method": "simple_averaging"},
        {"method": "weighted_averaging"},
        {"method": "stacking", "meta_model": "catboost", "use_oof": True},
        {"method": "stacking", "meta_model": "logistic_regression", "use_oof": True},
        {"method": "blending", "meta_model": "catboost", "holdout_frac": 0.2},
        {"method": "blending", "meta_model": "logistic_regression", "holdout_frac": 0.2},
        {"method": "voting", "voting_type": "soft"}
    ]
    result_dfs = []
    for pair in date_pairs:
        train = preprocessed_data.filter(
            pl.col("date") < pair[0]
        )
        val = preprocessed_data.filter(
            pl.col("date").is_between(
                pair[0], pair[1]
            )
        )
        fs_file_name = f"models/best_features_{pair[0].strftime('%Y%m%d')}.json"
        ensemble_file_name = f"models/ensemble_{pair[0].strftime('%Y%m%d')}.pkl"
        with open(fs_file_name, "r") as f:
            best_features = json.load(f)["features"]
        X_train = train.select(best_features)
        X_val = val.select(best_features)
        y_train = train.select("is_home_win")
        y_val = val.select("is_home_win")
        base = ensemble_pipeline.train_base_models(
            X_train=X_train,
            y_train=y_train,
            split_type="temporal",
            algorithms=["catboost", "lightgbm", "xgboost"],
            metric="neg_log_loss",
            n_trials=50,
        )

        # Ensemble configs iterate cheaply — no HPO repeated
        pipeline_results = [
            ensemble_pipeline.build_ensemble_from_base_models(base, config)
            for config in configs
        ]

        best_result = max(pipeline_results, key=lambda x: x["cv_score"])
    
        # Extract the best model and score
        best_ensemble = best_result["ensemble"]
        best_score = best_result["cv_score"]
    
        # Print which configuration won (optional but helpful)
        #print(f"Best CV Score for {pair[0]}: {best_score:.4f} using {best_result['stacking_options']}")

        # Get predictions only for the best one
        probabilities = best_ensemble.predict_proba(X_val)
    
        result_df = pl.DataFrame({
            "game_id": val["game_id"],
            "is_home_win": y_val, # Ensure it's a series for the DataFrame
            "proba": probabilities
        })
        joblib.dump(best_ensemble, ensemble_file_name)
    return (ensemble_file_name,)


@app.cell
def _(ensemble_file_name, joblib):
    ens = joblib.load(ensemble_file_name)
    return


@app.cell
def _():
    import sklearn
    sklearn.metrics.get_scorer_names()
    return


@app.cell
def _():
    # todo, bests model laden
    return


if __name__ == "__main__":
    app.run()
