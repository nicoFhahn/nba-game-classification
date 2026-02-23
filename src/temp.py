import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import calendar
    import json
    import time
    from datetime import date, timedelta

    import polars as pl
    import shap
    from supabase import Client, create_client
    from tqdm import tqdm

    import features
    import ml_helpers

    from elo_rating import elo_season
    from ml_pipeline import load_pipeline
    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_month_data
    from google.cloud import secretmanager
    from shap_analysis import calculate_shap_ensemble, plot_shap_summary

    return (
        create_client,
        date,
        fetch_entire_table,
        json,
        load_pipeline,
        ml_helpers,
        pl,
        secretmanager,
    )


@app.cell
def _(create_client, json, ml_helpers, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    games, schedule, player_boxscore, elo = ml_helpers.load_data(supabase)
    return elo, games, player_boxscore, schedule, supabase


@app.cell
def _(elo, games, ml_helpers, pl, player_boxscore, schedule):
    preprocessed_data = ml_helpers.preprocess_data(games, schedule, player_boxscore, elo).with_columns(
        pl.all().map_elements(
            lambda x: None if x in (float("inf"), float("-inf")) else x
        )
    )
    return (preprocessed_data,)


@app.cell
def _(preprocessed_data):
    preprocessed_data.write_parquet("preprocessed.parquet")
    return


@app.cell
def _(json, load_pipeline):
    with open("best_features_ensemble_20260201.json", "r") as f:
        bf = json.load(f)
    saved = load_pipeline('ensemble_20260201')
    threshold=saved['threshold']
    return bf, saved, threshold


@app.cell
def _(bf, date, pl, preprocessed_data, saved):
    df = preprocessed_data.filter(pl.col("date") >= date(2026, 2, 1))
    y = preprocessed_data["is_home_win"]
    X = preprocessed_data.select(bf["features"])
    pred = saved["ensemble"].predict(X.to_numpy())
    from sklearn.metrics import accuracy_score
    accuracy_score(y, pred)
    return (df,)


@app.cell
def _(fetch_entire_table, supabase):
    predictions = fetch_entire_table(supabase, "predictions")
    return (predictions,)


@app.cell
def _(bf, df, pl, predictions, saved):
    missing = df.filter(~pl.col("game_id").is_in(predictions["game_id"].to_list()))
    y2 = missing["is_home_win"]
    X2 = missing.select(bf["features"])
    pred2 = saved["ensemble"].predict_proba(X2.to_numpy())
    return missing, pred2


@app.cell
def _(missing, pl, pred2, schedule, supabase, threshold):
    supabase.table("predictions").insert(missing.select([
        "game_id", "home_team", "guest_team",
        "is_home_win"
    ]).with_columns([
        pl.Series(
            "proba",
            pred2
        )
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ]).join(
        schedule[["game_id", "date"]],
        on="game_id"
    ).with_columns(
        pl.col("date").cast(pl.String)
    ).to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
