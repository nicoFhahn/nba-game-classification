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
    from datetime import date, timedelta
    import polars as pl
    import numpy as np
    import polars.selectors as cs
    import lightgbm as lgbm
    import json
    import stepwise_feature_selection
    from google.cloud import secretmanager
    return (
        create_client,
        date,
        fetch_entire_table,
        json,
        pl,
        secretmanager,
        timedelta,
    )


@app.cell
def _(create_client, json, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    return (supabase,)


@app.cell
def _(fetch_entire_table, supabase):
    predictions = fetch_entire_table(supabase, "predictions").drop_nulls("is_home_win")
    elo = fetch_entire_table(supabase, "elo")
    return elo, predictions


@app.cell
def _(pl, predictions):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    team = "Philadelphia 76ers"
    temp = predictions.filter(
        (pl.col("home_team") == team) |
        (pl.col("guest_team") == team)
    )
    print(f"""
    Accuracy: {accuracy_score(temp["is_home_win"], temp["is_predicted_home_win"]):.3f}
    Precision: {precision_score(temp["is_home_win"], temp["is_predicted_home_win"]):.3f}
    Recall: {recall_score(temp["is_home_win"], temp["is_predicted_home_win"]):.3f}
    F1 Score: {f1_score(temp["is_home_win"], temp["is_predicted_home_win"]):.3f}
    """)
    return


@app.cell
def _(elo, pl):
    elo.with_columns(
        pl.col('game_id').str.slice(0, 8).str.strptime(pl.Date, '%Y%m%d').alias('date')
    ).group_by("team_id").tail(1).sort("elo_after", descending=True).select([
        "team_id", "elo_after"
    ])
    return


@app.cell
def _(date, elo, pl, timedelta):
    elo.with_columns(
        pl.col('game_id').str.slice(0, 8).str.strptime(pl.Date, '%Y%m%d').alias('date')
    ).filter(
        pl.col("date") <= (date.today() + timedelta(days=-30))
    ).group_by("team_id").tail(1).sort("elo_after", descending=True).select([
        "team_id", "elo_after"
    ])
    return


if __name__ == "__main__":
    app.run()
