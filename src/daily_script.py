import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import bbref
    import elo_rating

    from google.cloud import secretmanager
    from supabase import create_client
    from supabase_helper import fetch_entire_table, fetch_filtered_table
    from ml_pipeline import load_pipeline

    import importlib
    import polars as pl
    import predictions

    return (
        bbref,
        create_client,
        elo_rating,
        fetch_entire_table,
        importlib,
        json,
        load_pipeline,
        predictions,
        secretmanager,
    )


@app.cell
def _(create_client, json, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/5"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    return (supabase,)


@app.cell
def _(bbref, supabase):
    bbref.update_current_month(
        supabase,
        bbref.start_driver()
    )
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_boxscores(supabase)
    return


@app.cell
def _(elo_rating, importlib, supabase):
    importlib.reload(elo_rating)
    elo_rating.elo_update(supabase)
    # todo, earlier filtering to get the newest games from scrape missing boxscore & check whether they are in elo
    return


@app.cell
def _(predictions, supabase):
    predictions.update_team_records(supabase)
    return


@app.cell
def _(importlib, predictions, supabase):
    importlib.reload(predictions)
    predictions.update_existing_predictions(supabase)
    return


@app.cell
def _(predictions, supabase):
    games_to_predict = predictions.upcoming_game_data(supabase)
    return (games_to_predict,)


@app.cell
def _(
    fetch_entire_table,
    games_to_predict,
    json,
    load_pipeline,
    predictions,
    supabase,
):
    if games_to_predict is not None:
        with open("best_features_ensemble_20260201.json", "r") as f:
            bf = json.load(f)["features"]
        pipe = load_pipeline('ensemble_20260201')
        mod = pipe["ensemble"]
        threshold=pipe['threshold']
        schedule = fetch_entire_table(supabase, "schedule")
        upcoming_predictions = predictions.predict_upcoming_games(games_to_predict, mod, bf, threshold, schedule)
        supabase.table("predictions").insert(upcoming_predictions.to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
