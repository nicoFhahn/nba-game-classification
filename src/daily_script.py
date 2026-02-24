import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    # Use optimized dependency versions
    import bbref as bbref
    import elo_rating as elo_rating
    import predictions as predictions
    import supabase_helper as supabase_helper

    from google.cloud import secretmanager
    from supabase import create_client
    from ml_pipeline import load_pipeline

    import importlib
    import polars as pl

    return (
        bbref,
        create_client,
        elo_rating,
        importlib,
        json,
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
    supabase = create_client(url, key)
    return (supabase,)


@app.cell
def _(bbref, supabase):
    bbref.update_current_month(
        supabase,
        bbref.start_driver()
    )
    # checked - approved 
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_boxscores(supabase)
    # checked - approved
    return


@app.cell
def _(elo_rating, supabase):
    elo_rating.elo_update(supabase)
    # checked - approved
    return


@app.cell
def _(predictions, supabase):
    predictions.update_team_records(supabase)
    # checked - approved
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
def _(games_to_predict, predictions, supabase):
    if games_to_predict is not None:
        upcoming_predictions = predictions.predict_upcoming_games(
            games_to_predict, supabase, "https://nighthuhn-nba.hf.space"
        )
        supabase.table("predictions").insert(upcoming_predictions.to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
