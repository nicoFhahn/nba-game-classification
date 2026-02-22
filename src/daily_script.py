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

    import importlib
    import polars as pl
    import predictions

    return (
        bbref,
        create_client,
        elo_rating,
        importlib,
        json,
        pl,
        predictions,
        secretmanager,
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
def _(bbref, supabase):
    bbref.update_current_month(
        supabase,
        bbref.start_driver()
    )
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_team_boxscores(supabase)
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_player_boxscores(supabase, bbref.start_driver())
    return


@app.cell
def _(elo_rating, supabase):
    elo_rating.elo_update(supabase)
    return


@app.cell
def _(predictions, supabase):
    predictions.update_team_records(supabase)
    return


@app.cell
def _(predictions, supabase):
    predictions.update_existing_predictions(supabase)
    return


@app.cell
def _(importlib, predictions, supabase):
    importlib.reload(predictions)
    games_to_predict = predictions.upcoming_game_data(supabase)
    return


@app.cell
def _(final_df, json, load_pipeline, pl, season_schedule):
    with open("best_features_ensemble_20260201.json", "r") as f:
        bf = json.load(f)
    saved = load_pipeline('ensemble_20260201')
    X = final_df.select(bf["features"])
    new_predictions = saved['ensemble'].predict_proba(X.to_numpy())
    threshold=saved['threshold']
    pred_upcoming_games=final_df.select([
        "game_id", "home_team", "guest_team",
        "is_home_win"
    ]).with_columns([
        pl.Series(
            "proba",
            new_predictions
        )
    ]).group_by(["game_id", "home_team", "guest_team", "is_home_win"]).agg([
        pl.col("proba").mean()
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ]).join(
        season_schedule[["game_id", "date"]],
        on="game_id"
    ).with_columns(
        pl.col("date").cast(pl.String)
    )
    return


if __name__ == "__main__":
    app.run()
