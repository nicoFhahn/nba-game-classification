import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import bbref
    import elo_rating

    from google.cloud import secretmanager
    from supabase import create_client

    import importlib
    return bbref, create_client, elo_rating, importlib, json, secretmanager


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
def _(bbref, importlib, supabase):
    importlib.reload(bbref)
    bbref.update_current_month(
        supabase,
        bbref.start_driver()
    )
    return


@app.cell
def _(bbref, importlib, supabase):
    importlib.reload(bbref)
    bbref.scrape_missing_team_boxscores(supabase)
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_player_boxscores(supabase)
    return


@app.cell
def _(elo_rating, supabase):
    elo_rating.elo_update(supabase)
    return


@app.cell
def _(json, supabase):
    from supabase_helper import fetch_entire_table
    from ml_pipeline import load_pipeline
    import polars as pl
    import features
    print("Loading prediction data")
    schedule = fetch_entire_table(supabase, "schedule")
    games = fetch_entire_table(supabase, "boxscore")
    elo = fetch_entire_table(supabase, "elo")
    predictions = fetch_entire_table(supabase, "predictions")
    schedule = schedule.join(
        games,
        on="game_id",
        how="left"
    ).with_columns(
        pl.col("date").str.to_date()
    )
    print("Prediction data loaded")
    newest_predictions = predictions.filter(
        pl.col("is_home_win").is_null()
    )
    to_update = newest_predictions.drop("is_home_win").join(
        games.with_columns([
            (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
        ]).select([
            "game_id", "is_home_win"
        ]),
        on="game_id"
    ).to_dicts()
    if len(to_update) > 0:
        print("Updating newest predictions")
        supabase.table("predictions").upsert(to_update).execute()
    if newest_predictions.shape[0] > 0:
        cutoff_date = newest_predictions["date"].str.to_date().max()
    else:
        cutoff_date = predictions["date"].str.to_date().max()
    season_id = schedule.filter(
        pl.col("date") == cutoff_date
    )["season_id"][0]
    season_schedule = schedule.filter(
        pl.col("season_id") == season_id
    ).sort("date").select([
        "game_id", "date", "home_team", "guest_team", "pts_home", "pts_guest"
    ]).with_columns([
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
    ])
    season_games = season_schedule.select(
        "game_id", "date", "home_team", "guest_team"
    ).join(
        games,
        on="game_id",
        how="left"
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
    current_elo = elo.group_by("team_id").tail(1).select("team_id", "elo_after")
    upcoming_games = joined.filter(
        (pl.col("is_home_win").is_null()) &
        (pl.col("wins_this_year_home_team").is_not_null()) &
        (pl.col("wins_this_year_guest_team").is_not_null())
    ).join(
        current_elo.rename({
            "elo_after": "elo_home"
        }),
        left_on="home_team",
        right_on="team_id"
    ).join(
        current_elo.rename({
            "elo_after": "elo_guest"
        }),
        left_on="guest_team",
        right_on="team_id"
    )
    print("Loading ML Pipeline")
    with open("bf.json", "r") as f:
        bf = json.load(f)
    saved = load_pipeline('ensemble_model_v2')
    X = upcoming_games.select(bf["features"])
    new_predictions = saved['ensemble'].predict_proba(X.to_numpy())
    threshold=saved['threshold']
    pred_upcoming_games=upcoming_games.select([
        "game_id", "home_team", "guest_team",
        "is_home_win"
    ]).with_columns([
        pl.Series(
            "proba",
            new_predictions
        )
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ]).join(
        season_schedule[["game_id", "date"]],
        on="game_id"
    ).with_columns(
        pl.col("date").cast(pl.String)
    ).filter(
    ~pl.col("game_id").is_in(
        ~pl.col("game_id").is_in(predictions["game_id"].to_list())
    ))
    print("Adding new predictions")
    #supabase.table("predictions").insert(
    #    pred_upcoming_games.to_dicts()
    #).execute()
    return newest_predictions, pl, predictions


@app.cell
def _(newest_predictions, pl, predictions):
    newest_predictions.filter(
        ~pl.col("game_id").is_in(predictions["game_id"].to_list())
    )
    return


if __name__ == "__main__":
    app.run()
