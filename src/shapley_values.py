import marimo

__generated_with = "0.19.7"
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
    from elo_rating import elo_season
    from ml_pipeline import load_pipeline
    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_month_data
    from google.cloud import secretmanager
    from shap_analysis import calculate_shap_ensemble, plot_shap_summary
    return (
        create_client,
        features,
        fetch_entire_table,
        json,
        load_pipeline,
        pl,
        secretmanager,
        tqdm,
    )


@app.cell
def _(pl):
    to_insert = pl.read_csv("../arenas_geocoded.csv", separator=";").with_columns([
        pl.col("latitude").str.replace(",", ".").cast(pl.Float64),
        pl.col("longitude").str.replace(",", ".").cast(pl.Float64)
    ]).to_dicts()
    return (to_insert,)


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
def _(supabase, to_insert):
    supabase.table("location").insert(to_insert).execute()
    return


@app.cell
def _(fetch_entire_table, supabase):

    schedule = fetch_entire_table(supabase, "schedule")
    games = fetch_entire_table(supabase, "boxscore")
    elo = fetch_entire_table(supabase, "elo")
    schedule = schedule.join(
        games,
        on="game_id"
    )
    return elo, games, schedule


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
def _(df, json, pl, schedule):
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
    with open("bf.json", "r") as f:
        bf = json.load(f)
    X_train = train.select(bf["features"])
    X_test = test.select(bf["features"])
    X_val = val.select(bf["features"])
    y_train = train.select("is_home_win")
    y_test = test.select("is_home_win")
    y_val = val.select("is_home_win")
    return (X_train,)


@app.cell(disabled=True)
def _(X_train, load_pipeline):
    import importlib
    import shap_analysis
    importlib.reload(shap_analysis)
    saved = load_pipeline('ensemble_model_v2')
    shap_values, _, feature_names = shap_analysis.calculate_shap_ensemble(
        saved, 
        X_train,
        use_kernel=False
    )
    return feature_names, shap_analysis, shap_values


@app.cell(disabled=True)
def _(X_train, feature_names, shap_analysis, shap_values):
    shap_analysis.plot_shap_summary(
        shap_values, X_train, feature_names, 
        model_name='Ensemble'
    )
    return


@app.cell
def _(pl):
    import uuid

    # 1. Create the DataFrame from your SHAP values
    # shap_df = pl.from_numpy(shap_values, schema=feature_names)
    shap_df = pl.read_csv("shapley_values.csv")

    # 2. Generate a list of UUIDs (one for each row) and add as a new column
    # We use .with_columns to insert the 'row_id' at the start
    shap_df = shap_df.with_columns(
        row_id = pl.Series([str(uuid.uuid4()) for _ in range(len(shap_df))])
    )

    # 3. Save to CSV
    shap_df.write_csv("shapley_values.csv")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Todos
    #### Feature Engineering
    - Check whether I added a column to show how much the Elo of a team changed over the past 7 games -> Completed
    - Scrape game locations -> Add travel distance last game + last 7 games. Check location for bubble games -> Completed
    - Add boolean flag for playoff start -> Completed

    #### ML Modelling
    - Run the modelling pipeline with the expected team values
    - Add a boolean parameter to the pipeline that decides whether more recent observations should be more influential -> Added
    - Train the model until beginning of the season; train again after each month + make predictions
    - Turn this into an automated pipeline that trains the model first of the month.
    - At prediction for next games, pull all the injury reports for each team and then make predictions based on all possible player combinations (i.e. player 1 plays, 2 does not, 1 plays, 2 plays, ...)

    #### Database Topics
    - Location table. Arena name + coordinates -> Completed
    - Playoff dates table -> Completed
    - Injury Report Table

    #### UI
    - Elo over time. Chart / Table -> Completed
    - Shapley DB connection

    #### Server Automation
    - Add daily team record script to server
    - Add player boxscore to server -> added to daily bbref script
    - Scripts run as follows:
        - Pull the latest data (boxscores, injury reports, team rosters)
        - Update the recent predictions (add who won)
        - Check which games have to be predicted
        - For each game, calculate the expected team stats based on each pot. combination of players on injury report (but only include players that player over a certain amount of mpg)
        - Average the predictions for the final prediction of that game


    Update API Keys
    """)
    return


@app.cell
def _():
    # retrain the model up until 1 month ago (i.e. always retrain at the end of the month) - check performance
    # then train the model again and use weights to more heavily favor recent observations. add boolean parameter to pipeline function
    # feature development for players
    # once all features are there, see which players are/may not be playing and built multiple predictions for the same game
    # remove duplicates in player-boxscore
    # todo schedule nochmal ziehen für location - dann distanzen berechnen. travel since last game, travel last 7 days
    # todo - feature glossary durchschauen
    # todo - beeswarm plot überprüfen
    # todo - team record in der app -> auf server ziehen
    # todo - shapley value via supabase
    return


if __name__ == "__main__":
    app.run()
