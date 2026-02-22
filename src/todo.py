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
        features,
        fetch_entire_table,
        fetch_filtered_table,
        json,
        load_pipeline,
        pl,
        secretmanager,
        tqdm,
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


@app.function
def upcoming_games(schedule, boxscore):
    remaining_schedule = schedule.join(boxscore, on="game_id", how="anti")
    team_timeline = remaining_schedule.select(
        ["game_id", "home_team", "guest_team"]
    ).unpivot(index="game_id", variable_name="side", value_name="team")

    next_game_per_team = team_timeline.sort("game_id").group_by("team").first()
    upcoming_games = remaining_schedule.join(
        next_game_per_team.select(["team", "game_id"]),
        left_on=["home_team", "game_id"],
        right_on=["team", "game_id"],
        how="inner"
    ).join(
        next_game_per_team.select(["team", "game_id"]),
        left_on=["guest_team", "game_id"],
        right_on=["team", "game_id"],
        how="inner"
    )
    return upcoming_games


@app.cell
def _(fetch_entire_table, fetch_filtered_table, pl, supabase):
    current_season, current_boxscore = fetch_filtered_table(
        supabase, "schedule", "boxscore", "season_id", "game_id"
    )
    current_season, current_player_boxscore = fetch_filtered_table(
        supabase, "schedule", "player-boxscore", "season_id", "game_id"
    )
    games_to_predict = upcoming_games(current_season, current_boxscore)
    playoffs = fetch_entire_table(supabase, "playoffs")
    locations = fetch_entire_table(supabase, "location")
    current_season = current_season.join(
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
    return (
        current_boxscore,
        current_player_boxscore,
        current_season,
        games_to_predict,
    )


@app.cell
def _(
    current_boxscore,
    current_player_boxscore,
    current_season,
    features,
    games_to_predict,
    pl,
):
    season_schedule = current_season.join(
        current_boxscore,
        on="game_id",
        how="left"
    ).sort("date").select([
        "game_id", "date", "home_team", "guest_team", "pts_home", "pts_guest", "is_playoff_game",
        "latitude", "longitude"
    ]).with_columns([
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
    ])
    season_games = season_schedule.select(
        "game_id", "date", "home_team", "guest_team"
    ).join(
        current_boxscore,
        on="game_id",
        how="left"
    )
    teams = list(set(games_to_predict["home_team"])) + list(set(games_to_predict["guest_team"]))
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
    ).filter(
        pl.col("game_id").is_in(games_to_predict["game_id"].to_list())
    )
    season_boxscore = current_season[["game_id", "season_id"]].join(
        current_player_boxscore,
        on="game_id",
        how="left"
    )
    return joined, season_boxscore, season_schedule


@app.cell
def _(features, games_to_predict, season_boxscore, tqdm):
    import bbref
    # import importlib
    # importlib.reload(bbref)
    season_boxscores = season_boxscore.partition_by("player_id")
    season_player_stats = [features.player_season_stats(boxscore) for boxscore in season_boxscores]
    # todo: hier muss ich mir jetzt das aktuelle roster ziehen und alle partitions der avail. players bauen - dann für diese die  exp. stats calculaten
    driver = bbref.start_driver()
    team_ids = games_to_predict["home_abbrev"].to_list() + games_to_predict["guest_abbrev"].to_list()
    current_rosters = [bbref.scrape_current_roster(team_id, driver) for team_id in tqdm(team_ids)]
    return current_rosters, season_player_stats


@app.cell
def _(current_rosters, pl):
    available_rosters = pl.concat(current_rosters).filter(
        (pl.col("injury").is_null()) |
        (pl.col("injury").str.starts_with("Day"))
    ).with_columns(
        pl.col("injury").is_not_null()
    ).partition_by("team_id")
    return (available_rosters,)


@app.cell
def _(pl):
    from itertools import product
    def get_available_rosters(rosters: list[pl.DataFrame]) -> list[list[pl.DataFrame]]:
        """
        For each team roster, generate all possible player combinations
        based on injured players (each injured player may or may not play).

        Args:
            rosters: List of Polars DataFrames, one per team, with a boolean "injury" column.

        Returns:
            A list (one entry per team) of lists of DataFrames,
            where each DataFrame is one possible available roster.
        """
        available_rosters = []

        for roster in rosters:
            healthy = roster.filter(pl.col("injury") == False)
            injured = roster.filter(pl.col("injury") == True)

            injured_players = injured.to_dicts()

            # Generate all subsets of injured players (2^n combinations)
            team_combinations = []
            for inclusion in product([True, False], repeat=len(injured_players)):
                playing_injured = [
                    player for player, plays in zip(injured_players, inclusion) if plays
                ]

                if playing_injured:
                    playing_injured_df = pl.DataFrame(playing_injured, schema=roster.schema)
                    possible_roster = pl.concat([healthy, playing_injured_df])
                else:
                    possible_roster = healthy.clone()

                team_combinations.append(possible_roster)

            available_rosters.append(team_combinations)

        return available_rosters

    return (get_available_rosters,)


@app.cell
def _(
    available_rosters,
    features,
    get_available_rosters,
    pl,
    season_player_stats,
):
    potential_rosters = get_available_rosters(available_rosters)
    season_player_stats_df = pl.concat(season_player_stats)
    potential_stats = []

    for pr in potential_rosters:
        ts = []
        for roster in pr:
            temp = season_player_stats_df.filter(
                pl.col("player_id").is_in(roster["player_id"])
            ).sort("game_id").group_by("player_id").tail(1).with_columns(
                pl.lit("temp").alias("game_id")
            )
        
            expected = features.calculate_expected_team_stats(temp).tail(1)
        
            if not any(expected.equals(existing) for existing in ts):
                ts.append(expected)
            
        potential_stats.append(ts)
    return (potential_stats,)


@app.cell
def _(games_to_predict, pl, potential_stats):
    all_potential_stats = pl.concat([pl.concat(stats) for stats in potential_stats]).drop("num_players", "total_minutes_check", "game_id")
    all_potential_stats = pl.concat([
        games_to_predict.select(["game_id", "home_team"]).join(
            all_potential_stats,
            left_on="home_team",
            right_on="team_id"
        ).rename({"home_team": "team_id"}),
        games_to_predict.select(["game_id", "guest_team"]).join(
            all_potential_stats,
            left_on="guest_team",
            right_on="team_id"
        ).rename({"guest_team": "team_id"}),
    ])
    return (all_potential_stats,)


@app.cell
def _(fetch_entire_table, pl, supabase):
    elo = fetch_entire_table(supabase, "elo").with_columns(
        pl.col("elo_after").shift(7).over(pl.col("team_id")).alias("elo_7_games_before")
    ).with_columns(
        (pl.col("elo_after") - pl.col("elo_7_games_before")).alias("elo_change_7_absolute"),
        (pl.col("elo_after") / pl.col("elo_7_games_before") - 1).alias("elo_change_7_relative")
    ).drop("elo_7_games_before")
    current_elo = elo.group_by("team_id").tail(1).select("team_id", "elo_after", "elo_change_7_absolute", "elo_change_7_relative")
    return (current_elo,)


@app.cell
def _(all_potential_stats, current_elo, joined):
    final_df = joined.join(
        all_potential_stats.rename(
            lambda c: f"{c}_home_team"
        ),
        left_on=["game_id", "home_team"],
        right_on=["game_id_home_team", "team_id_home_team"]
    ).join(
        all_potential_stats.rename(
            lambda c: f"{c}_guest_team"
        ),
        left_on=["game_id", "guest_team"],
        right_on=["game_id_guest_team", "team_id_guest_team"]
    ).join(
        current_elo.rename({
            "elo_after": "elo_home",
            "elo_change_7_absolute": "elo_change_7_absolute_home",
            "elo_change_7_relative": "elo_change_7_relative_home"
        }),
        left_on="home_team",
        right_on="team_id"
    ).join(
        current_elo.rename({
            "elo_after": "elo_guest",
            "elo_change_7_absolute": "elo_change_7_absolute_guest",
            "elo_change_7_relative": "elo_change_7_relative_guest"
        }),
        left_on="guest_team",
        right_on="team_id"
    )
    return (final_df,)


@app.cell
def _(json):
    with open("best_features_ensemble_20260201.json", "r") as f:
        bf = json.load(f)
    return (bf,)


@app.cell
def _(bf, final_df, load_pipeline, pl, season_schedule):
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
    return (pred_upcoming_games,)


@app.cell
def _(pred_upcoming_games):
    pred_upcoming_games
    return


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
