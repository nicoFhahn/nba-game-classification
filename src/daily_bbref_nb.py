import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import calendar
    import json
    import time
    from datetime import date, timedelta

    import polars as pl
    from supabase import Client, create_client
    from tqdm import tqdm

    import features
    from bbref import scrape_game, scrape_month, scrape_season, start_driver
    from elo_rating import elo_season
    from ml_pipeline import load_pipeline
    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_month_data
    from google.cloud import secretmanager
    return (
        calendar,
        create_client,
        date,
        elo_season,
        features,
        fetch_entire_table,
        fetch_filtered_table,
        fetch_month_data,
        json,
        load_pipeline,
        pl,
        scrape_game,
        scrape_month,
        secretmanager,
        start_driver,
        time,
        timedelta,
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


@app.cell
def _(
    calendar,
    date,
    fetch_entire_table,
    fetch_month_data,
    supabase,
    timedelta,
):
    season_1 = fetch_entire_table(
        supabase, "schedule"
    ).sort("game_id")
    d = date.today() - timedelta(days=(date.today().day == 1))
    year = date.today().year + (date.today().month >= 10)
    month_number = d.month
    month_name = calendar.month_name[month_number].lower()
    schedule_current_month = fetch_month_data(
        supabase, "schedule", d, date_column='date', page_size=1000
    )
    print("fetched current month from supabase")
    return month_name, schedule_current_month, year


@app.cell
def _(
    month_name,
    pl,
    schedule_current_month,
    scrape_month,
    start_driver,
    supabase,
    year,
):
    link = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month_name}.html"
    driver_1 = start_driver()
    current_month = scrape_month(link, driver_1)
    print("fetched current month from bbref")
    current_month=current_month.rename({
        "Date": "date",
        "Home": "home_team",
        "Visitor": "guest_team",
        "Box_Score_URL": "game_url"
    }).with_columns([
        pl.lit(year).alias("season_id"),
        pl.lit(False).alias("is_scraped")
    ])[schedule_current_month.columns]
    response = (
        supabase
        .table('schedule')
        .delete()
        .in_('game_id', schedule_current_month["game_id"].to_list())
        .execute()
    )
    print("deleted old data")
    supabase.table('schedule').insert(
        current_month.with_columns(
            pl.col("date").cast(pl.String)
        ).to_dicts()
    ).execute()
    print("inserted new data")
    driver_1.quit()
    return


@app.cell
def _(fetch_entire_table, pl, supabase):
    season_2 = fetch_entire_table(supabase, "schedule").sort("game_id")
    games_1 = fetch_entire_table(supabase, "boxscore").sort("game_id")
    missing_games = season_2.filter(
        (~pl.col("game_id").is_in(games_1["game_id"].to_list())) &
        (pl.col("game_url") != "")
    )
    return (missing_games,)


@app.cell
def _(missing_games, scrape_game, start_driver, supabase, time, tqdm):
    if missing_games.shape[0] > 0:
        driver_2 = start_driver()

        for i, row in enumerate(tqdm(missing_games.to_dicts(), ncols=100)):

            # Restart driver every 50 iterations
            if i > 0 and i % 50 == 0:
                driver_2.quit()
                time.sleep(2)
                driver_2 = start_driver()

            scrape_game(
                row["game_url"],
                row["home_abbrev"],
                row["guest_abbrev"],
                row["game_id"],
                driver_2,
                supabase
            )

        driver_2.quit()
    return


@app.cell
def _(elo_season, fetch_filtered_table, missing_games, pl, supabase):
    if missing_games.shape[0] > 0:
        print("Fetching data for elo score calculation")
        schedule_current_season, games_current_season = fetch_filtered_table(
            supabase, "schedule", "boxscore", "season_id", "game_id"
        )
        schedule_current_season, elo_current_season = fetch_filtered_table(
            supabase, "schedule", "elo", "season_id", "game_id"
        )
        schedule_last_season, elo_last_season = fetch_filtered_table(
            supabase, "schedule", "elo", "season_id", "game_id",
            schedule_current_season["season_id"][0] - 1
        )
        games_w_schedule_current_season = schedule_current_season.select([
            "game_id", "date", "season_id", "home_team", "guest_team"
        ]).join(
            games_current_season.select([
                "game_id", "pts_home", "pts_guest"
            ]),
            on="game_id"
        ).with_columns([
            pl.col("date").str.to_date()
        ]).rename({
            "home_team": "home_team_id",
            "guest_team": "guest_team_id",
        })
        elo_w_schedule_last_season=elo_last_season.join(
            schedule_last_season[["game_id", "date"]],
            on="game_id"
        ).with_columns(
            pl.col("date").str.to_date()
        )
        elo_current_season_updated = elo_season(
            games_w_schedule_current_season,
            elo_w_schedule_last_season
        ).filter(
            ~pl.col("game_id").is_in(elo_current_season["game_id"].to_list())
        ).drop("date").to_dicts()
        if len(elo_current_season_updated) > 0:
            print("Updating elo")
            supabase.table("elo").insert(elo_current_season_updated).execute()
        else:
            print("No Elo Update required")
    else:
        print("No missing games - no elo update")
    return


@app.cell
def _(
    features,
    fetch_entire_table,
    json,
    load_pipeline,
    missing_games,
    pl,
    supabase,
):
    if missing_games.shape[0] > 0:
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
            newest_predictions["game_id"].to_list()
        ))
        print("Adding new predictions")
        supabase.table("predictions").insert(
            pred_upcoming_games.to_dicts()
        ).execute()
    else:
        print("No missing games - No predictions")
    return


if __name__ == "__main__":
    app.run()
