from itertools import product

from supabase_helper import fetch_entire_table, fetch_filtered_table

import polars as pl

import features
import bbref

def update_team_records(supabase):
    schedule_current_season, games_current_season = fetch_filtered_table(
        supabase, "schedule", "boxscore", "season_id", "game_id"
    )
    schedule_current_season = schedule_current_season.join(
        games_current_season,
        on="game_id",
        how="left"
    )
    team_records = fetch_entire_table(supabase, "team-record")
    game_results = schedule_current_season.with_columns(
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
    ).select(["game_id", "date", "home_team", "guest_team", "season_id", "is_home_win", "home_abbrev", "guest_abbrev"])
    home_side = game_results.select([
        pl.col("game_id"),
        pl.col("season_id"),
        pl.col("date"),
        pl.col("home_abbrev").alias("abbrev"),
        pl.col("home_team").alias("team"),
        pl.col("is_home_win").alias("win")
    ])
    guest_side = game_results.select([
        pl.col("game_id"),
        pl.col("season_id"),
        pl.col("date"),
        pl.col("guest_abbrev").alias("abbrev"),
        pl.col("guest_team").alias("team"),
        ~pl.col("is_home_win").alias("win")
    ])
    record_df = pl.concat([home_side, guest_side]).sort(
        ["team", "season_id", "date", "game_id"]
    ).with_columns([
        pl.col("win").cast(pl.Int32)
        .cum_sum()
        .shift(1)
        .fill_null(0)
        .over(["team", "season_id"])
        .alias("wins_before"),

        pl.col("win").not_().cast(pl.Int32)
        .cum_sum()
        .shift(1)
        .fill_null(0)
        .over(["team", "season_id"])
        .alias("losses_before")
    ]).filter(
        ~(
                (pl.col("win").is_null()) &
                (pl.col("wins_before") == 0) & (pl.col("losses_before") == 0)
        )
    ).with_columns(
        pl.concat_str(
            pl.col("game_id"), pl.lit("_"), pl.col("abbrev")
        ).alias("id")
    )
    new_inserts = record_df.filter(
        ~pl.concat_str(pl.col("game_id"), pl.col("team")).is_in(
            team_records.with_columns(pl.concat_str(pl.col("game_id"), pl.col("team")))["game_id"].to_list()
        )
    ).drop("abbrev")
    if new_inserts.shape[0] > 0:
        supabase.table('team-record').insert(new_inserts.drop(["date", "win"]).to_dicts()).execute()
        print("Updated team records")
    else:
        print("No new team records")

def update_existing_predictions(supabase):
    predictions = fetch_entire_table(supabase, "predictions")
    schedule_current_season, games_current_season = fetch_filtered_table(
        supabase, "schedule", "boxscore", "season_id", "game_id"
    )
    schedule_current_season = schedule_current_season.join(
        games_current_season,
        on="game_id",
        how="left"
    ).with_columns(
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
    )
    to_update = predictions.filter(
        pl.col("is_home_win").is_null()
    ).drop("is_home_win").join(
        schedule_current_season[["game_id", "is_home_win"]],
        on="game_id"
    ).filter(
        pl.col("is_home_win").is_not_null()
    ).to_dicts()
    if len(to_update) > 0:
        print("Updating newest predictions")
        supabase.table("predictions").upsert(to_update).execute()
    else:
        print("No new predictions")

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

def upcoming_game_data(supabase):
    current_season, current_boxscore = fetch_filtered_table(
        supabase, "schedule", "boxscore", "season_id", "game_id"
    )
    games_to_predict = upcoming_games(current_season, current_boxscore)
    current_predictions = fetch_entire_table(supabase, "predictions")
    games_to_predict = games_to_predict.filter(
        ~pl.col("game_id").is_in(current_predictions["game_id"].to_list()),
    )
    if games_to_predict.shape[0] == 0:
        print("No predictions need to be made")
        return
    current_season, current_player_boxscore = fetch_filtered_table(
        supabase, "schedule", "player-boxscore", "season_id", "game_id"
    )
    elo = fetch_entire_table(supabase, "elo")
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
    season_schedule = features.add_rest_features(season_schedule)
    season_schedule = features.add_clutch_features(season_schedule)
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
    season_boxscores = season_boxscore.partition_by("player_id")
    season_player_stats = [features.player_season_stats(boxscore) for boxscore in season_boxscores]
    driver = bbref.start_driver()
    team_ids = games_to_predict["home_abbrev"].to_list() + games_to_predict["guest_abbrev"].to_list()
    current_rosters = [bbref.scrape_current_roster(team_id, driver) for team_id in team_ids]
    available_rosters = pl.concat(current_rosters).filter(
        (pl.col("injury").is_null()) |
        (pl.col("injury").str.starts_with("Day"))
    ).with_columns(
        pl.col("injury").is_not_null()
    ).partition_by("team_id")
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
    all_potential_stats = pl.concat([pl.concat(stats) for stats in potential_stats]).drop(
        "game_id"
    )
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
    elo = elo.with_columns(
        pl.col("elo_after").shift(7).over(pl.col("team_id")).alias("elo_7_games_before")
    ).with_columns(
        (pl.col("elo_after") - pl.col("elo_7_games_before")).alias("elo_change_7_absolute"),
        (pl.col("elo_after") / pl.col("elo_7_games_before") - 1).alias("elo_change_7_relative")
    ).drop("elo_7_games_before")
    current_elo = elo.group_by("team_id").tail(1).select(
        "team_id", "elo_after", "elo_change_7_absolute", "elo_change_7_relative"
    )
    final_df = joined.join(
        all_potential_stats.rename(
            lambda c: f"{c}_home"
        ),
        left_on=["game_id", "home_team"],
        right_on=["game_id_home", "team_id_home"]
    ).join(
        all_potential_stats.rename(
            lambda c: f"{c}_guest"
        ),
        left_on=["game_id", "guest_team"],
        right_on=["game_id_guest", "team_id_guest"]
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
    return final_df

def predict_upcoming_games(df, model, features, threshold, schedule):
    X = df.select(features)
    predictions = model.predict_proba(X.to_numpy())
    game_predictions=df.select([
        "game_id", "home_team", "guest_team",
        "is_home_win"
    ]).with_columns([
        pl.Series(
            "proba",
            predictions
        )
    ]).group_by(["game_id", "home_team", "guest_team", "is_home_win"]).agg([
        pl.col("proba").mean()
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ]).join(
        schedule[["game_id", "date"]],
        on="game_id"
    ).with_columns(
        pl.col("date").cast(pl.String)
    )
    return game_predictions