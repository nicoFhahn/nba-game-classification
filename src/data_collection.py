import os
import polars as pl
import numpy as np
import random
from datetime import date
from nba_api.stats import endpoints
from tqdm import tqdm
from time import sleep

def daily_games(
        game_date: date
) -> pl.DataFrame:
    """
    Creates a dataframe containing all games of a given date
    :param game_date: The date for which to scrape the data
    :return: dataframe with date, game id and team ids
    """
    try:
        games = endpoints.scoreboardv2.ScoreboardV2(game_date=game_date)
        sleep(random.uniform(2, 5))
        game_list = games.get_dict()["resultSets"][0]["rowSet"]
        df = pl.DataFrame(
            data={
                "game_id": [game[2] for game in game_list],
                "home_team_id": [game[6] for game in game_list],
                "away_team_id": [game[7] for game in game_list]
            }
        ).with_columns([
            pl.lit(game_date).alias("date"),
            pl.col("home_team_id").cast(pl.String),
            pl.col("away_team_id").cast(pl.String)
        ])[["date", "game_id", "home_team_id", "away_team_id"]]
    except Exception as e:
        print(e)
        sleep(15)
        df = daily_games(game_date)
    return df

def boxscore(
        game_id: str
) -> pl.DataFrame:
    """
    Creates a dataframe containing a lot of different boxscore metrics for a given game
    :param game_id: The id of the game
    :return: dataframe with gameid + the boxscore details
    """
    try:
        bs_traditional = endpoints.boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_traditional = pl.concat([
            pl.DataFrame(
                bs_traditional.get_dict()["boxScoreTraditional"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_traditional.get_dict()["boxScoreTraditional"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_advanced = endpoints.boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_advanced = pl.concat([
            pl.DataFrame(
                bs_advanced.get_dict()["boxScoreAdvanced"]["homeTeam"]["statistics"]
            ).rename(
                lambda c: c + "_home"),
            pl.DataFrame(
                bs_advanced.get_dict()["boxScoreAdvanced"]["awayTeam"]["statistics"]
            ).rename(
                lambda c: c + "_away")
        ], how="horizontal")
        bs_four_factors = endpoints.boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_four_factors = pl.concat([
            pl.DataFrame(
                bs_four_factors.get_dict()["boxScoreFourFactors"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_four_factors.get_dict()["boxScoreFourFactors"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_hustle = endpoints.boxscorehustlev2.BoxScoreHustleV2(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_hustle = pl.concat([
            pl.DataFrame(
                bs_hustle.get_dict()["boxScoreHustle"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_hustle.get_dict()["boxScoreHustle"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_misc = endpoints.boxscoremiscv3.BoxScoreMiscV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_misc = pl.concat([
            pl.DataFrame(
                bs_misc.get_dict()["boxScoreMisc"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_misc.get_dict()["boxScoreMisc"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_player_track = endpoints.boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_player_track = pl.concat([
            pl.DataFrame(
                bs_player_track.get_dict()["boxScorePlayerTrack"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_player_track.get_dict()["boxScorePlayerTrack"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_scoring = endpoints.boxscorescoringv3.BoxScoreScoringV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_scoring = pl.concat([
            pl.DataFrame(
                bs_scoring.get_dict()["boxScoreScoring"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_scoring.get_dict()["boxScoreScoring"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        bs_usage = endpoints.boxscoreusagev3.BoxScoreUsageV3(game_id=game_id)
        sleep(random.uniform(2, 5))
        bs_usage = pl.concat([
            pl.DataFrame(
                bs_usage.get_dict()["boxScoreUsage"]["homeTeam"]["statistics"]
            ).rename(lambda c: c + "_home"),
            pl.DataFrame(
                bs_usage.get_dict()["boxScoreUsage"]["awayTeam"]["statistics"]
            ).rename(lambda c: c + "_away")
        ], how="horizontal")
        df = pl.concat([
            bs_traditional,
            bs_advanced.drop([
                "minutes_home", "minutes_away"
            ]),
            bs_four_factors.drop([
                "minutes_home", "minutes_away", "effectiveFieldGoalPercentage_home",
                "effectiveFieldGoalPercentage_away", "offensiveReboundPercentage_home",
                "offensiveReboundPercentage_away"
            ]),
            bs_hustle.drop([
                "minutes_home", "minutes_away", "points_home", "points_away"
            ]),
            bs_misc.drop([
                "minutes_home", "minutes_away", "blocks_home", "blocks_away", "foulsPersonal_home",
                "foulsPersonal_away"
            ]),
            bs_player_track.drop(
                ["minutes_home", "minutes_away", "assists_home", "assists_away"]
            ),
            bs_scoring.drop([
                "minutes_home", "minutes_away"
            ]),
            bs_usage.drop([
                "minutes_home", "minutes_away", "usagePercentage_home", "usagePercentage_away"
            ]),
        ], how="horizontal")
        df = df.with_columns([
            pl.lit(game_id).alias("game_id")
        ])
    except Exception as e:
        print(e)
        df = boxscore(game_id)
    return df

def team_details(
        team_id: str
) -> pl.DataFrame:
    """
    Gets team details for a given team
    :param team_id: id of the team
    :return: dataframe with team name, id, and arena capacity
    """
    details = endpoints.teamdetails.TeamDetails(team_id=team_id).get_dict()["resultSets"][0]["rowSet"][0]
    details = {
        "team_name": f"{details[4]} {details[2]}",
        "team_id": details[0],
        "arena_capacity": details[6]
    }
    sleep(random.uniform(2, 5))
    details = pl.DataFrame(details).with_columns(pl.col("arena_capacity").cast(pl.Int64))
    return details

def season(
        start_date: date,
        end_date: date,
        file_year: int,
        folder:str="data"
) -> pl.DataFrame:
    """
    Creates a dataframe containing all the games between two dates
    :param start_date:
    :param end_date:
    :param folder:
    :return:
    """
    date_list = np.arange(start_date, end_date)
    schedule_list = [daily_games(game_date) for game_date in tqdm(date_list)]
    filename = f"schedule_{file_year}.parquet"
    if filename in os.listdir(folder):
        schedule_df = pl.concat([
            pl.read_parquet(os.path.join(folder, filename)),
            pl.concat(schedule_list)
        ]).unique("game_id").sort("date")
    else:
        schedule_df = pl.concat(schedule_list)
    schedule_df.write_parquet(
        os.path.join(
            folder,
            filename
        )
    )
    games_list = []
    scraped_games = []
    filename = f"game_list_{file_year}.parquet"
    df_exists = False
    if filename in os.listdir(folder):
        games_df = pl.read_parquet(os.path.join(folder, filename))
        scraped_games = games_df["game_id"].to_list()
        df_exists = True
    # written like this cause sometimes it took ages to get the data and so I just wanted it to start at the last
    # game i missed
    for game_id in tqdm(schedule_df["game_id"]):
        if game_id in scraped_games:
            continue
        else:
            game_boxscore = boxscore(game_id)
            games_list.append(game_boxscore)
            if df_exists:
                current_games_df = pl.concat(games_list)
                pl.concat([
                    games_df, current_games_df
                ]).write_parquet(
                    os.path.join(folder, filename)
                )
            else:
                games_df = pl.concat(games_list)
                games_df.write_parquet(
                    os.path.join(folder, filename)
                )
            scraped_games.append(game_id)