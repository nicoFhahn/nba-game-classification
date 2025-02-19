from datetime import date, timedelta
from typing import List

import os
import polars as pl
import polars.selectors as cs

def load_schedule(
    all_star_date: date,
    play_in_start: date,
    play_in_end: date,
    filename_schedule: str,
    filename_games: str,
    folder: str = "data"
) -> pl.DataFrame:
  """
  Loads the schedule of a season
  :folder: folder where the data is saved
  :filename_schedule: name of the file containing the schedule
  :filename_games: name of the file containing the games
  :all_star_date: date the all star game takes place
  :play_in_start: date the nba play in begins
  :play_in_end: date the nba play in ends
  """
  schedule_df = pl.read_parquet(os.path.join(folder, filename_schedule))
  schedule_df = schedule_df.filter(
      pl.col("date") != all_star_date
  ).with_columns(
      pl.when(
          pl.col("date") < play_in_start
      ).then(
          pl.lit("Regular Season")
      ).otherwise(
          pl.when(
              pl.col("date") > play_in_end
          ).then(
              pl.lit("Playoffs")
          ).otherwise(
              pl.lit("Play-In")
          )
      ).alias("game_type")
  )
  games_df = pl.read_parquet(os.path.join(folder, filename_games))
  df = schedule_df.unique().join(
      games_df,
      on="game_id",
      how="inner",
      validate="1:1"
  )
  return df

def get_team_schedules(
        df: pl.DataFrame
) -> List[pl.DataFrame]:
    team_ids = set(df["home_team_id"])
    team_schedules = [
        df.filter(
            (pl.col("home_team_id") == team_id) |
            (pl.col("away_team_id") == team_id)
        ).sort("date")
        for team_id in team_ids
    ]
    return team_schedules

def team_boxscore(
    df: pl.DataFrame,
    team_id: int
    ) -> pl.DataFrame:
    """
    Extracts the stats/boxscore of one team (and a few stats of their opponents)
    for each game of the season
    :df: all games within one season
    :team_id: unique identifier of a team
    """
    home_df = df.filter(pl.col("home_team_id") == team_id).with_columns([
        (pl.col("points_home") > pl.col("points_away")).alias("is_win")
    ])
    away_df = df.filter(pl.col("away_team_id") == team_id).with_columns([
        (pl.col("points_home") < pl.col("points_away")).alias("is_win")
    ])
    team_stats_home_df = home_df.select(
        ~cs.matches("_away|is_win")
    ).rename(lambda c: c.replace("_home", ""))
    opp_stats_home_df = home_df.select([
        "reboundsDefensive_away", "reboundsOffensive_away",
        "reboundsTotal_away", "fieldGoalsMade_away", "fieldGoalsAttempted_away",
        "freeThrowsAttempted_away", "threePointersMade_away", "points_away",
        "turnovers_away", "possessions_away", "is_win"
    ]).rename(lambda c: c.replace("_away", "_opponent"))
    team_stats_away_df = away_df.select(
        ~cs.matches("_home|is_win")
    ).rename(lambda c: c.replace("_away", ""))
    opp_stats_away_df = away_df.select([
        "reboundsDefensive_home", "reboundsOffensive_home",
        "reboundsTotal_home", "fieldGoalsMade_home", "fieldGoalsAttempted_home",
        "freeThrowsAttempted_home", "threePointersMade_home", "points_home",
        "turnovers_home", "possessions_home", "is_win"
    ]).rename(lambda c: c.replace("_home", "_opponent"))
    home_df = pl.concat(
        [team_stats_home_df, opp_stats_home_df],
        how="horizontal"
    )
    away_df = pl.concat(
        [team_stats_away_df, opp_stats_away_df],
        how="horizontal"
    )
    df = pl.concat([home_df, away_df]).drop([
        "home_team_id", "away_team_id", "minutes"
    ]).sort("date").with_columns([
        pl.lit(team_id).cast(pl.String).alias("team_id"),
        (
            pl.col("is_win").cum_sum() / pl.col("is_win").cum_count()
        ).shift(1).alias("winning_percentage")
    ])
    return df


def team_season_stats(
    df: pl.DataFrame,
    window_sizes: list[int] = [3, 5, 10, 109]
    ) -> pl.DataFrame:
    """
    Extracts the stats of a team heading into a game
    for each game of the season
    :df: all games within one season of a specific team
    :window_sizes: windows of past games. should include 109 to include the
    stats up until one point in a season (109 as a team can play a max of 110)
    games in a season
    """
    absolute_columns = [
        "fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade",
        "threePointersAttempted", "freeThrowsMade", "freeThrowsAttempted",
        "reboundsOffensive", "reboundsDefensive", "reboundsTotal", "assists",
        "steals", "blocks", "turnovers", "foulsPersonal", "points",
        "plusMinusPoints", "estimatedPace", "pace", "pacePer40", "possessions",
        "contestedShots", "contestedShots2pt", "contestedShots3pt",
        "deflections", "chargesDrawn", "screenAssists", "screenAssistPoints",
        "looseBallsRecoveredOffensive", "looseBallsRecoveredDefensive",
        "looseBallsRecoveredTotal", "offensiveBoxOuts", "defensiveBoxOuts",
        "boxOutPlayerTeamRebounds", "boxOutPlayerRebounds", "boxOuts",
        "pointsOffTurnovers", "pointsSecondChance", "pointsFastBreak",
        "pointsPaint", "oppPointsOffTurnovers", "oppPointsSecondChance",
        "oppPointsFastBreak", "oppPointsPaint", "blocksAgainst", "foulsDrawn",
        "distance", "reboundChancesOffensive", "reboundChancesDefensive",
        "reboundChancesTotal", "touches", "secondaryAssists",
        "freeThrowAssists", "passes", "contestedFieldGoalsMade",
        "contestedFieldGoalsAttempted", "uncontestedFieldGoalsMade",
        "uncontestedFieldGoalsAttempted", "defendedAtRimFieldGoalsMade",
        "defendedAtRimFieldGoalsAttempted", "pointsMidrange2pt", "assisted2pt",
        "unassisted2pt", "assisted3pt", "unassisted3pt", "unassistedFGM"
    ]
    opponent_columns = [
        "reboundsDefensive_opponent", "reboundsOffensive_opponent",
        "reboundsTotal_opponent", "fieldGoalsMade_opponent",
        "fieldGoalsAttempted_opponent", "freeThrowsAttempted_opponent",
        "threePointersMade_opponent", "points_opponent", "possessions_opponent",
        "turnovers_opponent"
    ]
    per_100_columns = [
        "estimatedOffensiveRating", "offensiveRating",
        "estimatedDefensiveRating", "defensiveRating",
        "estimatedNetRating", "netRating"
    ]
    relative_columns = [
        "fieldGoalsPercentage", "threePointersPercentage",
        "freeThrowsPercentage", "assistPercentage", "assistToTurnover",
        "freeThrowAttemptRate", "contestedFieldGoalPercentage",
        "uncontestedFieldGoalsPercentage", "defendedAtRimFieldGoalPercentage",
        "percentageFieldGoalsAttempted2pt", "percentagePoints2pt",
        "percentagePointsMidrange2pt", "percentagePoints3pt",
        "percentagePointsFastBreak", "percentagePointsFreeThrow",
        "percentagePointsOffTurnovers", "percentagePointsPaint",
        "percentageAssisted2pt", "percentageUnassisted2pt",
        "percentageAssisted3pt", "percentageUnassisted3pt",
        "percentageAssistedFGM", "percentageUnassistedFGM"
    ]
    dropped_columns = [
        "usagePercentage", "estimatedUsagePercentage", "fieldGoalPercentage",
        "percentageFieldGoalsAttempted3pt", "percentageFieldGoalsMade",
        "percentageFieldGoalsAttempted", "percentageThreePointersMade",
        "percentageThreePointersAttempted", "percentageFreeThrowsMade",
        "percentageFreeThrowsAttempted", "percentageReboundsOffensive",
        "percentageReboundsDefensive", "percentageReboundsTotal",
        "percentageAssists", "percentageTurnovers", "percentageSteals",
        "percentageBlocks", "percentageBlocksAllowed",
        "percentagePersonalFouls", "percentagePersonalFoulsDrawn",
        "percentagePoints", "PIE", "effectiveFieldGoalPercentage",
        "trueShootingPercentage", "assistRatio", "offensiveReboundPercentage",
        "defensiveReboundPercentage", "reboundPercentage",
        "oppEffectiveFieldGoalPercentage", "oppFreeThrowAttemptRate",
        "oppOffensiveReboundPercentage", 	"estimatedTeamTurnoverPercentage",
        "turnoverRatio", "teamTurnoverPercentage", "oppTeamTurnoverPercentage"
    ]
    drop_cols = absolute_columns + per_100_columns + relative_columns + dropped_columns + opponent_columns
    df = df.sort(
        "date"
        ).with_columns(
            (
                pl.col("date") - pl.col("date").shift(1)
            ).dt.total_days().alias("days_since_last_game"),
        pl.col("is_win").shift(1).alias("has_won_last_game"),
        (
            pl.col("date").map_elements(
                lambda d: df.filter(
                    (pl.col("date") < d) &
                    (pl.col("date") >= (d - timedelta(days=7)))
              ).shape[0], return_dtype=pl.Int64
            )
        ).alias("games_last_7_days"),
        (
            pl.col("points") * pl.col("percentagePointsMidrange2pt")
        ).round(0).alias("pointsMidrange2pt"),
        (
            (pl.col("fieldGoalsMade") - pl.col("threePointersMade")) *
            pl.col("percentageAssisted2pt")
        ).round(0).alias("assisted2pt"),
        (
            (pl.col("fieldGoalsMade") - pl.col("threePointersMade")) *
            pl.col("percentageUnassisted2pt")
        ).round(0).alias("unassisted2pt"),
        (
            pl.col("threePointersMade") * pl.col("percentageAssisted3pt")
        ).round(0).alias("assisted3pt"),
        (
            pl.col("threePointersMade") * pl.col("percentageUnassisted3pt")
        ).round(0).alias("unassisted3pt"),
        (
            pl.col("fieldGoalsMade") * pl.col("percentageUnassistedFGM")
        ).round(0).alias("unassistedFGM")
    ).with_columns([
        pl.col(c).shift(1).alias(f"{c}_previous_game")
        for c in absolute_columns + per_100_columns + opponent_columns
    ]).with_columns(
        pl.col("has_won_last_game").rle_id().alias("streak")
    ).with_columns([
        pl.col("has_won_last_game").cum_count().over(
            pl.col("streak")
        ).alias("streak_length")
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            pl.col(f"{c}_previous_game").rolling_mean(
                window_size = ws, min_periods= 1 if ws >= 109 else ws
            ).alias(f"{c}_{ws}")
            for c in absolute_columns + per_100_columns + opponent_columns
        ]
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            (
                pl.col(f"fieldGoalsMade_{ws}") /
                pl.col(f"fieldGoalsAttempted_{ws}")
            ).alias(f"fieldGoalsPercentage_{ws}"),
            (
                pl.col(f"threePointersMade_{ws}") /
                pl.col(f"threePointersAttempted_{ws}")
            ).alias(f"threePointersPercentage_{ws}"),
            (
                pl.col(f"freeThrowsMade_{ws}") /
                pl.col(f"freeThrowsAttempted_{ws}")
            ).alias(f"freeThrowsPercentage_{ws}"),
            (
                pl.col(f"assists_{ws}") /
                pl.col(f"fieldGoalsMade_{ws}")
            ).alias(f"assistPercentage_{ws}"),
            (
                pl.col(f"assists_{ws}") /
                pl.col(f"turnovers_{ws}")
            ).alias(f"assistToTurnover_{ws}"),
            (
                pl.col(f"freeThrowsAttempted_{ws}") /
                pl.col(f"fieldGoalsAttempted_{ws}")
            ).alias(f"freeThrowAttemptRate_{ws}"),
            (
                pl.col(f"contestedFieldGoalsMade_{ws}") /
                pl.col(f"contestedFieldGoalsAttempted_{ws}")
            ).alias(f"contestedFieldGoalsPercentage_{ws}"),
            (
                pl.col(f"uncontestedFieldGoalsMade_{ws}") /
                pl.col(f"uncontestedFieldGoalsAttempted_{ws}")
            ).alias(f"uncontestedFieldGoalsPercentage_{ws}"),
            (
                pl.col(f"defendedAtRimFieldGoalsMade_{ws}") /
                pl.col(f"defendedAtRimFieldGoalsAttempted_{ws}")
            ).alias(f"defendedAtRimFieldGoalsPercentage_{ws}"),
            (
                1 - (
                      pl.col(f"threePointersAttempted_{ws}") /
                      pl.col(f"fieldGoalsAttempted_{ws}")
                    )
            ).alias(f"percentageFieldGoalsAttempted2pt_{ws}"),
            (
                (2 * (
                    pl.col(f"fieldGoalsMade_{ws}") -
                    pl.col(f"threePointersMade_{ws}")
                    )
                ) /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePoints2pt_{ws}"),
            (
                pl.col(f"pointsMidrange2pt_{ws}") /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePointsMidrange2pt_{ws}"),
            (
                pl.col(f"assisted2pt_{ws}") /
                (
                    pl.col(f"fieldGoalsMade_{ws}") -
                    pl.col(f"threePointersMade_{ws}")
                )
            ).alias(f"percentageAssisted2pt_{ws}"),
            (
                pl.col(f"unassisted2pt_{ws}") /
                (
                    pl.col(f"fieldGoalsMade_{ws}") -
                    pl.col(f"threePointersMade_{ws}")
                )
            ).alias(f"percentageUnassisted2pt_{ws}"),
            (
                pl.col(f"assisted3pt_{ws}") /
                pl.col(f"threePointersMade_{ws}")
            ).alias(f"percentageAssisted3pt_{ws}"),
            (
                pl.col(f"unassisted3pt_{ws}") /
                pl.col(f"threePointersMade_{ws}")
            ).alias(f"percentageunassisted3pt_{ws}"),
            (
                pl.col(f"unassistedFGM_{ws}") /
                pl.col(f"fieldGoalsMade_{ws}")
            ).alias(f"percentageunassistedFGM_{ws}"),
            (
                pl.col(f"pointsFastBreak_{ws}") /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePointsFastBreak_{ws}"),
            (
                pl.col(f"freeThrowsMade_{ws}") /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePointsFreeThrow_{ws}"),
            (
                pl.col(f"pointsOffTurnovers_{ws}") /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePointsOffTurnovers_{ws}"),
            (
                pl.col(f"pointsPaint_{ws}") /
                pl.col(f"points_{ws}")
            ).alias(f"percentagePointsPaint_{ws}"),
            (
                (
                    pl.col(f"fieldGoalsMade_{ws}") +
                    0.5 * pl.col(f"threePointersMade_{ws}")
                ) /
                pl.col(f"fieldGoalsAttempted_{ws}")
            ).alias(f"effectiveFieldGoalPercentage_{ws}"),
            (
                pl.col(f"points_{ws}") /
                (
                    2 * (
                          pl.col(f"fieldGoalsAttempted_{ws}") +
                          (0.44 * pl.col(f"freeThrowsAttempted_{ws}"))
                        )
                )
            ).alias(f"trueShootingPercentage_{ws}"),
            (
                pl.col(f"assists_{ws}") * 100 /
                (
                    pl.col(f"fieldGoalsAttempted_{ws}") +
                    0.44 * pl.col(f"freeThrowsAttempted_{ws}") +
                    pl.col(f"turnovers_{ws}") +
                    pl.col(f"assists_{ws}")
                )
            ).alias(f"assistRatio_{ws}"),
            (
                pl.col(f"reboundsTotal_{ws}") /
                (
                    pl.col(f"reboundsTotal_{ws}") +
                    pl.col(f"reboundsTotal_opponent_{ws}")
                )
            ).alias(f"reboundPercentage_{ws}"),
            (
                pl.col(f"reboundsDefensive_{ws}") /
                (
                    pl.col(f"reboundsDefensive_{ws}") +
                    pl.col(f"reboundsOffensive_opponent_{ws}")
                )
            ).alias(f"defensiveReboundPercentage_{ws}"),
            (
                pl.col(f"reboundsOffensive_{ws}") /
                (
                    pl.col(f"reboundsOffensive_{ws}") +
                    pl.col(f"reboundsDefensive_opponent_{ws}")
                )
            ).alias(f"offensiveReboundPercentage_{ws}"),
            (
                pl.col(f"freeThrowsAttempted_opponent_{ws}") /
                pl.col(f"fieldGoalsAttempted_opponent_{ws}")
            ).alias(f"freeThrowAttemptRate_opponent_{ws}"),
            (
                (
                    pl.col(f"fieldGoalsMade_opponent_{ws}") +
                    0.5 * pl.col(f"threePointersMade_opponent_{ws}")
                ) /
                pl.col(f"fieldGoalsAttempted_opponent_{ws}")
            ).alias(f"effectiveFieldGoalPercentage_opponent_{ws}"),
            (
                pl.col(f"points_opponent_{ws}") /
                (
                    2 * (
                          pl.col(f"fieldGoalsAttempted_opponent_{ws}") +
                          (0.44 * pl.col(f"freeThrowsAttempted_opponent_{ws}"))
                        )
                )
            ).alias(f"trueShootingPercentage_opponent_{ws}"),
            (
                pl.col(f"turnovers_{ws}") /
                pl.col(f"possessions_{ws}")
            ).alias(f"turnoverRatio_{ws}"),
            (
                pl.col(f"turnovers_opponent_{ws}") /
                pl.col(f"possessions_opponent_{ws}")
            ).alias(f"turnoverRatio_opponent_{ws}")
        ]
    ]).drop(drop_cols).with_columns([
        pl.when(
            pl.col("has_won_last_game")
        ).then(
            pl.col("streak_length")
        ).otherwise(
            pl.lit(0)
        ).alias("current_winning_streak"),
        pl.when(
            ~pl.col("has_won_last_game")
        ).then(
            pl.col("streak_length")
        ).otherwise(
            pl.lit(0)
        ).alias("current_losing_streak")
    ]).drop(["streak", "streak_length"])
    return df

def head_to_head(
    df_1: pl.DataFrame,
    df_2: pl.DataFrame = None
) -> pl.DataFrame:
    """
    Extracts the h2h winning percentage of the home team against the opposing
    team.
    :df_1: schedule of the current season
    :df_2: schedule of the previous season
    """
    df_1 = df_1.sort("date").with_columns([
        pl.struct([
            "date", "home_team_id", "away_team_id"
        ]).map_elements(
            lambda x: df_1.filter(
                (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("away_team_id") == x["away_team_id"])
                    ) |
                    (
                        (pl.col("home_team_id") == x["away_team_id"]) &
                        (pl.col("away_team_id") == x["home_team_id"])
                    )
                ) &
                (
                    pl.col("date") < x["date"]
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias("previous_games"),
        pl.struct([
            "date", "home_team_id", "away_team_id"
        ]).map_elements(
            lambda x: df_1.filter(
                (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("away_team_id") == x["away_team_id"])
                    ) |
                    (
                        (pl.col("home_team_id") == x["away_team_id"]) &
                        (pl.col("away_team_id") == x["home_team_id"])
                    )
                ) & (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("points_home") > pl.col("points_away"))
                    ) |
                    (
                        (pl.col("away_team_id") == x["home_team_id"]) &
                        (pl.col("points_home") < pl.col("points_away"))
                    )
                ) & (
                    pl.col("date") < x["date"]
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias("previous_wins"),
        pl.struct([
            "home_team_id", "away_team_id"
        ]).map_elements(
            lambda x: df_2.filter(
                (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("away_team_id") == x["away_team_id"])
                    ) |
                    (
                        (pl.col("home_team_id") == x["away_team_id"]) &
                        (pl.col("away_team_id") == x["home_team_id"])
                    )
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias("games_last_year"),
        pl.struct([
            "home_team_id", "away_team_id"
        ]).map_elements(
            lambda x: df_2.filter(
                (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("away_team_id") == x["away_team_id"])
                    ) |
                    (
                        (pl.col("home_team_id") == x["away_team_id"]) &
                        (pl.col("away_team_id") == x["home_team_id"])
                    )
                ) & (
                    (
                        (pl.col("home_team_id") == x["home_team_id"]) &
                        (pl.col("points_home") > pl.col("points_away"))
                    ) |
                    (
                        (pl.col("away_team_id") == x["home_team_id"]) &
                        (pl.col("points_home") < pl.col("points_away"))
                    )
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias("wins_last_year")
    ]).with_columns([
        (
            pl.col("previous_wins") / pl.col("previous_games")
        ).fill_nan(pl.lit(None)).alias("h2h_current_year"),
        (
            pl.col("wins_last_year") / pl.col("games_last_year")
        ).alias("h2h_previous_year")
    ])
    return df_1

def location_winning_percentage(
    df: pl.DataFrame
  ) -> pl.DataFrame:
  """
  Extracts the winning percentage of the home team at home and the road team
  on the road
  :df: schedule of a season
  """
  df = df.with_columns(
      pl.struct(
          ["date", "home_team_id"]
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col("date") < x["date"]) &
                (pl.col("home_team_id") == x["home_team_id"]) &
                (pl.col("is_home_win"))
            ).shape[0], return_dtype=pl.Int64
      ).alias("previous_home_wins"),
      pl.struct(
          ["date", "home_team_id"]
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col("date") < x["date"]) &
                (pl.col("home_team_id") == x["home_team_id"])
            ).shape[0], return_dtype=pl.Int64
      ).alias("previous_home_games"),
      pl.struct(
          ["date", "away_team_id"]
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col("date") < x["date"]) &
                (pl.col("away_team_id") == x["away_team_id"]) &
                (~pl.col("is_home_win"))
            ).shape[0], return_dtype=pl.Int64
      ).alias("previous_away_wins"),
      pl.struct(
          ["date", "away_team_id"]
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col("date") < x["date"]) &
                (pl.col("away_team_id") == x["away_team_id"])
            ).shape[0], return_dtype=pl.Int64
      ).alias("previous_away_games")
  ).with_columns([
      (
          pl.col("previous_home_wins") /
          pl.col("previous_home_games")
      ).fill_nan(None).alias("winning_percentage_home"),
      (
          pl.col("previous_away_wins") /
          pl.col("previous_away_games")
      ).fill_nan(None).alias("winning_percentage_away")
  ])
  return df

def merge_schedule_with_team_stats(
    schedule_df: pl.DataFrame,
    team_stats: list[pl.DataFrame]
  ):
  """
  Merges the data from a seasons schedule with the teams stats heading into
  the games + general details of the team
  :schedule_df: schedule of a season
  :team_stats: stats of the teams
  :team_details_df: general team details
  """
  team_stats_df = pl.concat(team_stats).drop(["date", "game_type"])
  schedule_df = schedule_df.with_columns(
      (pl.col("points_home") > pl.col("points_away")).alias("is_home_win")
  ).select([
      "date", "game_id", "home_team_id", "away_team_id", "game_type",
      "is_home_win", "previous_games", "h2h_current_year",
      "h2h_previous_year"
  ]).join(
      team_stats_df.rename(lambda c: c + "_home_team"),
      left_on=["game_id", "home_team_id"],
      right_on=["game_id_home_team", "team_id_home_team"]
  ).join(
      team_stats_df.rename(lambda c: c + "_away_team"),
      left_on=["game_id", "away_team_id"],
      right_on=["game_id_away_team", "team_id_away_team"]
  )
  schedule_df = schedule_df.sort("date").drop(["is_win_home_team", "is_win_away_team"])
  return schedule_df