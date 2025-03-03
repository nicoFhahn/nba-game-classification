from datetime import date, timedelta
from typing import List

from tqdm import tqdm
from supabase import Client
import polars as pl
import polars.selectors as cs
import data_collection

def load_schedule(
    all_star_date: date,
    play_in_start: date,
    play_in_end: date,
    connection: Client,
    season_id: str
) -> pl.DataFrame:
  '''
  Loads the schedule of a season
  :folder: folder where the data is saved
  :filename_schedule: name of the file containing the schedule
  :filename_games: name of the file containing the games
  :all_star_date: date the all star game takes place
  :play_in_start: date the nba play in begins
  :play_in_end: date the nba play in ends
  '''
  schedule_df = data_collection.collect_season_data(
      season_id,
      'schedule',
      connection
  )
  schedule_df = schedule_df.with_columns(
      pl.col('date').str.to_date()
  ).filter(
      pl.col('date') != all_star_date
  ).with_columns(
      pl.when(
          pl.col('date') < play_in_start
      ).then(
          pl.lit('Regular Season')
      ).otherwise(
          pl.when(
              pl.col('date') > play_in_end
          ).then(
              pl.lit('Playoffs')
          ).otherwise(
              pl.lit('Play-In')
          )
      ).alias('game_type')
  )
  boxscore_df = data_collection.collect_season_filtered_table(
      season_id,
      'boxscore',
      connection
  )
  df = schedule_df.unique().join(
      boxscore_df,
      on='game_id',
      how='left',
      validate='1:1'
  )
  return df

def get_team_schedules(
        df: pl.DataFrame
) -> List[pl.DataFrame]:
    team_ids = set(df['home_team_id'])
    team_schedules = [
        df.filter(
            (pl.col('home_team_id') == team_id) |
            (pl.col('away_team_id') == team_id)
        ).sort('date')
        for team_id in team_ids
    ]
    return team_schedules

def team_boxscore(
    df: pl.DataFrame,
    team_id: int
    ) -> pl.DataFrame:
    '''
    Extracts the stats/boxscore of one team (and a few stats of their opponents)
    for each game of the season
    :df: all games within one season
    :team_id: unique identifier of a team
    '''
    home_df = df.filter(pl.col('home_team_id') == team_id).with_columns([
        (pl.col('points_home') > pl.col('points_away')).alias('is_win')
    ])
    away_df = df.filter(pl.col('away_team_id') == team_id).with_columns([
        (pl.col('points_home') < pl.col('points_away')).alias('is_win')
    ])
    team_stats_home_df = home_df.select(
        ~cs.matches('_away|is_win')
    ).rename(lambda c: c.replace('_home', ''))
    opp_stats_home_df = home_df.select([
        'reboundsDefensive_away', 'reboundsOffensive_away',
        'reboundsTotal_away', 'fieldGoalsMade_away', 'fieldGoalsAttempted_away',
        'freeThrowsAttempted_away', 'threePointersMade_away', 'points_away',
        'turnovers_away', 'possessions_away', 'is_win'
    ]).rename(lambda c: c.replace('_away', '_opponent'))
    team_stats_away_df = away_df.select(
        ~cs.matches('_home|is_win')
    ).rename(lambda c: c.replace('_away', ''))
    opp_stats_away_df = away_df.select([
        'reboundsDefensive_home', 'reboundsOffensive_home',
        'reboundsTotal_home', 'fieldGoalsMade_home', 'fieldGoalsAttempted_home',
        'freeThrowsAttempted_home', 'threePointersMade_home', 'points_home',
        'turnovers_home', 'possessions_home', 'is_win'
    ]).rename(lambda c: c.replace('_home', '_opponent'))
    home_df = pl.concat(
        [team_stats_home_df, opp_stats_home_df],
        how='horizontal'
    )
    away_df = pl.concat(
        [team_stats_away_df, opp_stats_away_df],
        how='horizontal'
    )
    df = pl.concat([home_df, away_df]).drop([
        'home_team_id', 'away_team_id', 'minutes'
    ]).sort('date').with_columns([
        pl.lit(team_id).cast(pl.String).alias('team_id'),
        (
            pl.col('is_win').cum_sum() / pl.col('is_win').cum_count()
        ).shift(1).alias('winning_percentage')
    ])
    return df


def team_season_stats(
    df: pl.DataFrame,
    window_sizes: list[int] = [8, 109]
    ) -> pl.DataFrame:
    '''
    Extracts the stats of a team heading into a game
    for each game of the season
    :df: all games within one season of a specific team
    :window_sizes: windows of past games. should include 109 to include the
    stats up until one point in a season (109 as a team can play a max of 110)
    games in a season
    '''
    absolute_columns = [
        'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade',
        'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted',
        'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists',
        'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points',
        'plusMinusPoints', 'estimatedPace', 'pace', 'pacePer40', 'possessions',
        'contestedShots', 'contestedShots2pt', 'contestedShots3pt',
        'deflections', 'chargesDrawn', 'screenAssists', 'screenAssistPoints',
        'looseBallsRecoveredOffensive', 'looseBallsRecoveredDefensive',
        'looseBallsRecoveredTotal', 'offensiveBoxOuts', 'defensiveBoxOuts',
        'boxOutPlayerTeamRebounds', 'boxOutPlayerRebounds', 'boxOuts',
        'pointsOffTurnovers', 'pointsSecondChance', 'pointsFastBreak',
        'pointsPaint', 'oppPointsOffTurnovers', 'oppPointsSecondChance',
        'oppPointsFastBreak', 'oppPointsPaint', 'blocksAgainst', 'foulsDrawn',
        'distance', 'reboundChancesOffensive', 'reboundChancesDefensive',
        'reboundChancesTotal', 'touches', 'secondaryAssists',
        'freeThrowAssists', 'passes', 'contestedFieldGoalsMade',
        'contestedFieldGoalsAttempted', 'uncontestedFieldGoalsMade',
        'uncontestedFieldGoalsAttempted', 'defendedAtRimFieldGoalsMade',
        'defendedAtRimFieldGoalsAttempted', 'pointsMidrange2pt', 'assisted2pt',
        'unassisted2pt', 'assisted3pt', 'unassisted3pt', 'unassistedFGM'
    ]
    opponent_columns = [
        'reboundsDefensive_opponent', 'reboundsOffensive_opponent',
        'reboundsTotal_opponent', 'fieldGoalsMade_opponent',
        'fieldGoalsAttempted_opponent', 'freeThrowsAttempted_opponent',
        'threePointersMade_opponent', 'points_opponent', 'possessions_opponent',
        'turnovers_opponent'
    ]
    per_100_columns = [
        'estimatedOffensiveRating', 'offensiveRating',
        'estimatedDefensiveRating', 'defensiveRating',
        'estimatedNetRating', 'netRating'
    ]
    relative_columns = [
        'fieldGoalsPercentage', 'threePointersPercentage',
        'freeThrowsPercentage', 'assistPercentage', 'assistToTurnover',
        'freeThrowAttemptRate', 'contestedFieldGoalPercentage',
        'uncontestedFieldGoalsPercentage', 'defendedAtRimFieldGoalPercentage',
        'percentageFieldGoalsAttempted2pt', 'percentagePoints2pt',
        'percentagePointsMidrange2pt', 'percentagePoints3pt',
        'percentagePointsFastBreak', 'percentagePointsFreeThrow',
        'percentagePointsOffTurnovers', 'percentagePointsPaint',
        'percentageAssisted2pt', 'percentageUnassisted2pt',
        'percentageAssisted3pt', 'percentageUnassisted3pt',
        'percentageAssistedFGM', 'percentageUnassistedFGM'
    ]
    dropped_columns = [
        'usagePercentage', 'estimatedUsagePercentage', 'fieldGoalPercentage',
        'percentageFieldGoalsAttempted3pt', 'percentageFieldGoalsMade',
        'percentageFieldGoalsAttempted', 'percentageThreePointersMade',
        'percentageThreePointersAttempted', 'percentageFreeThrowsMade',
        'percentageFreeThrowsAttempted', 'percentageReboundsOffensive',
        'percentageReboundsDefensive', 'percentageReboundsTotal',
        'percentageAssists', 'percentageTurnovers', 'percentageSteals',
        'percentageBlocks', 'percentageBlocksAllowed',
        'percentagePersonalFouls', 'percentagePersonalFoulsDrawn',
        'percentagePoints', 'PIE', 'effectiveFieldGoalPercentage',
        'trueShootingPercentage', 'assistRatio', 'offensiveReboundPercentage',
        'defensiveReboundPercentage', 'reboundPercentage',
        'oppEffectiveFieldGoalPercentage', 'oppFreeThrowAttemptRate',
        'oppOffensiveReboundPercentage', 	'estimatedTeamTurnoverPercentage',
        'turnoverRatio', 'teamTurnoverPercentage', 'oppTeamTurnoverPercentage'
    ]
    drop_cols = absolute_columns + per_100_columns + relative_columns + dropped_columns + opponent_columns
    df = df.sort(
        'date'
        ).with_columns(
            (
                pl.col('date') - pl.col('date').shift(1)
            ).dt.total_days().alias('days_since_last_game'),
        pl.col('is_win').shift(1).alias('has_won_last_game'),
        (
            pl.col('date').map_elements(
                lambda d: df.filter(
                    (pl.col('date') < d) &
                    (pl.col('date') >= (d - timedelta(days=7)))
              ).shape[0], return_dtype=pl.Int64
            )
        ).alias('games_last_7_days'),
        (
            pl.col('points') * pl.col('percentagePointsMidrange2pt')
        ).round(0).alias('pointsMidrange2pt'),
        (
            (pl.col('fieldGoalsMade') - pl.col('threePointersMade')) *
            pl.col('percentageAssisted2pt')
        ).round(0).alias('assisted2pt'),
        (
            (pl.col('fieldGoalsMade') - pl.col('threePointersMade')) *
            pl.col('percentageUnassisted2pt')
        ).round(0).alias('unassisted2pt'),
        (
            pl.col('threePointersMade') * pl.col('percentageAssisted3pt')
        ).round(0).alias('assisted3pt'),
        (
            pl.col('threePointersMade') * pl.col('percentageUnassisted3pt')
        ).round(0).alias('unassisted3pt'),
        (
            pl.col('fieldGoalsMade') * pl.col('percentageUnassistedFGM')
        ).round(0).alias('unassistedFGM')
    ).with_columns([
        pl.col(c).shift(1).alias(f'{c}_previous_game')
        for c in absolute_columns + per_100_columns + opponent_columns
    ]).with_columns(
        pl.col('has_won_last_game').rle_id().alias('streak')
    ).with_columns([
        pl.col('has_won_last_game').cum_count().over(
            pl.col('streak')
        ).alias('streak_length')
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            pl.col(f'{c}_previous_game').rolling_mean(
                window_size = ws, min_periods= 1 if ws >= 109 else ws
            ).alias(f'{c}_{ws}')
            for c in absolute_columns + per_100_columns + opponent_columns
        ]
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            (
                pl.col(f'fieldGoalsMade_{ws}') /
                pl.col(f'fieldGoalsAttempted_{ws}')
            ).alias(f'fieldGoalsPercentage_{ws}'),
            (
                pl.col(f'threePointersMade_{ws}') /
                pl.col(f'threePointersAttempted_{ws}')
            ).alias(f'threePointersPercentage_{ws}'),
            (
                pl.col(f'freeThrowsMade_{ws}') /
                pl.col(f'freeThrowsAttempted_{ws}')
            ).alias(f'freeThrowsPercentage_{ws}'),
            (
                pl.col(f'assists_{ws}') /
                pl.col(f'fieldGoalsMade_{ws}')
            ).alias(f'assistPercentage_{ws}'),
            (
                pl.col(f'assists_{ws}') /
                pl.col(f'turnovers_{ws}')
            ).alias(f'assistToTurnover_{ws}'),
            (
                pl.col(f'freeThrowsAttempted_{ws}') /
                pl.col(f'fieldGoalsAttempted_{ws}')
            ).alias(f'freeThrowAttemptRate_{ws}'),
            (
                pl.col(f'contestedFieldGoalsMade_{ws}') /
                pl.col(f'contestedFieldGoalsAttempted_{ws}')
            ).alias(f'contestedFieldGoalsPercentage_{ws}'),
            (
                pl.col(f'uncontestedFieldGoalsMade_{ws}') /
                pl.col(f'uncontestedFieldGoalsAttempted_{ws}')
            ).alias(f'uncontestedFieldGoalsPercentage_{ws}'),
            (
                pl.col(f'defendedAtRimFieldGoalsMade_{ws}') /
                pl.col(f'defendedAtRimFieldGoalsAttempted_{ws}')
            ).alias(f'defendedAtRimFieldGoalsPercentage_{ws}'),
            (
                1 - (
                      pl.col(f'threePointersAttempted_{ws}') /
                      pl.col(f'fieldGoalsAttempted_{ws}')
                    )
            ).alias(f'percentageFieldGoalsAttempted2pt_{ws}'),
            (
                (2 * (
                    pl.col(f'fieldGoalsMade_{ws}') -
                    pl.col(f'threePointersMade_{ws}')
                    )
                ) /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePoints2pt_{ws}'),
            (
                pl.col(f'pointsMidrange2pt_{ws}') /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePointsMidrange2pt_{ws}'),
            (
                pl.col(f'assisted2pt_{ws}') /
                (
                    pl.col(f'fieldGoalsMade_{ws}') -
                    pl.col(f'threePointersMade_{ws}')
                )
            ).alias(f'percentageAssisted2pt_{ws}'),
            (
                pl.col(f'unassisted2pt_{ws}') /
                (
                    pl.col(f'fieldGoalsMade_{ws}') -
                    pl.col(f'threePointersMade_{ws}')
                )
            ).alias(f'percentageUnassisted2pt_{ws}'),
            (
                pl.col(f'assisted3pt_{ws}') /
                pl.col(f'threePointersMade_{ws}')
            ).alias(f'percentageAssisted3pt_{ws}'),
            (
                pl.col(f'unassisted3pt_{ws}') /
                pl.col(f'threePointersMade_{ws}')
            ).alias(f'percentageunassisted3pt_{ws}'),
            (
                pl.col(f'unassistedFGM_{ws}') /
                pl.col(f'fieldGoalsMade_{ws}')
            ).alias(f'percentageunassistedFGM_{ws}'),
            (
                pl.col(f'pointsFastBreak_{ws}') /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePointsFastBreak_{ws}'),
            (
                pl.col(f'freeThrowsMade_{ws}') /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePointsFreeThrow_{ws}'),
            (
                pl.col(f'pointsOffTurnovers_{ws}') /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePointsOffTurnovers_{ws}'),
            (
                pl.col(f'pointsPaint_{ws}') /
                pl.col(f'points_{ws}')
            ).alias(f'percentagePointsPaint_{ws}'),
            (
                (
                    pl.col(f'fieldGoalsMade_{ws}') +
                    0.5 * pl.col(f'threePointersMade_{ws}')
                ) /
                pl.col(f'fieldGoalsAttempted_{ws}')
            ).alias(f'effectiveFieldGoalPercentage_{ws}'),
            (
                pl.col(f'points_{ws}') /
                (
                    2 * (
                          pl.col(f'fieldGoalsAttempted_{ws}') +
                          (0.44 * pl.col(f'freeThrowsAttempted_{ws}'))
                        )
                )
            ).alias(f'trueShootingPercentage_{ws}'),
            (
                pl.col(f'assists_{ws}') * 100 /
                (
                    pl.col(f'fieldGoalsAttempted_{ws}') +
                    0.44 * pl.col(f'freeThrowsAttempted_{ws}') +
                    pl.col(f'turnovers_{ws}') +
                    pl.col(f'assists_{ws}')
                )
            ).alias(f'assistRatio_{ws}'),
            (
                pl.col(f'reboundsTotal_{ws}') /
                (
                    pl.col(f'reboundsTotal_{ws}') +
                    pl.col(f'reboundsTotal_opponent_{ws}')
                )
            ).alias(f'reboundPercentage_{ws}'),
            (
                pl.col(f'reboundsDefensive_{ws}') /
                (
                    pl.col(f'reboundsDefensive_{ws}') +
                    pl.col(f'reboundsOffensive_opponent_{ws}')
                )
            ).alias(f'defensiveReboundPercentage_{ws}'),
            (
                pl.col(f'reboundsOffensive_{ws}') /
                (
                    pl.col(f'reboundsOffensive_{ws}') +
                    pl.col(f'reboundsDefensive_opponent_{ws}')
                )
            ).alias(f'offensiveReboundPercentage_{ws}'),
            (
                pl.col(f'freeThrowsAttempted_opponent_{ws}') /
                pl.col(f'fieldGoalsAttempted_opponent_{ws}')
            ).alias(f'freeThrowAttemptRate_opponent_{ws}'),
            (
                (
                    pl.col(f'fieldGoalsMade_opponent_{ws}') +
                    0.5 * pl.col(f'threePointersMade_opponent_{ws}')
                ) /
                pl.col(f'fieldGoalsAttempted_opponent_{ws}')
            ).alias(f'effectiveFieldGoalPercentage_opponent_{ws}'),
            (
                pl.col(f'points_opponent_{ws}') /
                (
                    2 * (
                          pl.col(f'fieldGoalsAttempted_opponent_{ws}') +
                          (0.44 * pl.col(f'freeThrowsAttempted_opponent_{ws}'))
                        )
                )
            ).alias(f'trueShootingPercentage_opponent_{ws}'),
            (
                pl.col(f'turnovers_{ws}') /
                pl.col(f'possessions_{ws}')
            ).alias(f'turnoverRatio_{ws}'),
            (
                pl.col(f'turnovers_opponent_{ws}') /
                pl.col(f'possessions_opponent_{ws}')
            ).alias(f'turnoverRatio_opponent_{ws}')
        ]
    ]).drop(drop_cols).with_columns([
        pl.when(
            pl.col('has_won_last_game')
        ).then(
            pl.col('streak_length')
        ).otherwise(
            pl.lit(0)
        ).alias('current_winning_streak'),
        pl.when(
            ~pl.col('has_won_last_game')
        ).then(
            pl.col('streak_length')
        ).otherwise(
            pl.lit(0)
        ).alias('current_losing_streak')
    ]).drop(['streak', 'streak_length'])
    return df

def head_to_head(
    df_1: pl.DataFrame,
    df_2: pl.DataFrame = None
) -> pl.DataFrame:
    '''
    Extracts the h2h winning percentage of the home team against the opposing
    team.
    :df_1: schedule of the current season
    :df_2: schedule of the previous season
    '''
    df_1 = df_1.sort('date').with_columns([
        pl.struct([
            'date', 'home_team_id', 'away_team_id'
        ]).map_elements(
            lambda x: df_1.filter(
                (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('away_team_id') == x['away_team_id'])
                    ) |
                    (
                        (pl.col('home_team_id') == x['away_team_id']) &
                        (pl.col('away_team_id') == x['home_team_id'])
                    )
                ) &
                (
                    pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('previous_games'),
        pl.struct([
            'date', 'home_team_id', 'away_team_id'
        ]).map_elements(
            lambda x: df_1.filter(
                (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('away_team_id') == x['away_team_id'])
                    ) |
                    (
                        (pl.col('home_team_id') == x['away_team_id']) &
                        (pl.col('away_team_id') == x['home_team_id'])
                    )
                ) & (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('points_home') > pl.col('points_away'))
                    ) |
                    (
                        (pl.col('away_team_id') == x['home_team_id']) &
                        (pl.col('points_home') < pl.col('points_away'))
                    )
                ) & (
                    pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('previous_wins'),
        pl.struct([
            'home_team_id', 'away_team_id'
        ]).map_elements(
            lambda x: df_2.filter(
                (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('away_team_id') == x['away_team_id'])
                    ) |
                    (
                        (pl.col('home_team_id') == x['away_team_id']) &
                        (pl.col('away_team_id') == x['home_team_id'])
                    )
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('games_last_year'),
        pl.struct([
            'home_team_id', 'away_team_id'
        ]).map_elements(
            lambda x: df_2.filter(
                (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('away_team_id') == x['away_team_id'])
                    ) |
                    (
                        (pl.col('home_team_id') == x['away_team_id']) &
                        (pl.col('away_team_id') == x['home_team_id'])
                    )
                ) & (
                    (
                        (pl.col('home_team_id') == x['home_team_id']) &
                        (pl.col('points_home') > pl.col('points_away'))
                    ) |
                    (
                        (pl.col('away_team_id') == x['home_team_id']) &
                        (pl.col('points_home') < pl.col('points_away'))
                    )
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('wins_last_year')
    ]).with_columns([
        (
            pl.col('previous_wins') / pl.col('previous_games')
        ).fill_nan(pl.lit(None)).alias('h2h_current_year'),
        (
            pl.col('wins_last_year') / pl.col('games_last_year')
        ).alias('h2h_previous_year')
    ])
    return df_1

def location_winning_percentage(
    df: pl.DataFrame
  ) -> pl.DataFrame:
  '''
  Extracts the winning percentage of the home team at home and the road team
  on the road
  :df: schedule of a season
  '''
  df = df.with_columns(
      pl.struct(
          ['date', 'home_team_id']
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col('date') < x['date']) &
                (pl.col('home_team_id') == x['home_team_id']) &
                (pl.col('is_home_win'))
            ).shape[0], return_dtype=pl.Int64
      ).alias('previous_home_wins'),
      pl.struct(
          ['date', 'home_team_id']
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col('date') < x['date']) &
                (pl.col('home_team_id') == x['home_team_id'])
            ).shape[0], return_dtype=pl.Int64
      ).alias('previous_home_games'),
      pl.struct(
          ['date', 'away_team_id']
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col('date') < x['date']) &
                (pl.col('away_team_id') == x['away_team_id']) &
                (~pl.col('is_home_win'))
            ).shape[0], return_dtype=pl.Int64
      ).alias('previous_away_wins'),
      pl.struct(
          ['date', 'away_team_id']
      ).map_elements(
          lambda x:
            df.filter(
                (pl.col('date') < x['date']) &
                (pl.col('away_team_id') == x['away_team_id'])
            ).shape[0], return_dtype=pl.Int64
      ).alias('previous_away_games')
  ).with_columns([
      (
          pl.col('previous_home_wins') /
          pl.col('previous_home_games')
      ).fill_nan(None).alias('winning_percentage_home'),
      (
          pl.col('previous_away_wins') /
          pl.col('previous_away_games')
      ).fill_nan(None).alias('winning_percentage_away')
  ])
  return df

def merge_schedule_with_team_stats(
    schedule_df: pl.DataFrame,
    team_stats: list[pl.DataFrame]
  ):
  '''
  Merges the data from a seasons schedule with the teams stats heading into
  the games + general details of the team
  :schedule_df: schedule of a season
  :team_stats: stats of the teams
  :team_details_df: general team details
  '''
  team_stats_df = pl.concat(team_stats).drop(['date', 'game_type'])
  schedule_df = schedule_df.with_columns(
      (pl.col('points_home') > pl.col('points_away')).alias('is_home_win')
  ).select([
      'date', 'game_id', 'home_team_id', 'away_team_id', 'game_type',
      'is_home_win', 'previous_games', 'h2h_current_year',
      'h2h_previous_year'
  ]).join(
      team_stats_df.rename(lambda c: c + '_home_team'),
      left_on=['game_id', 'home_team_id'],
      right_on=['game_id_home_team', 'team_id_home_team']
  ).join(
      team_stats_df.rename(lambda c: c + '_away_team'),
      left_on=['game_id', 'away_team_id'],
      right_on=['game_id_away_team', 'team_id_away_team']
  )
  schedule_df = schedule_df.sort('date').drop(['is_win_home_team', 'is_win_away_team'])
  return schedule_df

def load_season(
    all_star_date: date,
    play_in_start: date,
    play_in_end: date,
    connection: Client,
    season_id: str,
    return_h2h: bool = False
) -> pl.DataFrame:
    schedule_last_season = data_collection.collect_season_data(
        str(int(season_id) - 1),
        'schedule',
        connection
    )
    boxscore_last_season = data_collection.collect_season_filtered_table(
        str(int(season_id) - 1),
        'boxscore',
        connection
    )
    schedule_current_season = load_schedule(
        all_star_date,
        play_in_start,
        play_in_end,
        connection,
        season_id
    )
    team_schedules_current_season = get_team_schedules(schedule_current_season)
    team_ids = list(set(schedule_current_season['home_team_id']))
    team_boxscores_current_season = [
        team_boxscore(
            team_schedules_current_season[i],
            team_ids[i]
        ) for i in range(len(team_ids))
    ]
    team_stats_current_season = [team_season_stats(team) for team in tqdm(team_boxscores_current_season)]
    schedule_last_season = schedule_last_season.join(
        boxscore_last_season[['game_id', 'points_home', 'points_away']],
        on='game_id'
    )
    schedule_current_season = head_to_head(
      schedule_current_season, schedule_last_season
    )
    if return_h2h:
        return schedule_current_season.select([
            'game_id', 'previous_wins', 'previous_games', 'wins_last_year', 'games_last_year'
        ])
    schedule_current_season = merge_schedule_with_team_stats(
      schedule_current_season, team_stats_current_season
    )
    schedule_current_season = location_winning_percentage(
        schedule_current_season
    )
    schedule_current_season = schedule_current_season.with_columns([
        pl.col('date').dt.month().cast(pl.String).alias('month'),
        pl.col('date').dt.weekday().cast(pl.String).alias('weekday')
    ])
    return schedule_current_season

def newest_games(connection, mod, season_id: str = '2024'):
    team_df = data_collection.collect_all_data('team_details', connection)
    mod.load_model()
    games = mod.full_data.with_columns([
        (pl.col('previous_home_games') - pl.col('previous_home_wins')).alias('previous_home_losses'),
        (pl.col('previous_away_games') - pl.col('previous_away_wins')).alias('previous_away_losses'),
        pl.when(
            (pl.col('current_winning_streak_home_team') == 0) & (pl.col('current_losing_streak_home_team') > 0)
        ).then(
            pl.concat_str(
                pl.col('current_losing_streak_home_team'), pl.lit('L')
            )
        ).otherwise(
            pl.concat_str(
                pl.col('current_winning_streak_home_team'), pl.lit('W')
            )
        ).alias('streak_home'),
        pl.when(
            (pl.col('current_winning_streak_away_team') == 0) & (pl.col('current_losing_streak_away_team') > 0)
        ).then(
            pl.concat_str(
                pl.col('current_losing_streak_away_team'), pl.lit('L')
            )
        ).otherwise(
            pl.concat_str(
                pl.col('current_winning_streak_away_team'), pl.lit('W')
            )
        ).alias('streak_away')
    ]).with_columns([
        pl.concat_str(
            pl.col('previous_home_wins'),
            pl.lit('-'),
            pl.col('previous_home_losses')
        ).alias('home_record'),
        pl.concat_str(
            pl.col('previous_away_wins'),
            pl.lit('-'),
            pl.col('previous_away_losses')
        ).alias('away_record'),
    ])
    h2h_df = data_collection.collect_season_filtered_table(season_id, 'h2h', connection)
    h2h_df = h2h_df.with_columns([
        (pl.col('previous_games') - pl.col('previous_wins')).alias('previous_losses'),
        (pl.col('games_last_year') - pl.col('wins_last_year')).alias('losses_last_year')
    ]).with_columns([
        pl.concat_str(
            pl.col('previous_wins'),
            pl.lit('-'),
            pl.col('previous_losses')
        ).alias('h2h_current_year'),
        pl.concat_str(
            pl.col('wins_last_year'),
            pl.lit('-'),
            pl.col('losses_last_year')
        ).alias('h2h_last_year')
    ]).select([
        'game_id', 'h2h_current_year', 'h2h_last_year'
    ])
    record_df = data_collection.collect_season_filtered_table(season_id, 'record', connection)
    record_df = record_df.with_columns([
        (pl.col('games_this_year_home_team') - pl.col('wins_this_year_home_team')).alias('losses_this_year_home_team'),
        (pl.col('games_this_year_away_team') - pl.col('wins_this_year_away_team')).alias('losses_this_year_away_team')
    ]).with_columns([
        pl.concat_str(
            pl.col('wins_this_year_home_team'),
            pl.lit('-'),
            pl.col('losses_this_year_home_team')
        ).alias('record_home_team'),
        pl.concat_str(
            pl.col('wins_this_year_away_team'),
            pl.lit('-'),
            pl.col('losses_this_year_away_team')
        ).alias('record_away_team')
    ]).select([
        'game_id', 'record_home_team', 'record_away_team'
    ])
    best_features = mod.load_best_features()[0]
    X_new = games.to_dummies([
        'game_type', 'month', 'weekday'
    ]).drop([
        'game_id', 'home_team_id', 'away_team_id'
    ]).filter(
        (pl.col('is_home_win').is_null()) &
        (pl.col('date') == games.filter(pl.col('is_home_win').is_null())['date'].min())
    ).select(pl.col(best_features)).drop('date')
    predictions = mod.model.predict(X_new.to_numpy())
    games = games.to_dummies([
        'game_type', 'month', 'weekday'
    ]).filter(
        (pl.col('is_home_win').is_null()) &
        (pl.col('date') == games.filter(pl.col('is_home_win').is_null())['date'].min())
    ).with_columns([
        pl.Series('probability', predictions)
    ]).with_columns([
        pl.when(
            pl.col('probability') < 0.5
        ).then(
            100 - 100 * pl.col('probability')
        ).otherwise(
            100 * pl.col('probability')
        ).alias('probability'),
        pl.when(
            pl.col('probability') >= 0.5
        ).then(
            pl.concat_str(
                pl.lit('https://raw.githubusercontent.com/nicoFhahn/nba-game-classification/main/streamlit/logos/'),
                pl.col('home_team_id'),
                pl.lit('.png')
            )
        ).otherwise(
            pl.concat_str(
                pl.lit('https://raw.githubusercontent.com/nicoFhahn/nba-game-classification/main/streamlit/logos/'),
                pl.col('away_team_id'),
                pl.lit('.png')
            )
        ).alias('winner_logo')
    ]).join(
        team_df.rename({'team_name': 'home_team_name'}),
        left_on='home_team_id',
        right_on='team_id'
    ).join(
        team_df.rename({'team_name': 'away_team_name'}),
        left_on='away_team_id',
        right_on='team_id'
    ).drop('h2h_current_year').join(
        h2h_df,
        on='game_id'
    ).join(
        record_df,
        on='game_id'
    ).select([
        'date', 'home_team_name', 'away_team_name', 'winner_logo', 'probability', 'h2h_current_year', 'h2h_last_year',
        'record_home_team', 'winning_percentage_home_team', 'record_away_team', 'winning_percentage_away_team',
        'home_record', 'winning_percentage_home', 'away_record', 'winning_percentage_away',
        'streak_home', 'streak_away', 'elo_home_team', 'elo_away_team', 'game_id'
    ]).with_columns([
        (100 * pl.col(c)).alias(c)
        for c in [
            'winning_percentage_home_team', 'winning_percentage_away_team',
            'winning_percentage_home', 'winning_percentage_away'
        ]
    ]).sort('game_id')
    return games

def record_current_season(
    all_star_date: date,
    play_in_start: date,
    play_in_end: date,
    connection: Client,
    season_id: str
):
    schedule_current_season = load_schedule(
        all_star_date,
        play_in_start,
        play_in_end,
        connection,
        season_id
    )
    schedule_current_season = schedule_current_season.select([
        'date', 'game_id', 'home_team_id', 'away_team_id', 'points_home', 'points_away'
    ]).with_columns([
        pl.struct([
            'date', 'home_team_id'
        ]).map_elements(
            lambda x: schedule_current_season.filter(
                (
                        (pl.col('home_team_id') == x['home_team_id']) |
                        (pl.col('away_team_id') == x['home_team_id'])
                ) &
                (
                        pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('games_this_year_home_team'),
        pl.struct([
            'date', 'away_team_id'
        ]).map_elements(
            lambda x: schedule_current_season.filter(
                (
                        (pl.col('home_team_id') == x['away_team_id']) |
                        (pl.col('away_team_id') == x['away_team_id'])
                ) &
                (
                        pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('games_this_year_away_team'),
        pl.struct([
            'date', 'home_team_id'
        ]).map_elements(
            lambda x: schedule_current_season.filter(
                (
                        (
                                (pl.col('home_team_id') == x['home_team_id']) &
                                (pl.col('points_home') > pl.col('points_away'))
                        ) |
                        (
                                (pl.col('away_team_id') == x['home_team_id']) &
                                (pl.col('points_home') < pl.col('points_away'))
                        )
                ) &
                (
                        pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('wins_this_year_home_team'),
        pl.struct([
            'date', 'away_team_id'
        ]).map_elements(
            lambda x: schedule_current_season.filter(
                (
                        (
                                (pl.col('home_team_id') == x['away_team_id']) &
                                (pl.col('points_home') > pl.col('points_away'))
                        ) |
                        (
                                (pl.col('away_team_id') == x['away_team_id']) &
                                (pl.col('points_home') < pl.col('points_away'))
                        )
                ) &
                (
                        pl.col('date') < x['date']
                )
            ).shape[0], return_dtype=pl.Int64
        ).alias('wins_this_year_away_team'),
    ]).sort('date').drop(['date', 'home_team_id', 'away_team_id'])
    return schedule_current_season