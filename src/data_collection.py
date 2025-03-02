from datetime import date
from time import sleep

import numpy as np
import polars as pl
from tqdm import tqdm
from nba_api.stats import endpoints
from supabase import Client


def daily_games(
        game_date: date
) -> pl.DataFrame:
    '''
    Creates a dataframe containing all games of a given date
    :param game_date: The date for which to scrape the data
    :return: dataframe with date, game id and team ids
    '''
    try:
        games = endpoints.scoreboardv2.ScoreboardV2(game_date=game_date)
        sleep(30)
        game_list = games.get_dict()['resultSets'][0]['rowSet']
        df = pl.DataFrame(
            data={
                'game_id': [game[2] for game in game_list],
                'home_team_id': [game[6] for game in game_list],
                'away_team_id': [game[7] for game in game_list],
                'season_id': game_list[0][8]
            }
        ).with_columns([
            pl.lit(game_date).alias('date'),
            pl.col('home_team_id').cast(pl.String),
            pl.col('away_team_id').cast(pl.String),
            pl.col('season_id').cast(pl.String)
        ])[['date', 'game_id', 'home_team_id', 'away_team_id', 'season_id']]
    except Exception as e:
        print(e)
        sleep(15)
        df = daily_games(game_date)
    return df

def boxscore(
        game_id: str
) -> pl.DataFrame:
    '''
    Creates a dataframe containing a lot of different boxscore metrics for a given game
    :param game_id: The id of the game
    :return: dataframe with gameid + the boxscore details
    '''
    try:
        bs_traditional = endpoints.boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        sleep(30)
        bs_traditional = pl.concat([
            pl.DataFrame(
                bs_traditional.get_dict()['boxScoreTraditional']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_traditional.get_dict()['boxScoreTraditional']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_advanced = endpoints.boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
        sleep(30)
        bs_advanced = pl.concat([
            pl.DataFrame(
                bs_advanced.get_dict()['boxScoreAdvanced']['homeTeam']['statistics']
            ).rename(
                lambda c: c + '_home'),
            pl.DataFrame(
                bs_advanced.get_dict()['boxScoreAdvanced']['awayTeam']['statistics']
            ).rename(
                lambda c: c + '_away')
        ], how='horizontal')
        bs_four_factors = endpoints.boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id=game_id)
        sleep(30)
        bs_four_factors = pl.concat([
            pl.DataFrame(
                bs_four_factors.get_dict()['boxScoreFourFactors']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_four_factors.get_dict()['boxScoreFourFactors']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_hustle = endpoints.boxscorehustlev2.BoxScoreHustleV2(game_id=game_id)
        sleep(30)
        bs_hustle = pl.concat([
            pl.DataFrame(
                bs_hustle.get_dict()['boxScoreHustle']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_hustle.get_dict()['boxScoreHustle']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_misc = endpoints.boxscoremiscv3.BoxScoreMiscV3(game_id=game_id)
        sleep(30)
        bs_misc = pl.concat([
            pl.DataFrame(
                bs_misc.get_dict()['boxScoreMisc']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_misc.get_dict()['boxScoreMisc']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_player_track = endpoints.boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id)
        sleep(30)
        bs_player_track = pl.concat([
            pl.DataFrame(
                bs_player_track.get_dict()['boxScorePlayerTrack']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_player_track.get_dict()['boxScorePlayerTrack']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_scoring = endpoints.boxscorescoringv3.BoxScoreScoringV3(game_id=game_id)
        sleep(30)
        bs_scoring = pl.concat([
            pl.DataFrame(
                bs_scoring.get_dict()['boxScoreScoring']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_scoring.get_dict()['boxScoreScoring']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        bs_usage = endpoints.boxscoreusagev3.BoxScoreUsageV3(game_id=game_id)
        sleep(30)
        bs_usage = pl.concat([
            pl.DataFrame(
                bs_usage.get_dict()['boxScoreUsage']['homeTeam']['statistics']
            ).rename(lambda c: c + '_home'),
            pl.DataFrame(
                bs_usage.get_dict()['boxScoreUsage']['awayTeam']['statistics']
            ).rename(lambda c: c + '_away')
        ], how='horizontal')
        df = pl.concat([
            bs_traditional,
            bs_advanced.drop([
                'minutes_home', 'minutes_away'
            ]),
            bs_four_factors.drop([
                'minutes_home', 'minutes_away', 'effectiveFieldGoalPercentage_home',
                'effectiveFieldGoalPercentage_away', 'offensiveReboundPercentage_home',
                'offensiveReboundPercentage_away'
            ]),
            bs_hustle.drop([
                'minutes_home', 'minutes_away', 'points_home', 'points_away'
            ]),
            bs_misc.drop([
                'minutes_home', 'minutes_away', 'blocks_home', 'blocks_away', 'foulsPersonal_home',
                'foulsPersonal_away'
            ]),
            bs_player_track.drop(
                ['minutes_home', 'minutes_away', 'assists_home', 'assists_away']
            ),
            bs_scoring.drop([
                'minutes_home', 'minutes_away'
            ]),
            bs_usage.drop([
                'minutes_home', 'minutes_away', 'usagePercentage_home', 'usagePercentage_away'
            ]),
        ], how='horizontal')
        df = df.with_columns([
            pl.lit(game_id).alias('game_id')
        ])

    except AttributeError as e:
        return None
    except Exception as e:
        print(e)
        df = boxscore(game_id)
    return df

def team_details(
        team_id: str
) -> pl.DataFrame:
    '''
    Gets team details for a given team
    :param team_id: id of the team
    :return: dataframe with team name, id, and arena capacity
    '''
    details = endpoints.teamdetails.TeamDetails(team_id=team_id).get_dict()['resultSets'][0]['rowSet'][0]
    details = {
        'team_name': f'{details[4]} {details[2]}',
        'team_id': details[0],
        'arena_capacity': details[6]
    }
    sleep(30)
    details = pl.DataFrame(details).with_columns(pl.col('arena_capacity').cast(pl.Int64))
    return details

def season(
        start_date: date,
        end_date: date,
        connection: Client,
        season_id: str
):
    '''
    Scrapes data for all games between two dates and pushes it into database
    :param start_date: beginning of the timeframe
    :param end_date: end of the tiemframe
    :param connection: a supabase connection
    :param season_id: unique id for the season
    :return: nothing
    '''
    date_list = np.arange(start_date, end_date)
    schedule_list = [daily_games(game_date) for game_date in tqdm(date_list)]
    schedule_df = collect_season_data(
        season_id,
        'schedule',
        connection
    )
    if len(schedule_list) > 0:
        new_schedule_df = pl.concat(schedule_list).filter(
            ~pl.col('game_id').is_in(schedule_df['game_id'])
        ).with_columns(
            pl.col('date').cast(pl.String)
        )
        if new_schedule_df.shape[0] > 0:
            schedule_df = pl.concat([
                schedule_df,
                new_schedule_df
            ])
            response = (
                connection.table('schedule').insert(
                    new_schedule_df.to_dicts()
                ).execute()
            )

    boxscore_df = collect_boxscore_table(season_id, connection)
    missing_games = schedule_df.join(
        boxscore_df,
        on='game_id',
        how='left'
    ).with_columns([
        pl.col('date').str.to_date()
    ]).filter(
        (pl.col('date') <= date.today()) & (pl.col('season_id_right').is_null())
    )['game_id']
    scraped_games = []
    for game_id in tqdm(missing_games):
            game_boxscore = boxscore(game_id)
            if game_boxscore is None:
                continue
            scraped_games.append(game_boxscore)
    scraped_games = pl.concat(scraped_games)
    response = (
        connection.table('boxscore').insert(
            scraped_games.to_dicts()
        ).execute()
    )


def collect_all_data(
        table_name: str,
        connection: Client
):
    '''
    Collects an entire table from supabase with chunks of size 1k
    :param table_name: name of the table
    :param connection: the supabase connection
    :return:
    '''
    chunk_size = 1000
    row_count = connection.table(table_name).select('*', head=True, count='exact').execute().count
    all_data = [
        item
        for start in range(0, row_count, chunk_size)
        for item in
        connection.table(table_name).select('*').range(start, min(start + chunk_size - 1, row_count - 1)).execute().data
    ]
    return pl.DataFrame(all_data, infer_schema_length=2000)

def collect_season_data(
        season_id: str,
        table_name: str,
        connection: Client
):
    '''
    Collects a filterd season table from supabase with chunks of size 1k
    :param season_id: id of the season
    :param table_name: name of the table
    :param connection: the supabase connection
    :return:
    '''
    chunk_size = 1000
    row_count = connection.table(table_name).select(
        '*', head=True, count='exact'
    ).eq('season_id', season_id).execute().count
    all_data = [
        item
        for start in range(0, row_count, chunk_size)
        for item in
        connection.table(table_name).select('*').eq('season_id', season_id).range(start, min(start + chunk_size - 1, row_count - 1)).execute().data
    ]
    return pl.DataFrame(all_data)

def collect_season_statistics(
        season_id: str,
        connection: Client
):
    chunk_size = 1000
    def collect_filter(
            season_id: str,
            table_name: str,
            connection: Client
    ):
        row_count = connection.table(table_name).select(
        '*, schedule!inner(season_id)', head=True, count='exact'
        ).eq('schedule.season_id', season_id).execute().count
        chunk_size = 1000
        all_data = [
            item
            for start in range(0, row_count, chunk_size)
            for item in
            connection.table(table_name).select(
                '*, schedule!inner(game_id,season_id)'
            ).eq(
                'schedule.season_id', season_id
            ).range(start, min(start + chunk_size - 1, row_count - 1)).execute().data
        ]
        return pl.DataFrame(all_data, infer_schema_length=2000).drop('game_id').unnest('schedule').drop('season_id')

    previous_df = collect_filter(season_id, 'statistics_previous', connection)
    recent_games_df = collect_filter(season_id, 'statistics_recent_games', connection)
    remainder_df = collect_filter(season_id, 'statistics_remainder', connection)
    season_df = collect_filter(season_id, 'statistics_season', connection)
    return previous_df, recent_games_df, remainder_df, season_df

def collect_season_filtered_table(
        season_id: str,
        table_name: str,
        connection: Client
):
    chunk_size = 1000
    row_count = connection.table(table_name).select(
        '*, schedule!inner(season_id)', head=True, count='exact'
    ).eq('schedule.season_id', season_id).execute().count
    all_data = [
        item
        for start in range(0, row_count, chunk_size)
        for item in
        connection.table(table_name).select(
            '*, schedule!inner(game_id,season_id)'
        ).eq(
            'schedule.season_id', season_id
        ).range(start, min(start + chunk_size - 1, row_count - 1)).execute().data
    ]
    return pl.DataFrame(all_data).drop('game_id').unnest('schedule').drop('season_id')

def collect_boxscore_table(
        season_id: str,
        connection: Client
):
    chunk_size = 1000
    row_count = connection.table('boxscore').select(
        'game_id,schedule!inner(season_id)', head=True, count='exact'
    ).eq('schedule.season_id', season_id).execute().count
    all_data = [
        item
        for start in range(0, row_count, chunk_size)
        for item in
        connection.table('boxscore').select(
            'game_id,schedule!inner(game_id,season_id)'
        ).eq(
            'schedule.season_id', season_id
        ).range(start, min(start + chunk_size - 1, row_count - 1)).execute().data
    ]
    return pl.DataFrame(all_data).drop('game_id').unnest('schedule')