from supabase_helper import fetch_filtered_table
from typing import Union, List
import polars as pl

def get_team_schedules(
        df: pl.DataFrame
) -> List[pl.DataFrame]:
    """
    Gets individual schedules for all the teams
    :param df: A seasons schedule
    :return: A list containing each teams games / schedule
    """
    team_ids = sorted(list(set(df['home_team_id']).union(set(df['guest_team_id']))))
    team_schedules = [
        df.filter(
            (pl.col('home_team_id') == team_id) |
            (pl.col('guest_team_id') == team_id)
        ).sort('date')
        for team_id in team_ids
    ]
    return team_schedules


def expected_outcome(
        elo_team: float,
        elo_opponent: float
) -> float:
    """
    Calculates the expected outcome based on the elo scores
    :param elo_team: elo of team 1
    :param elo_opponent: elo of team 2
    :return: float value of expected outcome
    """
    return 1 / (1 + 10 ** ((elo_opponent - elo_team) / 400))


def mov_multiplier(
        mov: float,
        elo_diff: float
) -> float:
    """
    calculates the margin of victory multiplier
    :param mov: margin of victory
    :param elo_diff: absolute difference in elo values
    :return: the multipler as a float
    """
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)


def elo_season(
        df: pl.DataFrame,
        initial_elo: Union[int, pl.DataFrame] = 1500,
        k_factor: int = 20,
        home_court_advantage: int = 100
) -> pl.DataFrame:
    """
    Calculates the elo rating of team through a season
    :param df: games of the season
    :param initial_elo: either initial int value to use as starting point or dataframe
    containing the elo scores from the previous season
    :param k_factor: k-value (20 based on 538 article)
    :param home_court_advantage: factor for homecourt advantage (100 based on 538 article)
    :return: df containing the elo before and after each game for each team
    """
    elo_ratings = {}
    team_ids = sorted(list(set(df['home_team_id'])))

    if isinstance(initial_elo, int):
        for team in team_ids:
            elo_ratings[team] = [initial_elo]
    else:
        initial_elo = initial_elo.sort('date').group_by(['team_id']).tail(1).with_columns([
            ((0.75 * pl.col('elo_after')) + (0.25 * 1505)).alias('elo')
        ]).select([
            pl.col(['team_id', 'elo'])
        ])
        for row in initial_elo.iter_rows():
            elo_ratings[row[0]] = [row[1]]
    for row in df.iter_rows():
        home_team = row[3]
        away_team = row[4]
        elo_home = elo_ratings[home_team][-1]
        elo_away = elo_ratings[away_team][-1]
        elo_home_adjusted = elo_home + home_court_advantage
        expected_home = expected_outcome(elo_home_adjusted, elo_away)
        expected_away = expected_outcome(elo_away, elo_home_adjusted)
        margin_of_victory = abs(row[5] - row[6])
        underdog_victory = ((elo_home_adjusted > elo_away) & (row[5] < row[6])) | (
                    (elo_home_adjusted < elo_away) & (row[5] > row[6]))
        if underdog_victory:
            elo_diff = -1 * abs(elo_home_adjusted - elo_away)
        else:
            elo_diff = abs(elo_home_adjusted - elo_away)
        mov_multi = mov_multiplier(margin_of_victory, elo_diff)
        if row[5] > row[6]:
            result_home, result_away = 1, 0
        else:
            result_home, result_away = 0, 1
        elo_ratings[home_team].append(elo_home + k_factor * mov_multi * (result_home - expected_home))
        elo_ratings[away_team].append(elo_away + k_factor * mov_multi * (result_away - expected_away))
    team_schedules = get_team_schedules(df)
    elo_df = pl.concat([
        team_schedules[i].select(pl.col(['game_id', 'date'])).with_columns([
            pl.lit(team_ids[i]).alias('team_id'),
            pl.Series('elo_before', elo_ratings[team_ids[i]][:-1], dtype=pl.Float64),
            pl.Series('elo_after', elo_ratings[team_ids[i]][1:], dtype=pl.Float64)
        ])
        for i in range(len(team_ids))
    ])
    return elo_df

def elo_update(supabase):
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
    elo_w_schedule_last_season = elo_last_season.join(
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