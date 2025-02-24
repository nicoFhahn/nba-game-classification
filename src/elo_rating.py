import os
import sys
import polars as pl
from typing import Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_wrangling import get_team_schedules
def expected_outcome(elo_team, elo_opponent):
    return 1 / (1 + 10 ** ((elo_opponent - elo_team) / 400))


def mov_multiplier(mov, elo_diff):
    return ((mov + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)


def elo_season(
        df: pl.DataFrame,
        initial_elo: Union[int, pl.DataFrame] = 1500,
        k_factor: int = 20,
        home_court_advantage: int = 100
) -> pl.DataFrame:
    elo_ratings = {}
    team_ids = list(set(df["home_team_id"]))

    if isinstance(initial_elo, int):
        for team in team_ids:
            elo_ratings[team] = [initial_elo]
    else:
        initial_elo = initial_elo.sort("date").group_by(["team_id"]).tail(1).with_columns([
            ((0.75 * pl.col("elo_after")) + (0.25 * 1505)).alias("elo")
        ]).select([
            pl.col(["team_id", "elo"])
        ])
        for row in initial_elo.iter_rows():
            elo_ratings[row[0]] = [row[1]]
    for row in df.iter_rows():
        home_team = row[2]
        away_team = row[3]
        elo_home = elo_ratings[home_team][-1]
        elo_away = elo_ratings[away_team][-1]
        elo_home_adjusted = elo_home + home_court_advantage
        expected_home = expected_outcome(elo_home_adjusted, elo_away)
        expected_away = expected_outcome(elo_away, elo_home_adjusted)
        margin_of_victory = abs(row[4] - row[5])
        underdog_victory = ((elo_home_adjusted > elo_away) & (row[4] < row[5])) | (
                    (elo_home_adjusted < elo_away) & (row[4] > row[5]))
        if underdog_victory:
            elo_diff = -1 * abs(elo_home_adjusted - elo_away)
        else:
            elo_diff = abs(elo_home_adjusted - elo_away)
        mov_multi = mov_multiplier(margin_of_victory, elo_diff)
        if row[4] > row[5]:
            result_home, result_away = 1, 0
        else:
            result_home, result_away = 0, 1
        elo_ratings[home_team].append(elo_home + k_factor * mov_multi * (result_home - expected_home))
        elo_ratings[away_team].append(elo_away + k_factor * mov_multi * (result_away - expected_away))
    team_schedules = get_team_schedules(df)
    elo_df = pl.concat([
        team_schedules[i].select(pl.col(["game_id", "date"])).with_columns([
            pl.lit(team_ids[i]).alias("team_id"),
            pl.Series("elo_before", elo_ratings[team_ids[i]][:-1], dtype=pl.Float64),
            pl.Series("elo_after", elo_ratings[team_ids[i]][1:], dtype=pl.Float64)
        ])
        for i in range(len(team_ids))
    ])
    return elo_df