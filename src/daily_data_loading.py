import json
import os
import polars as pl
from datetime import date, timedelta, datetime
from data_wrangling import load_season, record_current_season
from elo_rating import elo_season
from data_collection import season

files = os.listdir('data')
schedule_files = [file for file in files if file.startswith('schedule')]
latest_schedule_file = max(schedule_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
q = pl.scan_parquet(
    os.path.join('data', latest_schedule_file)
).select(pl.col('date')).max()
with open('data/season_dates.json', 'r') as f:
    season_dates = json.loads(f.read())
newest_date = q.collect()['date'][0]
if (date.today().year != latest_schedule_file.split('_')[-1].split('.')[0]) and date.today().month >= 9:
    season(
        start_date=date.today(),
        end_date=date.today() + timedelta(days=3),
        file_year=date.today().year,
        folder='data'
    )
else:
    season(
        start_date=newest_date,
        end_date=newest_date + timedelta(days=3),
        file_year=date.today().year - 1,
        folder='data'
    )
season_2024 = load_season(
    ['game_list_2023.parquet', 'game_list_2024.parquet'],
    ['schedule_2023.parquet', 'schedule_2024.parquet'],
    'data',
    datetime.strptime(season_dates['2024']['all_star'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_start'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_end'], '%Y-%m-%d').date()
)
season_2024.write_parquet('data/season_2024.parquet')
h2h_current_year = load_season(
    ['game_list_2023.parquet', 'game_list_2024.parquet'],
    ['schedule_2023.parquet', 'schedule_2024.parquet'],
    'data',
    datetime.strptime(season_dates['2024']['all_star'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_start'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_end'], '%Y-%m-%d').date(),
    return_h2h=True
)
h2h_current_year.write_parquet('data/h2h_2024.parquet')
rec_current_year = record_current_season(
    datetime.strptime(season_dates['2024']['all_star'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_start'], '%Y-%m-%d').date(),
    datetime.strptime(season_dates['2024']['play_in_end'], '%Y-%m-%d').date(),
    'schedule_2024.parquet',
    'game_list_2024.parquet',
    'data'
)
rec_current_year.write_parquet('data/record_2024.parquet')
df_list = [
    record_current_season(
        datetime.strptime(season_dates[key]['all_star'], '%Y-%m-%d').date(),
        datetime.strptime(season_dates[key]['play_in_start'], '%Y-%m-%d').date(),
        datetime.strptime(season_dates[key]['play_in_end'], '%Y-%m-%d').date(),
        f'schedule_{key}.parquet',
        f'game_list_{key}.parquet',
        'data'
    ).drop_nulls('points_home') for key in season_dates.keys()
]
elo_df_list = []
for i in range(len(df_list)):
    if i == 0:
        elo_df_list.append(
            elo_season(df_list[i])
        )
    else:
        elo_df_list.append(
            elo_season(df_list[i], elo_df_list[i - 1])
        )
elo_df = pl.concat(elo_df_list)
current_elo = elo_df.sort("date").group_by("team_id").tail(1).select(["team_id", "elo_after"])
elo_df.write_parquet('data/elo_score.parquet')

season_df = pl.concat([
    pl.read_parquet('data/season_2021.parquet'),
    pl.read_parquet('data/season_2022.parquet'),
    pl.read_parquet('data/season_2023.parquet'),
    pl.read_parquet('data/season_2024.parquet')
])
season_df = season_df.join(
    elo_df.drop(['elo_after', 'date']),
    left_on=['game_id', 'home_team_id'],
    right_on=['game_id', 'team_id'],
    how='left'
).join(
    elo_df.drop(['elo_after', 'date']),
    left_on=['game_id', 'away_team_id'],
    right_on=['game_id', 'team_id'],
    how='left'
).join(
    current_elo,
    left_on="home_team_id",
    right_on="team_id"
).join(
    current_elo,
    left_on="away_team_id",
    right_on="team_id"
).with_columns([
    pl.coalesce(pl.col("elo_before"), pl.col("elo_after")).alias("elo_home_team"),
    pl.coalesce(pl.col("elo_before_right"), pl.col("elo_after_right")).alias("elo_away_team")
]).drop([
    "elo_before", "elo_before_right", "elo_after", "elo_after_right"
])
season_df.write_parquet('data/season_df.parquet')