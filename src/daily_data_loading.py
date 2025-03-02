import json
from datetime import timedelta, datetime

import polars as pl
from supabase import create_client

from data_wrangling import load_season, record_current_season
from elo_rating import elo_season
from data_collection import (
    season,
    collect_season_statistics,
    collect_season_filtered_table,
    collect_all_data
)

with open('streamlit/credentials.json', 'r') as f:
    creds = json.loads(f.read())
    connection = create_client(creds['postgres']['project_url'], creds['postgres']['api_key'])

season_dates = pl.DataFrame(connection.table('season').select('*').execute().data).with_columns([
    pl.col('all_star_date').str.to_date(),
    pl.col('play_in_start').str.to_date(),
    pl.col('play_in_end').str.to_date()
])
res = connection.table('schedule').select('date, season_id').order('date', desc=True).limit(1).execute().data[0]
newest_date = datetime.strptime(res['date'], '%Y-%m-%d').date()
season_id = res['season_id']
season(
    start_date=newest_date,
    end_date=newest_date + timedelta(days=3),
    connection=connection,
    season_id=season_id
)

season_2024 = load_season(
    season_dates.filter(pl.col('season_id') == season_id)['all_star_date'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_start'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_end'][0],
    connection=connection,
    season_id=season_id
)
previous_df, recent_games_df, remainder_df, season_df = collect_season_statistics(season_id, connection)
new_data_1 = season_2024.filter(~pl.col('game_id').is_in(previous_df['game_id'])).select(previous_df.columns).to_dicts()
new_data_2 = season_2024.filter(~pl.col('game_id').is_in(recent_games_df['game_id'])).select(recent_games_df.columns).to_dicts()
new_data_3 = season_2024.filter(~pl.col('game_id').is_in(remainder_df['game_id'])).select(remainder_df.columns).to_dicts()
new_data_4 = season_2024.filter(~pl.col('game_id').is_in(season_df['game_id'])).select(season_df.columns).to_dicts()
if len(new_data_1) > 0:
    response = (
        connection.table('statistics_previous').insert(
            new_data_1
        ).execute()
    )
if len(new_data_2) > 0:
    response = (
        connection.table('statistics_recent_games').insert(
            new_data_2
        ).execute()
    )
if len(new_data_3) > 0:
    response = (
        connection.table('statistics_remainder').insert(
            new_data_3
        ).execute()
    )
if len(new_data_4) > 0:
    response = (
        connection.table('statistics_season').insert(
            new_data_4
        ).execute()
    )
h2h_current_year = load_season(
    season_dates.filter(pl.col('season_id') == season_id)['all_star_date'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_start'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_end'][0],
    connection=connection,
    season_id=season_id,
    return_h2h=True
)
h2h_supabase = collect_season_filtered_table(season_id, 'h2h', connection)
new_data_5  = h2h_current_year.filter(~pl.col('game_id').is_in(h2h_supabase['game_id'])).to_dicts()
if len(new_data_5) > 0:
    response = (
        connection.table('h2h').insert(
            new_data_5
        ).execute()
    )
rec_current_year = record_current_season(
    season_dates.filter(pl.col('season_id') == season_id)['all_star_date'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_start'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_end'][0],
    connection=connection,
    season_id=season_id,
).drop_nulls()
rec_current_year_supabase = collect_season_filtered_table(season_id, 'record', connection)
new_data_6  = rec_current_year.filter(~pl.col('game_id').is_in(rec_current_year_supabase['game_id'])).to_dicts()
if len(new_data_6) > 0:
    response = (
        connection.table('record').insert(
            new_data_6
        ).execute()
    )
df_list = []
for s_id in season_dates['season_id']:
    if s_id == season_id:
        df_list.append(rec_current_year)
    else:
        df_list.append(
            collect_season_filtered_table(s_id, 'record', connection)
        )
df_list = [
    rec_current_year if s_id == season_id else collect_season_filtered_table(s_id, 'record', connection)
    for s_id in season_dates['season_id']
]
schedule_df = collect_all_data('schedule', connection)
df_list = [df.join(schedule_df, on='game_id').drop('season_id') for df in df_list]
df_list[-1] = df_list[-1][df_list[0].columns]
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
elo_df_supabase = collect_all_data('elo', connection)
new_data_7 = elo_df.join(
    elo_df_supabase[['game_id', 'team_id']],
    on=['game_id', 'team_id'],
    how='left'
).filter(pl.col('game_id').is_null()).to_dicts()
if len(new_data_7) > 0:
    response = (
        connection.table('elo').insert(
            new_data_7
        ).execute()
    )