import os
from datetime import timedelta, datetime

import polars as pl
from supabase import create_client
from google.cloud import storage, secretmanager

from data_wrangling import load_season, record_current_season
from elo_rating import elo_season
from modelling import lgbm_model
from data_collection import (
    season,
    collect_season_statistics,
    collect_season_filtered_table,
    collect_all_data
)

secret_client = secretmanager.SecretManagerServiceClient()
response = secret_client.access_secret_version(request={'name':'projects/898760610238/secrets/supabase/versions/1'})
creds = eval(response.payload.data.decode("UTF-8"))
connection = create_client(creds['postgres']['project_url'], creds['postgres']['api_key'])

season_dates = pl.DataFrame(connection.table('season').select('*').execute().data).with_columns([
    pl.col('all_star_date').str.to_date(),
    pl.col('play_in_start').str.to_date(),
    pl.col('play_in_end').str.to_date()
])
res = connection.table('schedule').select('date, season_id').order('date', desc=True).limit(1).execute().data[0]
newest_date = datetime.strptime(res['date'], '%Y-%m-%d').date()
season_id = res['season_id']
print('Collecting Schedule and Boxscores')
season(
    start_date=newest_date,
    end_date=newest_date + timedelta(days=3),
    connection=connection,
    season_id=season_id
)
print('Collecting Season Statistics')
season_2024 = load_season(
    season_dates.filter(pl.col('season_id') == season_id)['all_star_date'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_start'][0],
    season_dates.filter(pl.col('season_id') == season_id)['play_in_end'][0],
    connection=connection,
    season_id=season_id
)
previous_df, recent_games_df, remainder_df, season_df = collect_season_statistics(season_id, connection)
new_data_1 = season_2024.filter(~pl.col('game_id').is_in(previous_df['game_id'])).select(previous_df.columns)
new_data_2 = season_2024.filter(~pl.col('game_id').is_in(recent_games_df['game_id'])).select(recent_games_df.columns)
new_data_3 = season_2024.filter(~pl.col('game_id').is_in(remainder_df['game_id'])).select(remainder_df.columns)
new_data_4 = season_2024.filter(~pl.col('game_id').is_in(season_df['game_id'])).select(season_df.columns)
if new_data_1.shape[0] > 0:
    response = (
        connection.table('statistics_previous').insert(
            new_data_1.to_dicts()
        ).execute()
    )
if new_data_2.shape[0] > 0:
    response = (
        connection.table('statistics_recent_games').insert(
            new_data_2.to_dicts()
        ).execute()
    )
if new_data_3.shape[0] > 0:
    response = (
        connection.table('statistics_remainder').insert(
            new_data_3.to_dicts()
        ).execute()
    )
if new_data_4.shape[0] > 0:
    response = (
        connection.table('statistics_season').insert(
            new_data_4.to_dicts()
        ).execute()
    )
update_date = remainder_df[["game_id", "is_home_win"]].join(
    season_2024[["game_id", "date", "is_home_win"]],
    on="game_id"
).filter(
    pl.col("is_home_win").is_null() & (pl.col("is_home_win_right").is_not_null())
)["date"].min()
update_data_1 = season_2024.filter(
    (pl.col("date") >= update_date) & (~pl.col("game_id").is_in(new_data_1["game_id"]))
).drop_nulls("fieldGoalsMade_previous_game_home_team").select(previous_df.columns)
update_data_2 = season_2024.filter(
    (pl.col("date") >= update_date) & (~pl.col("game_id").is_in(new_data_2["game_id"]))
).select(recent_games_df.columns)
update_data_3 = season_2024.filter(
    (pl.col("date") >= update_date) & (~pl.col("game_id").is_in(new_data_3["game_id"]))
).select(remainder_df.columns)
update_data_4 = season_2024.filter(
    (pl.col("date") >= update_date) & (~pl.col("game_id").is_in(new_data_4["game_id"]))
).drop_nulls("fieldGoalsMade_109_home_team").select(season_df.columns)
if update_data_1.shape[0] > 0:
    response = (
        connection.table('statistics_previous').upsert(
            update_data_1.to_dicts()
        ).execute()
    )
if update_data_2.shape[0] > 0:
    response = (
        connection.table('statistics_recent_games').upsert(
            update_data_2.to_dicts()
        ).execute()
    )
if update_data_3.shape[0] > 0:
    response = (
        connection.table('statistics_remainder').upsert(
            update_data_3.to_dicts()
        ).execute()
    )
if update_data_4.shape[0] > 0:
    response = (
        connection.table('statistics_season').upsert(
            update_data_4.to_dicts()
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
    elo_df_supabase[['game_id', 'team_id', 'elo_before']],
    on=['game_id', 'team_id'],
    how='left'
).filter(pl.col('elo_before-right').is_null()).drop('elo_before_right').to_dicts()
if len(new_data_7) > 0:
    response = (
        connection.table('elo').insert(
            new_data_7
        ).execute()
    )
print('Loading LGBM Model')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), '..', 'streamlit', 'credentials', 'cloud-key.json'
)
client = storage.Client()
bucket = client.get_bucket("lgbm")
mod = lgbm_model(
    connection = connection,
    bucket = bucket,
    data_origin="supabase"
)
mod.load_model()
mod.predict()