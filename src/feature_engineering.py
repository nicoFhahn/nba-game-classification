import polars as pl
import json
from datetime import datetime
from data_wrangling import load_season

with open("data/season_dates.json", "r") as f:
    season_dates = json.loads(f.read())
season_2021 = load_season(
    ["game_list_2020.parquet", "game_list_2021.parquet"],
    ["schedule_2020.parquet", "schedule_2021.parquet"],
    "data",
    datetime.strptime(season_dates["2021"]["all_star"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2021"]["play_in_start"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2021"]["play_in_end"], '%Y-%m-%d').date()
)
season_2022 = load_season(
    ["game_list_2021.parquet", "game_list_2022.parquet"],
    ["schedule_2021.parquet", "schedule_2022.parquet"],
    "data",
    datetime.strptime(season_dates["2022"]["all_star"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2022"]["play_in_start"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2022"]["play_in_end"], '%Y-%m-%d').date()
)
season_2023 = load_season(
    ["game_list_2022.parquet", "game_list_2023.parquet"],
    ["schedule_2022.parquet", "schedule_2023.parquet"],
    "data",
    datetime.strptime(season_dates["2023"]["all_star"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2023"]["play_in_start"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2023"]["play_in_end"], '%Y-%m-%d').date()
)
season_2024 = load_season(
    ["game_list_2023.parquet", "game_list_2024.parquet"],
    ["schedule_2023.parquet", "schedule_2024.parquet"],
    "data",
    datetime.strptime(season_dates["2024"]["all_star"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2024"]["play_in_start"], '%Y-%m-%d').date(),
    datetime.strptime(season_dates["2024"]["play_in_end"], '%Y-%m-%d').date()
)
season_list = [
    season_2021, season_2022, season_2023, season_2024
]
season_df = pl.concat(season_list)
elo_df = pl.read_parquet("data/elo_score.parquet").select(pl.col(["game_id", "team_id", "elo_before"])).rename({
    "elo_before": "elo"
})
season_df = season_df.join(
    elo_df,
    left_on=["game_id", "home_team_id"],
    right_on=["game_id", "team_id"]
).join(
    elo_df,
    left_on=["game_id", "away_team_id"],
    right_on=["game_id", "team_id"]
).rename({
    "elo": "elo_home_team",
    "elo_right": "elo_away_team"
})
season_df.write_parquet("data/season_df.parquet")