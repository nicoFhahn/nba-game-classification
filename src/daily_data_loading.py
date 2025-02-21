import os
import polars as pl
from datetime import date, timedelta
from data_wrangling import load_season, record_current_season
from data_collection import season
files = os.listdir("data")
schedule_files = [file for file in files if file.startswith("schedule")]
latest_schedule_file = max(schedule_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
q = pl.scan_parquet(
    os.path.join("data", latest_schedule_file)
).select(pl.col("date")).max()
newest_date = q.collect()["date"][0]
if (date.today().year != latest_schedule_file.split('_')[-1].split(".")[0]) and date.today().month >= 9:
    season(
        start_date = date.today(),
        end_date = date.today() + timedelta(days=1),
        file_year = date.today().year,
        folder = "data"
    )
else:
    season(
        start_date = newest_date,
        end_date = date.today() + timedelta(days=1),
        file_year = date.today().year - 1,
        folder = "data"
    )
season_2024 = load_season(
    ["game_list_2023.parquet", "game_list_2024.parquet"],
    ["schedule_2023.parquet", "schedule_2024.parquet"],
    "data",
    date(2025, 2, 16), date(2025, 4, 15), date(2025, 4, 18)
)
season_2024.write_parquet("data/season_2024.parquet")
h2h_current_year = load_season(
    ["game_list_2023.parquet", "game_list_2024.parquet"],
    ["schedule_2023.parquet", "schedule_2024.parquet"],
    "data",
    date(2025, 2, 16), date(2025, 4, 15), date(2025, 4, 18),
    return_h2h=True
)
h2h_current_year.write_parquet("data/h2h_2024.parquet")
rec_current_year = record_current_season(
    date(2025, 2, 16),
    date(2025, 4, 15),
    date(2025, 4, 18),
    "schedule_2024.parquet",
    "game_list_2024.parquet",
    "data"
)
rec_current_year.write_parquet("data/record_2024.parquet")