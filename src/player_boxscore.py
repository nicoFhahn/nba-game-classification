import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import calendar
    import json
    import time
    from datetime import date, timedelta


    import polars as pl
    from supabase import Client, create_client
    from tqdm import tqdm

    from bbref import start_driver, scrape_missing_player_boxscores
    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_distinct_column
    from google.cloud import secretmanager
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from requests.exceptions import HTTPError
    from selenium.common.exceptions import TimeoutException, WebDriverException
    return (
        create_client,
        fetch_distinct_column,
        fetch_entire_table,
        json,
        pl,
        secretmanager,
        start_driver,
    )


@app.cell
def _(create_client, json, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    return (supabase,)


@app.cell
def _(fetch_distinct_column, fetch_entire_table, start_driver, supabase):
    driver = start_driver()
    schedule = fetch_entire_table(supabase, "schedule").sort("game_id")
    player_boxscore = fetch_distinct_column(supabase, "player-boxscore", "game_id")
    # scrape_missing_player_boxscores(supabase, driver)
    return (driver,)


@app.cell
def _(driver, missing_games):
    row = missing_games.to_dicts()[0]
    home_id_basic = f"box-{row['home_abbrev']}-game-basic"
    home_id_advanced = f"box-{row['home_abbrev']}-game-advanced"
    guest_id_basic = f"box-{row['guest_abbrev']}-game-basic"
    guest_id_advanced = f"box-{row['guest_abbrev']}-game-advanced"
    print(row["game_url"])
    driver.get(row["game_url"])
    return (row,)


@app.cell
def _():
    return


@app.cell(disabled=True)
def _(
    pl,
    res_guest_advanced,
    res_guest_basic,
    res_home_advanced,
    res_home_basic,
    row,
):
    res_home_basic_df = pl.DataFrame(res_home_basic["players_stats"])
    res_home_basic_df = res_home_basic_df.with_columns(
        pl.col("mp").str.split(":").list.eval(
            pl.element().first().cast(pl.Float64, strict=False) +
            pl.element().last().cast(pl.Float64, strict=False) / 60
        ).list.first()
    ).with_columns(
        pl.col(res_home_basic_df.columns[1:21]).cast(pl.Float64, strict=False)
    ).with_columns(
        pl.col("player_id").cast(pl.String)
    )

    res_home_advanced_df = pl.DataFrame(res_home_advanced["players_stats"])
    res_home_advanced_df = res_home_advanced_df.with_columns(
        pl.col("mp").str.split(":").list.eval(
            pl.element().first().cast(pl.Float64, strict=False) +
            pl.element().last().cast(pl.Float64, strict=False) / 60
        ).list.first()
    ).with_columns(
        pl.col(res_home_advanced_df.columns[1:16]).cast(pl.Float64, strict=False)
    ).with_columns(
        pl.col("player_id").cast(pl.String)
    )

    res_home_df = res_home_basic_df.join(
        res_home_advanced_df,
        on=["player_id", "player_name", "mp"]
    ).with_columns(
        pl.lit(row["home_team"]).alias("team_id"),
        pl.lit(row["game_id"]).alias("game_id")
    )

    res_guest_basic_df = pl.DataFrame(res_guest_basic["players_stats"])
    res_guest_basic_df = res_guest_basic_df.with_columns(
        pl.col("mp").str.split(":").list.eval(
            pl.element().first().cast(pl.Float64, strict=False) +
            pl.element().last().cast(pl.Float64, strict=False) / 60
        ).list.first()
    ).with_columns(
        pl.col(res_guest_basic_df.columns[1:21]).cast(pl.Float64, strict=False)
    ).with_columns(
        pl.col("player_id").cast(pl.String)
    )

    res_guest_advanced_df = pl.DataFrame(res_guest_advanced["players_stats"])
    res_guest_advanced_df = res_guest_advanced_df.with_columns(
        pl.col("mp").str.split(":").list.eval(
            pl.element().first().cast(pl.Float64, strict=False) +
            pl.element().last().cast(pl.Float64, strict=False) / 60
        ).list.first()
    ).with_columns(
        pl.col(res_guest_advanced_df.columns[1:16]).cast(pl.Float64, strict=False)
    ).with_columns(
        pl.col("player_id").cast(pl.String)
    )

    res_guest_df = res_guest_basic_df.join(
        res_guest_advanced_df,
        on=["player_id", "player_name", "mp"]
    ).with_columns(
        pl.lit(row["guest_team"]).alias("team_id"),
        pl.lit(row["game_id"]).alias("game_id")
    )

    res_df = pl.concat([res_home_df, res_guest_df])
    return


@app.cell(disabled=True)
def _(fetch_entire_table, supabase):
    pbs = fetch_entire_table(supabase, "player-boxscore")
    unique_pbs = pbs.group_by(["team_id", "game_id", "player_id"]).head(1)
    return pbs, unique_pbs


@app.cell(disabled=True)
def _(pbs, pl, supabase, unique_pbs):
    duplicate_ids = pbs.filter(
        ~pl.col("id").is_in(unique_pbs["id"].to_list())
    )["id"]
    deleted_count = 0
    errors = []

    for idx, duplicate_id in enumerate(duplicate_ids):
        try:
            supabase.table("player-boxscore").delete().eq("id", duplicate_id).execute()
            deleted_count += 1
            if (deleted_count) % 100 == 0:
                print(f"Deleted {deleted_count}/{len(duplicate_ids)}")
        except Exception as e:
            errors.append((duplicate_id, str(e)))
            if len(errors) <= 5:  # Only print first 5 errors
                print(f"Error deleting {duplicate_id}: {e}")

    print(f"Total deleted: {deleted_count}")
    print(f"Total errors: {len(errors)}")
    return


if __name__ == "__main__":
    app.run()
