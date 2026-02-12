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

    from bbref import start_driver
    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_month_data
    from google.cloud import secretmanager
    return (
        create_client,
        fetch_entire_table,
        json,
        pl,
        secretmanager,
        start_driver,
        time,
        tqdm,
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
def _(By):
    def scrape_table_stats(table_id, driver):
        table = driver.find_element(By.ID, table_id)

        # Get column names from thead
        thead = table.find_element(By.TAG_NAME, "thead")
        thead_rows = thead.find_elements(By.TAG_NAME, "tr")
        header_cells = thead_rows[1].find_elements(By.TAG_NAME, "th")[1:]
        column_names = [cell.get_attribute("data-stat") for cell in header_cells]

        # Get team stats from tfoot
        tfoot = table.find_element(By.TAG_NAME, "tfoot")
        tfoot_rows = tfoot.find_elements(By.TAG_NAME, "tr")
        stat_cells = tfoot_rows[0].find_elements(By.TAG_NAME, "td")
        stat_values = [cell.text.strip() for cell in stat_cells]
        team_stats = dict(zip(column_names, stat_values))

        # Get player stats from tbody
        tbody = table.find_element(By.TAG_NAME, "tbody")
        player_rows = tbody.find_elements(By.TAG_NAME, "tr")

        players_stats = []
        for row in player_rows:
            # Skip rows that don't have the standard structure (e.g., section headers)
            if not row.find_elements(By.TAG_NAME, "td"):
                continue

            # Get player_id and player_name from the first th element
            first_th = row.find_element(By.TAG_NAME, "th")
            player_id = first_th.get_attribute("data-append-csv")
            player_name = first_th.text.strip()

            # Get all stat values from td elements
            stat_cells = row.find_elements(By.TAG_NAME, "td")
            stat_values = [cell.text.strip() for cell in stat_cells]

            # Create player stats dict
            player_stats = dict(zip(column_names, stat_values))
            player_stats['player_id'] = player_id
            player_stats['player_name'] = player_name

            players_stats.append(player_stats)

        return {
            'team_stats': team_stats,
            'players_stats': players_stats
        }
    return (scrape_table_stats,)


@app.cell
def _():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import WebDriverException
    return By, EC, WebDriverWait


@app.cell
def _(fetch_entire_table, pl, supabase):
    schedule = fetch_entire_table(supabase, "schedule").sort("game_id")
    player_boxscore = fetch_entire_table(supabase, "player-boxscore").sort("game_id")
    missing_games = schedule.filter(
        ~pl.col("game_id").is_in(player_boxscore["game_id"].unique().to_list())
    )
    return (missing_games,)


@app.cell
def _(
    By,
    EC,
    WebDriverWait,
    game_url,
    missing_games,
    pl,
    scrape_table_stats,
    start_driver,
    supabase,
    time,
    tqdm,
):
    driver = start_driver()
    for i, row in enumerate(tqdm(missing_games.to_dicts(), ncols=100)):

        # Restart driver every 50 iterations
        if i > 0 and i % 50 == 0:
            driver.quit()
            time.sleep(2)
            driver = start_driver()
        home_id_basic = f"box-{row["home_abbrev"]}-game-basic"
        home_id_advanced = f"box-{row["home_abbrev"]}-game-advanced"
        guest_id_basic = f"box-{row["guest_abbrev"]}-game-basic"
        guest_id_advanced = f"box-{row["guest_abbrev"]}-game-advanced"
        max_retries = 3
        retry_delay = 1800  # seconds

        for attempt in range(max_retries):
            try:
                driver.get(row["game_url"])
                break  # Success, exit the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to load {game_url} after {max_retries} attempts")
                    raise  # Re-raise the exception after all retries exhausted

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, home_id_basic)))
        wait.until(EC.presence_of_element_located((By.ID, guest_id_basic)))
        wait.until(EC.presence_of_element_located((By.ID, home_id_advanced)))
        wait.until(EC.presence_of_element_located((By.ID, guest_id_advanced)))
        res_home_basic = scrape_table_stats(home_id_basic, driver)
        res_home_advanced = scrape_table_stats(home_id_advanced, driver)
        res_guest_basic = scrape_table_stats(guest_id_basic, driver)
        res_guest_advanced = scrape_table_stats(guest_id_advanced, driver)
        res_home_basic_df = pl.DataFrame(res_home_basic["players_stats"])
        res_home_basic_df = res_home_basic_df.with_columns(
            pl.col("mp").str.split(":").list.eval(
                pl.element().first().cast(pl.Float64, strict=False) +
                pl.element().last().cast(pl.Float64, strict=False) / 60
            ).list.first()
        ).with_columns(
            pl.col(res_home_basic_df.columns[1:21]).cast(pl.Float64, strict=False)
        )
        res_home_advanced_df = pl.DataFrame(res_home_advanced["players_stats"])
        res_home_advanced_df = res_home_advanced_df.with_columns(
            pl.col("mp").str.split(":").list.eval(
                pl.element().first().cast(pl.Float64, strict=False) +
                pl.element().last().cast(pl.Float64, strict=False) / 60
            ).list.first()
        ).with_columns(
            pl.col(res_home_advanced_df.columns[1:16]).cast(pl.Float64, strict=False)
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
        )
        res_guest_advanced_df = pl.DataFrame(res_guest_advanced["players_stats"])
        res_guest_advanced_df = res_guest_advanced_df.with_columns(
            pl.col("mp").str.split(":").list.eval(
                pl.element().first().cast(pl.Float64, strict=False) +
                pl.element().last().cast(pl.Float64, strict=False) / 60
            ).list.first()
        ).with_columns(
            pl.col(res_guest_advanced_df.columns[1:16]).cast(pl.Float64, strict=False)
        )
        res_guest_df = res_guest_basic_df.join(
            res_guest_advanced_df,
            on=["player_id", "player_name", "mp"]
        ).with_columns(
            pl.lit(row["guest_team"]).alias("team_id"),
            pl.lit(row["game_id"]).alias("game_id")
        )
        res_df = pl.concat([res_home_df, res_guest_df])
        if res_df["plus_minus"].dtype==pl.String:
            res_df = res_df.with_columns([
                pl.col("plus_minus").str.replace("\+", "").cast(pl.Float64, strict=False)
            ])
        supabase.table("player-boxscore").insert(res_df.to_dicts()).execute()
        # time.sleep(2)
    return


if __name__ == "__main__":
    app.run()
