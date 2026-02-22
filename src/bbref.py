import calendar
import traceback
import polars as pl
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import date, timedelta
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, WebDriverException
from supabase_helper import fetch_entire_table, fetch_distinct_column, fetch_month_data
import time
import time
from selenium.common.exceptions import WebDriverException, TimeoutException

import time
import polars as pl
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

def start_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--incognito')  # Fresh session each time
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_season(end_year: int, driver):
    url = f'https://www.basketball-reference.com/leagues/NBA_{end_year}_games.html'
    driver.get(url)
    time.sleep(3)
    month_filter_elements = driver.find_element(By.CLASS_NAME, "filter").find_elements(By.TAG_NAME, "a")
    month_hrefs = [el.get_attribute("href") for el in month_filter_elements]
    month_dfs = [scrape_month(href, driver) for href in month_hrefs]
    season_df = pl.concat(month_dfs)
    season_df = season_df.with_columns([
        pl.lit(end_year).alias("season_id")
    ]).rename({
        "Date": "date",
        "Visitor": "guest_team",
        "Home": "home_team",
        "Box_Score_URL": "game_url"
    })
    return season_df

def scrape_month(month_url: str, driver):
    driver.get(month_url)
    wait = WebDriverWait(driver, 10)
    table = wait.until(EC.presence_of_element_located((By.ID, "schedule")))
    rows = table.find_elements(By.TAG_NAME, "tr")
    dates = []
    visitors = []
    homes = []
    home_abbrevs = []
    visitor_abbrevs = []
    box_score_links = []
    locations = []
    
    for row in rows:
        th_elements = row.find_elements(By.TAG_NAME, "th")
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(th_elements) == 0 or len(cells) < 6:
            continue
        try:
            date = th_elements[0].text.strip()
            visitor = cells[1].text.strip()
            home = cells[3].text.strip()
            arena = cells[9].text.strip()
            
            visitor_link = cells[1].find_elements(By.TAG_NAME, "a")
            if not visitor_link:
                continue
            visitor_href = visitor_link[0].get_attribute("href")
            visitor_abbrev = visitor_href.split("/teams/")[1].split("/")[0] if "/teams/" in visitor_href else ""
            
            home_link = cells[3].find_elements(By.TAG_NAME, "a")
            if not home_link:
                continue
            home_href = home_link[0].get_attribute("href")
            home_abbrev = home_href.split("/teams/")[1].split("/")[0] if "/teams/" in home_href else ""
            
            link_elements = cells[5].find_elements(By.TAG_NAME, "a")
            box_score_link = link_elements[0].get_attribute("href") if link_elements else ""
            
            if date and visitor and home and home_abbrev:
                dates.append(date)
                visitors.append(visitor)
                homes.append(home)
                home_abbrevs.append(home_abbrev)
                visitor_abbrevs.append(visitor_abbrev)
                box_score_links.append(box_score_link)
                locations.append(arena)
        except Exception as e:
            continue
    
    df = pl.DataFrame({
        'Date': dates,
        'Visitor': visitors,
        'Home': homes,
        'home_abbrev': home_abbrevs,
        'guest_abbrev': visitor_abbrevs,
        'Box_Score_URL': box_score_links,
        'arena': locations
    }).with_columns([
        pl.col("Date").str.strptime(pl.Date, "%a, %b %d, %Y").alias("Date")
    ]).with_columns([
        (pl.col("Date").dt.strftime("%Y%m%d") + pl.col("home_abbrev")).alias("game_id")
    ])
    return df

def scrape_table_stats(table_id, driver):
    table = driver.find_element(By.ID, table_id)
    
    thead = table.find_element(By.TAG_NAME, "thead")
    thead_rows = thead.find_elements(By.TAG_NAME, "tr")
    header_cells = thead_rows[1].find_elements(By.TAG_NAME, "th")[1:]
    column_names = [cell.get_attribute("data-stat") for cell in header_cells]
    
    tfoot = table.find_element(By.TAG_NAME, "tfoot")
    tfoot_rows = tfoot.find_elements(By.TAG_NAME, "tr")
    stat_cells = tfoot_rows[0].find_elements(By.TAG_NAME, "td")
    stat_values = [cell.text.strip() for cell in stat_cells]
    
    stats_dict = dict(zip(column_names, stat_values))
    
    return stats_dict

def scrape_game(game_url, home_abbrev, guest_abbrev, game_id, driver, supabase):
    response = (
        supabase.table(
            "boxscore"
        ).select("game_id").eq("game_id", game_id).limit(1).execute()
    )
    exists = bool(response.data)
    if not exists:
        # Retry driver.get() until it succeeds
        max_retries = 3
        retry_delay = 1800  # seconds

        for attempt in range(max_retries):
            try:
                driver.get(game_url)
                break  # Success, exit the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to load {game_url} after {max_retries} attempts")
                    raise  # Re-raise the exception after all retries exhausted

        wait = WebDriverWait(driver, 10)

        home_id_basic = f"box-{home_abbrev}-game-basic"
        guest_id_basic = f"box-{guest_abbrev}-game-basic"
        home_id_advanced = f"box-{home_abbrev}-game-advanced"
        guest_id_advanced = f"box-{guest_abbrev}-game-advanced"

        wait.until(EC.presence_of_element_located((By.ID, home_id_basic)))
        wait.until(EC.presence_of_element_located((By.ID, guest_id_basic)))
        wait.until(EC.presence_of_element_located((By.ID, home_id_advanced)))
        wait.until(EC.presence_of_element_located((By.ID, guest_id_advanced)))
        home_stats_basic = pl.DataFrame(scrape_table_stats(home_id_basic, driver))
        home_stats_basic = home_stats_basic.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).alias(c)
            for c in home_stats_basic.columns
        ]).drop([
            "game_score", "plus_minus"
        ]).rename(
            lambda c: f"{c}_home"
        )
        home_stats_advanced = pl.DataFrame(scrape_table_stats(home_id_advanced, driver))
        home_stats_advanced = home_stats_advanced.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).alias(c)
            for c in home_stats_advanced.columns
        ]).select([
            "tov_pct", "off_rtg", "def_rtg"
        ]).rename(
            lambda c: f"{c}_home"
        )
        guest_stats_basic = pl.DataFrame(scrape_table_stats(guest_id_basic, driver))
        guest_stats_basic = guest_stats_basic.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).alias(c)
            for c in guest_stats_basic.columns
        ]).drop([
            "game_score", "plus_minus"
        ]).rename(
            lambda c: f"{c}_guest"
        )
        guest_stats_advanced = pl.DataFrame(scrape_table_stats(guest_id_advanced, driver))
        guest_stats_advanced = guest_stats_advanced.with_columns([
            pl.col(c).cast(pl.Float64, strict=False).alias(c)
            for c in guest_stats_advanced.columns
        ]).select([
            "tov_pct", "off_rtg", "def_rtg"
        ]).rename(
            lambda c: f"{c}_guest"
        )
        boxscore_df = pl.concat([
            home_stats_basic, home_stats_advanced,
            guest_stats_basic, guest_stats_advanced
        ], how="horizontal").with_columns(
            pl.lit(game_id).alias("game_id")
        )
        supabase.table("boxscore").insert(boxscore_df.to_dicts()).execute()
        time.sleep(4)

def scrape_player_table_stats(table_id, driver):
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

def scrape_missing_player_boxscores(supabase, driver):
    while True:
        try:
            # Refetch missing games on each restart
            schedule = fetch_entire_table(supabase, "schedule").sort("game_id")
            player_boxscore = fetch_distinct_column(supabase, "player-boxscore", "game_id")
            missing_games = schedule.filter(
                (~pl.col("game_id").is_in(player_boxscore)) &
                (pl.col("game_url") != "")
            )

            if len(missing_games) == 0:
                print("No missing games remaining!")
                break

            print(f"Processing {len(missing_games)} missing games...")

            for i, row in enumerate(tqdm(missing_games.to_dicts(), ncols=100)):
                # Restart driver every 50 iterations
                if i > 0 and i % 50 == 0:
                    driver.quit()
                    time.sleep(2)
                    driver = start_driver()

                home_id_basic = f"box-{row['home_abbrev']}-game-basic"
                home_id_advanced = f"box-{row['home_abbrev']}-game-advanced"
                guest_id_basic = f"box-{row['guest_abbrev']}-game-basic"
                guest_id_advanced = f"box-{row['guest_abbrev']}-game-advanced"
                max_retries = 30
                retry_delay = 60  # seconds

                # Retry logic for loading the page
                for attempt in range(max_retries):
                    try:
                        driver.get(row["game_url"])
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"Failed to load {row['game_url']} after {max_retries} attempts")
                            raise

                # Retry logic for waiting for page elements and scraping
                scrape_success = False
                for attempt in range(max_retries):
                    try:
                        wait = WebDriverWait(driver, 20)
                        wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")

                        wait.until(EC.presence_of_element_located((By.ID, home_id_basic)))
                        wait.until(EC.presence_of_element_located((By.ID, guest_id_basic)))
                        wait.until(EC.presence_of_element_located((By.ID, home_id_advanced)))
                        wait.until(EC.presence_of_element_located((By.ID, guest_id_advanced)))

                        res_home_basic = scrape_player_table_stats(home_id_basic, driver)
                        res_home_advanced = scrape_player_table_stats(home_id_advanced, driver)
                        res_guest_basic = scrape_player_table_stats(guest_id_basic, driver)
                        res_guest_advanced = scrape_player_table_stats(guest_id_advanced, driver)

                        scrape_success = True
                        break

                    except (TimeoutException, WebDriverException) as e:
                        error_text = str(e)
                        page_source = driver.page_source if driver else ""

                        if "504" in error_text or "504" in page_source or "Gateway Timeout" in page_source:
                            if attempt < max_retries - 1:
                                print(
                                    f"504 Gateway Timeout on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                driver.get(row["game_url"])
                            else:
                                print(
                                    f"Failed to scrape {row['game_url']} after {max_retries} attempts (504 Gateway Timeout)")
                                break
                        else:
                            print(f"Error on attempt {attempt + 1}: {e}")
                            print(driver.find_element(By.TAG_NAME, "body").text)
                            break
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        print(driver.find_element(By.TAG_NAME, "body").text)
                        break

                if not scrape_success:
                    print(f"Skipping game {row['game_id']} due to scraping failure")
                    continue

                # Data processing
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
                ).drop(pl.col("player_id"))

                res_home_df = res_home_basic_df.join(
                    res_home_advanced_df,
                    on=["player_name", "mp"]
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
                ).drop(pl.col("player_id"))

                res_guest_df = res_guest_basic_df.join(
                    res_guest_advanced_df,
                    on=["player_name", "mp"]
                ).with_columns(
                    pl.lit(row["guest_team"]).alias("team_id"),
                    pl.lit(row["game_id"]).alias("game_id")
                )

                res_df = pl.concat([res_home_df, res_guest_df])

                # Retry logic for database insert
                for attempt in range(max_retries):
                    try:
                        supabase.table("player-boxscore").insert(res_df.to_dicts()).execute()
                        break
                    except Exception as e:
                        error_text = str(e)
                        if "504" in error_text or "Gateway Timeout" in error_text:
                            if attempt < max_retries - 1:
                                print(
                                    f"504 Gateway Timeout on database insert, attempt {attempt + 1}. Retrying in 60 seconds...")
                                time.sleep(60)
                            else:
                                print(f"Failed to insert data for game {row['game_id']} after {max_retries} attempts")
                                raise
                        else:
                            print(f"Database error: {e}")
                            raise

                time.sleep(2)

            # If we reach here, all games processed successfully
            break

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"CRASH DETECTED")
            print(f"Error: {e}")
            print(f"Traceback:")
            traceback.print_exc()
            print(f"{'=' * 80}\n")

            # Clean up driver
            try:
                driver.quit()
            except:
                pass

            # Wait before restart
            print(f"Waiting 30 seconds before restarting...")
            time.sleep(30)

            # Restart driver and loop will refetch missing games
            print(f"Restarting and refetching missing games...")
            driver = start_driver()

def update_current_month(supabase, driver):
    d = date.today() - timedelta(days=(date.today().day == 1))
    year = date.today().year + (date.today().month >= 10)
    month_number = d.month
    month_name = calendar.month_name[month_number].lower()
    schedule_current_month = fetch_month_data(
        supabase, "schedule", d, date_column='date', page_size=1000
    )
    print("fetched current month from supabase")
    link = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month_name}.html"
    current_month = scrape_month(link, driver)
    print("fetched current month from bbref")
    current_month = current_month.rename({
        "Date": "date",
        "Home": "home_team",
        "Visitor": "guest_team",
        "Box_Score_URL": "game_url"
    }).with_columns([
        pl.lit(year).alias("season_id"),
        pl.lit(False).alias("is_scraped")
    ])[schedule_current_month.columns]
    response = (
        supabase
        .table('schedule')
        .delete()
        .in_('game_id', schedule_current_month["game_id"].to_list())
        .execute()
    )
    print("deleted old data")
    supabase.table('schedule').insert(
        current_month.with_columns(
            pl.col("date").cast(pl.String)
        ).to_dicts()
    ).execute()
    print("inserted new data")
    driver.quit()

def scrape_missing_team_boxscores(supabase):
    while True:
        try:
            season = fetch_entire_table(supabase, "schedule").sort("game_id")
            games = fetch_entire_table(supabase, "boxscore").sort("game_id")
            missing_games = season.filter(
                (~pl.col("game_id").is_in(games["game_id"].to_list())) &
                (pl.col("game_url") != "")
            )
            if missing_games.shape[0] > 0:
                driver = start_driver()

                for i, row in enumerate(tqdm(missing_games.to_dicts(), ncols=100)):

                    # Restart driver every 50 iterations
                    if i > 0 and i % 50 == 0:
                        driver.quit()
                        time.sleep(2)
                        driver = start_driver()

                    scrape_game(
                        row["game_url"],
                        row["home_abbrev"],
                        row["guest_abbrev"],
                        row["game_id"],
                        driver,
                        supabase
                    )

                driver.quit()
            else:
                # No more missing games, exit the loop
                break
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Retrying...")
            time.sleep(2)


def scrape_current_roster(team_id, driver):
    team_url = f"https://www.basketball-reference.com/teams/{team_id}/2026.html"
    attempts = 0
    max_retries = 5

    while attempts < max_retries:
        try:
            driver.get(team_url)
            wait = WebDriverWait(driver, 10)

            # 1. Wait for the main roster table (This must exist)
            roster_table = wait.until(EC.presence_of_element_located((By.ID, "roster")))

            # Extract Roster Data
            player_names = []
            player_ids = []
            player_tbody = roster_table.find_element(By.TAG_NAME, "tbody")
            player_rows = player_tbody.find_elements(By.TAG_NAME, "tr")

            for row in player_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    continue
                # The first <td> usually contains the name link
                link_element = cells[0].find_element(By.TAG_NAME, "a")
                player_names.append(link_element.text)
                p_id = link_element.get_attribute("href").split("/")[-1].split(".html")[0]
                player_ids.append(p_id)

            roster_df = pl.DataFrame({
                "player_name": player_names,
                "player_id": player_ids
            })

            # 2. Handle Injury Table (Optional - might not be on page)
            # We use find_elements (plural) so it returns an empty list instead of crashing if not found
            injury_elements = driver.find_elements(By.ID, "injuries")

            if injury_elements:
                injury_table = injury_elements[0]
                injury_tbody = injury_table.find_element(By.TAG_NAME, "tbody")
                injury_rows = injury_tbody.find_elements(By.TAG_NAME, "tr")

                player_injuries = []
                player_ids_injuries = []

                for row in injury_rows:
                    # Injuries often use <th> for the name link
                    name_link = row.find_element(By.TAG_NAME, "th").find_element(By.TAG_NAME, "a")
                    p_id_inj = name_link.get_attribute("href").split("/")[-1].split(".html")[0]
                    player_ids_injuries.append(p_id_inj)

                    injury_report = row.find_elements(By.TAG_NAME, "td")[-1].text
                    player_injuries.append(injury_report)

                injury_df = pl.DataFrame({
                    "player_id": player_ids_injuries,
                    "injury": player_injuries
                })
            else:
                # Create an empty DF with correct types if no injuries found
                injury_df = pl.DataFrame({
                    "player_id": pl.Series([], dtype=pl.Utf8),
                    "injury": pl.Series([], dtype=pl.Utf8)
                })

            # 3. Join and Return
            current_roster = roster_df.join(
                injury_df,
                on="player_id",
                how="left"
            ).with_columns(
                pl.lit(team_id).alias("team_id")
            )

            return current_roster

        except (TimeoutException, WebDriverException) as e:
            attempts += 1
            wait_time = attempts * 5
            print(f"Attempt {attempts} failed for {team_id}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # If the loop finishes without returning
    print(f"Max retries reached for {team_id}. Returning empty DataFrame.")
    return pl.DataFrame()