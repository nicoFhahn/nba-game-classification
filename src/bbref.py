import polars as pl
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from selenium.common.exceptions import WebDriverException
import time

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
    
    for row in rows:
        th_elements = row.find_elements(By.TAG_NAME, "th")
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(th_elements) == 0 or len(cells) < 6:
            continue
        try:
            date = th_elements[0].text.strip()
            visitor = cells[1].text.strip()
            home = cells[3].text.strip()
            
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
        except Exception as e:
            continue
    
    df = pl.DataFrame({
        'Date': dates,
        'Visitor': visitors,
        'Home': homes,
        'home_abbrev': home_abbrevs,
        'guest_abbrev': visitor_abbrevs,
        'Box_Score_URL': box_score_links
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
        home_id_advanced= f"box-{home_abbrev}-game-advanced"
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