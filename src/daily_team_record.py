import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from datetime import date, timedelta

    import polars as pl
    from supabase import Client, create_client

    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_month_data
    from google.cloud import secretmanager
    return (
        create_client,
        fetch_entire_table,
        fetch_filtered_table,
        json,
        pl,
        secretmanager,
    )


@app.cell
def _(
    create_client,
    fetch_entire_table,
    fetch_filtered_table,
    json,
    secretmanager,
):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    schedule_current_season, games_current_season = fetch_filtered_table(
        supabase, "schedule", "boxscore", "season_id", "game_id"
    )
    schedule_current_season = schedule_current_season.join(
        games_current_season,
        on="game_id",
        how="left"
    )
    team_records = fetch_entire_table(supabase, "team-record")
    return schedule_current_season, supabase, team_records


@app.cell
def _(pl, schedule_current_season):
    game_results = schedule_current_season.with_columns(
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_home_win")
    ).select(["game_id", "date", "home_team", "guest_team", "season_id", "is_home_win", "home_abbrev", "guest_abbrev"])
    return (game_results,)


@app.cell
def _(game_results, pl):
    home_side = game_results.select([
        pl.col("game_id"),
        pl.col("season_id"),
        pl.col("date"),
        pl.col("home_abbrev").alias("abbrev"),
        pl.col("home_team").alias("team"),
        pl.col("is_home_win").alias("win")
    ])
    guest_side = game_results.select([
        pl.col("game_id"),
        pl.col("season_id"),
        pl.col("date"),
        pl.col("guest_abbrev").alias("abbrev"),
        pl.col("guest_team").alias("team"),
        ~pl.col("is_home_win").alias("win")
    ])
    record_df = pl.concat([home_side, guest_side]).sort(
        ["team", "season_id", "date", "game_id"]
    ).with_columns([
        pl.col("win").cast(pl.Int32)
          .cum_sum()
          .shift(1)
          .fill_null(0)
          .over(["team", "season_id"])
          .alias("wins_before"),

        pl.col("win").not_().cast(pl.Int32)
          .cum_sum()
          .shift(1)
          .fill_null(0)
          .over(["team", "season_id"])
          .alias("losses_before")
    ]).filter(
        ~(
            (pl.col("win").is_null()) &
            (pl.col("wins_before") == 0) & (pl.col("losses_before") == 0)
        )
    ).with_columns(
        pl.concat_str(
            pl.col("game_id"), pl.lit("_"), pl.col("abbrev")
        ).alias("id")
    )
    return (record_df,)


@app.cell
def _(pl, record_df, team_records):
    new_inserts = record_df.filter(
        ~pl.concat_str(pl.col("game_id"),pl.col("team")).is_in(
            team_records.with_columns(pl.concat_str(pl.col("game_id"), pl.col("team")))["game_id"].to_list()
        )
    ).drop("abbrev")
    return (new_inserts,)


@app.cell
def _(new_inserts, supabase):
    supabase.table('team-record').insert(new_inserts.drop(["date", "win"]).to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
