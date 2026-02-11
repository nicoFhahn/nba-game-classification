import polars as pl
import polars.selectors as cs
from datetime import timedelta


def add_overall_winning_pct(df):
    # Sort by date to ensure chronological order
    df = df.sort("date")
    
    # Create a helper column for wins
    df = df.with_columns([
        pl.col("is_home_win").alias("home_win"),
        (~pl.col("is_home_win")).alias("guest_win")
    ])
    
    # Process home team stats
    home_stats = (
        df.select(["date", "home_team", "home_win"])
        .rename({"home_team": "team", "home_win": "win"})
    )
    
    # Process guest team stats
    guest_stats = (
        df.select(["date", "guest_team", "guest_win"])
        .rename({"guest_team": "team", "guest_win": "win"})
    )
    
    # Combine both perspectives
    all_games = pl.concat([home_stats, guest_stats]).sort("date")
    
    # Calculate cumulative stats per team, SHIFTED to show stats BEFORE each game
    cumulative_stats = (
        all_games
        .group_by("team")
        .agg([
            pl.col("date"),
            pl.col("win").cum_sum().shift(1, fill_value=0).alias("cumulative_wins"),
            pl.int_range(pl.len()).alias("cumulative_games")
        ])
        .explode(["date", "cumulative_wins", "cumulative_games"])
    )
    
    # Join back to get stats as of each date for home team
    df = df.join(
        cumulative_stats,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left"
    ).rename({
        "cumulative_games": "games_this_year_home_team",
        "cumulative_wins": "wins_this_year_home_team"
    })
    
    # Join back to get stats as of each date for guest team
    df = df.join(
        cumulative_stats,
        left_on=["date", "guest_team"],
        right_on=["date", "team"],
        how="left"
    ).rename({
        "cumulative_games": "games_this_year_guest_team",
        "cumulative_wins": "wins_this_year_guest_team"
    })
    
    # Calculate win percentages
    df = df.with_columns([
        (pl.col("wins_this_year_home_team") / pl.col("games_this_year_home_team"))
            .fill_nan(None)
            .alias("win_pct_home"),
        (pl.col("wins_this_year_guest_team") / pl.col("games_this_year_guest_team"))
            .fill_nan(None)
            .alias("win_pct_guest")
    ])
    
    return df

def add_location_winning_pct(df):
    # Sort by date to ensure chronological order
    df = df.sort("date")
    
    # Process home team stats (only home games)
    home_stats = (
        df.select(["date", "home_team", "is_home_win"])
        .rename({"home_team": "team", "is_home_win": "win"})
        .with_columns(pl.lit("home").alias("location"))
    )
    
    # Process guest team stats (only away games)
    guest_stats = (
        df.select(["date", "guest_team", "is_home_win"])
        .rename({"guest_team": "team"})
        .with_columns([
            (~pl.col("is_home_win")).alias("win"),
            pl.lit("away").alias("location")
        ])
        .select(["date", "team", "win", "location"])  # Select only the columns we need
    )
    
    # Combine both perspectives
    all_games = pl.concat([home_stats, guest_stats]).sort("date")
    
    # Calculate cumulative stats per team at each location, SHIFTED to show stats BEFORE each game
    cumulative_stats_home = (
        all_games
        .filter(pl.col("location") == "home")
        .group_by("team")
        .agg([
            pl.col("date"),
            pl.col("win").cum_sum().shift(1, fill_value=0).alias("cumulative_wins_home"),
            pl.int_range(pl.len()).alias("cumulative_games_home")
        ])
        .explode(["date", "cumulative_wins_home", "cumulative_games_home"])
    )
    
    cumulative_stats_away = (
        all_games
        .filter(pl.col("location") == "away")
        .group_by("team")
        .agg([
            pl.col("date"),
            pl.col("win").cum_sum().shift(1, fill_value=0).alias("cumulative_wins_away"),
            pl.int_range(pl.len()).alias("cumulative_games_away")
        ])
        .explode(["date", "cumulative_wins_away", "cumulative_games_away"])
    )
    
    # Join back to get home stats for home team
    df = df.join(
        cumulative_stats_home,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left"
    ).rename({
        "cumulative_games_home": "home_games_this_year_home_team",
        "cumulative_wins_home": "home_wins_this_year_home_team"
    })
    
    # Join back to get away stats for guest team
    df = df.join(
        cumulative_stats_away,
        left_on=["date", "guest_team"],
        right_on=["date", "team"],
        how="left"
    ).rename({
        "cumulative_games_away": "away_games_this_year_guest_team",
        "cumulative_wins_away": "away_wins_this_year_guest_team"
    })
    
    # Calculate win percentages
    df = df.with_columns([
        (pl.col("home_wins_this_year_home_team") / pl.col("home_games_this_year_home_team"))
            .fill_nan(None)
            .alias("home_win_pct_home"),
        (pl.col("away_wins_this_year_guest_team") / pl.col("away_games_this_year_guest_team"))
            .fill_nan(None)
            .alias("away_win_pct_guest")
    ])
    
    return df

def h2h(df):
    # Sort by date to ensure chronological order
    df = df.sort("date")
    
    # Create all matchups - we need to look at both directions
    # First, games where current home_team played against current guest_team
    matchups_home_perspective = df.select([
        "date",
        "home_team",
        "guest_team",
        "is_home_win"
    ]).with_columns([
        pl.col("is_home_win").cast(pl.Int64).alias("home_team_won"),
        (~pl.col("is_home_win")).cast(pl.Int64).alias("guest_team_won")
    ])
    
    # Second, games where current home_team was guest and current guest_team was home
    matchups_reversed = df.select([
        "date",
        "guest_team",
        "home_team", 
        "is_home_win"
    ]).rename({
        "guest_team": "home_team",
        "home_team": "guest_team"
    }).with_columns([
        (~pl.col("is_home_win")).cast(pl.Int64).alias("home_team_won"),
        pl.col("is_home_win").cast(pl.Int64).alias("guest_team_won")
    ])
    
    # Combine both perspectives
    all_matchups = pl.concat([matchups_home_perspective, matchups_reversed]).sort("date")
    
    # Calculate cumulative H2H stats
    h2h_stats = (
        all_matchups
        .group_by(["home_team", "guest_team"])
        .agg([
            pl.col("date"),
            pl.col("home_team_won").cum_sum().shift(1, fill_value=0).alias("h2h_wins_home"),
            pl.col("guest_team_won").cum_sum().shift(1, fill_value=0).alias("h2h_wins_guest"),
            pl.int_range(0, pl.len()).alias("h2h_games")
        ])
        .explode(["date", "h2h_wins_home", "h2h_wins_guest", "h2h_games"])
    )
    
    # Join back to original dataframe
    df = df.join(
        h2h_stats,
        on=["date", "home_team", "guest_team"],
        how="left"
    )
    
    # Calculate H2H win percentage
    df = df.with_columns([
        (pl.col("h2h_wins_home") / pl.col("h2h_games"))
            .fill_nan(None)
            .alias("h2h_win_pct")
    ])
    
    return df


def team_boxscores(df, team):
    home_df = df.filter(
        pl.col("home_team") == team
    ).with_columns([
        (pl.col("pts_home") > pl.col("pts_guest")).alias("is_win")
    ])
    guest_df = df.filter(
        pl.col("guest_team") == team
    ).with_columns([
        (pl.col("pts_home") < pl.col("pts_guest")).alias("is_win")
    ])
    team_stats_home_df = home_df.select(
            ~cs.matches('_guest|is_win')
    ).rename(lambda c: c.replace('_home', ''))
    opp_stats_home_df = home_df.select([
        "drb_guest", "orb_guest", "trb_guest",
        "fg_guest", "fga_guest", "fta_guest",
        "def_rtg_guest", "off_rtg_guest",
        "fg3_guest", "pts_guest", "tov_guest", "is_win"
    ]).rename(lambda c: c.replace('_guest', '_opponent'))
    team_stats_away_df = guest_df.select(
        ~cs.matches('_home|is_win')
    ).rename(lambda c: c.replace('_guest', ''))
    opp_stats_away_df = guest_df.select([
        "drb_home", "orb_home", "trb_home",
        "fg_home", "fga_home", "fta_home",
        "def_rtg_home", "off_rtg_home",
        "fg3_home", "pts_home", "tov_home", "is_win"
    ]).rename(lambda c: c.replace('_home', '_opponent'))
    home_df = pl.concat(
        [team_stats_home_df, opp_stats_home_df],
        how='horizontal'
    )
    away_df = pl.concat(
        [team_stats_away_df, opp_stats_away_df],
        how='horizontal'
    )
    df = pl.concat([home_df, away_df]).drop([
        'home_team', 'guest_team', 'mp'
    ]).sort('date').with_columns(
        (0.5 * (pl.col("fga") + 0.44 * pl.col("fta") - pl.col("orb") + pl.col("tov"))).alias("pace"),
        (pl.col("off_rtg") - pl.col("def_rtg")).alias("net_rtg"),
        (pl.col("off_rtg_opponent") - pl.col("def_rtg_opponent")).alias("net_rtg_opponent"),
        (pl.col("tov") - pl.col("tov_opponent")).alias("tov_diff"),
        (
            (pl.col("fga") + 0.44 * pl.col("fta")) - 
            (pl.col("fga_opponent") + 0.44 * pl.col("fta_opponent"))
        ).alias("poss_diff"),
        pl.lit(team).alias("team")
    )
    return df


def team_season_stats(df, window_sizes = [7, 109]):
    absolute_columns = [
        "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "drb", "trb",
        "ast", "stl", "blk", "tov", "pts", "pace", "tov_diff", "poss_diff"
    ]
    per_100_columns = [
        "tov_pct", "off_rtg", "def_rtg", "net_rtg"
    ]
    relative_columns = [
        "fg_pct", "fg3_pct", "ft_pct"
    ]
    opponent_columns = [
        "drb_opponent", "orb_opponent", "trb_opponent",
        "fg_opponent", "fga_opponent", "fta_opponent",
        "fg3_opponent", "pts_opponent", "tov_opponent",
        "def_rtg_opponent", "off_rtg_opponent", "net_rtg_opponent"
    ]
    dropped_cols = ["pf"]
    drop_cols = absolute_columns + per_100_columns + relative_columns + opponent_columns + dropped_cols
    df = df.with_columns(
        (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("days_since_last_game"),
        pl.col('is_win').shift(1).alias('has_won_last_game'),
        (
            pl.col('date').map_elements(
                lambda d: df.filter(
                    (pl.col('date') < d) &
                    (pl.col('date') >= (d - timedelta(days=7)))
              ).shape[0], return_dtype=pl.Int64
            )
        ).alias('games_last_7_days')
    ).with_columns([
        pl.col(c).shift(1).alias(f'{c}_previous_game')
        for c in absolute_columns + per_100_columns + opponent_columns
    ] + [
        (pl.col('pts') - pl.col('pts_opponent')).shift(1).alias('pts_diff_previous_game')
    ]).with_columns(
        pl.col('has_won_last_game').rle_id().alias('streak')
    ).with_columns([
        pl.col('has_won_last_game').cum_count().over(
            pl.col('streak')
        ).alias('streak_length')
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            pl.col(f'{c}_previous_game').rolling_mean(
                window_size=ws, min_samples=1 if ws >= 109 else ws
            ).alias(f'{c}_{ws}')
            for c in absolute_columns + per_100_columns + opponent_columns
        ] + [
            pl.col(f'{c}_previous_game').ewm_mean(
                span=ws, min_periods=1 if ws >= 109 else ws
            ).alias(f'{c}_ewm_{ws}')
            for c in absolute_columns + per_100_columns + opponent_columns
        ] + [
            pl.col(f'{c}_previous_game').rolling_std(
                window_size=ws, min_samples=1 if ws >= 109 else ws
            ).alias(f'{c}_{ws}_std')
            for c in absolute_columns + per_100_columns + opponent_columns
        ] + [
            (pl.col('pts_diff_previous_game') >= 15).cast(pl.Int32).rolling_sum(
                window_size=ws, min_samples=1 if ws >= 109 else ws
            ).alias(f'blowout_wins_{ws}')
        ] + [
            (pl.col('pts_diff_previous_game') <= -15).cast(pl.Int32).rolling_sum(
                window_size=ws, min_samples=1 if ws >= 109 else ws
            ).alias(f'blowout_losses_{ws}')
        ] + [
                (
                    pl.col(f'off_rtg_previous_game').rolling_mean(
                        window_size=ws, min_samples=1 if ws >= 109 else ws
                    ) - pl.col(f'def_rtg_opponent_previous_game').rolling_mean(
                        window_size=ws, min_samples=1 if ws >= 109 else ws
                    )
                ).alias(f'off_rtg_sos_adj_{ws}')
            ] + [
                (
                    pl.col(f'def_rtg_previous_game').rolling_mean(
                        window_size=ws, min_samples=1 if ws >= 109 else ws
                    ) - pl.col(f'off_rtg_opponent_previous_game').rolling_mean(
                        window_size=ws, min_samples=1 if ws >= 109 else ws
                    )
                ).alias(f'def_rtg_sos_adj_{ws}')
            ]
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            (
                pl.col(f'fg_{ws}') /
                pl.col(f'fga_{ws}')
            ).alias(f'fg_pct_{ws}'),
            (
                pl.col(f'fg3_{ws}') /
                pl.col(f'fg3a_{ws}')
            ).alias(f'fg3_pct_{ws}'),
            (
                pl.col(f'ft_{ws}') /
                pl.col(f'fta_{ws}')
            ).alias(f'ft_pct_{ws}'),
            (
                pl.col(f'ast_{ws}') /
                pl.col(f'fg_{ws}')
            ).alias(f'ast_pct_{ws}'),
            (
                pl.col(f'ast_{ws}') /
                pl.col(f'tov_{ws}')
            ).alias(f'ast_to_tov_{ws}'),
            (
                pl.col(f'fta_{ws}') /
                pl.col(f'fga_{ws}')
            ).alias(f'fta_rate_{ws}'),
            (
                1 - (
                      pl.col(f'fg3a_{ws}') /
                      pl.col(f'fga_{ws}')
                    )
            ).alias(f'fg2a_pct_{ws}'),
            (
                (2 * (
                    pl.col(f'fg_{ws}') -
                    pl.col(f'fg3_{ws}')
                    )
                ) /
                pl.col(f'pts_{ws}')
            ).alias(f'pts_pct_2_{ws}'),
            (
                pl.col(f'ft_{ws}') /
                pl.col(f'pts_{ws}')
            ).alias(f'ptc_pct_ft_{ws}'),
            (
                (
                    pl.col(f'fg_{ws}') +
                    0.5 * pl.col(f'fg3_{ws}')
                ) /
                pl.col(f'fga_{ws}')
            ).alias(f'eff_fg_pct_{ws}'),
            (
                pl.col(f'pts_{ws}') /
                (
                    2 * (
                          pl.col(f'fga_{ws}') +
                          (0.44 * pl.col(f'fta_{ws}'))
                        )
                )
            ).alias(f'ts_pct_{ws}'),
            (
                pl.col(f'ast_{ws}') * 100 /
                (
                    pl.col(f'fga_{ws}') +
                    0.44 * pl.col(f'fta_{ws}') +
                    pl.col(f'tov_{ws}') +
                    pl.col(f'ast_{ws}')
                )
            ).alias(f'ast_rat_{ws}'),
            (
                pl.col(f'trb_{ws}') /
                (
                    pl.col(f'trb_{ws}') +
                    pl.col(f'trb_opponent_{ws}')
                )
            ).alias(f'trb_pct_{ws}'),
            (
                pl.col(f'drb_{ws}') /
                (
                    pl.col(f'drb_{ws}') +
                    pl.col(f'orb_opponent_{ws}')
                )
            ).alias(f'drb_pct_{ws}'),
            (
                pl.col(f'orb_{ws}') /
                (
                    pl.col(f'orb_{ws}') +
                    pl.col(f'drb_opponent_{ws}')
                )
            ).alias(f'orb_pct_{ws}'),
            (
                pl.col(f'fta_opponent_{ws}') /
                pl.col(f'fga_opponent_{ws}')
            ).alias(f'fta_rate_opponent_{ws}'),
            (
                (
                    pl.col(f'fg_opponent_{ws}') +
                    0.5 * pl.col(f'fg3_opponent_{ws}')
                ) /
                pl.col(f'fga_opponent_{ws}')
            ).alias(f'eff_fg_pct_opponent_{ws}'),
            (
                pl.col(f'pts_opponent_{ws}') /
                (
                    2 * (
                          pl.col(f'fga_opponent_{ws}') +
                          (0.44 * pl.col(f'fta_opponent_{ws}'))
                        )
                )
            ).alias(f'ts_pct_opponent_{ws}')
        ]
    ]).drop(drop_cols).with_columns([
        (pl.col(f'ts_pct_{ws}') - pl.col(f'ts_pct_opponent_{ws}')).alias(f'ts_diff_{ws}')
        for ws in window_sizes
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            pl.col(f'{c}_{ws}').rolling_std(
                window_size = ws, min_samples= 1 if ws >= 109 else ws
            ).alias(f'{c}_{ws}_std')
            for c in relative_columns + ["ts_pct", "eff_fg_pct"]
        ]
    ]).with_columns([
        pl.when(
            pl.col('has_won_last_game')
        ).then(
            pl.col('streak_length')
        ).otherwise(
            pl.lit(0)
        ).alias('current_winning_streak'),
        pl.when(
            ~pl.col('has_won_last_game')
        ).then(
            pl.col('streak_length')
        ).otherwise(
            pl.lit(0)
        ).alias('current_losing_streak')
    ]).drop(['streak', 'streak_length'])
    cols = [col for col in df.columns if col.endswith(f'_{window_sizes[0]}')]
    new_cols = [
        (
            pl.col(col) / pl.col(col.replace(f'_{window_sizes[0]}', f'_{window_sizes[1]}'))
        ).alias(
            col.replace(f'_{window_sizes[0]}', '_vs_avg')
        )
        for col in cols
    ]
    df=df.with_columns(new_cols)
    return df