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
            pl.col("win").sort_by("date").cum_sum().shift(1, fill_value=0).alias("cumulative_wins"),
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


"""
Expected Team Statistics Calculator - Full Enhanced Polars Implementation

Calculates expected team stats PLUS advanced performance indicators:
- Roster quality counts (plus/minus thresholds)
- Consistency metrics (variance/volatility)
- Form indicators (hot/cold streaks)
- Depth quality metrics
- Risk/uncertainty measures
"""

import polars as pl
from typing import List

"""
Expected Team Statistics Calculator - With Realistic Minutes Distribution

Key improvement: Distributes 240 total minutes across available players based on 
their historical usage rates, respecting the 48-minute maximum per player.
"""

import polars as pl
from typing import List


def calculate_expected_team_stats(
        df: pl.DataFrame,
        top_n_players: int = 8
) -> pl.DataFrame:
    """
    Calculate expected team statistics with realistic minutes distribution.

    The key improvement: Instead of using raw historical minutes, this distributes
    exactly 240 minutes (48 min × 5 positions) across available players based on
    their usage rates, with a 48-minute cap per player.

    Args:
        df: Polars DataFrame with player-level statistics
        top_n_players: Number of rotation players to use (default: 8)

    Returns:
        Polars DataFrame with expected team statistics and performance indicators
    """

    # Define statistics
    counting_stats = [
        "fg", "fga", "fg3", "fg3a", "ft", "fta",
        "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pts"
    ]
    rate_stats = ["game_score", "plus_minus", "off_rtg", "def_rtg", "bpm"]
    all_stats = counting_stats + rate_stats

    # Weights: 15% last game, 35% last 7, 50% season
    W_LAST = 0.15
    W_SEVEN = 0.35
    W_SEASON = 0.50

    # Keep players with either mp_7 OR mp_109
    df_filtered = df.filter(
        pl.col("mp_7").is_not_null() | pl.col("mp_109").is_not_null()
    )

    # Historical minutes (for usage rate calculation)
    df_filtered = df_filtered.with_columns([
        pl.when(pl.col("mp_7").is_not_null())
        .then(pl.col("mp_7"))
        .otherwise(pl.col("mp_109"))
        .alias("historical_minutes")
    ])

    # Calculate weighted stats
    weighted_exprs = [
        (
                W_LAST * pl.col(f"{stat}_previous_game").fill_null(0) +
                W_SEVEN * pl.col(f"{stat}_7").fill_null(0) +
                W_SEASON * pl.col(f"{stat}_109").fill_null(0)
        ).alias(f"{stat}_weighted")
        for stat in all_stats
    ]

    df_weighted = df_filtered.with_columns(weighted_exprs)

    # Select top N players per game by historical minutes
    df_top = (
        df_weighted
        .sort("historical_minutes", descending=True)
        .group_by("game_id")
        .agg([
            pl.col("team_id").first(),
            pl.col("player_id").head(top_n_players),
            *[pl.col(f"{stat}_weighted").head(top_n_players) for stat in all_stats],
            pl.col("historical_minutes").head(top_n_players),
            # Keep mp_7 and mp_109 for additional calculations
            pl.col("mp_7").head(top_n_players),
            pl.col("mp_109").head(top_n_players),
            # For consistency and form metrics
            *[pl.col(f"{stat}_7_std").head(top_n_players) for stat in
              ["pts", "off_rtg", "def_rtg", "plus_minus"]],
            *[pl.col(f"{stat}_7").head(top_n_players) for stat in
              ["pts", "off_rtg", "plus_minus"]],
            *[pl.col(f"{stat}_109").head(top_n_players) for stat in
              ["pts", "off_rtg", "plus_minus"]],
        ])
    )

    # Explode to player-game level
    explode_cols = [
        "player_id",
        *[f"{stat}_weighted" for stat in all_stats],
        "historical_minutes",
        "mp_7",
        "mp_109",
        "pts_7_std", "off_rtg_7_std", "def_rtg_7_std", "plus_minus_7_std",
        "pts_7", "off_rtg_7", "plus_minus_7",
        "pts_109", "off_rtg_109", "plus_minus_109",
    ]
    df_exploded = df_top.explode(explode_cols)

    # CALCULATE REALISTIC EXPECTED MINUTES PER GAME
    # Step 1: Calculate usage rates (proportion of historical minutes)
    df_exploded = df_exploded.with_columns([
        (pl.col("historical_minutes") / pl.col("historical_minutes").sum().over("game_id"))
        .alias("usage_rate")
    ])

    # Step 2: Distribute 240 minutes based on usage rates
    TOTAL_GAME_MINUTES = 240.0
    MAX_PLAYER_MINUTES = 48.0

    df_exploded = df_exploded.with_columns([
        (pl.col("usage_rate") * TOTAL_GAME_MINUTES)
        .clip(upper_bound=MAX_PLAYER_MINUTES)
        .alias("expected_minutes_unconstrained")
    ])

    # Step 3: If total > 240 after capping, redistribute excess minutes
    # (This handles cases where multiple stars would exceed 48 min)
    df_exploded = df_exploded.with_columns([
        pl.col("expected_minutes_unconstrained").sum().over("game_id").alias("total_unconstrained")
    ])

    df_exploded = df_exploded.with_columns([
        pl.when(pl.col("total_unconstrained") > TOTAL_GAME_MINUTES)
        .then(
            # Scale down proportionally
            pl.col("expected_minutes_unconstrained") *
            (TOTAL_GAME_MINUTES / pl.col("total_unconstrained"))
        )
        .otherwise(pl.col("expected_minutes_unconstrained"))
        .alias("expected_minutes")
    ])

    # Build aggregation expressions
    agg_exprs = [
        pl.col("team_id").first(),
        pl.col("player_id").count().alias("num_players"),
        pl.col("expected_minutes").sum().alias("total_minutes_check"),  # Should be ~240
    ]

    # Counting stats: weighted by expected minutes / historical minutes
    for stat in counting_stats:
        agg_exprs.append(
            (pl.col(f"{stat}_weighted") * pl.col("expected_minutes") / pl.col("historical_minutes"))
            .sum()
            .alias(f"exp_{stat}")
        )

    # Rate stats: minutes-weighted average
    total_minutes = pl.col("expected_minutes").sum()
    for stat in rate_stats:
        agg_exprs.append(
            ((pl.col(f"{stat}_weighted") * pl.col("expected_minutes")).sum() / total_minutes)
            .alias(f"exp_{stat}")
        )

    # CONSISTENCY METRICS
    agg_exprs.extend([
        ((pl.col("pts_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_pts_volatility"),
        ((pl.col("off_rtg_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_off_rtg_volatility"),
        ((pl.col("def_rtg_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_def_rtg_volatility"),
        ((pl.col("plus_minus_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_plus_minus_volatility"),
    ])

    # FORM INDICATORS
    agg_exprs.extend([
        ((pl.col("pts_7").fill_null(pl.col("pts_109")).fill_null(0) -
          pl.col("pts_109").fill_null(0)) * pl.col("expected_minutes")).sum()
        .alias("pts_recent_vs_season"),

        (((pl.col("off_rtg_7").fill_null(pl.col("off_rtg_109")).fill_null(0) -
           pl.col("off_rtg_109").fill_null(0)) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("off_rtg_recent_vs_season"),

        (((pl.col("plus_minus_7").fill_null(pl.col("plus_minus_109")).fill_null(0) -
           pl.col("plus_minus_109").fill_null(0)) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("plus_minus_recent_vs_season"),
    ])

    result = df_exploded.group_by("game_id").agg(agg_exprs)

    # Calculate shooting percentages
    result = result.with_columns([
        (pl.col("exp_fg") / pl.col("exp_fga")).alias("exp_fg_pct"),
        (pl.col("exp_fg3") / pl.col("exp_fg3a")).alias("exp_fg3_pct"),
        (pl.col("exp_ft") / pl.col("exp_fta")).alias("exp_ft_pct")
    ])

    # Add plus/minus player counts
    result = add_plus_minus_counts(df_filtered, result)

    # Add advanced performance indicators
    result = add_advanced_indicators(df_filtered, result)

    return result.sort("game_id")


"""
Expected Team Statistics Calculator - With Realistic Minutes Distribution

Key improvement: Distributes 240 total minutes across available players based on 
their historical usage rates, respecting the 48-minute maximum per player.
"""

import polars as pl
from typing import List


def calculate_expected_team_stats(
        df: pl.DataFrame,
        top_n_players: int = 8
) -> pl.DataFrame:
    """
    Calculate expected team statistics with realistic minutes distribution.

    The key improvement: Instead of using raw historical minutes, this distributes
    exactly 240 minutes (48 min × 5 positions) across available players based on
    their usage rates, with a 48-minute cap per player.

    Args:
        df: Polars DataFrame with player-level statistics
        top_n_players: Number of rotation players to use (default: 8)

    Returns:
        Polars DataFrame with expected team statistics and performance indicators
    """

    # Define statistics
    counting_stats = [
        "fg", "fga", "fg3", "fg3a", "ft", "fta",
        "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pts"
    ]
    rate_stats = ["game_score", "plus_minus", "off_rtg", "def_rtg", "bpm"]
    all_stats = counting_stats + rate_stats

    # Weights: 15% last game, 35% last 7, 50% season
    W_LAST = 0.15
    W_SEVEN = 0.35
    W_SEASON = 0.50

    # Keep players with either mp_7 OR mp_109
    df_filtered = df.filter(
        pl.col("mp_7").is_not_null() | pl.col("mp_109").is_not_null()
    )

    # Historical minutes (for usage rate calculation)
    df_filtered = df_filtered.with_columns([
        pl.when(pl.col("mp_7").is_not_null())
        .then(pl.col("mp_7"))
        .otherwise(pl.col("mp_109"))
        .alias("historical_minutes")
    ])

    # Calculate weighted stats with proper fallback logic
    # When recent data (previous_game or _7) is missing, fall back to season average
    weighted_exprs = [
        (
                W_LAST * pl.col(f"{stat}_previous_game").fill_null(pl.col(f"{stat}_109")).fill_null(0) +
                W_SEVEN * pl.col(f"{stat}_7").fill_null(pl.col(f"{stat}_109")).fill_null(0) +
                W_SEASON * pl.col(f"{stat}_109").fill_null(0)
        ).alias(f"{stat}_weighted")
        for stat in all_stats
    ]

    df_weighted = df_filtered.with_columns(weighted_exprs)

    # Select top N players per game by historical minutes
    df_top = (
        df_weighted
        .sort("historical_minutes", descending=True)
        .group_by("game_id")
        .agg([
            pl.col("team_id").first(),
            pl.col("player_id").head(top_n_players),
            *[pl.col(f"{stat}_weighted").head(top_n_players) for stat in all_stats],
            pl.col("historical_minutes").head(top_n_players),
            # Keep mp_7 and mp_109 for additional calculations
            pl.col("mp_7").head(top_n_players),
            pl.col("mp_109").head(top_n_players),
            # For consistency and form metrics
            *[pl.col(f"{stat}_7_std").head(top_n_players) for stat in
              ["pts", "off_rtg", "def_rtg", "plus_minus"]],
            *[pl.col(f"{stat}_7").head(top_n_players) for stat in
              ["pts", "off_rtg", "plus_minus"]],
            *[pl.col(f"{stat}_109").head(top_n_players) for stat in
              ["pts", "off_rtg", "plus_minus"]],
        ])
    )

    # Explode to player-game level
    explode_cols = [
        "player_id",
        *[f"{stat}_weighted" for stat in all_stats],
        "historical_minutes",
        "mp_7",
        "mp_109",
        "pts_7_std", "off_rtg_7_std", "def_rtg_7_std", "plus_minus_7_std",
        "pts_7", "off_rtg_7", "plus_minus_7",
        "pts_109", "off_rtg_109", "plus_minus_109",
    ]
    df_exploded = df_top.explode(explode_cols)

    # CALCULATE REALISTIC EXPECTED MINUTES PER GAME
    # Step 1: Calculate usage rates (proportion of historical minutes)
    df_exploded = df_exploded.with_columns([
        (pl.col("historical_minutes") / pl.col("historical_minutes").sum().over("game_id"))
        .alias("usage_rate")
    ])

    # Step 2: Distribute 240 minutes based on usage rates
    TOTAL_GAME_MINUTES = 240.0
    MAX_PLAYER_MINUTES = 48.0

    df_exploded = df_exploded.with_columns([
        (pl.col("usage_rate") * TOTAL_GAME_MINUTES)
        .clip(upper_bound=MAX_PLAYER_MINUTES)
        .alias("expected_minutes_unconstrained")
    ])

    # Step 3: If total > 240 after capping, redistribute excess minutes
    # (This handles cases where multiple stars would exceed 48 min)
    df_exploded = df_exploded.with_columns([
        pl.col("expected_minutes_unconstrained").sum().over("game_id").alias("total_unconstrained")
    ])

    df_exploded = df_exploded.with_columns([
        pl.when(pl.col("total_unconstrained") > TOTAL_GAME_MINUTES)
        .then(
            # Scale down proportionally
            pl.col("expected_minutes_unconstrained") *
            (TOTAL_GAME_MINUTES / pl.col("total_unconstrained"))
        )
        .otherwise(pl.col("expected_minutes_unconstrained"))
        .alias("expected_minutes")
    ])

    # Build aggregation expressions
    agg_exprs = [
        pl.col("team_id").first(),
        pl.col("player_id").count().alias("num_players"),
        pl.col("expected_minutes").sum().alias("total_minutes_check"),  # Should be ~240
    ]

    # Counting stats: weighted by expected minutes / historical minutes
    for stat in counting_stats:
        agg_exprs.append(
            (pl.col(f"{stat}_weighted") * pl.col("expected_minutes") / pl.col("historical_minutes"))
            .sum()
            .alias(f"exp_{stat}")
        )

    # Rate stats: minutes-weighted average
    total_minutes = pl.col("expected_minutes").sum()
    for stat in rate_stats:
        agg_exprs.append(
            ((pl.col(f"{stat}_weighted") * pl.col("expected_minutes")).sum() / total_minutes)
            .alias(f"exp_{stat}")
        )

    # CONSISTENCY METRICS
    agg_exprs.extend([
        ((pl.col("pts_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_pts_volatility"),
        ((pl.col("off_rtg_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_off_rtg_volatility"),
        ((pl.col("def_rtg_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_def_rtg_volatility"),
        ((pl.col("plus_minus_7_std").fill_null(0) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("avg_plus_minus_volatility"),
    ])

    # FORM INDICATORS
    agg_exprs.extend([
        ((pl.col("pts_7").fill_null(pl.col("pts_109")).fill_null(0) -
          pl.col("pts_109").fill_null(0)) * pl.col("expected_minutes")).sum()
        .alias("pts_recent_vs_season"),

        (((pl.col("off_rtg_7").fill_null(pl.col("off_rtg_109")).fill_null(0) -
           pl.col("off_rtg_109").fill_null(0)) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("off_rtg_recent_vs_season"),

        (((pl.col("plus_minus_7").fill_null(pl.col("plus_minus_109")).fill_null(0) -
           pl.col("plus_minus_109").fill_null(0)) * pl.col("expected_minutes")).sum() / total_minutes)
        .alias("plus_minus_recent_vs_season"),
    ])

    result = df_exploded.group_by("game_id").agg(agg_exprs)

    # Calculate shooting percentages
    result = result.with_columns([
        (pl.col("exp_fg") / pl.col("exp_fga")).alias("exp_fg_pct"),
        (pl.col("exp_fg3") / pl.col("exp_fg3a")).alias("exp_fg3_pct"),
        (pl.col("exp_ft") / pl.col("exp_fta")).alias("exp_ft_pct")
    ])

    # Add plus/minus player counts
    result = add_plus_minus_counts(df_filtered, result)

    # Add advanced performance indicators
    result = add_advanced_indicators(df_filtered, result)

    return result.sort("game_id").drop(["total_minutes_check", "num_players"])


def add_plus_minus_counts(
        df_filtered: pl.DataFrame,
        result: pl.DataFrame
) -> pl.DataFrame:
    """Add player count features based on plus_minus_109 thresholds."""

    thresholds = [0, 2, 4, 6, 8, 10]

    pm_counts = df_filtered.group_by("game_id").agg([
        pl.col("team_id").first(),
        *[(pl.col("plus_minus_109").fill_null(-999) >= threshold)
          .sum()
          .alias(f"players_pm109_gte_{threshold}")
          for threshold in thresholds]
    ])

    result = result.join(pm_counts, on="game_id", how="left")

    if "team_id_right" in result.columns:
        result = result.drop("team_id_right")

    return result


def add_advanced_indicators(
        df_filtered: pl.DataFrame,
        result: pl.DataFrame
) -> pl.DataFrame:
    """Add advanced performance indicators."""

    # Use historical minutes for these calculations
    df_with_minutes = df_filtered.with_columns([
        pl.when(pl.col("mp_7").is_not_null())
        .then(pl.col("mp_7"))
        .otherwise(pl.col("mp_109"))
        .alias("minutes_for_calc")
    ])

    advanced = df_with_minutes.group_by("game_id").agg([
        pl.col("team_id").first(),

        # MINUTES CONCENTRATION
        (pl.col("minutes_for_calc").std() / (pl.col("minutes_for_calc").mean() + 0.01))
        .alias("minutes_concentration"),

        (pl.col("minutes_for_calc").top_k(3).sum() / pl.col("minutes_for_calc").sum())
        .alias("top3_minutes_share"),

        # DEPTH QUALITY
        (pl.when(pl.col("minutes_for_calc").rank(descending=True) > 5)
         .then(pl.col("plus_minus_109"))
         .otherwise(None)
         .mean())
        .alias("bench_avg_plus_minus"),

        (pl.col("minutes_for_calc") >= 10.0).sum().alias("rotation_size"),

        # MOMENTUM/FORM
        ((pl.col("pts_7").fill_null(pl.col("pts_109")) > pl.col("pts_109").fill_null(0)).sum())
        .alias("players_scoring_hot"),

        ((pl.col("plus_minus_7").fill_null(pl.col("plus_minus_109")) >
          pl.col("plus_minus_109").fill_null(-999)).sum())
        .alias("players_trending_positive"),

        # CONSISTENCY/RISK
        (1.0 / (pl.col("pts_7_std").fill_null(pl.col("pts_109_std")).fill_null(99).mean() + 0.1))
        .alias("team_consistency_score"),

        ((pl.col("pts_7_std").fill_null(pl.col("pts_109_std")).fill_null(0) >
          pl.col("pts_7").fill_null(pl.col("pts_109")).fill_null(0)).sum() / pl.count())
        .alias("high_volatility_player_pct"),
    ])

    result = result.join(advanced, on="game_id", how="left")

    if "team_id_right" in result.columns:
        result = result.drop("team_id_right")

    return result


def prepare_home_away_features(
        home_df: pl.DataFrame,
        away_df: pl.DataFrame,
        game_id: str
) -> pl.DataFrame:
    """Prepare home/away features for a single game."""

    home_stats = calculate_expected_team_stats(home_df)
    away_stats = calculate_expected_team_stats(away_df)

    home_game = home_stats.filter(pl.col("game_id") == game_id)
    away_game = away_stats.filter(pl.col("game_id") == game_id)

    exclude_cols = ["game_id", "team_id"]

    home_game = home_game.select([
        pl.col("game_id"),
        *[pl.col(col).alias(f"home_{col}")
          for col in home_game.columns if col not in exclude_cols]
    ])

    away_game = away_game.select([
        pl.col("game_id"),
        *[pl.col(col).alias(f"away_{col}")
          for col in away_game.columns if col not in exclude_cols]
    ])

    return home_game.join(away_game, on="game_id")



def player_season_stats(df, window_sizes = [7, 109]):
    absolute_columns = [
        "mp", "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "drb",
        "trb", "ast", "stl", "blk", "tov", "pts", "game_score",
        "plus_minus"
    ]
    per_100_columns = ["off_rtg", "def_rtg", "bpm"]
    relative_columns = [
        "fg_pct", "fg3_pct", "ft_pct", "orb_pct", "drb_pct", "trb_pct",
        "ast_pct", "stl_pct", "blk_pct", "usg_pct", "fg3a_per_fga_pct",
        "fta_per_fga_pct", "tov_pct"
    ]
    dropped_cols = ["ts_pct", "efg_pct", "pf"]
    df = df.drop(["id", "player_name"]).with_columns([
        pl.col(c).shift(1).over(pl.col("player_id")).alias(f'{c}_previous_game')
        for c in absolute_columns + per_100_columns
    ]).with_columns([
        expr
        for ws in window_sizes
        for expr in [
            pl.col(f'{c}_previous_game').rolling_mean(
                window_size=ws, min_samples=1 if (ws >= 109 or c == "mp") else ws
            ).over(pl.col("player_id")).alias(f'{c}_{ws}')
            for c in absolute_columns + per_100_columns
        ] + [
            pl.col(f'{c}_previous_game').ewm_mean(
                span=ws, min_periods=1 if ws >= 109 else ws
            ).over(pl.col("player_id")).alias(f'{c}_ewm_{ws}')
            for c in absolute_columns + per_100_columns
        ] + [
            pl.col(f'{c}_previous_game').rolling_std(
                window_size=ws, min_samples=1 if ws >= 109 else ws
            ).over(pl.col("player_id")).alias(f'{c}_{ws}_std')
            for c in absolute_columns + per_100_columns
        ]
    ]).drop(
        absolute_columns + per_100_columns + relative_columns +
        dropped_cols
    )
    return df

"""
NBA Travel Distance Calculator
================================
For each game, computes for both the home team and the guest team:
  - km_since_last_game  : distance (km) traveled from the location of their
                          previous game to the current game venue
  - km_last_7_days      : total distance (km) traveled in the 7-day window
                          ending on (and including) the current game date
  - km_season_total     : cumulative distance (km) traveled since the start
                          of the season up to and including this game

Travel model
------------
A team travels TO the location of each game.  Because the CSV only records
game venues (not team home arenas), the very first game for every team has
km_since_last_game = 0 (no prior location known).

Haversine formula is used for all distances.

Usage
-----
    python nba_travel.py                          # prints first 20 rows
    python nba_travel.py --output results.csv     # saves full results to CSV
"""

import math
import argparse
from datetime import timedelta

import polars as pl


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two (lat, lon) points."""
    R = 6_371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_travel(df) -> pl.DataFrame:
    """
    Read the schedule CSV and return a DataFrame with travel metrics for both
    home and guest teams attached to every row.

    New columns added:
        home_km_since_last_game
        home_km_last_7_days
        home_km_season_total
        guest_km_since_last_game
        guest_km_last_7_days
        guest_km_season_total
    """


    # ------------------------------------------------------------------
    # 2. Build a "team view": one row per (team, game) with the venue
    #    they're playing AT (home or away).
    # ------------------------------------------------------------------
    home_view = df.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("latitude"),
        pl.col("longitude"),
    ])

    guest_view = df.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("guest_team").alias("team"),
        pl.col("latitude"),
        pl.col("longitude"),
    ])

    team_games = (
        pl.concat([home_view, guest_view])
        .sort(["team", "date", "game_id"])
    )

    # ------------------------------------------------------------------
    # 3. Compute per-team metrics using Python – Polars' lazy / native
    #    window functions don't support arbitrary Haversine aggregations
    #    over variable-length temporal windows, so we process each team
    #    individually and collect results.
    # ------------------------------------------------------------------
    records: list[dict] = []

    for team, group in team_games.group_by("team", maintain_order=True):
        # group is already sorted by date because team_games is sorted
        rows = group.sort(["date", "game_id"]).to_dicts()

        season_total = 0.0

        for i, row in enumerate(rows):
            # --- distance since last game ---
            if i == 0:
                dist_last = 0.0
            else:
                prev = rows[i - 1]
                dist_last = haversine_km(
                    prev["latitude"], prev["longitude"],
                    row["latitude"],  row["longitude"],
                )

            season_total += dist_last

            # --- distance in last 7 days ---
            cutoff = row["date"] - timedelta(days=7)
            window_dist = 0.0
            # Walk backwards to sum legs that fall within the 7-day window.
            # A "leg" is attributed to the game you're travelling TO, so we
            # look at all legs i where the destination date is within window.
            for j in range(i, 0, -1):
                dest_date = rows[j]["date"]
                if dest_date < cutoff:
                    break
                prev_j = rows[j - 1]
                leg = haversine_km(
                    prev_j["latitude"], prev_j["longitude"],
                    rows[j]["latitude"], rows[j]["longitude"],
                )
                window_dist += leg

            records.append({
                "game_id":           row["game_id"],
                "team":              row["team"],
                "km_since_last_game": round(dist_last, 2),
                "km_last_7_days":    round(window_dist, 2),
                "km_season_total":   round(season_total, 2),
            })

    travel = pl.DataFrame(records)

    # ------------------------------------------------------------------
    # 4. Join metrics back onto the original schedule
    # ------------------------------------------------------------------
    result = (
        df
        # home metrics
        .join(
            travel
            .rename({
                "team":              "home_team",
                "km_since_last_game": "home_km_since_last_game",
                "km_last_7_days":    "home_km_last_7_days",
                "km_season_total":   "home_km_season_total",
            })
            .select(["game_id", "home_team",
                     "home_km_since_last_game",
                     "home_km_last_7_days",
                     "home_km_season_total"]),
            on=["game_id", "home_team"],
            how="left",
        )
        # guest metrics
        .join(
            travel
            .rename({
                "team":              "guest_team",
                "km_since_last_game": "guest_km_since_last_game",
                "km_last_7_days":    "guest_km_last_7_days",
                "km_season_total":   "guest_km_season_total",
            })
            .select(["game_id", "guest_team",
                     "guest_km_since_last_game",
                     "guest_km_last_7_days",
                     "guest_km_season_total"]),
            on=["game_id", "guest_team"],
            how="left",
        )
    )

    return result


def add_rest_features(df):
    """
    Add rest and fatigue features: back-to-backs and recent game density.
    """
    df = df.sort("date")

    # helper for team rest
    team_dates = []
    teams = sorted(set(df["home_team"].unique().to_list() + df["guest_team"].unique().to_list()))
    for team in teams:
        team_df = df.filter((pl.col("home_team") == team) | (pl.col("guest_team") == team)).select(
            "date").unique().sort("date")
        team_df = team_df.with_columns([
            pl.lit(team).alias("team"),
            (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("days_rest")
        ])
        team_dates.append(team_df)

    rest_df = pl.concat(team_dates)

    # Process game density
    density_list = []
    for team in teams:
        t_df = rest_df.filter(pl.col("team") == team)
        dates = t_df["date"].to_list()

        g3 = []
        g5 = []
        for d in dates:
            # count games strictly before d
            c3 = t_df.filter((pl.col("date") < d) & (pl.col("date") >= d - timedelta(days=3))).shape[0]
            c5 = t_df.filter((pl.col("date") < d) & (pl.col("date") >= d - timedelta(days=5))).shape[0]
            g3.append(c3)
            g5.append(c5)

        t_df = t_df.with_columns([
            pl.Series("games_last_3_days", g3),
            pl.Series("games_last_5_days", g5)
        ])
        density_list.append(t_df)

    full_rest_df = pl.concat(density_list)

    # Join for home team
    df = df.join(
        full_rest_df.rename({
            "days_rest": "home_days_rest",
            "games_last_3_days": "home_games_last_3_density",
            "games_last_5_days": "home_games_last_5_density"
        }),
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left"
    ).with_columns([
        (pl.col("home_days_rest") == 1).alias("is_b2b_home")
    ])

    # Join for guest team
    df = df.join(
        full_rest_df.rename({
            "days_rest": "guest_days_rest",
            "games_last_3_days": "guest_games_last_3_density",
            "games_last_5_days": "guest_games_last_5_density"
        }),
        left_on=["date", "guest_team"],
        right_on=["date", "team"],
        how="left"
    ).with_columns([
        (pl.col("guest_days_rest") == 1).alias("is_b2b_guest")
    ])

    return df


def add_clutch_features(df, window_size=109):
    """
    Calculate performance in close games (diff <= 5).
    """
    df = df.sort("date")

    # Identify clutch games and winner
    df = df.with_columns([
        (pl.col("pts_home") - pl.col("pts_guest")).abs().alias("point_diff")
    ]).with_columns([
        ((pl.col("point_diff") <= 5) & (pl.col("point_diff") > 0)).alias("is_clutch_game")
    ])

    team_clutch = []
    all_teams = sorted(set(df["home_team"].unique().to_list() + df["guest_team"].unique().to_list()))

    for team in all_teams:
        t_df = df.filter((pl.col("home_team") == team) | (pl.col("guest_team") == team)).sort("date")

        t_df = t_df.with_columns([
            pl.when(pl.col("home_team") == team)
            .then(pl.col("pts_home") > pl.col("pts_guest"))
            .otherwise(pl.col("pts_guest") > pl.col("pts_home"))
            .alias("is_team_win")
        ])

        # Calculate rolling clutch win pct
        # We only care about games that WERE clutch
        clutch_only = t_df.filter(pl.col("is_clutch_game"))
        if clutch_only.shape[0] > 0:
            t_df = t_df.join(
                clutch_only.with_columns(
                    pl.col("is_team_win").rolling_mean(
                        window_size=window_size, min_samples=1
                    ).alias("clutch_win_pct")
                ).select(["game_id", "clutch_win_pct"]),
                on="game_id",
                how="left"
            ).with_columns(
                pl.col("clutch_win_pct").shift(1).forward_fill()
            )
        else:
            t_df = t_df.with_columns(pl.lit(None).cast(pl.Float64).alias("clutch_win_pct"))

        team_clutch.append(t_df.select(["date", pl.lit(team).alias("team"), "clutch_win_pct"]))

    clutch_df = pl.concat(team_clutch).unique(["date", "team"])

    df = df.join(
        clutch_df.rename({"clutch_win_pct": "clutch_win_pct_home"}),
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left"
    ).join(
        clutch_df.rename({"clutch_win_pct": "clutch_win_pct_guest"}),
        left_on=["date", "guest_team"],
        right_on=["date", "team"],
        how="left"
    ).drop("point_diff")

    return df
