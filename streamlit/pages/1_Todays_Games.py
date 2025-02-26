import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import data_wrangling
st.set_page_config(layout="wide")

df_today = data_wrangling.newest_games().drop('game_id')
st.dataframe(
    df_today,
    column_config={
        "date": st.column_config.DateColumn(
            label="Date"
        ),
        "home_team_name": st.column_config.TextColumn(
            label="Home Team"
        ),
        "away_team_name": st.column_config.TextColumn(
            label="Road Team"
        ),
        "winner_logo": st.column_config.ImageColumn(
            label="Predicted Winner"
        ),
        "probability": st.column_config.NumberColumn(
            label="Probability",
            format='%.0f %%',
            help="The predicted probability of the model for the predicted winner"
        ),
        "h2h_current_year": st.column_config.TextColumn(
            label="H2H - This Season"
        ),
        "h2h_last_year": st.column_config.TextColumn(
            label="H2H - Last Season"
        ),
        "record_home_team": st.column_config.TextColumn(
            "Record Home Team"
        ),
        "winning_percentage_home_team": st.column_config.NumberColumn(
            label="W% Home Team",
            format='%.0f %%',
            width=100
        ),
        "record_away_team": st.column_config.TextColumn(
            "Record Road Team"
        ),
        "winning_percentage_away_team": st.column_config.NumberColumn(
            label="W% Road Team",
            format='%.0f %%',
            width=100
        ),
        "home_record": st.column_config.TextColumn(
            label="Home Record",
            help="The record of the home team at home"
        ),
        "winning_percentage_home": st.column_config.NumberColumn(
            label="W% at Home",
            format='%.0f %%',
            help="The winning percentage of the home team at home",
            width=100
        ),
        "away_record": st.column_config.TextColumn(
            label="Home Record",
            help="The record of the road team on the road"
        ),
        "winning_percentage_away": st.column_config.NumberColumn(
            label="W% at Home",
            format='%.0f %%',
            help="The winning percentage of the road team on the road",
            width=100
        ),
        "streak_home": st.column_config.TextColumn(
            label="Current Streak Home",
            width=120
        ),
        "streak_away": st.column_config.TextColumn(
            label="Current Streak Road",
            width=120
        ),
        "elo_home_team": st.column_config.NumberColumn(
            label="Elo Home",
            format='%.0f',
            help="Elo Rating of the home team"
        ),
        "elo_away_team": st.column_config.NumberColumn(
            label="Elo Road",
            format='%.0f',
            help="Elo Rating of the road team"
        )
    },
    use_container_width=True,
    row_height=80,
    height=80 * df_today.shape[0] + 80
)
