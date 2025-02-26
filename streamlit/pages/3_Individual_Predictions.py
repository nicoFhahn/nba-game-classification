import streamlit as st
import sys
import os
import polars as pl
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import data_wrangling
import modelling
import shap
st.set_page_config(layout="wide")

df_today = data_wrangling.newest_games()
mod = modelling.lgbm_model()
mod.load_model()
mod.load_data()
games = mod.full_data
best_features = mod.load_best_features()[0]
train_data = pl.concat([
    mod.data_sets['train'], mod.data_sets['test']
]).sort('game_id')
X_new = games.to_dummies([
    "game_type", "month", "weekday"
]).drop([
    "home_team_id", "away_team_id"
]).filter(
    (pl.col("is_home_win").is_null()) &
    (pl.col("date") == games.filter(pl.col("is_home_win").is_null())["date"].min())
).sort('game_id').select(pl.col(best_features)).drop("date")
df_today = df_today.with_columns([
    pl.concat_str(
        pl.col("home_team_name"),
        pl.lit("vs."),
        pl.col("away_team_name"),
        separator=" "
    ).alias("matchup")
])

explainer = shap.Explainer(
    mod.model,
    pl.concat(
        [mod.X['train'], mod.X['test']]
    ).select(
        pl.col(best_features)
    ).drop("date").to_numpy(),
    feature_names=mod.X['train'].select(pl.col(best_features)).drop("date").columns
)
shap_values_today = explainer(X_new.to_numpy())
col_1, col_2 = st.columns(2)
with col_1:
    selection = st.selectbox(
        "Select Matchup",
        df_today["matchup"]
    )
    st.dataframe(
        df_today.filter(pl.col("matchup") == selection)[[
            "home_team_name", "away_team_name", "winner_logo", "probability",
            "record_home_team", "record_away_team", "elo_home_team", "elo_away_team"
        ]],
        column_config={
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
            "record_home_team": st.column_config.TextColumn(
                "Record Home Team"
            ),
            "record_away_team": st.column_config.TextColumn(
                "Record Road Team"
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
        height=140
    )
    st.markdown("""
    The chart to the right illustrates how each feature contributes to the model's prediction for a 
    specific game, starting from a base value (average prediction) and accumulating 
    feature contributions to reach the final prediction score. 
    The prediction value f(x) displays the log-odds whereas E[f(x)] displays the base value.
    (A base value of 0.31 would transform to a 0.576 probability for a home win)
    """)
with col_2:
    st.markdown(
        """<h3> Waterfall Plot </h3>""", unsafe_allow_html=True
    )
    fig_1, ax = plt.subplots()
    shap.plots.waterfall(
        shap_values_today[df_today["matchup"].to_list().index(selection)],
        max_display=10
    )
    st.pyplot(fig_1)