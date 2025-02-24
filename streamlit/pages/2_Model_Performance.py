import sys
import os
import streamlit as st
import polars as pl
import streamlit_shadcn_ui as ui
from streamlit_extras.metric_cards import style_metric_cards
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import modelling

st.set_page_config(layout="wide")

mod = modelling.lgbm_model()
mod.evaluate_performance()
team_df = pl.read_parquet(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "team_details.parquet"
    )
)
performance_df = mod.performance['by_team']
performance_df = performance_df.join(
    team_df,
    on="team_id"
).with_columns([
    pl.concat_str(
        pl.lit("https://raw.githubusercontent.com/nicoFhahn/nba-game-classification/main/streamlit/logos/"),
        pl.col("team_id"),
        pl.lit(".png")
    ).alias("team_logo")
]).sort("accuracy", descending=True).select([
    "team_logo", "team_name", "accuracy", "precision", "recall", "f1_score",
    "true_positives", "true_negatives", "false_positives", "false_negatives"
])

col_1, col_2, col_3, col_4 = st.columns(4)
with col_1:
    metric_title_1 = "Accuracy"
    ui.metric_card(
        title=metric_title_1,
        content=f'{mod.performance['over_time']['accuracy'][-1]:.2f}',
        description='The ratio of correctly predicted outcomes to total games'
    )
    st.write("plot")
with col_2:
    metric_title_1 = "Precision"
    ui.metric_card(
        title=metric_title_1,
        content=f'{mod.performance['over_time']['precision'][-1]:.2f}',
        description='The ratio of correctly predicted home wins to total predicted home wins'
    )
    st.write("plot")
with col_3:
    metric_title_1 = "Recall"
    ui.metric_card(
        title=metric_title_1,
        content=f'{mod.performance['over_time']['recall'][-1]:.2f}',
        description='The ratio of correctly predicted home wins to total home wins'
    )
    st.write("plot")
with col_4:
    metric_title_1 = "F1-Score"
    ui.metric_card(
        title=metric_title_1,
        content=f'{mod.performance['over_time']['f1_score'][-1]:.2f}',
        description='The harmonic mean between precision and recall'
    )
    st.write("plot")

st.dataframe(
    performance_df,
    column_config={
        "team_logo": st.column_config.ImageColumn(
            label=""
        ),
        "team_name": st.column_config.TextColumn(
            label="Team"
        ),
        "accuracy": st.column_config.NumberColumn(
            label="Accuracy",
            format="%.2f",
            width=10
        ),
        "precision": st.column_config.NumberColumn(
            label="Precision",
            format="%.2f",
            width=10
        ),
        "recall": st.column_config.NumberColumn(
            label="Recall",
            format="%.2f",
            width=10
        ),
        "f1_score": st.column_config.NumberColumn(
            label="F1-Score",
            format="%.2f",
            width=10
        ),
        "true_positives": st.column_config.NumberColumn(
            label="True Positives",
            width=10,
            help="The number of predicted home wins that were home wins"
        ),
        "true_negatives": st.column_config.NumberColumn(
            label="True Negatives",
            width=10,
            help="The number of predicted home losses that were home losses"
        ),
        "false_positives": st.column_config.NumberColumn(
            label="False Positives",
            width=10,
            help="The number of predicted home wins that were home losses"
        ),
        "false_negatives": st.column_config.NumberColumn(
            label="False Negatives",
            width=10,
            help="The number of predicted home losses that were home wins"
        )
    },
    use_container_width=True,
    row_height=80,
    height=80 * performance_df.shape[0] + 80
)