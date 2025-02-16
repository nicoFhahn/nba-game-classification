import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")


st.write("Model Performance")
col_1, col_2, col_3 = st.columns(3)
with col_1:
    metric_title_1 = "Accuracy"
    ui.metric_card(
        title=metric_title_1,
        content=0.77
    )

style_metric_cards(
    border_left_color="#7C1F31"
)