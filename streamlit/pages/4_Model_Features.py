import sys
import os

import polars as pl
import matplotlib.pyplot as plt
import streamlit as st
import shap

from src.helpers import get_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import data_wrangling

st.set_page_config(layout='wide', page_title='Model Features')
if 'model' not in st.session_state:
    with st.spinner('Loading model and data', show_time=True):
        mod, connection = get_model()
        st.session_state['model'] = mod
        st.session_state['connection'] = connection

df_today = data_wrangling.newest_games(
    st.session_state['connection'],
    st.session_state['model']
)
mod = st.session_state['model']
mod.load_model()
games = mod.full_data
best_features = mod.load_best_features()[0]
train_data = pl.concat([
    mod.data_sets['train'], mod.data_sets['test']
]).sort('game_id')
explainer = shap.Explainer(
    mod.model,
    pl.concat(
        [mod.X['train'], mod.X['test']]
    ).select(
        pl.col(best_features)
    ).drop('date').to_numpy(),
    feature_names=mod.X['train'].select(pl.col(best_features)).drop('date').columns
)
shap_values_train = explainer(
        pl.concat(
            [mod.X['train'], mod.X['test']]
        ).select(
            pl.col(best_features)
        ).drop('date').sample(400, seed=7918).to_numpy()
    )
st.markdown(
    '''
    All of the plots shown on this page were calculated using a sample of 400 observations from the training data.
    
    Variables containing '_8' denote 8-game running averages and variables containing '_109' denote season running
    averages (as a team can play a max of 110 games in a season)
    '''
)
col_1, col_2 = st.columns(2)
with col_1:
    st.markdown(
        '''<h3> Beeswarm Plot </h3>''', unsafe_allow_html=True
    )
    fig_1, ax = plt.subplots(figsize=(14, 8))
    shap.plots.beeswarm(shap_values_train)
    st.pyplot(
        fig_1,
        use_container_width=False
    )
    st.markdown(
        '''<h3> Heatmap </h3>''', unsafe_allow_html=True
    )
    fig_2, ax = plt.subplots()
    shap.plots.heatmap(shap_values_train)
    st.pyplot(
        fig_2,
        use_container_width=False
    )
with col_2:
    st.markdown(
        '''<h3> Bar Plot </h3>''', unsafe_allow_html=True
    )
    fig_3, ax = plt.subplots(figsize=(14, 8))
    shap.plots.bar(shap_values_train)
    st.pyplot(
        fig_3,
        use_container_width=False,
    )