import streamlit as st
from src.helpers import get_model

st.set_page_config(
    page_title='NBA Games Prediction'
)
#with open('.streamlit/styles.css') as f:
#    css = f.read()

#st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

if 'model' not in st.session_state:
    with st.spinner('Loading model and data', show_time=True):
        mod, connection = get_model()
        st.session_state['model'] = mod
        st.session_state['connection'] = connection

st.markdown('''
This app shows the performance of a LGBM model that aims at predicting the outcome of games in the 
National Basketball Association. 

It has been trained on data starting from the 2020 season.

The data comes from the NBAs official API and is updated near daily.

The model is retrained every month.
''')