import streamlit as st
from src.helpers import get_model

st.set_page_config(
    page_title='NBA Games Prediction'
)

if 'model' not in st.session_state:
    with st.spinner('Loading model and data', show_time=True):
        mod, connection = get_model()
        st.session_state['model'] = mod
        st.session_state['connection'] = connection