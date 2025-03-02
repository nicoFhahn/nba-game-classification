import sys
import os
import json

import streamlit as st
from supabase import create_client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import modelling


@st.cache_resource(ttl=3600)
def get_model():
    with open('streamlit/credentials.json', 'r') as f:
        creds = json.loads(f.read())
        connection = create_client(creds['postgres']['project_url'], creds['postgres']['api_key'])
    mod = modelling.lgbm_model(connection)
    return mod, connection