import sys
import os
import streamlit as st
from supabase import create_client
from google.cloud import storage
from google.cloud import secretmanager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import modelling


@st.cache_resource(ttl=3600)
def get_model():
    client = storage.Client()
    bucket = client.get_bucket("lgbm")
    secret_client = secretmanager.SecretManagerServiceClient()
    response = secret_client.access_secret_version(
        request={'name': 'projects/898760610238/secrets/supabase/versions/1'}
    )
    creds = eval(response.payload.data.decode("UTF-8"))
    connection = create_client(creds['postgres']['project_url'], creds['postgres']['api_key'])
    mod = modelling.lgbm_model(connection, bucket)
    return mod, connection