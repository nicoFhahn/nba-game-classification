import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import calendar
    import json
    import time
    from datetime import date, timedelta

    import polars as pl
    from supabase import Client, create_client
    from tqdm import tqdm

    import features

    from supabase_helper import fetch_entire_table, fetch_filtered_table, fetch_distinct_column
    from google.cloud import secretmanager
    from catboost import CatBoostClassifier
    from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor
    import joblib
    import tabpfn_client
    import importlib
    import ml_helpers

    return create_client, json, pl, secretmanager, tabpfn_client


@app.cell
def _(create_client, json, secretmanager, tabpfn_client):
    name = "projects/898760610238/secrets/supabase/versions/7"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    tabpfn_client.set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGYyM2VmYzktYzNlNy00YWNjLTljOGMtMGYyNWVlZGU1YTFiIiwiZXhwIjoxODAzNzQzMTQ3fQ.SrxoILOo8WwLT9d1rF1XXMqQP_IhZrd7DG5z7gf1MO8")
    return


@app.cell
def _(pl):
    from sqlalchemy import create_engine, text

    # 1. Load your data
    df = pl.read_csv("shapley_cat.csv")
    pw = "cuhsom-2dYmhu-qygpeq"

    # 2. Define your Supabase Connection URI
    # Replace placeholders with your actual Supabase credentials
    #DB_URI = f"postgresql://postgres:{pw}@db.fwnaeycchrvmiurojlkd.supabase.co:5432/postgres"
    DB_URI = f"postgresql://postgres:{pw}@db.fwnaeycchrvmiurojlkd.supabase.co:6543/postgres?sslmode=require"

    # 3. Create the SQLAlchemy engine
    engine = create_engine(DB_URI)

    # 4. Write to Supabase
    # 'if_table_exists' can be "fail", "replace", or "append"
    df.write_database(
        table_name="shapley",
        connection=engine,
        if_table_exists="replace" 
    )

    # 3. Execute SQL to Enable RLS
    with engine.connect() as conn:
        # Enable RLS on the table
        conn.execute(text("ALTER TABLE shapley ENABLE ROW LEVEL SECURITY;"))
    
        # Optional: Add a 'Select' policy so authenticated users can see data
        # Without a policy, even you won't see data in the API (though you'll see it in the Dashboard)
        conn.execute(text("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies WHERE tablename = 'shapley' AND policyname = 'Allow auth select'
                ) THEN
                    CREATE POLICY "Allow auth select" ON "public"."shapley"
                    FOR SELECT TO authenticated USING (true);
                END IF;
            END $$;
        """))
        conn.commit()

    print("Table 'shapley' created and RLS is now active.")
    return


if __name__ == "__main__":
    app.run()
