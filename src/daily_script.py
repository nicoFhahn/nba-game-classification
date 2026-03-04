import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    # Use optimized dependency versions
    import bbref as bbref
    import elo_rating as elo_rating
    import predictions as predictions
    import supabase_helper as supabase_helper

    from google.cloud import secretmanager
    from supabase import create_client
    import tabpfn_client

    import polars as pl
    import joblib

    return (
        bbref,
        create_client,
        elo_rating,
        joblib,
        json,
        predictions,
        secretmanager,
        tabpfn_client,
    )


@app.cell
def _(create_client, json, secretmanager, tabpfn_client):
    name = "projects/898760610238/secrets/supabase/versions/7"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase = create_client(url, key)
    tabpfn_client.set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGYyM2VmYzktYzNlNy00YWNjLTljOGMtMGYyNWVlZGU1YTFiIiwiZXhwIjoxODAzNzQzMTQ3fQ.SrxoILOo8WwLT9d1rF1XXMqQP_IhZrd7DG5z7gf1MO8")
    # check why this key is not working
    return (supabase,)


@app.cell
def _(bbref, supabase):
    bbref.update_current_month(
        supabase,
        bbref.start_driver()
    )
    # checked - approved 
    return


@app.cell
def _(bbref, supabase):
    bbref.scrape_missing_boxscores(supabase)
    # checked - approved
    return


@app.cell
def _(elo_rating, supabase):
    elo_rating.elo_update(supabase)
    # checked - approved
    return


@app.cell
def _(predictions, supabase):
    predictions.update_team_records(supabase)
    # checked - approved
    return


@app.cell
def _(predictions, supabase):
    predictions.update_existing_predictions(supabase)
    return


@app.cell
def _(predictions, supabase):
    games_to_predict = predictions.upcoming_game_data(supabase)
    return (games_to_predict,)


@app.cell
def _(games_to_predict, joblib, json, predictions, supabase):
    if games_to_predict.shape[0] > 0:
        with open("models/best_features.json", "r") as f:
            best_features = json.load(f)["features"]
        catboost_model = joblib.load("models/cat_boost.pkl")
        lgbm_model = joblib.load("models/lgbm.pkl")
        xgb_model = joblib.load("models/xgboost.pkl")
        extra_model = joblib.load("models/extra.pkl")
        ensemble_model = joblib.load("models/ensemble.pkl")
        tabpfn_model = joblib.load("models/tab_pfn.pkl")
        model_dict = {
            "catboost": catboost_model,
            "lgbm": lgbm_model,
            "xgb": xgb_model,
            "extra": extra_model,
            "ensemble": ensemble_model,
            "tabpfn": tabpfn_model
        }
        upcoming_predictions = predictions.predict_upcoming_games(
            games_to_predict, best_features, model_dict
        )
        supabase.table("prediction-comparison").insert(upcoming_predictions.to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
