import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from supabase import create_client, Client
    from supabase_helper import fetch_entire_table
    from ml_pipeline import main_pipeline, load_pipeline
    import features
    from tqdm import tqdm
    from datetime import date, datetime
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    import polars as pl
    import numpy as np
    import polars.selectors as cs
    import lightgbm as lgbm
    import json
    import stepwise_feature_selection
    import importlib
    import ml_helpers
    from pathlib import Path
    import ml_pipeline
    from google.cloud import secretmanager
    import warnings

    # Suppress sklearn warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    # Suppress LightGBM feature name warnings
    warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

    return (
        create_client,
        date,
        importlib,
        json,
        load_pipeline,
        ml_helpers,
        pl,
        secretmanager,
    )


@app.cell
def _(create_client, json, secretmanager):
    name = "projects/898760610238/secrets/supabase/versions/2"
    client = secretmanager.SecretManagerServiceClient()
    g_response = client.access_secret_version(request={"name": name})
    payload = g_response.payload.data.decode("UTF-8")
    payload_dict = json.loads(payload)
    url = payload_dict["postgres"]["project_url"]
    key = payload_dict["postgres"]["api_key"]
    supabase=create_client(url, key)
    return (supabase,)


@app.cell
def _(date, importlib, ml_helpers, supabase):
    importlib.reload(ml_helpers)
    cutoff_dates = [date(2026, 2, 1)]
    for c in cutoff_dates:
        pipe = ml_helpers.run_pipeline(
            supabase,
            cutoff_date = c,
            use_weights= False,
            run_fs = True,
            n_trials = 300,
            max_estimators=1500,
            n_jobs_optuna=6,
            output_dir=f"ensemble_{c.strftime('%Y%m%d')}",
            random_state=2352,
            use_ensemble=True
        )
    # need to verify data is in correct order
    # 0.698
    # antigravity project and ask it to improve my feature generation
    return


@app.cell
def _(X_val_best, load_pipeline):
    saved = load_pipeline('ensemble_model_v2')
    predictions = saved['ensemble'].predict_proba(X_val_best.to_numpy())
    threshold=saved['threshold']
    return predictions, threshold


@app.cell
def _(pl, predictions, threshold, val):
    pred_df = val.select(["game_id", "date", "home_team", "guest_team", "is_home_win"]).with_columns([
        pl.Series(
            "proba",
            predictions
        )
    ]).with_columns([
        (pl.col("proba") >= threshold).alias("is_predicted_home_win")
    ])
    return (pred_df,)


@app.cell
def _(pred_df_2):
    pred_df_2.sort("proba")
    return


@app.cell
def _():
    # below 55% confidence -> low
    # below 68% confidence -> medium
    # above -> high
    # todo: change confidence to classes
    # add one tab for future predictions, one tab for past predictions
    # add rolling performance metrics
    # check whether the data is live
    # automate data uploads
    # build table for future games
    return


@app.cell
def _(pl, pred_df, supabase):
    supabase.table(
        "predictions"
    ).insert(pred_df.with_columns([
        pl.col("date").cast(pl.String)
    ]).to_dicts()).execute()
    return


if __name__ == "__main__":
    app.run()
