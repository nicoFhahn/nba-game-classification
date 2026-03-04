import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Standard library
    import calendar
    import json
    import time
    from datetime import date, timedelta
    import importlib
    import os

    # Third-party
    import tabpfn_client
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    from catboost import CatBoostClassifier
    from google.cloud import secretmanager
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, log_loss
    from supabase import Client, create_client
    from tabpfn_client import init, TabPFNClassifier, TabPFNRegressor
    from tqdm import tqdm
    import joblib
    import optuna

    # Local modules
    import features
    import ml_helpers
    from supabase_helper import (
        fetch_entire_table,
        fetch_filtered_table,
        fetch_distinct_column,
    )

    return (
        TabPFNClassifier,
        accuracy_score,
        create_client,
        date,
        fetch_entire_table,
        importlib,
        joblib,
        json,
        log_loss,
        ml_helpers,
        os,
        pl,
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
    supabase=create_client(url, key)
    tabpfn_client.set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGYyM2VmYzktYzNlNy00YWNjLTljOGMtMGYyNWVlZGU1YTFiIiwiZXhwIjoxODAzNzQzMTQ3fQ.SrxoILOo8WwLT9d1rF1XXMqQP_IhZrd7DG5z7gf1MO8")
    return (supabase,)


@app.cell
def _(fetch_entire_table, pl, supabase):
    temp = pl.read_csv("results_updated.csv")
    current_preds = fetch_entire_table(supabase, "prediction-comparison")
    missing = current_preds.filter(
        ~pl.col("game_id").is_in(temp["game_id"].to_list())
    )
    return missing, temp


@app.cell
def _(importlib, ml_helpers, supabase):
    importlib.reload(ml_helpers)
    games, schedule, player_boxscore, elo = ml_helpers.load_data(supabase, season_ids = [2026], use_player_boxscore=True)
    return elo, games, player_boxscore, schedule


@app.cell
def _(elo, games, ml_helpers, pl, player_boxscore, schedule):
    preprocessed_data = ml_helpers.preprocess_data(games, schedule, player_boxscore, elo).with_columns(
        pl.all().map_elements(
            lambda x: None if x in (float("inf"), float("-inf")) else x
        )
    )
    return (preprocessed_data,)


@app.cell
def _(missing, pl, preprocessed_data):
    preprocessed_data.filter(
        pl.col("game_id").is_in(missing["game_id"].to_list())
    )
    return


@app.cell
def _(joblib, json):
    with open("models/best_features_20260301_log_loss.json", "r") as f:
        bf = json.load(f)
    catboost_mod = joblib.load("models/cat_boost_20260301_log_loss.pkl")
    lgbm_mod = joblib.load("models/lgbm_20260301_log_loss.pkl")
    xgb_mod = joblib.load("models/xgboost_20260301_log_loss.pkl")
    extra_mod = joblib.load("models/extra_20260301_log_loss.pkl")
    ensemble_mod = joblib.load("models/ensemble_20260301_log_loss.pkl")
    tabpfn_mod = joblib.load("models/tab_pfn_20260301_log_loss.pkl")
    return (
        bf,
        catboost_mod,
        ensemble_mod,
        extra_mod,
        lgbm_mod,
        tabpfn_mod,
        xgb_mod,
    )


@app.cell
def _(
    bf,
    catboost_mod,
    ensemble_mod,
    extra_mod,
    lgbm_mod,
    missing,
    pl,
    preprocessed_data,
    tabpfn_mod,
    xgb_mod,
):
    missing2 = preprocessed_data.filter(pl.col("game_id").is_in(missing["game_id"].to_list()))
    X3 = missing2.select(bf["features"])
    c_pred = catboost_mod.predict_proba(X3)
    l_pred = lgbm_mod.predict_proba(X3)
    x_pred = xgb_mod.predict_proba(X3)
    ex_pred = extra_mod.predict_proba(X3)
    e_pred = ensemble_mod.predict_proba(X3)
    t_pred = tabpfn_mod.predict_proba(X3.to_numpy())
    return c_pred, e_pred, ex_pred, l_pred, missing2, t_pred, x_pred


@app.cell
def _(c_pred, e_pred, ex_pred, l_pred, missing2, pl, t_pred, x_pred):
    temp2 = pl.DataFrame({
        "game_id": missing2["game_id"],
        "is_home_win": missing2["is_home_win"],
        "proba_catboost": c_pred,
        "proba_lgbm": l_pred,
        "proba_xgb": x_pred,
        "proba_extra": ex_pred,
        "proba_ensemble": e_pred,
        "proba_tabpfn": t_pred
    }).with_columns([
        pl.col("proba_ensemble").arr.last(),
        pl.col("proba_catboost").arr.last(),
        pl.col("proba_lgbm").arr.last(),
        pl.col("proba_xgb").arr.last(),
        pl.col("proba_extra").arr.last(),
        pl.col("proba_tabpfn").arr.last()
    ]).with_columns([
        (pl.col("proba_catboost") >= 0.5).alias("is_predicted_home_win_catboost"),
        (pl.col("proba_lgbm") >= 0.5).alias("is_predicted_home_win_lgbm"),
        (pl.col("proba_xgb") >= 0.5).alias("is_predicted_home_win_xgb"),
        (pl.col("proba_extra") >= 0.5).alias("is_predicted_home_win_extra"),
        (pl.col("proba_tabpfn") >= 0.5).alias("is_predicted_home_win_tabpfn"),
        (pl.col("proba_ensemble") >= 0.5).alias("is_predicted_home_win_ensemble")
    ])
    return (temp2,)


@app.cell
def _(importlib, supabase):
    from supabase_helper import fetch_distinct_column_filtered
    import predictions
    importlib.reload(predictions)
    games_to_predict = predictions.upcoming_game_data(supabase)
    return games_to_predict, predictions


@app.cell
def _(
    bf,
    catboost_mod,
    ensemble_mod,
    extra_mod,
    games_to_predict,
    importlib,
    lgbm_mod,
    predictions,
    tabpfn_mod,
    xgb_mod,
):
    importlib.reload(predictions)
    upcoming_predictions = predictions.predict_upcoming_games(
        games_to_predict, bf["features"], {
            "catboost": catboost_mod,
            "lgbm": lgbm_mod,
            "xgb": xgb_mod,
            "extra": extra_mod,
            "ensemble": ensemble_mod,
            "tabpfn": tabpfn_mod
        }
    )
    return (upcoming_predictions,)


@app.cell
def _(pl, supabase, temp, temp2, upcoming_predictions):
    supabase.table("prediction-comparison").upsert(
        pl.concat([pl.concat([
            temp2, upcoming_predictions[temp2.columns]
        ]).with_columns(
            pl.col("proba_xgb").cast(pl.Float64)
        )[temp.columns], temp]).to_dicts()
    ).execute()
    return


@app.cell(disabled=True)
def _(
    TabPFNClassifier,
    date,
    importlib,
    joblib,
    json,
    log_loss,
    os,
    pl,
    preprocessed_data,
):
    import stepwise_feature_selection
    import ensemble_pipeline
    import simple_pipeline
    importlib.reload(ensemble_pipeline)
    importlib.reload(simple_pipeline)
    date_pairs = [
        (date(2025, 10, 1), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 28)),
        (date(2026, 3, 1), date(2026, 3, 31))
    ]
    configs = [
        {"method": "simple_averaging"},
        {"method": "weighted_averaging"},
        {"method": "stacking", "meta_model": "catboost", "use_oof": True},
        {"method": "stacking", "meta_model": "logistic_regression", "use_oof": True},
        {"method": "blending", "meta_model": "catboost", "holdout_frac": 0.2},
        {"method": "blending", "meta_model": "logistic_regression", "holdout_frac": 0.2},
        {"method": "voting", "voting_type": "soft"}
    ]
    simple_result_dfs = []
    ensemble_result_dfs = []
    n_trials=50
    for pair in date_pairs:
        fs_file_name = f"models/best_features_{pair[0].strftime('%Y%m%d')}_log_loss.json"
        if not os.path.exists(fs_file_name):
            s1 = stepwise_feature_selection.LGBMStepwiseFeatureSelector(
                importance_type='shap',
                cv_strategy='timeseries',   
                max_features=50,             
                use_ensemble=True,           
                metric=log_loss,
                n_folds=3,
                verbose=True
            )
            X_train = train.drop(["game_id", "home_team", "guest_team", "is_home_win"])
            y_train = train.select("is_home_win")
            s1.fit(
                X_train.drop("date").to_numpy(),
                y_train.to_numpy().ravel()
            )
            selected_indices = [
                int(feat.replace('feature_', ''))
                for feat in s1.best_features_
            ]
            original_feature_names = X_train.columns
            bf1 = [
                original_feature_names[idx]
                for idx in selected_indices
            ]
            bf  = {
                "features": bf1
            }
            with open(fs_file_name, "w") as f:
                json.dump(bf, f)
        else:
            with open(fs_file_name, "r") as f:
                bf = json.load(f)
        train = preprocessed_data.filter(pl.col("date") < pair[0])
        val = preprocessed_data.filter(
            pl.col("date").is_between(
                pair[0], pair[1]
            )
        )
        X_train = train.select(bf["features"])
        X_val = val.select(bf["features"])
        y_train = train.select("is_home_win")
        y_val = val.select("is_home_win")
        model_name_tab = f"models/tab_pfn_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        model_name_cat = f"models/cat_boost_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        model_name_lgbm = f"models/lgbm_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        model_name_xgb = f"models/xgboost_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        model_name_extra = f"models/extra_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        if not os.path.exists(model_name_cat):
            catboost_mod, catboost_params = simple_pipeline.train_pipeline(
                X_train, y_train, model_name = "catboost", n_trials=n_trials
            )
            joblib.dump(catboost_mod, model_name_cat)
        else:
            catboost_mod = joblib.load(model_name_cat)
        if not os.path.exists(model_name_lgbm):
            lgbm_mod, lgbm_params = simple_pipeline.train_pipeline(
                X_train, y_train, model_name = "lightgbm", n_trials=n_trials
            )
            joblib.dump(lgbm_mod, model_name_lgbm)
        else:
            lgbm_mod = joblib.load(model_name_lgbm)
        if not os.path.exists(model_name_xgb):
            xgb_mod, xgb_params = simple_pipeline.train_pipeline(
                X_train, y_train, model_name = "xgboost", n_trials=n_trials
            )
            joblib.dump(xgb_mod, model_name_xgb)
        else:
            xgb_mod = joblib.load(model_name_xgb)
        if not os.path.exists(model_name_extra):
            extra_mod, extra_params = simple_pipeline.train_pipeline(
                X_train, y_train, model_name = "extratrees", n_trials=n_trials
            )
            joblib.dump(extra_mod, model_name_extra)
        else:
            extra_mod = joblib.load(model_name_extra)
        if not os.path.exists(model_name_tab):
            tabpfn_mod = TabPFNClassifier()
            tabpfn_mod.fit(X_train.to_numpy(), y_train.to_numpy().ravel())
            joblib.dump(tabpfn_mod, model_name_tab)
        else:
            tabpfn_mod = joblib.load(model_name_tab)
        if val.shape[0] > 0:
            catboost_pred = catboost_mod.predict_proba(X_val)
            lgbm_pred = lgbm_mod.predict_proba(X_val)
            xgb_pred = xgb_mod.predict_proba(X_val)
            extra_pred = extra_mod.predict_proba(X_val)
            tabpfn_pred = tabpfn_mod.predict_proba(X_val.to_numpy())
            pred_df = pl.DataFrame({
                "game_id": val["game_id"],
                "is_home_win": y_val,
                "proba_catboost": catboost_pred,
                "proba_lgbm": lgbm_pred,
                "proba_xgb": xgb_pred,
                "proba_extra": extra_pred,
                "proba_tabpfn": tabpfn_pred
            })
            simple_result_dfs.append(pred_df)
        ensemble_file_name = f"models/ensemble_{pair[0].strftime('%Y%m%d')}_log_loss.pkl"
        if not os.path.exists(ensemble_file_name):
            base = ensemble_pipeline.train_base_models(
                X_train=X_train,
                y_train=y_train,
                split_type="temporal",
                algorithms=["catboost", "lightgbm", "xgboost"],
                metric="neg_log_loss",
                n_trials=n_trials,
            )

            # Ensemble configs iterate cheaply — no HPO repeated
            pipeline_results = [
                ensemble_pipeline.build_ensemble_from_base_models(base, config)
                for config in configs
            ]

            best_result = max(pipeline_results, key=lambda x: x["cv_score"])

            # Extract the best model and score
            best_ensemble = best_result["ensemble"]
            best_score = best_result["cv_score"]
            joblib.dump(best_ensemble, ensemble_file_name)
        else:
            best_ensemble = joblib.load(ensemble_file_name)

        # Get predictions only for the best one
        if val.shape[0] > 0:
            probabilities = best_ensemble.predict_proba(X_val)

            result_df = pl.DataFrame({
                "game_id": val["game_id"],
                "is_home_win": y_val, # Ensure it's a series for the DataFrame
                "proba": probabilities
            })
            ensemble_result_dfs.append(result_df)
    return (
        bf,
        catboost_mod,
        ensemble_result_dfs,
        extra_mod,
        lgbm_mod,
        simple_result_dfs,
        tabpfn_mod,
        train,
        xgb_mod,
    )


@app.cell(disabled=True)
def _(ensemble_result_dfs, pl, simple_result_dfs):
    rdf = pl.concat(ensemble_result_dfs).join(
        pl.concat(simple_result_dfs).drop("is_home_win"),
        on="game_id"
    ).with_columns([
        pl.col("proba").arr.last().alias("proba_ensemble"),
        pl.col("proba_catboost").arr.last(),
        pl.col("proba_lgbm").arr.last(),
        pl.col("proba_xgb").arr.last(),
        pl.col("proba_extra").arr.last(),
        pl.col("proba_tabpfn").arr.last()
    ]).drop("proba").with_columns([
        (pl.col("proba_catboost") >= 0.5).alias("is_predicted_home_win_catboost"),
        (pl.col("proba_lgbm") >= 0.5).alias("is_predicted_home_win_lgbm"),
        (pl.col("proba_xgb") >= 0.5).alias("is_predicted_home_win_xgb"),
        (pl.col("proba_extra") >= 0.5).alias("is_predicted_home_win_extra"),
        (pl.col("proba_tabpfn") >= 0.5).alias("is_predicted_home_win_tabpfn"),
        (pl.col("proba_ensemble") >= 0.5).alias("is_predicted_home_win_ensemble")
    ])
    return (rdf,)


@app.cell
def _(rdf):
    #todo, again mit 50 trials für catboost + xgboost + lightgbm
    rdf.write_csv("results_updated.csv")
    return


@app.cell
def _(accuracy_score, log_loss, rdf):
    print(f"""
    Log Loss
    CatBoost: {log_loss(rdf["is_home_win"], rdf["proba_catboost"]):.4f}
    LGBM: {log_loss(rdf["is_home_win"], rdf["proba_lgbm"]):.4f}
    XGBoost: {log_loss(rdf["is_home_win"], rdf["proba_xgb"]):.4f}
    ExtraTrees: {log_loss(rdf["is_home_win"], rdf["proba_extra"]):.4f}
    TabPFN: {log_loss(rdf["is_home_win"], rdf["proba_tabpfn"]):.4f}
    Ensemble: {log_loss(rdf["is_home_win"], rdf["proba_ensemble"]):.4f}
    Log Accuracy
    CatBoost: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_catboost"]):.4f}
    LGBM: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_lgbm"]):.4f}
    XGBoost: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_xgb"]):.4f}
    ExtraTrees: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_extra"]):.4f}
    TabPFN: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_tabpfn"]):.4f}
    Ensemble: {accuracy_score(rdf["is_home_win"], rdf["is_predicted_home_win_ensemble"]):.4f}
    """)
    return


@app.cell
def _(date, joblib, json, pl, preprocessed_data):
    import shap
    cb_mod = joblib.load("models/cat_boost_20260301.pkl")
    with open("models/best_features_20260301.json", "r") as f2:
        best_features = json.load(f2)["features"]
    X_train_pd = preprocessed_data.filter(pl.col("date") < date(2026, 2, 1)).select(best_features).to_pandas()

    # Create explainer
    explainer = shap.TreeExplainer(cb_mod)

    # Get SHAP values
    shap_values = explainer.shap_values(X_train_pd)


    # shap_values is a numpy array of shape (n_samples, n_features)
    return X_train_pd, best_features, explainer, shap, shap_values


@app.cell
def _(X_train_pd, explainer, shap):
    shap.plots.beeswarm(explainer(X_train_pd))
    return


@app.cell
def _(best_features, pl, shap_values):
    #import psycopg2
    pl.DataFrame(
        shap_values,
        schema=best_features
    ).write_csv("shapley_cat.csv")
    return


@app.cell
def _():
    # todo - evaluate predictive performance for catboost and tabpfn
    # todo - train the stacked models for the same timeframe. evaluate predictive performance on test -> keep best stacked model
    # todo - new predictions table w. predictions for all model types
    # todo - shapley values for latest catboost model
    return


if __name__ == "__main__":
    app.run()
