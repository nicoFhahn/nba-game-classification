from datetime import datetime
import json
import os
import glob
import joblib
import polars as pl

def load_model(folder: str):
    pkl_files = glob.glob(f"{folder}/*.pkl")
    mod = joblib.load(max(pkl_files, key=os.path.getmtime))
    return mod

def load_data(folder: str):
    df = pl.read_parquet(os.path.join(folder, "season_df.parquet"))
    df = df.to_dummies([
        "game_type", "month", "weekday"
    ])
    return df

def load_evaluation(folder: str):
    eval_files = glob.glob(f"{folder}/lgbm*.json")
    with open(max(eval_files, key=os.path.getmtime), "r") as f:
        evaluation = json.loads(f.read())
    last_train_date = datetime.strptime(evaluation["last_train_date"], "%Y-%m-%d").date()
    best_features = evaluation["features"]["trained_on"][evaluation["performance"]["accuracy"].index(
        max(evaluation["performance"]["accuracy"])
    )]
    return best_features, last_train_date

def make_predictions(df, best_features, last_train_date, mod):
    X_new = df.filter(
        pl.col("date") > last_train_date
    )
    game_ids = X_new["game_id"]
    X_new = X_new.select(
        pl.col(best_features)
    ).drop("date")
    predictions = mod.predict(X_new.to_numpy())
    prediction_df = pl.DataFrame({
        "game_id": game_ids,
        "probability": predictions,
        "is_home_win": predictions >= 0.5
    })
    return prediction_df