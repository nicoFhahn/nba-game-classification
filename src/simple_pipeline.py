import numpy as np
import polars as pl
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── per-model search spaces ────────────────────────────────────────────────────

def _catboost_params(trial, cat_features):
    return dict(
        iterations          = trial.suggest_int("iterations", 100, 1000),
        depth               = trial.suggest_int("depth", 3, 8),
        learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        border_count        = trial.suggest_int("border_count", 32, 255),
        subsample           = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bylevel   = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        min_child_samples   = trial.suggest_int("min_child_samples", 5, 100),
        random_strength     = trial.suggest_float("random_strength", 0, 10),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1),
        grow_policy         = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        auto_class_weights  = trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", None]),
        cat_features        = cat_features,
        verbose             = 0,
    )

def _lgbm_params(trial, cat_features):
    return dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 300),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
        class_weight      = trial.suggest_categorical("class_weight", ["balanced", None]),
        verbose           = -1,
        force_row_wise    = True,
    )

def _xgboost_params(trial, cat_features):
    return dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
        gamma             = trial.suggest_float("gamma", 0, 5),
        scale_pos_weight  = trial.suggest_float("scale_pos_weight", 0.5, 10),
        verbosity         = 0,
    )

def _extratrees_params(trial, cat_features):
    return dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
        max_depth         = trial.suggest_int("max_depth", 3, 20),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20),
        class_weight      = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        n_jobs            = -1,
    )

# ── model registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "catboost":   (CatBoostClassifier,   _catboost_params),
    "lightgbm":   (LGBMClassifier,       _lgbm_params),
    "xgboost":    (XGBClassifier,        _xgboost_params),
    "extratrees": (ExtraTreesClassifier, _extratrees_params),
}

# ── preprocessing ──────────────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode any object/string columns for non-CatBoost models."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df

def _prepare_X(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """CatBoost handles categoricals natively; all others need encoded ints."""
    if model_name == "catboost":
        return df
    return _encode_categoricals(df)

# ── fit helpers ────────────────────────────────────────────────────────────────

def _fit_model(model_name, model, X_tr, y_tr, X_val, y_val):
    if model_name == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  early_stopping_rounds=50, verbose=0)
    elif model_name == "lightgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
    elif model_name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:  # extratrees — no early stopping
        model.fit(X_tr, y_tr)

# ── objective ──────────────────────────────────────────────────────────────────

def objective(trial, model_name: str, X: pl.DataFrame, y: pl.Series, cat_features=None):
    ModelClass, param_fn = MODEL_REGISTRY[model_name]
    params = param_fn(trial, cat_features)

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, val_idx in tscv.split(range(len(X))):
        X_tr  = _prepare_X(model_name, X[train_idx].to_pandas())
        X_val = _prepare_X(model_name, X[val_idx].to_pandas())
        y_tr  = y[train_idx].to_pandas()
        y_val = y[val_idx].to_pandas()

        model = ModelClass(**params)
        _fit_model(model_name, model, X_tr, y_tr, X_val, y_val)
        scores.append(accuracy_score(y_val, model.predict(X_val)))

    return np.mean(scores)

# ── main pipeline ──────────────────────────────────────────────────────────────

def train_pipeline(
    X: pl.DataFrame,
    y: pl.Series,
    model_name: str = "catboost",
    cat_features=None,
    n_trials: int = 50,
):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    ModelClass, param_fn = MODEL_REGISTRY[model_name]

    # Phase 1: tune
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model_name, X, y, cat_features),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"[{model_name}] Best CV accuracy : {study.best_value:.4f}")
    print(f"[{model_name}] Best params      : {best_params}")

    # Phase 2: retrain on full data
    if model_name == "catboost":
        best_params["cat_features"] = cat_features

    final_model = ModelClass(**best_params)
    X_pd = _prepare_X(model_name, X.to_pandas())

    if model_name == "catboost":
        final_model.fit(X_pd, y.to_pandas(), verbose=0)
    else:
        final_model.fit(X_pd, y.to_pandas())

    return final_model, best_params