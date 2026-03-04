"""
ensemble_pipeline.py
====================
A flexible, production-quality machine learning pipeline supporting:
  - Multiple base models (CatBoost, LightGBM, XGBoost, TabPFN)
  - Optuna hyperparameter optimisation
  - Temporal and random cross-validation splits
  - Multiple ensemble strategies (averaging, stacking, blending, voting, …)
  - Generic sklearn metric handling
  - Polars-native inputs

Usage
-----
from ensemble_pipeline import run_pipeline

result = run_pipeline(
    X_train=X_pl,
    y_train=y_pl,
    split_type="temporal",          # "temporal" | "random"
    algorithms=["catboost", "lightgbm"],
    metric="f1_weighted",           # any valid sklearn scorer name
    stacking_options={
        "method": "stacking",       # see ENSEMBLE_METHODS.md
        "meta_model": "lightgbm",   # only used for stacking
        "use_oof": True,
        "n_trials": 50,
    },
)

best_model   = result["ensemble"]
base_models  = result["base_models"]
best_params  = result["best_params"]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SPLITS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Lazy imports – only import a library when the user actually requests it
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Polars → pandas conversion (dtype-safe)
# ---------------------------------------------------------------------------

def _polars_to_pandas(df: pl.DataFrame) -> "pd.DataFrame":
    """
    Convert a Polars DataFrame to pandas with every column backed by a plain
    numpy dtype (int, float64, bool / uint8).  No pandas nullable extension
    types survive this function.

    Why this matters
    ----------------
    * ``pl.Boolean`` columns can arrive as pandas ``object`` dtype (Polars ≤ 0.19)
      or as the nullable ``BooleanDtype`` extension (Polars ≥ 0.20).
    * Nullable extension dtypes (``Float32``, ``Int64``, …) return an **object**
      array full of ``pd.NA`` sentinels when ``.to_numpy()`` is called — even
      though they *look* numeric.  CatBoost, LightGBM, and XGBoost all reject
      ``pd.NA``; they require genuine ``np.nan`` (a float) for missing values.

    Conversion rules
    ----------------
    1. ``pl.Boolean``, no nulls  → numpy ``uint8``  (0 / 1)
    2. ``pl.Boolean``, has nulls → numpy ``float64`` (0.0 / 1.0 / NaN)
    3. Any other pandas extension dtype still present after step 1-2
       (e.g. ``Int8``, ``Float32``) → numpy ``float64`` with NaN for nulls.
    """
    import pandas as pd

    pdf = df.to_pandas()

    # ---- Step 1 & 2: handle Polars Boolean columns -------------------------
    for col in pdf.columns:
        if df[col].dtype != pl.Boolean:
            continue

        if not df[col].is_null().any():
            # No nulls: safe to cast directly to uint8
            pdf[col] = pdf[col].astype("uint8")
        else:
            # Has nulls: go through nullable BooleanDtype so None → pd.NA,
            # then immediately land in float64 where pd.NA → np.nan.
            pdf[col] = pdf[col].astype("boolean").astype("float64")

    # ---- Step 3: flush any remaining extension-array columns ---------------
    # After the Polars → pandas conversion (and the Boolean handling above),
    # other nullable types like Float32 / Int16 may still be present.
    # .to_numpy() on these produces object arrays with pd.NA — we fix that here.
    for col in pdf.columns:
        if pd.api.types.is_extension_array_dtype(pdf[col].dtype):
            pdf[col] = pdf[col].to_numpy(dtype="float64", na_value=np.nan)

    return pdf


def _import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError as e:
        raise ImportError("catboost is not installed. Run: pip install catboost") from e


def _import_lightgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier
    except ImportError as e:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from e


def _import_xgboost():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier
    except ImportError as e:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from e


def _import_tabpfn():
    try:
        from tabpfn_client import TabPFNClassifier
        return TabPFNClassifier
    except ImportError as e:
        raise ImportError(
            "tabpfn-client is not installed. Run: pip install tabpfn-client"
        ) from e


# ---------------------------------------------------------------------------
# Split factory
# ---------------------------------------------------------------------------

def make_splitter(split_type: str, n_splits: int = N_SPLITS, n_classes: Optional[int] = None):
    """Return an sklearn cross-validator that matches the requested split type."""
    if split_type == "temporal":
        return TimeSeriesSplit(n_splits=n_splits)
    elif split_type == "random":
        if n_classes and n_classes > 1:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"split_type must be 'temporal' or 'random', got '{split_type}'")


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _optuna_direction(metric: str) -> str:
    """
    Return the correct Optuna optimisation direction for a given sklearn metric.

    sklearn scorers follow the convention that *higher is always better* — loss
    metrics are stored as their negative (e.g. ``neg_log_loss``).  However,
    users sometimes pass the raw loss name (``log_loss``, ``brier_score``).
    We detect those and tell Optuna to minimise; everything else is maximised.
    """
    _MINIMISE_METRICS = {
        # Raw loss names a user might pass directly
        "log_loss", "brier_score", "brier_score_loss",
        "mean_squared_error", "mse",
        "mean_absolute_error", "mae",
        "mean_squared_log_error", "msle",
        # sklearn scorer convention: names starting with "neg_" are stored
        # negated, so the scorer already returns higher-is-better — but if
        # someone passes the bare loss name we still want to minimise.
    }
    return "minimize" if metric.lower() in _MINIMISE_METRICS else "maximize"


def make_scorer(metric: str):
    """
    Return an sklearn scorer with ``._metric_name`` attached so
    ``evaluate_scorer`` can determine predict_proba vs predict without
    relying on unstable sklearn internal attributes.
    """
    s = get_scorer(metric)
    s._metric_name = metric
    return s


def evaluate_scorer(scorer, model, X, y) -> float:
    """
    Score *model* on (X, y) without going through sklearn's scorer wrapper.

    sklearn ≥ 1.4 validates estimator type inside the wrapper and raises
    ``ValueError: Got a regressor with response_method=predict_proba`` for our
    ensemble classes.  We bypass this entirely by:

    1. Reading ``scorer._metric_name`` (attached by ``make_scorer``) to decide
       whether to call ``predict_proba`` or ``predict``.
    2. Calling the underlying ``scorer._score_func`` directly with ``scorer._sign``
       for the negation convention.  If those internals are unavailable (very old
       sklearn) we fall back to explicit metric functions.
    """
    _PROBA_METRICS = {
        "neg_log_loss", "log_loss",
        "neg_brier_score", "brier_score",
        "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
        "roc_auc_ovr_weighted", "roc_auc_ovo_weighted",
        "average_precision",
    }

    metric_name = getattr(scorer, "_metric_name", "")
    needs_proba = (
        metric_name in _PROBA_METRICS
        or any(k in metric_name for k in ("log_loss", "brier", "roc_auc", "average_precision"))
    )

    y_pred     = model.predict_proba(X) if needs_proba else model.predict(X)
    sign       = getattr(scorer, "_sign", 1)
    score_func = getattr(scorer, "_score_func", None)
    kwargs     = getattr(scorer, "_kwargs", {})

    if score_func is not None:
        return sign * score_func(y, y_pred, **kwargs)

    # Fallback for very old sklearn where internal attrs are absent
    if needs_proba:
        from sklearn.metrics import log_loss
        return -log_loss(y, y_pred)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, y_pred)


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

def _catboost_search_space(trial: optuna.Trial, cat_features=None) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "auto_class_weights": trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced", None]
        ),
        "verbose": 0,
        "cat_features": cat_features,
        "random_seed": RANDOM_STATE,
    }


def _lightgbm_search_space(trial: optuna.Trial, **_) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }


def _xgboost_search_space(trial: optuna.Trial, **_) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }


def _tabpfn_search_space(trial: optuna.Trial, **_) -> dict:
    # TabPFN has minimal hyperparameters exposed via the client
    return {
        "N_ensemble_configurations": trial.suggest_int("N_ensemble_configurations", 4, 32),
    }


# Map algorithm name → (search space fn, import fn, early-stopping kwargs builder)
_ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "catboost": {
        "search_space": _catboost_search_space,
        "import": _import_catboost,
        "early_stop": lambda params: {
            "eval_set": None,       # filled at fit-time
            "early_stopping_rounds": 50,
            "verbose": 0,
        },
        "supports_early_stop": True,
    },
    "lightgbm": {
        "search_space": _lightgbm_search_space,
        "import": _import_lightgbm,
        "early_stop": lambda params: {
            "callbacks": None,      # filled at fit-time via lgb callbacks
        },
        "supports_early_stop": True,
    },
    "xgboost": {
        "search_space": _xgboost_search_space,
        "import": _import_xgboost,
        "early_stop": lambda params: {
            "early_stopping_rounds": 50,
            "verbose": False,
        },
        "supports_early_stop": True,
    },
    "tabpfn": {
        "search_space": _tabpfn_search_space,
        "import": _import_tabpfn,
        "early_stop": lambda params: {},
        "supports_early_stop": False,
    },
}


# ---------------------------------------------------------------------------
# Per-model fit helpers (handle early-stopping nuances)
# ---------------------------------------------------------------------------

def _fit_catboost(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)


def _fit_lightgbm(model, X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)


def _fit_xgboost(model, X_tr, y_tr, X_val, y_val):
    # XGBoost < 2.0 : early_stopping_rounds lives in .fit()
    # XGBoost ≥ 2.0 : it was moved to the constructor; passing it to .fit()
    #                  raises TypeError.  We detect the version and branch.
    import xgboost as xgb
    xgb_major = int(xgb.__version__.split(".")[0])
    if xgb_major >= 2:
        model.set_params(early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )


def _fit_tabpfn(model, X_tr, y_tr, X_val=None, y_val=None):
    model.fit(X_tr, y_tr)


_FIT_DISPATCH = {
    "catboost": _fit_catboost,
    "lightgbm": _fit_lightgbm,
    "xgboost": _fit_xgboost,
    "tabpfn": _fit_tabpfn,
}


# ---------------------------------------------------------------------------
# Optuna objective builder (generic across all models)
# ---------------------------------------------------------------------------

def _build_objective(
    algorithm: str,
    X_pd: "pd.DataFrame",
    y_np: np.ndarray,
    splitter,
    scorer,
    cat_features=None,
):
    """Return a closure that Optuna can call as an objective function.

    We slice *X_pd* (a pandas DataFrame) rather than a raw numpy array so that
    every fold model is trained with the real feature names.  Without this,
    sklearn/LightGBM/XGBoost record generic names like 'Column_0' during HPO
    and then raise a feature-name mismatch at inference time when the user
    passes a named DataFrame.
    """
    registry = _ALGORITHM_REGISTRY[algorithm]
    ModelClass = registry["import"]()
    fit_fn = _FIT_DISPATCH[algorithm]
    supports_es = registry["supports_early_stop"]
    # numpy array used only for splitter.split() — it never sees the models
    X_np_for_split = X_pd.to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = registry["search_space"](trial, cat_features=cat_features)
        fold_scores: List[float] = []

        for train_idx, val_idx in splitter.split(X_np_for_split, y_np):
            # Slice as DataFrame → feature names travel with the data
            X_tr = X_pd.iloc[train_idx]
            X_val = X_pd.iloc[val_idx]
            y_tr, y_val = y_np[train_idx], y_np[val_idx]

            model = ModelClass(**params)

            if supports_es:
                fit_fn(model, X_tr, y_tr, X_val, y_val)
            else:
                fit_fn(model, X_tr, y_tr)

            fold_scores.append(evaluate_scorer(scorer, model, X_val, y_val))

        return float(np.mean(fold_scores))

    return objective


# ---------------------------------------------------------------------------
# Phase 1 – Hyperparameter optimisation
# ---------------------------------------------------------------------------

def optimise_model(
    algorithm: str,
    X: pl.DataFrame,
    y: pl.Series,
    split_type: str,
    metric: str,
    n_trials: int = 50,
    cat_features: Optional[List[str]] = None,
) -> Tuple[Any, dict]:
    """
    Run Optuna HPO for *algorithm* and return (fitted_model, best_params).

    The model is **retrained on the full dataset** using the best parameters
    found during cross-validation (Phase 2 within this function).
    """
    X_pd = _polars_to_pandas(X)
    y_pd = y.to_pandas()
    X_np = X_pd.to_numpy()
    y_np = y_pd.to_numpy()

    n_classes = len(np.unique(y_np))
    splitter = make_splitter(split_type, n_classes=n_classes)
    scorer = make_scorer(metric)

    objective = _build_objective(
        algorithm, X_pd, y_np, splitter, scorer, cat_features=cat_features
    )

    study = optuna.create_study(direction=_optuna_direction(metric))
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        failed_msgs = []
        for t in study.trials[:3]:          # show first 3 failure reasons
            if t.state == optuna.trial.TrialState.FAIL:
                failed_msgs.append(str(t.value))
        raise RuntimeError(
            f"[{algorithm}] All {n_trials} Optuna trials failed — no best params available.\n"
            f"First failure hints: {failed_msgs}\n"
            "Check that your data has no unsupported dtypes and that the model "
            "library version is compatible."
        )

    best_params = study.best_params
    print(f"[{algorithm}] Best CV {metric}: {study.best_value:.4f}")
    print(f"[{algorithm}] Best params: {best_params}")

    # --- Phase 2: retrain on full data ---
    registry = _ALGORITHM_REGISTRY[algorithm]
    ModelClass = registry["import"]()

    # Reconstruct full params (search space fn may add non-trial keys)
    # We replay via a FrozenTrial stub; easier to just pass best_params directly
    # but some keys (cat_features, verbose, …) were added by the search space fn.
    # Safe approach: call search_space with a FrozenParamsTrial shim.
    class _FixedTrial:
        """Minimal optuna Trial shim that replays best_params."""
        def __init__(self, params):
            self._params = params

        def suggest_int(self, name, *a, **kw):       return self._params.get(name, a[0])
        def suggest_float(self, name, *a, **kw):     return self._params.get(name, a[0])
        def suggest_categorical(self, name, choices): return self._params.get(name, choices[0])

    full_params = registry["search_space"](_FixedTrial(best_params), cat_features=cat_features)

    final_model = ModelClass(**full_params)
    fit_fn = _FIT_DISPATCH[algorithm]
    # No validation set for final fit
    if algorithm in ("catboost",):
        final_model.fit(X_pd, y_pd)
    elif algorithm in ("lightgbm",):
        final_model.fit(X_pd, y_pd)
    elif algorithm in ("xgboost",):
        final_model.fit(X_pd, y_pd)
    else:
        fit_fn(final_model, X_np, y_np)

    return final_model, best_params


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------

class EnsembleBase(BaseEstimator, ClassifierMixin):
    """Abstract base for all ensemble wrappers."""

    # Explicitly declare classifier type so sklearn scorers (neg_log_loss,
    # brier_score, roc_auc, …) can call predict_proba without raising
    # "Got a regressor" errors in sklearn ≥ 1.4.
    _estimator_type = "classifier"

    def fit(self, X, y) -> "EnsembleBase":
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def _collect_probas(self, X) -> np.ndarray:
        """Shape: (n_samples, n_classes, n_models)."""
        probas = [m.predict_proba(X) for m in self.base_models_]
        return np.stack(probas, axis=-1)

    def _collect_preds(self, X) -> np.ndarray:
        """Shape: (n_samples, n_models)."""
        return np.stack([m.predict(X) for m in self.base_models_], axis=1)


# ------------------------------------------------------------------
# 1. Simple averaging
# ------------------------------------------------------------------

class SimpleAveragingEnsemble(EnsembleBase):
    """Average predicted probabilities from all base models equally."""

    def __init__(self, base_models: List[Any]):
        self.base_models = base_models
        self.base_models_ = base_models   # already fitted

    def fit(self, X, y) -> "SimpleAveragingEnsemble":
        # Base models already fitted; nothing to do.
        return self

    def predict_proba(self, X) -> np.ndarray:
        probas = self._collect_probas(X)          # (n, c, k)
        return probas.mean(axis=-1)               # (n, c)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ------------------------------------------------------------------
# 2. Weighted averaging (weights optimised via CV)
# ------------------------------------------------------------------

class WeightedAveragingEnsemble(EnsembleBase):
    """
    Weighted average of base model probabilities.

    Weights are found by minimising negative-metric on held-out OOF predictions
    using scipy.optimize. The final ensemble is always retrained on full data.
    """

    def __init__(
        self,
        base_models: List[Any],
        scorer,
        weights: Optional[np.ndarray] = None,
    ):
        self.base_models = base_models
        self.base_models_ = base_models
        self.scorer = scorer
        self.weights_ = weights

    def fit(self, X, y) -> "WeightedAveragingEnsemble":
        if self.weights_ is None:
            self.weights_ = optimise_weights(
                self.base_models_, X, y, self.scorer
            )
        return self

    def predict_proba(self, X) -> np.ndarray:
        probas = self._collect_probas(X)                       # (n, c, k)
        w = np.array(self.weights_)
        w = w / w.sum()
        return (probas * w[np.newaxis, np.newaxis, :]).sum(axis=-1)  # (n, c)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def optimise_weights(
    base_models: List[Any],
    X: np.ndarray,
    y: np.ndarray,
    scorer,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Use scipy L-BFGS-B to find per-model weights that maximise CV metric.
    Returns a 1-D weight array (one weight per model, sums to 1).
    """
    k = len(base_models)
    oof_probas = np.stack(
        [m.predict_proba(X) for m in base_models], axis=-1
    )  # (n, c, k)

    def neg_metric(raw_w):
        w = np.exp(raw_w) / np.exp(raw_w).sum()   # softmax → valid simplex
        blended = (oof_probas * w[np.newaxis, np.newaxis, :]).sum(axis=-1)
        preds = blended.argmax(axis=1)
        # Wrap predictions in a dummy estimator for scorer compatibility
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier().fit(X[:1], y[:1])
        dummy.predict_proba = lambda _X: blended
        dummy.predict = lambda _X: preds
        try:
            return -scorer(dummy, X, y)
        except Exception:
            from sklearn.metrics import accuracy_score
            return -accuracy_score(y, preds)

    x0 = np.zeros(k)
    result = minimize(neg_metric, x0, method="L-BFGS-B")
    raw_best = result.x
    weights = np.exp(raw_best) / np.exp(raw_best).sum()
    print(f"[WeightedEnsemble] Optimised weights: {np.round(weights, 4)}")
    return weights


# ------------------------------------------------------------------
# 3. Stacking
# ------------------------------------------------------------------

class StackingEnsemble(EnsembleBase):
    """
    Two-layer stacking.

    Base models generate out-of-fold (OOF) probability predictions that are
    used as features for a meta-model. Optionally uses full OOF predictions.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Any,
        splitter,
        use_oof: bool = True,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.splitter = splitter
        self.use_oof = use_oof
        self.base_models_: List[Any] = []
        self.meta_model_: Any = None

    def fit(self, X, y) -> "StackingEnsemble":
        """
        X may be a pandas DataFrame or a numpy array.
        We normalise to DataFrame so that fold models and the final retrained
        models share the same feature names — preventing mismatch at inference.
        """
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        n_models = len(self.base_models)

        oof_features = np.zeros((n_samples, n_classes * n_models))
        X_np_for_split = X.to_numpy()

        self.base_models_ = []

        for idx, model in enumerate(self.base_models):
            oof_pred = np.zeros((n_samples, n_classes))

            for train_idx, val_idx in self.splitter.split(X_np_for_split, y):
                X_tr  = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_tr  = y[train_idx]

                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)
                oof_pred[val_idx] = fold_model.predict_proba(X_val)

            col_start = idx * n_classes
            oof_features[:, col_start : col_start + n_classes] = oof_pred

            # Retrain on full data for inference
            full_model = clone(model)
            full_model.fit(X, y)
            self.base_models_.append(full_model)

        # Train meta-model on OOF features
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(oof_features, y)
        return self

    def _make_meta_features(self, X) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.base_models_]
        return np.hstack(probas)   # (n, n_classes * n_models)

    def predict_proba(self, X) -> np.ndarray:
        meta_features = self._make_meta_features(X)
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ------------------------------------------------------------------
# 4. Blending
# ------------------------------------------------------------------

class BlendingEnsemble(EnsembleBase):
    """
    Holdout-based blending.

    A fixed fraction of training data is held out to train the meta-model,
    while the rest trains the base models.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Any,
        holdout_frac: float = 0.2,
        random_state: int = RANDOM_STATE,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.holdout_frac = holdout_frac
        self.random_state = random_state
        self.base_models_: List[Any] = []
        self.meta_model_: Any = None

    def fit(self, X, y) -> "BlendingEnsemble":
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        holdout_idx = rng.choice(n, size=int(n * self.holdout_frac), replace=False)
        train_idx = np.setdiff1d(np.arange(n), holdout_idx)

        X_base = X.iloc[train_idx];  y_base = y[train_idx]
        X_hold = X.iloc[holdout_idx]; y_hold = y[holdout_idx]

        self.base_models_ = []
        blend_features = []

        for model in self.base_models:
            m = clone(model)
            m.fit(X_base, y_base)
            self.base_models_.append(m)
            blend_features.append(m.predict_proba(X_hold))

        blend_X = np.hstack(blend_features)
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(blend_X, y_hold)
        return self

    def _make_blend_features(self, X) -> np.ndarray:
        return np.hstack([m.predict_proba(X) for m in self.base_models_])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_model_.predict_proba(self._make_blend_features(X))

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ------------------------------------------------------------------
# 5. Voting classifier
# ------------------------------------------------------------------

class VotingEnsemble(EnsembleBase):
    """
    Hard or soft majority voting over base model predictions.
    """

    def __init__(self, base_models: List[Any], voting: str = "soft"):
        assert voting in ("soft", "hard"), "voting must be 'soft' or 'hard'"
        self.base_models = base_models
        self.base_models_ = base_models
        self.voting = voting

    def fit(self, X, y) -> "VotingEnsemble":
        return self  # Base models already fitted

    def predict_proba(self, X) -> np.ndarray:
        if self.voting == "hard":
            raise AttributeError("predict_proba not available for hard voting.")
        probas = self._collect_probas(X)   # (n, c, k)
        return probas.mean(axis=-1)

    def predict(self, X) -> np.ndarray:
        if self.voting == "soft":
            return self.predict_proba(X).argmax(axis=1)
        # Hard voting: majority class
        preds = self._collect_preds(X)     # (n, k)
        from scipy import stats
        result, _ = stats.mode(preds, axis=1)
        return result.ravel()


# ---------------------------------------------------------------------------
# Ensemble registry
# ---------------------------------------------------------------------------

_ENSEMBLE_REGISTRY: Dict[str, Any] = {
    "simple_averaging": SimpleAveragingEnsemble,
    "weighted_averaging": WeightedAveragingEnsemble,
    "stacking": StackingEnsemble,
    "blending": BlendingEnsemble,
    "voting": VotingEnsemble,
}


def list_ensemble_methods() -> List[str]:
    return list(_ENSEMBLE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Meta-model factory (for stacking / blending)
# ---------------------------------------------------------------------------

def _build_meta_model(name: str):
    """Build a lightweight meta-model by name."""
    name = name.lower()
    if name == "lightgbm":
        LGBMClassifier = _import_lightgbm()
        return LGBMClassifier(n_estimators=100, verbose=-1, random_state=RANDOM_STATE)
    elif name == "xgboost":
        XGBClassifier = _import_xgboost()
        return XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    elif name == "catboost":
        CatBoostClassifier = _import_catboost()
        return CatBoostClassifier(iterations=100, verbose=0, random_seed=RANDOM_STATE)
    elif name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError(
            f"Unknown meta_model '{name}'. Choose from: lightgbm, xgboost, catboost, logistic_regression"
        )


# ---------------------------------------------------------------------------
# Ensemble cross-validation
# ---------------------------------------------------------------------------

def evaluate_ensemble_cv(
    ensemble: EnsembleBase,
    X_pd: "pd.DataFrame",
    y_np: np.ndarray,
    split_type: str,
    scorer,
    stacking_options: dict,
    n_splits: int = N_SPLITS,
) -> float:
    """
    Return a cross-validated score for the fitted *ensemble* without leaking
    any information from the final fit.

    Strategy
    --------
    For each CV fold we build a **fresh clone** of the same ensemble type,
    train it on the fold's training split, and score it on the held-out split.
    This mirrors exactly what an end-user cares about: how well the ensemble
    generalises to unseen data.

    Why not just call sklearn's cross_val_score on the ensemble object?
    sklearn's cross_val_score calls clone() internally which works for simple
    sklearn estimators, but our ensemble classes hold pre-fitted base models as
    constructor arguments — clone() would copy the *unfitted* class with no base
    models.  Instead we re-run build_ensemble per fold using the already-fitted
    base models as a template (re-cloning them inside each fold).
    """
    import pandas as pd

    n_classes = len(np.unique(y_np))
    splitter = make_splitter(split_type, n_splits=n_splits, n_classes=n_classes)
    X_np_for_split = X_pd.to_numpy()

    # Determine which ensemble method and options to use
    method = stacking_options.get("method", "simple_averaging")

    fold_scores: List[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X_np_for_split, y_np)):
        X_tr = X_pd.iloc[train_idx]
        X_val = X_pd.iloc[val_idx]
        y_tr = y_np[train_idx]
        y_val = y_np[val_idx]

        # Clone each base model and refit on this fold's training data
        fold_base_models = []
        for base_model in ensemble.base_models:
            m = clone(base_model)
            m.fit(X_tr, y_tr)
            fold_base_models.append(m)

        # Build a fresh ensemble using the fold-fitted base models
        fold_splitter = make_splitter(split_type, n_splits=n_splits, n_classes=n_classes)
        fold_ensemble = _instantiate_ensemble(
            method=method,
            base_models=fold_base_models,
            stacking_options=stacking_options,
            splitter=fold_splitter,
            scorer=scorer,
        )
        fold_ensemble.fit(X_tr, y_tr)

        fold_scores.append(evaluate_scorer(scorer, fold_ensemble, X_val, y_val))

    return float(np.mean(fold_scores))


def _instantiate_ensemble(
    method: str,
    base_models: List[Any],
    stacking_options: dict,
    splitter,
    scorer,
) -> EnsembleBase:
    """Create an unfitted ensemble instance (no .fit call)."""
    method = method.lower()
    if method == "simple_averaging":
        return SimpleAveragingEnsemble(base_models)
    elif method == "weighted_averaging":
        return WeightedAveragingEnsemble(base_models, scorer)
    elif method == "stacking":
        meta_name = stacking_options.get("meta_model", "logistic_regression")
        use_oof = stacking_options.get("use_oof", True)
        return StackingEnsemble(base_models, _build_meta_model(meta_name), splitter, use_oof=use_oof)
    elif method == "blending":
        meta_name = stacking_options.get("meta_model", "logistic_regression")
        holdout_frac = stacking_options.get("holdout_frac", 0.2)
        return BlendingEnsemble(base_models, _build_meta_model(meta_name), holdout_frac=holdout_frac)
    elif method == "voting":
        return VotingEnsemble(base_models, voting=stacking_options.get("voting_type", "soft"))
    else:
        raise ValueError(f"Unknown ensemble method: {method}")




def build_ensemble(
    method: str,
    base_models: List[Any],
    X_pd: "pd.DataFrame",
    y_np: np.ndarray,
    stacking_options: dict,
    scorer,
    split_type: str,
) -> EnsembleBase:
    """
    Instantiate and fit the requested ensemble on the full training data.

    Accepts *X_pd* as a pandas DataFrame so that column names are preserved
    when ensemble methods refit base models internally (stacking, blending).
    """
    method = method.lower()

    if method not in _ENSEMBLE_REGISTRY:
        raise ValueError(
            f"Unknown ensemble method '{method}'. "
            f"Available: {list_ensemble_methods()}"
        )

    n_classes = len(np.unique(y_np))
    splitter = make_splitter(split_type, n_classes=n_classes)

    if method == "simple_averaging":
        ensemble = SimpleAveragingEnsemble(base_models)

    elif method == "weighted_averaging":
        ensemble = WeightedAveragingEnsemble(base_models, scorer)

    elif method == "stacking":
        meta_name = stacking_options.get("meta_model", "logistic_regression")
        use_oof = stacking_options.get("use_oof", True)
        meta_model = _build_meta_model(meta_name)
        ensemble = StackingEnsemble(base_models, meta_model, splitter, use_oof=use_oof)

    elif method == "blending":
        meta_name = stacking_options.get("meta_model", "logistic_regression")
        holdout_frac = stacking_options.get("holdout_frac", 0.2)
        meta_model = _build_meta_model(meta_name)
        ensemble = BlendingEnsemble(base_models, meta_model, holdout_frac=holdout_frac)

    elif method == "voting":
        voting_type = stacking_options.get("voting_type", "soft")
        ensemble = VotingEnsemble(base_models, voting=voting_type)

    else:
        raise ValueError(f"Unhandled ensemble method: {method}")

    print(f"[Ensemble] Fitting '{method}' ensemble on full training data …")
    ensemble.fit(X_pd, y_np)
    return ensemble


# ---------------------------------------------------------------------------
# Public sub-step: HPO + base model training (separated from ensemble build)
# ---------------------------------------------------------------------------

def train_base_models(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    split_type: str,
    algorithms: List[str],
    metric: str,
    n_trials: int = 50,
    cat_features: Optional[List[str]] = None,
) -> dict:
    """
    Run hyperparameter optimisation and retrain each base model on the full
    dataset.  Returns a dict that can be passed directly to
    ``build_ensemble_from_base_models``.

    Use this when you want to try **multiple ensemble configs** on the same
    training window without repeating expensive HPO for every config:

    .. code-block:: python

        base = ensemble_pipeline.train_base_models(
            X_train=X_train, y_train=y_train,
            split_type="temporal", algorithms=["catboost", "lightgbm", "xgboost"],
            metric="accuracy", n_trials=50,
        )
        for config in configs:
            result = ensemble_pipeline.build_ensemble_from_base_models(
                base_models_result=base,
                X_train=X_train, y_train=y_train,
                split_type="temporal", metric="accuracy",
                stacking_options=config,
            )

    Returns
    -------
    dict with keys:
      - base_models   : dict {algorithm: fitted_model}
      - best_params   : dict {algorithm: best_params_dict}
      - X_pd          : dtype-safe pandas DataFrame (for reuse)
      - y_np          : numpy label array (for reuse)
      - scorer        : the sklearn scorer used
      - metric        : the metric name string
      - split_type    : the split type string (for ensemble CV)
    """
    unknown = set(algorithms) - set(_ALGORITHM_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Valid: {list(_ALGORITHM_REGISTRY)}")

    scorer = make_scorer(metric)
    X_pd = _polars_to_pandas(X_train)
    y_np = y_train.to_pandas().to_numpy()

    base_models_dict: Dict[str, Any] = {}
    best_params_dict: Dict[str, dict] = {}

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"[Pipeline] Optimising {algo.upper()} ({n_trials} trials) …")
        print(f"{'='*60}")
        model, params = optimise_model(
            algorithm=algo,
            X=X_train,
            y=y_train,
            split_type=split_type,
            metric=metric,
            n_trials=n_trials,
            cat_features=cat_features,
        )
        base_models_dict[algo] = model
        best_params_dict[algo] = params

    return {
        "base_models": base_models_dict,
        "best_params": best_params_dict,
        "X_pd": X_pd,
        "y_np": y_np,
        "scorer": scorer,
        "metric": metric,
        "split_type": split_type,
    }


def build_ensemble_from_base_models(
    base_models_result: dict,
    stacking_options: Optional[dict] = None,
) -> dict:
    """
    Build and evaluate an ensemble from already-trained base models.

    Accepts the dict returned by ``train_base_models`` so HPO never runs twice
    for the same training window.

    Parameters
    ----------
    base_models_result : dict
        The dict returned by ``train_base_models``.
    stacking_options : dict, optional
        Ensemble configuration (same keys as in ``run_pipeline``).

    Returns
    -------
    dict with keys:
      - ensemble      : fitted ensemble model
      - base_models   : dict {algorithm: fitted_model}
      - best_params   : dict {algorithm: best_params_dict}
      - scorer        : the sklearn scorer used
      - cv_score      : cross-validated ensemble score
    """
    if stacking_options is None:
        stacking_options = {}

    X_pd       = base_models_result["X_pd"]
    y_np       = base_models_result["y_np"]
    scorer     = base_models_result["scorer"]
    metric     = base_models_result["metric"]
    split_type = base_models_result["split_type"]

    base_models_list = list(base_models_result["base_models"].values())
    ensemble_method  = stacking_options.get("method", "simple_averaging")

    print(f"\n{'='*60}")
    print(f"[Pipeline] Building '{ensemble_method}' ensemble …")
    print(f"{'='*60}")

    ensemble = build_ensemble(
        method=ensemble_method,
        base_models=base_models_list,
        X_pd=X_pd,
        y_np=y_np,
        stacking_options=stacking_options,
        scorer=scorer,
        split_type=split_type,
    )

    print(f"\n{'='*60}")
    print(f"[Pipeline] Cross-validating ensemble ({metric}) …")
    print(f"{'='*60}")

    cv_score = evaluate_ensemble_cv(
        ensemble=ensemble,
        X_pd=X_pd,
        y_np=y_np,
        split_type=split_type,
        scorer=scorer,
        stacking_options=stacking_options,
    )

    print(f"[Ensemble] CV {metric}: {cv_score:.4f}")
    return {
        "ensemble": ensemble,
        "base_models": base_models_result["base_models"],
        "best_params": base_models_result["best_params"],
        "scorer": scorer,
        "cv_score": cv_score,
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    split_type: str,
    algorithms: List[str],
    metric: str,
    stacking_options: Optional[dict] = None,
    n_trials: int = 50,
    cat_features: Optional[List[str]] = None,
) -> dict:
    """
    End-to-end ML pipeline: HPO → base model training → ensemble → CV score.

    For running multiple ensemble configs on the same training window without
    repeating HPO, use ``train_base_models`` + ``build_ensemble_from_base_models``
    directly instead of calling this function in a loop.

    Parameters
    ----------
    X_train : pl.DataFrame
    y_train : pl.Series
    split_type : str
        'temporal' or 'random'.
    algorithms : list of str
        Any subset of: 'catboost', 'lightgbm', 'xgboost', 'tabpfn'.
    metric : str
        Any valid sklearn scorer name (e.g. 'accuracy', 'f1_weighted').
    stacking_options : dict, optional
        - method       : ensemble method (default 'simple_averaging')
        - meta_model   : meta-learner for stacking/blending
        - use_oof      : bool (default True)
        - holdout_frac : float (default 0.2)
        - voting_type  : 'soft' | 'hard' (default 'soft')
        - n_trials     : overrides top-level n_trials
    n_trials : int
        Optuna trials per model (default 50).
    cat_features : list of str, optional
        Categorical column names (CatBoost only).

    Returns
    -------
    dict with keys: ensemble, base_models, best_params, scorer, cv_score
    """
    if stacking_options is None:
        stacking_options = {}

    _n_trials = stacking_options.get("n_trials", n_trials)

    base_result = train_base_models(
        X_train=X_train,
        y_train=y_train,
        split_type=split_type,
        algorithms=algorithms,
        metric=metric,
        n_trials=_n_trials,
        cat_features=cat_features,
    )

    return build_ensemble_from_base_models(
        base_models_result=base_result,
        stacking_options=stacking_options,
    )


# ---------------------------------------------------------------------------
# Quick self-test (run: python ensemble_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Running self-test with synthetic data …")
    X_raw, y_raw = make_classification(
        n_samples=300, n_features=10, n_informative=6,
        n_classes=2, random_state=42
    )
    X_pl = pl.DataFrame(X_raw, schema=[f"f{i}" for i in range(X_raw.shape[1])])
    y_pl = pl.Series("target", y_raw)

    result = run_pipeline(
        X_train=X_pl,
        y_train=y_pl,
        split_type="random",
        algorithms=["lightgbm", "xgboost"],
        metric="accuracy",
        stacking_options={
            "method": "weighted_averaging",
            "n_trials": 10,
        },
    )
    ensemble = result["ensemble"]
    X_test_np = X_raw[:10]
    preds = ensemble.predict(X_test_np)
    print(f"Sample predictions: {preds}")
    print("Self-test complete.")