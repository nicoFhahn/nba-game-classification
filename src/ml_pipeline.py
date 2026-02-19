"""
Comprehensive ML Pipeline with Optuna Optimization
- Train & Tune: ExtraTrees, RandomForest, XGBoost, LightGBM, CatBoost, HistGradientBoosting
- Build weighted ensemble with optimized weights
- Optimize decision threshold
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, 
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    log_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import warnings
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: OPTUNA OBJECTIVE FUNCTIONS FOR EACH MODEL
# ============================================================================

class ModelTuner:
    """Unified class for tuning all models with Optuna"""

    def __init__(self, X_train, y_train, cv_folds=5, random_state=42, max_estimators=1000, sample_weights=None):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.max_estimators = max_estimators
        self.sample_weights = sample_weights
        self.best_models = {}
        self.best_params = {}
        self.studies = {}

    def _cross_val_score_weighted(self, model):
        """
        Perform cross-validation with optional sample weights

        Returns:
        --------
        float : Mean ROC-AUC score across folds
        """
        scores = []

        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]

            # Train with weights if provided
            if self.sample_weights is not None:
                fold_weights = self.sample_weights[train_idx]
                model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
            else:
                model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, y_pred_proba)
            scores.append(score)

        return np.mean(scores)

    def save_checkpoint(self, output_dir):
        """Save all models, parameters, and studies to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)
        for name, model in self.best_models.items():
            with open(models_dir / f'{name}.pkl', 'wb') as f:
                pickle.dump(model, f)

        # Save best parameters
        with open(output_path / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        # Save studies (Optuna optimization history)
        studies_dir = output_path / 'studies'
        studies_dir.mkdir(exist_ok=True)
        for name, study in self.studies.items():
            with open(studies_dir / f'{name}_study.pkl', 'wb') as f:
                pickle.dump(study, f)

        # Save metadata
        metadata = {
            'random_state': self.random_state,
            'max_estimators': self.max_estimators,
            'cv_folds': self.cv.n_splits,
            'timestamp': datetime.now().isoformat(),
            'trained_models': list(self.best_models.keys())
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Checkpoint saved to: {output_path}")
        print(f"  - Models saved: {len(self.best_models)}")
        print(f"  - Studies saved: {len(self.studies)}")

    @classmethod
    def load_checkpoint(cls, output_dir, X_train, y_train):
        """Load models, parameters, and studies from disk to continue training"""
        output_path = Path(output_dir)

        if not output_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {output_dir}")

        # Load metadata
        with open(output_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create instance with saved settings
        instance = cls(
            X_train, y_train,
            cv_folds=metadata['cv_folds'],
            random_state=metadata['random_state'],
            max_estimators=metadata['max_estimators']
        )

        # Load best parameters
        if (output_path / 'best_params.json').exists():
            with open(output_path / 'best_params.json', 'r') as f:
                instance.best_params = json.load(f)

        # Load models
        models_dir = output_path / 'models'
        if models_dir.exists():
            for model_file in models_dir.glob('*.pkl'):
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    instance.best_models[model_name] = pickle.load(f)

        # Load studies
        studies_dir = output_path / 'studies'
        if studies_dir.exists():
            for study_file in studies_dir.glob('*_study.pkl'):
                model_name = study_file.stem.replace('_study', '')
                with open(study_file, 'rb') as f:
                    instance.studies[model_name] = pickle.load(f)

        print(f"\n✓ Checkpoint loaded from: {output_path}")
        print(f"  - Models loaded: {len(instance.best_models)}")
        print(f"  - Studies loaded: {len(instance.studies)}")
        print(f"  - Timestamp: {metadata['timestamp']}")

        return instance

    def objective_extratrees(self, trial):
        """Optuna objective for ExtraTrees"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, self.max_estimators, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }

        model = ExtraTreesClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_randomforest(self, trial):
        """Optuna objective for RandomForest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, self.max_estimators, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_xgboost(self, trial):
        """Optuna objective for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, self.max_estimators, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }

        model = xgb.XGBClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_lightgbm(self, trial):
        """Optuna objective for LightGBM"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, self.max_estimators, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.random_state,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_catboost(self, trial):
        """Optuna objective for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, self.max_estimators, step=100),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_state': self.random_state,
            'verbose': False
        }

        model = cb.CatBoostClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_histgradient(self, trial):
        """Optuna objective for HistGradientBoosting"""
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, self.max_estimators, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'max_bins': trial.suggest_int('max_bins', 128, 255),
            'random_state': self.random_state
        }

        model = HistGradientBoostingClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_logistic(self, trial):
        """Optuna objective for Logistic Regression"""
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
            'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
            'solver': 'saga',  # saga supports all penalties
            'max_iter': 1000,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # ElasticNet requires l1_ratio
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        # None penalty doesn't support saga, use lbfgs
        if params['penalty'] is None:
            params['solver'] = 'lbfgs'

        model = LogisticRegression(**params)
        return self._cross_val_score_weighted(model)

    def objective_sgd(self, trial):
        """Optuna objective for SGD Classifier"""
        params = {
            'loss': trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber', 'perceptron']),
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'max_iter': 1000,
            'tol': 1e-3,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # ElasticNet requires l1_ratio
        if params['penalty'] == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        model = SGDClassifier(**params)
        return self._cross_val_score_weighted(model)

    def objective_gaussiannb(self, trial):
        """Optuna objective for Gaussian Naive Bayes"""
        params = {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-11, 1e-5, log=True)
        }

        model = GaussianNB(**params)
        return self._cross_val_score_weighted(model)

    def objective_bernoullinb(self, trial):
        """Optuna objective for Bernoulli Naive Bayes"""
        params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'binarize': trial.suggest_float('binarize', 0.0, 1.0),
            'fit_prior': trial.suggest_categorical('fit_prior', [True, False])
        }

        model = BernoulliNB(**params)
        return self._cross_val_score_weighted(model)

    def tune_model(self, model_name, n_trials=100, n_jobs=1, optuna_verbosity=1):
        """Tune a specific model using Optuna

        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        n_trials : int
            Number of Optuna trials
        n_jobs : int
            Number of parallel jobs for Optuna. Use -1 for all CPUs.
        optuna_verbosity : int
            0=no output, 1=progress bar only, 2=info, 3=debug
        """
        print(f"\n{'='*70}")
        print(f"Tuning {model_name}...")
        print(f"{'='*70}")

        objectives = {
            'ExtraTrees': self.objective_extratrees,
            'RandomForest': self.objective_randomforest,
            'XGBoost': self.objective_xgboost,
            'LightGBM': self.objective_lightgbm,
            'CatBoost': self.objective_catboost,
            'HistGradientBoosting': self.objective_histgradient,
            'LogisticRegression': self.objective_logistic,
            'SGDClassifier': self.objective_sgd,
            'GaussianNB': self.objective_gaussiannb,
            'BernoulliNB': self.objective_bernoullinb
        }

        # Check if we have an existing study to continue from
        if model_name in self.studies:
            study = self.studies[model_name]
            current_trials = len(study.trials)
            print(f"Continuing from existing study with {current_trials} trials...")
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state),
                study_name=f'{model_name}_optimization'
            )

        show_progress = optuna_verbosity >= 1
        study.optimize(objectives[model_name], n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress)

        print(f"\nBest {model_name} CV ROC-AUC: {study.best_value:.4f}")
        print(f"Total trials completed: {len(study.trials)}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Train final model with best parameters
        best_params = study.best_params
        self.best_params[model_name] = best_params

        if model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(**best_params, random_state=self.random_state, tree_method='hist')
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(**best_params, random_state=self.random_state, verbose=-1)
        elif model_name == 'CatBoost':
            model = cb.CatBoostClassifier(**best_params, random_state=self.random_state, verbose=False)
        elif model_name == 'HistGradientBoosting':
            model = HistGradientBoostingClassifier(**best_params, random_state=self.random_state)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'SGDClassifier':
            model = SGDClassifier(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'GaussianNB':
            model = GaussianNB(**best_params)
        elif model_name == 'BernoulliNB':
            model = BernoulliNB(**best_params)

        # Train with sample weights if provided
        if self.sample_weights is not None:
            model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        else:
            model.fit(self.X_train, self.y_train)

        self.best_models[model_name] = model
        self.studies[model_name] = study

        return study

    def tune_all_models(self, n_trials=100, n_jobs=1, optuna_verbosity=1):
        """Tune all models

        Parameters:
        -----------
        n_trials : int
            Number of Optuna trials per model
        n_jobs : int
            Number of parallel jobs for Optuna. Use -1 for all CPUs.
        optuna_verbosity : int
            0=no output, 1=progress bar only, 2=info, 3=debug
        """
        models = ['ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting',
                  'LogisticRegression', 'SGDClassifier', 'GaussianNB', 'BernoulliNB']
        studies = {}

        for model_name in models:
            studies[model_name] = self.tune_model(model_name, n_trials, n_jobs, optuna_verbosity)

        return studies


# ============================================================================
# PART 2: WEIGHTED ENSEMBLE WITH OPTIMIZED WEIGHTS
# ============================================================================

class WeightedEnsemble:
    """Weighted ensemble with Optuna-optimized weights"""

    def __init__(self, models_dict, X_val, y_val, random_state=42):
        self.models = models_dict
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.weights = None
        self.model_names = list(models_dict.keys())

        # Get predictions from all models
        self.predictions = {}
        for name, model in self.models.items():
            self.predictions[name] = model.predict_proba(X_val)[:, 1]

    def objective_weights(self, trial):
        """Optuna objective for ensemble weights"""
        # Suggest weights for each model
        weights = []
        for name in self.model_names:
            w = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            weights.append(w)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Calculate weighted ensemble predictions
        ensemble_pred = np.zeros(len(self.y_val))
        for i, name in enumerate(self.model_names):
            ensemble_pred += weights[i] * self.predictions[name]

        # Calculate ROC-AUC
        score = roc_auc_score(self.y_val, ensemble_pred)
        return score

    def optimize_weights(self, n_trials=200, optuna_verbosity=1):
        """Optimize ensemble weights using Optuna

        Parameters:
        -----------
        n_trials : int
            Number of Optuna trials
        optuna_verbosity : int
            0=no output, 1=progress bar only, 2=info, 3=debug
        """
        print(f"\n{'='*70}")
        print("Optimizing Ensemble Weights...")
        print(f"{'='*70}")

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            study_name='ensemble_weights_optimization'
        )

        show_progress = optuna_verbosity >= 1
        study.optimize(self.objective_weights, n_trials=n_trials, show_progress_bar=show_progress)

        # Extract and normalize best weights
        weights = []
        for name in self.model_names:
            weights.append(study.best_params[f'weight_{name}'])

        weights = np.array(weights)
        self.weights = weights / weights.sum()

        print(f"\nBest Ensemble ROC-AUC: {study.best_value:.4f}")
        print("\nOptimized Weights:")
        for name, weight in zip(self.model_names, self.weights):
            print(f"  {name}: {weight:.4f}")

        return study

    def predict_proba(self, X):
        """Predict probabilities using weighted ensemble"""
        if self.weights is None:
            raise ValueError("Weights not optimized yet. Call optimize_weights() first.")

        ensemble_pred = np.zeros(len(X))
        for i, name in enumerate(self.model_names):
            pred = self.models[name].predict_proba(X)[:, 1]
            ensemble_pred += self.weights[i] * pred

        return ensemble_pred

    def predict(self, X, threshold=0.5):
        """Predict classes using weighted ensemble"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# ============================================================================
# PART 3: THRESHOLD OPTIMIZATION
# ============================================================================

class ThresholdOptimizer:
    """Optimize decision threshold for best performance"""

    def __init__(self, y_true, y_proba):
        self.y_true = y_true
        self.y_proba = y_proba
        self.best_threshold = 0.5
        self.threshold_metrics = {}

    def optimize_threshold(self, metric='f1', beta=1.0):
        """
        Optimize threshold based on specified metric

        Parameters:
        -----------
        metric : str
            'f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy', 'youden', 'fbeta', or 'log_loss'
        beta : float
            Beta parameter for F-beta score (only used if metric='fbeta')
        """
        print(f"\n{'='*70}")
        print(f"Optimizing Threshold for {metric.upper()}...")
        print(f"{'='*70}")

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)

        # For log_loss, we want to minimize (so we'll negate it for maximization)
        if metric == 'log_loss':
            best_score = np.inf  # We want to minimize log_loss
        else:
            best_score = -np.inf

        best_threshold = 0.5

        # Test different thresholds
        threshold_range = np.linspace(0.01, 0.99, 99)
        scores = []

        for threshold in threshold_range:
            y_pred = (self.y_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(self.y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(self.y_true, y_pred)
            elif metric == 'fbeta':
                score = ((1 + beta**2) * precision_score(self.y_true, y_pred) * recall_score(self.y_true, y_pred)) / \
                        (beta**2 * precision_score(self.y_true, y_pred) + recall_score(self.y_true, y_pred))
            elif metric == 'precision':
                score = precision_score(self.y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(self.y_true, y_pred)
            elif metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                score = (sensitivity + specificity) / 2
            elif metric == 'youden':
                # Youden's J statistic = Sensitivity + Specificity - 1
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                score = sensitivity + specificity - 1
            elif metric == 'log_loss':
                # For log_loss, we use probabilities with the threshold to create binary predictions
                # Then calculate log_loss using the original probabilities
                score = log_loss(self.y_true, self.y_proba)
            else:
                raise ValueError(f"Unknown metric: {metric}. Supported metrics: 'f1', 'accuracy', 'precision', 'recall', 'balanced_accuracy', 'youden', 'fbeta', 'log_loss'")

            scores.append(score)

            # Update best score
            if metric == 'log_loss':
                if score < best_score:  # Minimize log_loss
                    best_score = score
                    best_threshold = threshold
            else:
                if score > best_score:  # Maximize other metrics
                    best_score = score
                    best_threshold = threshold

        self.best_threshold = best_threshold
        self.threshold_metrics[metric] = {
            'threshold': best_threshold,
            'score': best_score,
            'all_thresholds': threshold_range,
            'all_scores': scores
        }

        print(f"\nBest Threshold: {best_threshold:.4f}")
        if metric == 'log_loss':
            print(f"Best {metric.upper()} Score: {best_score:.4f} (lower is better)")
        else:
            print(f"Best {metric.upper()} Score: {best_score:.4f}")

        # Print metrics at best threshold
        y_pred_best = (self.y_proba >= best_threshold).astype(int)
        print(f"\nMetrics at best threshold:")
        print(f"  Precision: {precision_score(self.y_true, y_pred_best):.4f}")
        print(f"  Recall: {recall_score(self.y_true, y_pred_best):.4f}")
        print(f"  F1-Score: {f1_score(self.y_true, y_pred_best):.4f}")
        print(f"  Accuracy: {accuracy_score(self.y_true, y_pred_best):.4f}")
        print(f"  Log Loss: {log_loss(self.y_true, self.y_proba):.4f}")

        return best_threshold, best_score

    def get_metrics_at_threshold(self, threshold):
        """Get all metrics at a specific threshold"""
        y_pred = (self.y_proba >= threshold).astype(int)

        return {
            'threshold': threshold,
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred),
            'recall': recall_score(self.y_true, y_pred),
            'f1': f1_score(self.y_true, y_pred),
            'roc_auc': roc_auc_score(self.y_true, self.y_proba),
            'log_loss': log_loss(self.y_true, self.y_proba),
            'confusion_matrix': confusion_matrix(self.y_true, y_pred)
        }


# ============================================================================
# PART 4: EVALUATION AND COMPARISON
# ============================================================================

def evaluate_models(models_dict, X_test, y_test, threshold=0.5):
    """Evaluate all models on test set"""
    print(f"\n{'='*70}")
    print("MODEL EVALUATION ON TEST SET")
    print(f"{'='*70}")

    results = {}

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        results[name] = {
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }

    # Create comparison DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('ROC-AUC', ascending=False)

    print("\n" + results_df.to_string())
    print(f"\n{'='*70}")

    return results_df


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main_pipeline(X_train, y_train, X_test, y_test, n_trials=100, max_estimators=1000,
                  threshold_metric='f1', n_jobs_optuna=1, random_state=42, optuna_verbosity=1,
                  output_dir=None, load_checkpoint=False, save_frequency='model',
                  use_weights=False, weight_decay=0.99, date_column=None):
    """
    Complete ML pipeline

    Parameters:
    -----------
    X_train, y_train : Training data (Polars/Pandas DataFrame or numpy array)
    X_test, y_test : Test data (used for validation split and final evaluation)
    n_trials : Number of Optuna trials per model
    max_estimators : Maximum number of estimators/iterations for tree models
    threshold_metric : Metric to optimize threshold for ('f1', 'accuracy', 'log_loss', 'precision', 'recall', etc.)
    n_jobs_optuna : Number of parallel jobs for Optuna optimization. Use -1 for all CPUs.
    random_state : Random seed for reproducibility
    optuna_verbosity : Optuna logging level (0=silent, 1=progress bar only, 2=info, 3=debug)
    output_dir : Directory to save checkpoints. If None, creates 'ml_pipeline_output_{timestamp}'
    load_checkpoint : If True, load existing checkpoint from output_dir and continue tuning
    save_frequency : When to save checkpoints ('model'=after each model, 'end'=only at end, 'never'=don't save)
    use_weights : If True, weight recent observations more heavily (exponential decay)
    weight_decay : Decay factor for sample weights (default 0.99). Higher = more emphasis on recent data.
                   Only used if use_weights=True. Range: 0.9 (strong decay) to 0.999 (mild decay)
    date_column : Name of date column in X_train (if DataFrame). If provided and use_weights=True,
                  weights are calculated based on actual dates instead of row order.
                  Date column will be automatically removed from features before training.
    """

    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'ml_pipeline_output_{timestamp}'

    output_path = Path(output_dir)

    # Set Optuna verbosity level
    if optuna_verbosity == 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    elif optuna_verbosity == 1:
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Only show progress bar
    elif optuna_verbosity == 2:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    elif optuna_verbosity == 3:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Extract and store date column if provided (before converting to numpy)
    dates_train = None
    if date_column is not None and hasattr(X_train, 'columns'):
        if date_column in X_train.columns:
            # Extract dates
            if hasattr(X_train, 'to_pandas'):  # Polars
                dates_train = X_train[date_column].to_pandas()
            else:  # Pandas
                dates_train = X_train[date_column]

            # Remove date column from features
            print(f"\n{'='*70}")
            print(f"EXTRACTING DATE COLUMN: '{date_column}'")
            print(f"{'='*70}")
            print(f"Date range: {dates_train.min()} to {dates_train.max()}")
            print(f"Removing '{date_column}' from features before training")

            # Remove from X_train
            if hasattr(X_train, 'drop'):  # Polars or Pandas
                X_train = X_train.drop(date_column)
            else:
                raise ValueError(f"date_column specified but X_train doesn't support .drop()")
        else:
            raise ValueError(f"date_column '{date_column}' not found in X_train. Available columns: {X_train.columns}")

    # Convert Polars DataFrames to numpy arrays if needed
    if hasattr(X_train, 'to_numpy'):  # Polars or Pandas DataFrame
        X_train = X_train.to_numpy()
    if hasattr(X_test, 'to_numpy'):
        X_test = X_test.to_numpy()
    if hasattr(y_train, 'to_numpy'):
        y_train = y_train.to_numpy()
    if hasattr(y_test, 'to_numpy'):
        y_test = y_test.to_numpy()

    # Flatten target arrays if needed
    if len(y_train.shape) > 1:
        y_train = y_train.ravel()
    if len(y_test.shape) > 1:
        y_test = y_test.ravel()

    # Split test set for validation (ensemble optimization) and final test
    from sklearn.model_selection import train_test_split
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=random_state, stratify=y_test
    )

    print(f"\n{'='*70}")
    print("DATASET INFORMATION")
    print(f"{'='*70}")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test_final.shape[0]} samples")
    print(f"Target distribution (train): {np.bincount(y_train)}")

    # Calculate sample weights if requested
    sample_weights = None
    if use_weights:
        print(f"\n{'='*70}")
        print("CALCULATING SAMPLE WEIGHTS")
        print(f"{'='*70}")

        if dates_train is not None:
            # Use actual dates for weighting
            print(f"Using date-based weighting from column '{date_column}'")
            print(f"Weight decay factor: {weight_decay}")

            # Convert dates to numeric (days since earliest date)
            import pandas as pd
            dates_numeric = pd.to_datetime(dates_train)
            days_since_first = (dates_numeric - dates_numeric.min()).dt.days.values

            # Calculate weights based on time gaps
            # weight = decay^(days_ago)
            max_days = days_since_first.max()
            days_ago = max_days - days_since_first  # Most recent = 0 days ago

            # Use daily decay
            sample_weights = weight_decay ** days_ago

            # Normalize
            sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())

            print(f"\nDate-based weight statistics:")
            print(f"  Date range: {dates_numeric.min().strftime('%Y-%m-%d')} to {dates_numeric.max().strftime('%Y-%m-%d')}")
            print(f"  Total days: {max_days}")
            print(f"  Oldest sample ({dates_numeric.min().strftime('%Y-%m-%d')}):  weight = {sample_weights.min():.4f}")
            print(f"  Median sample: weight = {np.median(sample_weights):.4f}")
            print(f"  Newest sample ({dates_numeric.max().strftime('%Y-%m-%d')}): weight = {sample_weights.max():.4f}")
            print(f"  Weight ratio (newest/oldest): {sample_weights.max() / sample_weights.min():.2f}x")

        else:
            # Use row-based weighting (original behavior)
            print(f"Using row-based weighting (no date column provided)")
            print(f"Weight decay factor: {weight_decay}")
            print(f"Assuming data is sorted: oldest first, newest last")

            n_samples = len(y_train)
            sample_weights = np.array([weight_decay ** (n_samples - i - 1) for i in range(n_samples)])

            # Normalize
            sample_weights = sample_weights * (n_samples / sample_weights.sum())

            print(f"\nRow-based weight statistics:")
            print(f"  Oldest sample (row 0):     weight = {sample_weights[0]:.4f}")
            print(f"  Median sample (row {n_samples//2}): weight = {np.median(sample_weights):.4f}")
            print(f"  Newest sample (row {n_samples-1}):  weight = {sample_weights[-1]:.4f}")
            print(f"  Weight ratio (newest/oldest): {sample_weights[-1] / sample_weights[0]:.2f}x")

        print(f"  Sum of weights: {sample_weights.sum():.0f} (normalized to sample count)")

    else:
        print(f"\nSample weights: Not using (all samples weighted equally)")

    # ========================================================================
    # STEP 1: Tune individual models
    # ========================================================================

    # Load checkpoint if requested
    if load_checkpoint and output_path.exists():
        tuner = ModelTuner.load_checkpoint(output_dir, X_train, y_train)
        # Update sample weights if using weights now
        if use_weights:
            tuner.sample_weights = sample_weights
        print(f"\nContinuing optimization with {n_trials} additional trials per model...")
    else:
        tuner = ModelTuner(X_train, y_train, cv_folds=5, random_state=random_state,
                          max_estimators=max_estimators, sample_weights=sample_weights)

    # Determine which models to train
    models_to_train = ['ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting',
                       'LogisticRegression', 'SGDClassifier', 'GaussianNB', 'BernoulliNB']
    
    # Tune models individually and save after each if requested
    for model_name in models_to_train:
        study = tuner.tune_model(model_name, n_trials=n_trials, n_jobs=n_jobs_optuna, optuna_verbosity=optuna_verbosity)
        
        # Save checkpoint after each model if save_frequency is 'model'
        if save_frequency == 'model':
            tuner.save_checkpoint(output_dir)
    
    # Store all studies
    studies = tuner.studies
    
    # ========================================================================
    # STEP 2: Evaluate individual models
    # ========================================================================
    individual_results = evaluate_models(tuner.best_models, X_val, y_val)
    
    # ========================================================================
    # STEP 3: Build weighted ensemble
    # ========================================================================
    ensemble = WeightedEnsemble(tuner.best_models, X_val, y_val, random_state=random_state)
    ensemble_study = ensemble.optimize_weights(n_trials=200, optuna_verbosity=optuna_verbosity)
    
    # Save ensemble
    if save_frequency in ['model', 'end']:
        ensemble_dir = output_path / 'ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        with open(ensemble_dir / 'ensemble.pkl', 'wb') as f:
            pickle.dump(ensemble, f)
        with open(ensemble_dir / 'ensemble_study.pkl', 'wb') as f:
            pickle.dump(ensemble_study, f)
    
    # ========================================================================
    # STEP 4: Optimize threshold
    # ========================================================================
    ensemble_proba = ensemble.predict_proba(X_val)
    threshold_opt = ThresholdOptimizer(y_val, ensemble_proba)
    
    # Optimize for specified metric
    threshold_best, score_best = threshold_opt.optimize_threshold(threshold_metric)
    
    # ========================================================================
    # STEP 5: Final evaluation on test set
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*70}")
    
    # Individual models
    final_results = evaluate_models(tuner.best_models, X_test_final, y_test_final, threshold=threshold_best)
    
    # Ensemble with optimized threshold
    ensemble_proba_test = ensemble.predict_proba(X_test_final)
    ensemble_pred_test = (ensemble_proba_test >= threshold_best).astype(int)
    
    print("\n" + "="*70)
    print(f"ENSEMBLE PERFORMANCE (threshold optimized for {threshold_metric.upper()})")
    print("="*70)
    print(f"Threshold: {threshold_best:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test_final, ensemble_proba_test):.4f}")
    print(f"Accuracy: {accuracy_score(y_test_final, ensemble_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test_final, ensemble_pred_test):.4f}")
    print(f"Recall: {recall_score(y_test_final, ensemble_pred_test):.4f}")
    print(f"F1-Score: {f1_score(y_test_final, ensemble_pred_test):.4f}")
    print(f"Log Loss: {log_loss(y_test_final, ensemble_proba_test):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_final, ensemble_pred_test))
    
    print("\nClassification Report:")
    print(classification_report(y_test_final, ensemble_pred_test))
    
    # ========================================================================
    # Save final results
    # ========================================================================
    if save_frequency in ['model', 'end']:
        # Save threshold optimizer
        threshold_dir = output_path / 'threshold'
        threshold_dir.mkdir(parents=True, exist_ok=True)
        with open(threshold_dir / 'threshold_optimizer.pkl', 'wb') as f:
            pickle.dump(threshold_opt, f)
        
        # Save final results
        results_dir = output_path / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(results_dir / 'model_comparison.csv')
        
        # Save final predictions
        predictions_df = pd.DataFrame({
            'y_true': y_test_final,
            'y_pred': ensemble_pred_test,
            'y_proba': ensemble_proba_test
        })
        predictions_df.to_csv(results_dir / 'final_predictions.csv', index=False)
        
        # Save configuration
        config = {
            'n_trials': n_trials,
            'max_estimators': max_estimators,
            'threshold_metric': threshold_metric,
            'n_jobs_optuna': n_jobs_optuna,
            'random_state': random_state,
            'best_threshold': float(threshold_best),
            'final_roc_auc': float(roc_auc_score(y_test_final, ensemble_proba_test)),
            'final_accuracy': float(accuracy_score(y_test_final, ensemble_pred_test)),
            'final_f1': float(f1_score(y_test_final, ensemble_pred_test)),
            'final_log_loss': float(log_loss(y_test_final, ensemble_proba_test))
        }
        with open(output_path / 'final_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"All results saved to: {output_path}")
        print(f"{'='*70}")
    
    # ========================================================================
    # Return everything
    # ========================================================================
    return {
        'tuner': tuner,
        'models': tuner.best_models,
        'best_params': tuner.best_params,
        'ensemble': ensemble,
        'threshold_optimizer': threshold_opt,
        'best_threshold': threshold_best,
        'threshold_metric': threshold_metric,
        'results': final_results,
        'studies': studies,
        'output_dir': str(output_path)
    }


# ============================================================================
# HELPER FUNCTIONS FOR LOADING SAVED PIPELINES
# ============================================================================

def load_pipeline(output_dir):
    """
    Load a complete saved pipeline for inference
    
    Parameters:
    -----------
    output_dir : str
        Directory containing the saved pipeline
    
    Returns:
    --------
    dict with:
        - ensemble: WeightedEnsemble object for predictions
        - models: dict of individual models
        - best_params: dict of best hyperparameters
        - threshold: optimized decision threshold
        - config: pipeline configuration
        - predict: helper function for safe predictions
        - predict_proba: helper function for safe probability predictions
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    print(f"\nLoading pipeline from: {output_path}")
    
    # Load ensemble
    ensemble_path = output_path / 'ensemble' / 'ensemble.pkl'
    if ensemble_path.exists():
        with open(ensemble_path, 'rb') as f:
            ensemble = pickle.load(f)
        print("✓ Ensemble loaded")
    else:
        ensemble = None
        print("✗ Ensemble not found")
    
    # Load individual models
    models = {}
    models_dir = output_path / 'models'
    if models_dir.exists():
        for model_file in models_dir.glob('*.pkl'):
            model_name = model_file.stem
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
        print(f"✓ {len(models)} individual models loaded")
    
    # Load best parameters
    params_path = output_path / 'best_params.json'
    if params_path.exists():
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print("✓ Best parameters loaded")
    else:
        best_params = {}
    
    # Load configuration
    config_path = output_path / 'final_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = config.get('best_threshold', 0.5)
        print(f"✓ Configuration loaded (threshold: {threshold:.4f})")
    else:
        config = {}
        threshold = 0.5
    
    # Create helper functions that auto-convert data
    def safe_predict(X, threshold=threshold, use_ensemble=True):
        """
        Predict classes with automatic data conversion
        
        Parameters:
        -----------
        X : array-like, DataFrame
            Features (will be auto-converted to numpy if needed)
        threshold : float
            Decision threshold
        use_ensemble : bool
            If True, use ensemble. If False, returns dict of individual model predictions.
        
        Returns:
        --------
        predictions : array or dict
        """
        # Convert to numpy if needed
        if hasattr(X, 'to_numpy'):
            X_np = X.to_numpy()
        else:
            X_np = X
        
        if use_ensemble:
            if ensemble is None:
                raise ValueError("Ensemble not available. Set use_ensemble=False to use individual models.")
            return ensemble.predict(X_np, threshold=threshold)
        else:
            preds = {}
            for name, model in models.items():
                preds[name] = model.predict(X_np)
            return preds
    
    def safe_predict_proba(X, use_ensemble=True):
        """
        Predict probabilities with automatic data conversion
        
        Parameters:
        -----------
        X : array-like, DataFrame
            Features (will be auto-converted to numpy if needed)
        use_ensemble : bool
            If True, use ensemble. If False, returns dict of individual model probabilities.
        
        Returns:
        --------
        probabilities : array or dict
        """
        # Convert to numpy if needed
        if hasattr(X, 'to_numpy'):
            X_np = X.to_numpy()
        else:
            X_np = X
        
        if use_ensemble:
            if ensemble is None:
                raise ValueError("Ensemble not available. Set use_ensemble=False to use individual models.")
            return ensemble.predict_proba(X_np)
        else:
            probas = {}
            for name, model in models.items():
                probas[name] = model.predict_proba(X_np)[:, 1]
            return probas
    
    return {
        'ensemble': ensemble,
        'models': models,
        'best_params': best_params,
        'threshold': threshold,
        'config': config,
        'predict': safe_predict,
        'predict_proba': safe_predict_proba
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    # Assuming you have X_train, y_train, X_test, y_test ready:
    
    results = main_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,  # Note: y_test should exist, not just y_train
        n_trials=100  # Increase for better results (e.g., 200-500)
    )
    
    # Access trained models
    best_models = results['models']
    ensemble = results['ensemble']
    
    # Make predictions with ensemble
    new_predictions = ensemble.predict(X_new, threshold=results['best_threshold'])
    new_probabilities = ensemble.predict_proba(X_new)
    
    # Access best parameters
    print(results['best_params'])
    
    # Access optimization studies for analysis
    studies = results['studies']
    """
    pass