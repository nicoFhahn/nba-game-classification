import numpy as np
import pandas as pd
import lightgbm as lgbm
import shap
from typing import List, Dict, Tuple, Literal
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class LGBMStepwiseFeatureSelector:
    """
    Stepwise feature selection for LightGBM with cross-validation.
    Reduces features by 10% at each step using either built-in importance or SHAP values.
    """
    
    def __init__(
        self,
        importance_type: Literal['gain', 'split', 'shap'] = 'gain',
        n_folds: int = 5,
        random_state: int = 234,
        verbose: bool = True,
        metric=None
    ):
        """
        Initialize the feature selector.
        
        Parameters:
        -----------
        importance_type : str
            Method to calculate feature importance:
            - 'gain': LightGBM's built-in gain-based importance
            - 'split': LightGBM's built-in split-based importance
            - 'shap': SHAP values (averaged across all samples)
        n_folds : int
            Number of cross-validation folds
        random_state : int
            Random state for reproducibility
        verbose : bool
            Whether to print progress information
        metric : callable or None
            Sklearn metric function to optimize (e.g., accuracy_score, roc_auc_score, f1_score).
            Should have signature: metric(y_true, y_pred) -> float
            If None, uses accuracy_score by default.
            For metrics that need probabilities (like roc_auc_score), use y_pred as probabilities.
            For metrics that need binary predictions (like accuracy_score, f1_score), y_pred will be binarized.
        """
        self.importance_type = importance_type
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.metric = metric
        self.metric_name = None
        self.higher_is_better = True
        self.needs_proba = False
        self.history_ = []
        self.best_features_ = None
        self.best_score_ = None
        
        # Set metric if not provided
        if self.metric is None:
            from sklearn.metrics import accuracy_score
            self.metric = accuracy_score
            self.metric_name = 'accuracy'
            self.higher_is_better = True
            self.needs_proba = False
        else:
            # Try to infer metric properties
            self.metric_name = getattr(metric, '__name__', 'custom_metric')
            
            # Determine if metric needs probabilities or binary predictions
            # Common probability-based metrics
            proba_metrics = ['roc_auc_score', 'log_loss', 'brier_score_loss', 
                           'average_precision_score']
            self.needs_proba = self.metric_name in proba_metrics
            
            # Determine if higher is better
            # Metrics where lower is better
            lower_is_better_metrics = ['log_loss', 'brier_score_loss', 'mean_squared_error',
                                      'mean_absolute_error', 'mean_absolute_percentage_error']
            self.higher_is_better = self.metric_name not in lower_is_better_metrics
        
    def _lgbm_metric_wrapper(self, y_pred: np.ndarray, dtrain: lgbm.Dataset) -> Tuple[str, float, bool]:
        """
        Wrapper to use sklearn metric as LightGBM custom metric.
        
        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted probabilities from LightGBM
        dtrain : lgbm.Dataset
            Training dataset containing true labels
            
        Returns:
        --------
        tuple : (metric_name, metric_value, is_higher_better)
        """
        y_true = dtrain.get_label()
        
        # Prepare predictions based on metric requirements
        if self.needs_proba:
            # Use probabilities directly for metrics like roc_auc_score
            y_pred_for_metric = y_pred
        else:
            # Binarize predictions for metrics like accuracy_score, f1_score
            y_pred_for_metric = (y_pred > 0.5).astype(int)
        
        # Calculate metric
        try:
            score = self.metric(y_true, y_pred_for_metric)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error calculating metric: {e}")
            score = 0.0
        
        return self.metric_name, score, self.higher_is_better
    
    def _get_feature_importance(
        self,
        models: List[lgbm.Booster],
        X: pd.DataFrame,
        y: np.ndarray,
        train_indices: List[np.ndarray]
    ) -> pd.Series:
        """
        Calculate feature importance using the specified method.
        
        Parameters:
        -----------
        models : List[lgbm.Booster]
            Trained models from each fold
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target variable
        train_indices : List[np.ndarray]
            Training indices for each fold (needed for SHAP)
            
        Returns:
        --------
        pd.Series : Feature importance scores (averaged across folds)
        """
        feature_names = X.columns.tolist()
        
        if self.importance_type in ['gain', 'split']:
            # Use LightGBM's built-in importance
            importance_scores = []
            for model in models:
                imp = model.feature_importance(importance_type=self.importance_type)
                importance_scores.append(imp)
            
            # Average across folds
            avg_importance = np.mean(importance_scores, axis=0)
            return pd.Series(avg_importance, index=feature_names)
        
        elif self.importance_type == 'shap':
            # Use SHAP values
            shap_values_list = []
            
            for model, train_idx in zip(models, train_indices):
                X_train_fold = X.iloc[train_idx]
                
                # Calculate SHAP values for this fold
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train_fold)
                
                # For binary classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                # Get mean absolute SHAP value for each feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_values_list.append(mean_abs_shap)
            
            # Average across folds
            avg_shap = np.mean(shap_values_list, axis=0)
            return pd.Series(avg_shap, index=feature_names)
        
        else:
            raise ValueError(f"Unknown importance_type: {self.importance_type}")
    
    def _evaluate_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str]
    ) -> Dict:
        """
        Evaluate a set of features using cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Full feature matrix
        y : np.ndarray
            Target variable
        features : List[str]
            Features to evaluate
            
        Returns:
        --------
        dict : Dictionary containing CV scores, models, and indices
        """
        X_subset = X[features]
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        models = []
        train_indices = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_subset)):
            X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create LightGBM datasets
            dtrain = lgbm.Dataset(X_train, label=y_train)
            dval = lgbm.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Train model
            model = lgbm.train(
                params={
                    "objective": "binary",
                    "metric": None,
                    "first_metric_only": True,
                    "verbose": -1,
                    "random_state": self.random_state
                },
                train_set=dtrain,
                valid_sets=[dval],
                feval=self._lgbm_metric_wrapper
            )
            
            # Evaluate on validation set using the sklearn metric
            y_pred = model.predict(X_val)
            
            # Prepare predictions based on metric requirements
            if self.needs_proba:
                y_pred_for_metric = y_pred
            else:
                y_pred_for_metric = (y_pred > 0.5).astype(int)
            
            # Calculate metric
            score = self.metric(y_val, y_pred_for_metric)
            
            cv_scores.append(score)
            models.append(model)
            train_indices.append(train_idx)
        
        return {
            'scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'models': models,
            'train_indices': train_indices
        }
    
    def fit(self, X, y) -> 'LGBMStepwiseFeatureSelector':
        """
        Perform stepwise feature selection.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix. If numpy array, will be converted to DataFrame 
            with default feature names (feature_0, feature_1, ...)
        y : np.ndarray or array-like
            Target variable (binary). Will be converted to 1D numpy array.
            
        Returns:
        --------
        self : Returns the instance itself
        """
        # Convert X to DataFrame if it's a numpy array
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                # Create DataFrame with default feature names
                n_features = X.shape[1] if len(X.shape) > 1 else 1
                feature_names = [f'feature_{i}' for i in range(n_features)]
                X = pd.DataFrame(X, columns=feature_names)
                if self.verbose:
                    print(f"Converting numpy array to DataFrame with default feature names")
            else:
                raise ValueError("X must be a pandas DataFrame or numpy array")
        
        # Convert y to numpy array and ensure it's 1D
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.ravel()
        
        current_features = X.columns.tolist()
        step = 0
        
        if self.verbose:
            print(f"Starting stepwise feature selection with {len(current_features)} features")
            print(f"Importance method: {self.importance_type}")
            print(f"Optimization metric: {self.metric_name}")
            print(f"Cross-validation folds: {self.n_folds}")
            print("=" * 80)
        
        while len(current_features) > 1:
            step += 1
            
            if self.verbose:
                print(f"\nStep {step}: Evaluating {len(current_features)} features")
            
            # Evaluate current feature set
            results = self._evaluate_features(X, y, current_features)
            
            # Record history
            self.history_.append({
                'step': step,
                'n_features': len(current_features),
                'features': current_features.copy(),
                'mean_score': results['mean_score'],
                'std_score': results['std_score'],
                'cv_scores': results['scores']
            })
            
            if self.verbose:
                print(f"  Mean CV {self.metric_name}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            
            # Update best features if this is the best score so far
            if self.best_score_ is None:
                self.best_score_ = results['mean_score']
                self.best_features_ = current_features.copy()
                if self.verbose:
                    print(f"  *** New best score! ***")
            else:
                # Check if current score is better based on metric direction
                is_better = (results['mean_score'] > self.best_score_ if self.higher_is_better 
                           else results['mean_score'] < self.best_score_)
                
                if is_better:
                    self.best_score_ = results['mean_score']
                    self.best_features_ = current_features.copy()
                    if self.verbose:
                        print(f"  *** New best score! ***")
            
            
            # Calculate feature importance
            importance = self._get_feature_importance(
                results['models'],
                X[current_features],
                y,
                results['train_indices']
            )
            
            # Calculate number of features to remove (10% of current features, at least 1)
            n_to_remove = max(1, int(len(current_features) * 0.1))
            
            # If removing 10% would leave us with 0 features, just remove enough to leave 1
            if len(current_features) - n_to_remove < 1:
                n_to_remove = len(current_features) - 1
            
            # Get least important features
            least_important = importance.nsmallest(n_to_remove).index.tolist()
            
            if self.verbose:
                print(f"  Removing {n_to_remove} least important features: {least_important}")
            
            # Remove least important features
            current_features = [f for f in current_features if f not in least_important]
            
            # Break if we're down to 1 feature
            if len(current_features) <= 1:
                # Evaluate the final feature
                if len(current_features) == 1:
                    final_results = self._evaluate_features(X, y, current_features)
                    self.history_.append({
                        'step': step + 1,
                        'n_features': 1,
                        'features': current_features.copy(),
                        'mean_score': final_results['mean_score'],
                        'std_score': final_results['std_score'],
                        'cv_scores': final_results['scores']
                    })
                    if self.verbose:
                        print(f"\nFinal step: 1 feature remaining")
                        print(f"  Mean CV {self.metric_name}: {final_results['mean_score']:.4f} (+/- {final_results['std_score']:.4f})")
                break
        
        if self.verbose:
            print("\n" + "=" * 80)
            print(f"Feature selection complete!")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best number of features: {len(self.best_features_)}")
            print(f"Best features: {self.best_features_}")
        
        return self
    
    def get_selection_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of the selection process.
        
        Returns:
        --------
        pd.DataFrame : Summary of each step
        """
        summary = pd.DataFrame([
            {
                'step': h['step'],
                'n_features': h['n_features'],
                f'mean_{self.metric_name}': h['mean_score'],
                f'std_{self.metric_name}': h['std_score']
            }
            for h in self.history_
        ])
        return summary
    
    def plot_selection_curve(self):
        """
        Plot the feature selection curve showing metric score vs number of features.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install it with: pip install matplotlib")
            return
        
        summary = self.get_selection_summary()
        
        metric_col = f'mean_{self.metric_name}'
        std_col = f'std_{self.metric_name}'
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            summary['n_features'],
            summary[metric_col],
            yerr=summary[std_col],
            marker='o',
            capsize=5,
            capthick=2
        )
        plt.axvline(x=len(self.best_features_), color='r', linestyle='--', 
                   label=f'Best: {len(self.best_features_)} features')
        plt.xlabel('Number of Features')
        plt.ylabel(f'Cross-Validation {self.metric_name.replace("_", " ").title()}')
        plt.title(f'Stepwise Feature Selection ({self.importance_type} importance)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example with synthetic data (replace with your NBA data)
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
    
    # Create synthetic dataset
    X_array, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        random_state=234
    )
    
    # Convert to DataFrame with feature names
    X = pd.DataFrame(
        X_array,
        columns=[f'feature_{i}' for i in range(X_array.shape[1])]
    )
    
    print("Example 1: Using accuracy_score (default)")
    print("-" * 80)
    selector_accuracy = LGBMStepwiseFeatureSelector(
        importance_type='gain',
        n_folds=5,
        random_state=234,
        verbose=True,
        metric=accuracy_score  # or leave as None for default
    )
    selector_accuracy.fit(X, y)
    
    print("\n\nSelection Summary:")
    print(selector_accuracy.get_selection_summary())
    
    print("\n" + "=" * 80)
    print("\nExample 2: Using F1 Score")
    print("-" * 80)
    selector_f1 = LGBMStepwiseFeatureSelector(
        importance_type='gain',
        n_folds=5,
        random_state=234,
        verbose=True,
        metric=f1_score
    )
    selector_f1.fit(X, y)
    
    print("\n\nSelection Summary:")
    print(selector_f1.get_selection_summary())
    
    print("\n" + "=" * 80)
    print("\nExample 3: Using ROC AUC Score")
    print("-" * 80)
    selector_auc = LGBMStepwiseFeatureSelector(
        importance_type='gain',
        n_folds=5,
        random_state=234,
        verbose=True,
        metric=roc_auc_score  # This metric needs probabilities
    )
    selector_auc.fit(X, y)
    
    print("\n\nSelection Summary:")
    print(selector_auc.get_selection_summary())
    
    print("\n" + "=" * 80)
    print("\nExample 4: Using Precision Score")
    print("-" * 80)
    selector_precision = LGBMStepwiseFeatureSelector(
        importance_type='shap',
        n_folds=5,
        random_state=234,
        verbose=True,
        metric=precision_score
    )
    selector_precision.fit(X, y)
    
    print("\n\nSelection Summary:")
    print(selector_precision.get_selection_summary())
    
    
    print("\n" + "=" * 80)
    print("\nExample 5: Using numpy arrays as input with custom metric")
    print("-" * 80)
    
    # Simulate your use case: X_train.to_numpy(), y_train.to_numpy().ravel()
    X_numpy = X.to_numpy()
    y_numpy = y.ravel()
    
    selector_numpy = LGBMStepwiseFeatureSelector(
        importance_type='gain',
        n_folds=5,
        random_state=234,
        verbose=True,
        metric=f1_score
    )
    
    # This now works with numpy arrays!
    selector_numpy.fit(X_numpy, y_numpy)
    
    print("\n\nSelection Summary:")
    print(selector_numpy.get_selection_summary())
    print(f"\nNote: Features were automatically named as feature_0, feature_1, etc.")
    print(f"Best features: {selector_numpy.best_features_[:5]}...")  # Show first 5