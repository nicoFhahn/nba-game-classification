"""
SHAP Analysis for ML Pipeline
Supports both WeightedEnsemble and StackingEnsemble
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def _extract_feature_names(X):
    """Extract feature names from DataFrame or create generic names"""
    if hasattr(X, 'columns'):
        # Polars or Pandas DataFrame
        if hasattr(X.columns, 'to_list'):
            return X.columns.to_list()  # Pandas
        else:
            return list(X.columns)  # Polars
    else:
        # Numpy array - create generic names
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        return [f'feature_{i}' for i in range(n_features)]


def calculate_shap_single_model(model, X_train, model_name='Model'):
    """
    Calculate SHAP values for a single model

    Parameters:
    -----------
    model : trained model
        Single trained model
    X_train : array-like
        Training data for SHAP analysis
    model_name : str
        Name of the model (for logging)

    Returns:
    --------
    shap_values : numpy array
        SHAP values
    base_value : float
        Expected value
    feature_names : list
        Feature names
    """
    print(f"\n{'='*70}")
    print(f"Calculating SHAP for {model_name}")
    print(f"{'='*70}")

    # Convert to numpy if needed
    if hasattr(X_train, 'to_numpy'):
        X_train_np = X_train.to_numpy()
    else:
        X_train_np = X_train

    # Extract feature names
    feature_names = _extract_feature_names(X_train)

    print(f"Data shape: {X_train_np.shape}")

    # Try TreeExplainer first (fast for tree models)
    try:
        print("Trying TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_np)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Binary classification, take positive class
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # (n_samples, n_features, 2) -> (n_samples, n_features)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        print(f"✓ TreeExplainer successful")

    except Exception as e:
        print(f"⚠️  TreeExplainer failed: {str(e)[:100]}")
        print("Falling back to KernelExplainer (slower)...")

        # Fallback to KernelExplainer
        background = shap.sample(X_train_np, min(50, len(X_train_np)))

        def predict_fn(X):
            proba = model.predict_proba(X)
            return proba[:, 1] if len(proba.shape) > 1 else proba

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_train_np)
        base_value = explainer.expected_value

        print(f"✓ KernelExplainer successful")

    print(f"SHAP values shape: {shap_values.shape}")

    return shap_values, base_value, feature_names


def calculate_shap_individual_models(saved, X_train, models_to_analyze=None):
    """
    Calculate SHAP values for all individual models

    Parameters:
    -----------
    saved : dict
        Pipeline loaded with load_pipeline()
    X_train : array-like
        Training data for SHAP analysis
    models_to_analyze : list, optional
        List of model names to analyze. If None, analyze all models.

    Returns:
    --------
    results : dict
        Dictionary with SHAP values for each model
        {model_name: {'shap_values': ..., 'base_value': ..., 'feature_names': ...}}
    """
    models = saved['models']

    if models_to_analyze is None:
        models_to_analyze = list(models.keys())

    print(f"\n{'='*70}")
    print(f"Calculating SHAP for Individual Models")
    print(f"{'='*70}")
    print(f"Models to analyze: {', '.join(models_to_analyze)}")

    results = {}

    for model_name in models_to_analyze:
        if model_name not in models:
            print(f"\n⚠️  Model '{model_name}' not found. Skipping...")
            continue

        model = models[model_name]

        try:
            shap_values, base_value, feature_names = calculate_shap_single_model(
                model, X_train, model_name
            )

            results[model_name] = {
                'shap_values': shap_values,
                'base_value': base_value,
                'feature_names': feature_names
            }

        except Exception as e:
            print(f"\n✗ Failed to calculate SHAP for {model_name}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✓ Completed SHAP analysis for {len(results)}/{len(models_to_analyze)} models")
    print(f"{'='*70}")

    return results


def calculate_shap_ensemble(saved, X_train, use_kernel=False, background_size=100, sample_size=None):
    """
    Calculate SHAP values for ensemble model
    Supports both WeightedEnsemble and StackingEnsemble

    Parameters:
    -----------
    saved : dict
        Pipeline loaded with load_pipeline()
    X_train : array-like
        Training data for SHAP analysis
    use_kernel : bool
        If True, use KernelExplainer on ensemble (slow but exact)
        If False, use weighted/averaged SHAP from individual models (fast, approximate)
    background_size : int
        Number of samples for background data (KernelExplainer only)
    sample_size : int, optional
        Subsample X_train for faster computation

    Returns:
    --------
    shap_values : numpy array
        SHAP values
    base_value : float or None
        Base value (expected value)
    feature_names : list
        Feature names
    """
    ensemble = saved['ensemble']
    models = saved['models']

    # Convert to numpy if needed
    if hasattr(X_train, 'to_numpy'):
        X_train_np = X_train.to_numpy()
    else:
        X_train_np = X_train

    # Extract feature names
    feature_names = _extract_feature_names(X_train)

    # Subsample if requested
    if sample_size is not None and len(X_train_np) > sample_size:
        indices = np.random.choice(len(X_train_np), sample_size, replace=False)
        X_train_np = X_train_np[indices]
        print(f"Subsampled to {sample_size} samples for faster computation")

    print("="*70)
    print("Calculating SHAP for Ensemble")
    print("="*70)

    # Determine ensemble type
    is_weighted = hasattr(ensemble, 'weights')
    is_stacking = hasattr(ensemble, 'stacking_clf')

    if is_weighted:
        print("Ensemble type: Weighted Average")
        print(f"Models: {', '.join(ensemble.model_names)}")
        print(f"Weights: {dict(zip(ensemble.model_names, ensemble.weights))}")
        weights = ensemble.weights
    elif is_stacking:
        print("Ensemble type: Stacking")
        print(f"Base models: {', '.join(ensemble.model_names)}")
        print(f"Meta-learner: {ensemble.meta_learner_type}")
        # For stacking, use equal weights as approximation
        weights = np.ones(len(ensemble.model_names)) / len(ensemble.model_names)
        print("Note: Using equal average of base models (approximation)")
    else:
        raise ValueError("Unknown ensemble type")

    print(f"Data shape: {X_train_np.shape}")

    if use_kernel:
        # METHOD 1: Exact KernelExplainer (slow)
        print("\n🐌 Using KernelExplainer (exact but SLOW)")
        print(f"Background size: {background_size}")
        print("⚠️  This can take several minutes to hours...")

        # Create background data
        if len(X_train_np) > background_size:
            background = shap.sample(X_train_np, background_size)
        else:
            background = X_train_np

        # Prediction function
        def predict_fn(X):
            return ensemble.predict_proba(X)

        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, background)

        # Calculate SHAP
        shap_values = explainer.shap_values(X_train_np)

        # Handle 3D output
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        print(f"✓ SHAP values calculated: {shap_values.shape}")

    else:
        # METHOD 2: Fast weighted/averaged method
        if is_weighted:
            print("\n⚡ Using Weighted Average Method (FAST)")
            print("✓ Combines individual model SHAP values with ensemble weights")
        else:
            print("\n⚡ Using Equal Average Method (FAST)")
            print("✓ Averages individual model SHAP values equally")

        print("   Note: This is an approximation, not exact ensemble SHAP")

        all_shap_values = []
        successful_models = []

        for model_name, model in models.items():
            print(f"\nCalculating SHAP for {model_name}...")

            try:
                # Try TreeExplainer first
                if model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees',
                                 'HistGradientBoosting', 'RandomForest']:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X_train_np)

                        # Handle different output formats
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[1]
                        elif len(shap_vals.shape) == 3:
                            shap_vals = shap_vals[:, :, 1]

                        print(f"  ✓ TreeExplainer: {shap_vals.shape}")

                    except Exception as e:
                        print(f"  ⚠️  TreeExplainer failed: {str(e)[:80]}")
                        print(f"  ↳ Falling back to KernelExplainer...")

                        # Fallback
                        background = shap.sample(X_train_np, min(50, len(X_train_np)))
                        explainer = shap.KernelExplainer(
                            lambda x: model.predict_proba(x)[:, 1],
                            background
                        )
                        shap_vals = explainer.shap_values(X_train_np)
                        print(f"  ✓ KernelExplainer: {shap_vals.shape}")

                else:
                    # Linear/probabilistic models - use KernelExplainer
                    background = shap.sample(X_train_np, min(50, len(X_train_np)))
                    explainer = shap.KernelExplainer(
                        lambda x: model.predict_proba(x)[:, 1],
                        background
                    )
                    shap_vals = explainer.shap_values(X_train_np)
                    print(f"  ✓ KernelExplainer: {shap_vals.shape}")

                all_shap_values.append(shap_vals)
                successful_models.append(model_name)

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue

        if len(all_shap_values) == 0:
            raise ValueError("Failed to calculate SHAP for any model")

        # Combine SHAP values
        print(f"\n{'='*70}")
        print("Combining Individual SHAP Values")
        print(f"{'='*70}")
        print(f"Successfully calculated SHAP for: {', '.join(successful_models)}")

        if is_weighted:
            # Weighted average
            print(f"Using weighted average")

            # Only use weights for successful models
            model_to_weight = dict(zip(ensemble.model_names, weights))
            successful_weights = [model_to_weight[name] for name in successful_models]

            # Normalize weights
            total_weight = sum(successful_weights)
            successful_weights = [w / total_weight for w in successful_weights]

            print(f"Adjusted weights: {dict(zip(successful_models, successful_weights))}")

            # Weight each model's SHAP values
            weighted_shap = []
            for shap_vals, weight in zip(all_shap_values, successful_weights):
                weighted_shap.append(shap_vals * weight)

            shap_values = np.sum(weighted_shap, axis=0)
            print(f"✓ Weighted ensemble SHAP: {shap_values.shape}")

        else:
            # Equal average (for stacking or when weights not available)
            print(f"Using equal average of {len(all_shap_values)} models")
            shap_values = np.mean(all_shap_values, axis=0)
            print(f"✓ Averaged ensemble SHAP: {shap_values.shape}")

        base_value = None

    return shap_values, base_value, feature_names


def plot_shap_summary(shap_values, X_train, feature_names, model_name='Model',
                      max_display=20, save_path=None):
    """
    Create SHAP summary plot

    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X_train : array-like
        Training data
    feature_names : list
        Feature names
    model_name : str
        Name for the plot title
    max_display : int
        Maximum number of features to display
    save_path : str, optional
        Path to save the plot
    """
    # Convert to numpy if needed
    if hasattr(X_train, 'to_numpy'):
        X_train_np = X_train.to_numpy()
    else:
        X_train_np = X_train

    print(f"\n{'='*70}")
    print(f"Creating SHAP Summary Plot for {model_name}")
    print(f"{'='*70}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_train_np,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title(f'SHAP Summary - {model_name}', fontsize=14, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")

    plt.show()
    print(f"✓ Summary plot displayed")


def plot_shap_bar(shap_values, feature_names, model_name='Model',
                  max_display=20, save_path=None):
    """
    Create SHAP bar plot (mean absolute SHAP values)

    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    feature_names : list
        Feature names
    model_name : str
        Name for the plot title
    max_display : int
        Maximum number of features to display
    save_path : str, optional
        Path to save the plot
    """
    print(f"\n{'='*70}")
    print(f"Creating SHAP Bar Plot for {model_name}")
    print(f"{'='*70}")

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    indices = np.argsort(mean_abs_shap)[::-1][:max_display]

    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        mean_abs_shap[indices],
        color='steelblue'
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'Feature Importance (SHAP) - {model_name}', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")

    plt.show()
    print(f"✓ Bar plot displayed")


def get_feature_importance_df(shap_values, feature_names):
    """
    Get feature importance as a pandas DataFrame

    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    feature_names : list
        Feature names

    Returns:
    --------
    df : pandas DataFrame
        Feature importance sorted by mean absolute SHAP value
    """
    import pandas as pd

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })

    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("SHAP Analysis Module")
    print("Use calculate_shap_ensemble() or calculate_shap_individual_models()")