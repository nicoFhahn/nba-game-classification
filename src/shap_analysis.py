"""
SHAP (SHapley Additive exPlanations) Analysis for ML Pipeline

This script shows how to calculate SHAP values for:
a) Training data (global feature importance)
b) New predictions (individual explanations)

SHAP provides better explanations than simple feature importance because it:
- Shows how each feature contributes to individual predictions
- Is theoretically grounded in game theory
- Works for any model type
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from ml_pipeline import load_pipeline

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_feature_names(X_data):
    """
    Extract feature names from DataFrame (handles both Pandas and Polars)
    
    Parameters:
    -----------
    X_data : DataFrame or array
        Input data
    
    Returns:
    --------
    feature_names : list
        List of feature names
    """
    if hasattr(X_data, 'columns'):
        columns = X_data.columns
        # Polars columns is already a list, Pandas has to_list() method
        if isinstance(columns, list):
            return columns
        elif hasattr(columns, 'to_list'):
            return columns.to_list()
        elif hasattr(columns, 'tolist'):
            return columns.tolist()
        else:
            return list(columns)
    else:
        # Numpy array or no column names
        return [f'Feature_{i}' for i in range(X_data.shape[1])]

# ============================================================================
# SETUP: Install SHAP if needed
# ============================================================================

"""
First, install SHAP:
    pip install shap

For faster computations with tree models:
    pip install shap[plots]
"""

# ============================================================================
# OPTION 1: SHAP for Individual Models (Recommended for Tree Models)
# ============================================================================

def calculate_shap_individual_models(saved_pipeline, X_data, model_name='XGBoost', 
                                     background_size=100):
    """
    Calculate SHAP values for a specific tree-based model
    
    This is FAST for tree models (XGBoost, LightGBM, CatBoost, etc.)
    Uses TreeExplainer which is exact and efficient.
    
    Parameters:
    -----------
    saved_pipeline : dict
        Output from load_pipeline()
    X_data : DataFrame or array
        Data to explain (will be auto-converted to numpy)
    model_name : str
        Which model to explain: 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'HistGradientBoosting'
    background_size : int
        Number of background samples for explainer (not needed for tree models, included for consistency)
    
    Returns:
    --------
    shap_values : array
        SHAP values for each feature and sample
    explainer : shap.Explainer
        SHAP explainer object
    """
    
    # Convert to numpy if needed
    if hasattr(X_data, 'to_numpy'):
        X_np = X_data.to_numpy()
        feature_names = _extract_feature_names(X_data)
    else:
        X_np = X_data
        feature_names = [f'Feature_{i}' for i in range(X_data.shape[1])]
    
    # Get the model
    model = saved_pipeline['models'][model_name]
    
    print(f"\nCalculating SHAP values for {model_name}...")
    print(f"Data shape: {X_np.shape}")
    
    # Use TreeExplainer for tree-based models (FAST!)
    if model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'HistGradientBoosting']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_np)
        
        # For binary classification, some models return list of arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
    else:
        raise ValueError(f"Use calculate_shap_ensemble for non-tree models")
    
    print(f"✓ SHAP values calculated: {shap_values.shape}")
    
    return shap_values, explainer, feature_names


# ============================================================================
# OPTION 2: SHAP for Ensemble (Works for weighted ensemble)
# ============================================================================

def calculate_shap_ensemble(saved_pipeline, X_data, background_size=100, 
                            use_kernel=True):
    """
    Calculate SHAP values for the entire weighted ensemble
    
    This explains the final ensemble predictions (after weighted averaging).
    
    Two methods available:
    1. KernelExplainer (use_kernel=True): Model-agnostic, SLOW but works always
    2. Linear combination of individual SHAP (use_kernel=False): FAST approximation
    
    NOTE: If you get errors with XGBoost and SHAP (base_score issue), the function
    will automatically fall back to KernelExplainer for problematic models.
    
    Alternative: Upgrade SHAP: pip install --upgrade shap
    
    Parameters:
    -----------
    saved_pipeline : dict
        Output from load_pipeline()
    X_data : DataFrame or array
        Data to explain
    background_size : int
        Number of background samples (smaller = faster but less accurate)
        Only used for KernelExplainer (use_kernel=True)
    use_kernel : bool
        If True, use KernelExplainer (exact but slow)
        If False, use weighted average of individual model SHAP (fast approximation)
    
    Returns:
    --------
    shap_values : array
        SHAP values for each feature and sample
    explainer : shap.Explainer or None
        SHAP explainer object (None if use_kernel=False)
    feature_names : list
        Feature names
    """
    
    # Convert to numpy if needed
    if hasattr(X_data, 'to_numpy'):
        X_np = X_data.to_numpy()
        feature_names = _extract_feature_names(X_data)
    else:
        X_np = X_data
        feature_names = [f'Feature_{i}' for i in range(X_data.shape[1])]
    
    ensemble = saved_pipeline['ensemble']
    
    if use_kernel:
        # METHOD 1: KernelExplainer (Exact but SLOW)
        print("\n" + "="*70)
        print("Calculating SHAP for Ensemble using KernelExplainer")
        print("="*70)
        print("⚠️  This is SLOW but gives exact SHAP values for ensemble")
        print("    For faster results, use use_kernel=False")
        
        # Create background dataset
        if X_np.shape[0] > background_size:
            background_indices = np.random.choice(X_np.shape[0], background_size, replace=False)
            background = X_np[background_indices]
        else:
            background = X_np
        
        print(f"\nData shape: {X_np.shape}")
        print(f"Background size: {background.shape[0]}")
        print("Computing... (this may take a while)")
        
        # Create prediction function for ensemble
        def predict_fn(X):
            return saved_pipeline['predict_proba'](X)
        
        # Use KernelExplainer
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_np)
        
        print(f"✓ SHAP values calculated: {shap_values.shape}")
        
    else:
        # METHOD 2: Weighted average of individual SHAP values (FAST approximation)
        print("\n" + "="*70)
        print("Calculating SHAP for Ensemble using Weighted Average Method")
        print("="*70)
        print("✓ This is FAST - uses weighted average of individual model SHAP values")
        print("   Note: This is an approximation, not exact ensemble SHAP")
        
        print(f"\nData shape: {X_np.shape}")
        print(f"Ensemble weights: {dict(zip(ensemble.model_names, ensemble.weights))}")
        
        # Calculate SHAP for each model and combine with ensemble weights
        all_shap = []
        
        for model_name in ensemble.model_names:
            print(f"\n  Computing SHAP for {model_name}...")
            model = ensemble.models[model_name]
            
            try:
                # Use TreeExplainer for tree models
                model_explainer = shap.TreeExplainer(model)
                model_shap = model_explainer.shap_values(X_np)
                
                # Handle list output (binary classification)
                if isinstance(model_shap, list):
                    model_shap = model_shap[1]
                
                all_shap.append(model_shap)
                print(f"    ✓ Shape: {model_shap.shape}")
                
            except (ValueError, Exception) as e:
                # Fallback for models with compatibility issues (e.g., XGBoost with SHAP)
                if "could not convert string to float" in str(e) or "base_score" in str(e):
                    print(f"    ⚠️  TreeExplainer failed (XGBoost/SHAP compatibility issue)")
                    print(f"       Using KernelExplainer as fallback (slower)...")
                    
                    # Use KernelExplainer as fallback
                    background_indices = np.random.choice(X_np.shape[0], min(50, X_np.shape[0]), replace=False)
                    background = X_np[background_indices]
                    
                    def predict_fn(X):
                        return model.predict_proba(X)[:, 1]
                    
                    kernel_explainer = shap.KernelExplainer(predict_fn, background)
                    model_shap = kernel_explainer.shap_values(X_np, silent=True)
                    
                    all_shap.append(model_shap)
                    print(f"    ✓ Shape: {model_shap.shape} (using KernelExplainer)")
                else:
                    # Re-raise if it's a different error
                    raise
        
        # Combine SHAP values using ensemble weights
        print(f"\n  Combining with ensemble weights...")
        
        # First, ensure all SHAP arrays have the same shape
        for i, (model_name, shap_array) in enumerate(zip(ensemble.model_names, all_shap)):
            # Handle 3D arrays (some explainers return [n_samples, n_features, n_classes])
            if len(shap_array.shape) == 3:
                # Take positive class (last dimension, index 1)
                all_shap[i] = shap_array[:, :, 1]
                print(f"    {model_name}: Reshaped from {shap_array.shape} to {all_shap[i].shape}")
            elif len(shap_array.shape) == 2:
                # Already correct shape
                print(f"    {model_name}: Shape {shap_array.shape} (OK)")
            else:
                raise ValueError(f"Unexpected SHAP shape for {model_name}: {shap_array.shape}")
        
        # Now combine with weights
        shap_values = np.zeros_like(all_shap[0])
        for i, (model_name, weight) in enumerate(zip(ensemble.model_names, ensemble.weights)):
            shap_values += weight * all_shap[i]
            print(f"    {model_name}: weight = {weight:.4f}")
        
        explainer = None  # No single explainer object
        
        print(f"\n✓ Ensemble SHAP values calculated: {shap_values.shape}")
    
    return shap_values, explainer, feature_names


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_shap_summary(shap_values, X_data, feature_names, model_name='Model', 
                      max_display=20, save_path=None):
    """
    Create a SHAP summary plot showing global feature importance
    
    This shows:
    - Which features are most important overall
    - How feature values affect predictions (color coding)
    """
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, 
        X_data.to_numpy() if hasattr(X_data, 'to_numpy') else X_data,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Summary plot saved to: {save_path}")
    
    plt.show()


def plot_shap_waterfall(shap_values, X_data, feature_names, sample_idx=0, 
                        model_name='Model', save_path=None):
    """
    Create a waterfall plot for a single prediction
    
    Shows how each feature contributes to pushing the prediction
    from the base value (average prediction) to the final prediction.
    """
    # Convert to numpy if needed
    X_np = X_data.to_numpy() if hasattr(X_data, 'to_numpy') else X_data
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=shap_values.mean(axis=0) if len(shap_values.shape) > 1 else 0,
        data=X_np[sample_idx],
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f'SHAP Waterfall Plot - {model_name} (Sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Waterfall plot saved to: {save_path}")
    
    plt.show()


def plot_shap_force(shap_values, X_data, feature_names, sample_idx=0, 
                    base_value=None, save_path=None):
    """
    Create a force plot for a single prediction
    
    Shows which features push the prediction higher (red) or lower (blue).
    """
    # Convert to numpy if needed
    X_np = X_data.to_numpy() if hasattr(X_data, 'to_numpy') else X_data
    
    if base_value is None:
        base_value = shap_values.mean()
    
    # Force plot
    shap.force_plot(
        base_value,
        shap_values[sample_idx],
        X_np[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Force plot saved to: {save_path}")
    
    plt.show()


def get_top_features(shap_values, feature_names, top_n=10):
    """
    Get top N most important features based on mean absolute SHAP values
    
    Returns:
    --------
    DataFrame with features ranked by importance
    """
    # Calculate mean absolute SHAP value for each feature
    importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df[['rank', 'feature', 'importance']]


# ============================================================================
# COMPLETE EXAMPLE WORKFLOW
# ============================================================================

def shap_analysis_workflow(pipeline_dir, X_train, X_test, 
                           model_name='XGBoost', use_ensemble=False,
                           use_kernel=False):
    """
    Complete SHAP analysis workflow
    
    Parameters:
    -----------
    pipeline_dir : str
        Path to saved pipeline
    X_train : DataFrame or array
        Training data (for global importance)
    X_test : DataFrame or array
        Test data (for individual predictions)
    model_name : str
        Which model to analyze (if use_ensemble=False)
    use_ensemble : bool
        If True, analyze ensemble. If False, analyze individual model.
    use_kernel : bool
        Only for ensemble: If True, use exact KernelExplainer (slow).
        If False, use fast weighted average method (recommended).
    """
    
    print("="*70)
    print("SHAP ANALYSIS WORKFLOW")
    print("="*70)
    
    # Load pipeline
    print("\n1. Loading pipeline...")
    saved = load_pipeline(pipeline_dir)
    
    # Calculate SHAP values on training data (for global importance)
    print("\n2. Calculating SHAP values on training data...")
    
    if use_ensemble:
        shap_train, explainer, feature_names = calculate_shap_ensemble(
            saved, X_train, background_size=100, use_kernel=use_kernel
        )
        model_label = "Ensemble" + (" (Exact)" if use_kernel else " (Fast)")
    else:
        shap_train, explainer, feature_names = calculate_shap_individual_models(
            saved, X_train, model_name=model_name
        )
        model_label = model_name
    
    # Global feature importance
    print("\n3. Global Feature Importance (Training Data):")
    top_features = get_top_features(shap_train, feature_names, top_n=10)
    print(top_features.to_string(index=False))
    
    # Visualizations on training data
    print("\n4. Creating visualizations...")
    
    # Summary plot (global importance)
    plot_shap_summary(
        shap_train, X_train, feature_names, 
        model_name=f"{model_label} (Training Data)",
        save_path=f'shap_summary_{model_label.lower().replace(" ", "_")}.png'
    )
    
    # Calculate SHAP values on test data (for individual predictions)
    print("\n5. Calculating SHAP values on test data...")
    
    if use_ensemble:
        shap_test, _, _ = calculate_shap_ensemble(
            saved, X_test, background_size=100, use_kernel=use_kernel
        )
    else:
        # For tree models, can reuse explainer
        X_test_np = X_test.to_numpy() if hasattr(X_test, 'to_numpy') else X_test
        shap_test = explainer.shap_values(X_test_np)
        if isinstance(shap_test, list):
            shap_test = shap_test[1]
    
    # Individual prediction explanations
    print("\n6. Explaining individual predictions...")
    
    # Example: Explain first 3 test samples
    for i in range(min(3, X_test.shape[0])):
        print(f"\n   Sample {i}:")
        plot_shap_waterfall(
            shap_test, X_test, feature_names, 
            sample_idx=i, model_name=model_label,
            save_path=f'shap_waterfall_{model_label.lower().replace(" ", "_")}_sample{i}.png'
        )
    
    print("\n" + "="*70)
    print("✓ SHAP Analysis Complete!")
    print("="*70)
    
    return {
        'shap_train': shap_train,
        'shap_test': shap_test,
        'explainer': explainer,
        'feature_names': feature_names,
        'top_features': top_features
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    # Example 1: SHAP for single model (FAST - recommended for tree models)
    
    from ml_pipeline import load_pipeline
    
    # Load pipeline
    saved = load_pipeline('my_experiment')
    
    # Calculate SHAP for XGBoost on training data
    shap_values, explainer, feature_names = calculate_shap_individual_models(
        saved, X_train, model_name='XGBoost'
    )
    
    # Visualize global importance
    plot_shap_summary(shap_values, X_train, feature_names, model_name='XGBoost')
    
    # Get top features
    top_features = get_top_features(shap_values, feature_names, top_n=10)
    print(top_features)
    
    # Explain individual prediction
    shap_test = explainer.shap_values(X_test.to_numpy())
    if isinstance(shap_test, list):
        shap_test = shap_test[1]
    
    plot_shap_waterfall(shap_test, X_test, feature_names, sample_idx=0, 
                       model_name='XGBoost')
    
    
    # Example 2: SHAP for ensemble (SLOWER but explains ensemble)
    
    saved = load_pipeline('my_experiment')
    
    # Calculate SHAP for ensemble
    shap_values, explainer, feature_names = calculate_shap_ensemble(
        saved, X_train, background_size=100
    )
    
    # Same visualizations as above
    plot_shap_summary(shap_values, X_train, feature_names, model_name='Ensemble')
    
    
    # Example 3: Complete workflow (does everything)
    
    results = shap_analysis_workflow(
        pipeline_dir='my_experiment',
        X_train=X_train,
        X_test=X_test,
        model_name='XGBoost',  # Analyze XGBoost
        use_ensemble=False     # Set True to analyze ensemble
    )
    
    # Access results
    shap_train = results['shap_train']
    top_features = results['top_features']
    
    
    # Example 4: Compare SHAP values across all models
    
    saved = load_pipeline('my_experiment')
    
    for model_name in ['ExtraTrees', 'XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting']:
        print(f"\n{'='*70}")
        print(f"Analyzing {model_name}")
        print(f"{'='*70}")
        
        shap_values, explainer, feature_names = calculate_shap_individual_models(
            saved, X_train, model_name=model_name
        )
        
        top_features = get_top_features(shap_values, feature_names, top_n=5)
        print(f"\nTop 5 features for {model_name}:")
        print(top_features.to_string(index=False))
    """
    pass