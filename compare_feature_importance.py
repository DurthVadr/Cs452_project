"""
Compare feature importance between the simple and complex NBA prediction approaches.

This script loads models from both approaches, extracts feature importance data,
and creates visualizations to compare them.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

# Set up matplotlib for better visualization
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Ensure the output directory exists
os.makedirs('comparison_plots', exist_ok=True)

def load_model_and_features(model_path, feature_path=None):
    """
    Load a model and its associated features.

    Args:
        model_path: Path to the model file
        feature_path: Path to the feature names file (optional)

    Returns:
        model: The loaded model
        features: List of feature names
    """
    try:
        model = joblib.load(model_path)

        # Load features if path is provided
        if feature_path and os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                features = [line.strip() for line in f.readlines()]
        else:
            features = None

        return model, features
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

def extract_feature_importance(model, features):
    """
    Extract feature importance from a model.

    Args:
        model: Trained model
        features: List of feature names

    Returns:
        importance_df: DataFrame with feature names and importance values
    """
    if model is None or features is None:
        return None

    # Extract feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't have feature importance attributes")
        return None

    # Check if lengths match
    if len(features) != len(importances):
        print(f"Warning: Feature length ({len(features)}) doesn't match importance length ({len(importances)})")
        # Use the shorter length
        min_length = min(len(features), len(importances))
        features = features[:min_length]
        importances = importances[:min_length]

    # Create DataFrame with feature names and importance values
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df

def plot_feature_importance_comparison(simple_importance, complex_importance, top_n=15):
    """
    Create a comparison plot of feature importance between simple and complex models.

    Args:
        simple_importance: DataFrame with feature importance for simple model
        complex_importance: DataFrame with feature importance for complex model
        top_n: Number of top features to display

    Returns:
        fig: The matplotlib figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # Plot simple model feature importance
    simple_top = simple_importance.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=simple_top, ax=ax1, palette='Blues_d')
    ax1.set_title('Simple Model Feature Importance', fontsize=16)
    ax1.set_xlabel('Importance', fontsize=14)
    ax1.set_ylabel('Feature', fontsize=14)

    # Plot complex model feature importance
    complex_top = complex_importance.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=complex_top, ax=ax2, palette='Reds_d')
    ax2.set_title('Complex Model Feature Importance', fontsize=16)
    ax2.set_xlabel('Importance', fontsize=14)
    ax2.set_ylabel('Feature', fontsize=14)

    plt.tight_layout()
    return fig

def plot_common_features(simple_importance, complex_importance, top_n=10):
    """
    Create a plot comparing importance of common features between models.

    Args:
        simple_importance: DataFrame with feature importance for simple model
        complex_importance: DataFrame with feature importance for complex model
        top_n: Number of top common features to display

    Returns:
        fig: The matplotlib figure
    """
    # Find common features
    common_features = set(simple_importance['Feature']).intersection(set(complex_importance['Feature']))

    # Create DataFrame with common features
    common_data = []

    for feature in common_features:
        simple_value = simple_importance[simple_importance['Feature'] == feature]['Importance'].values[0]
        complex_value = complex_importance[complex_importance['Feature'] == feature]['Importance'].values[0]

        common_data.append({
            'Feature': feature,
            'Simple Model': simple_value,
            'Complex Model': complex_value
        })

    # Create DataFrame from the list of dictionaries
    common_df = pd.DataFrame(common_data)

    if common_df.empty:
        print("No common features found between models")
        return None

    # Sort by average importance
    common_df['Average'] = (common_df['Simple Model'] + common_df['Complex Model']) / 2
    common_df = common_df.sort_values('Average', ascending=False).head(top_n)

    # Melt for easier plotting
    plot_df = pd.melt(common_df, id_vars=['Feature'], value_vars=['Simple Model', 'Complex Model'],
                     var_name='Model', value_name='Importance')

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='Feature', y='Importance', hue='Model', data=plot_df, ax=ax,
               palette=['#1f77b4', '#d62728'])

    ax.set_title('Feature Importance Comparison (Common Features)', fontsize=16)
    ax.set_xlabel('Feature', fontsize=14)
    ax.set_ylabel('Importance', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig

def main():
    try:
        # Define paths for simple approach (a_project)
        simple_rf_model_path = 'a_project/all_models/random_forest.pkl'
        simple_gb_model_path = 'a_project/all_models/gradient_boosting.pkl'
        simple_feature_path = 'a_project/data/processed/feature_names.txt'

        # Define paths for complex approach (main project)
        complex_rf_model_path = 'models/rf_model.pkl'
        complex_gb_model_path = 'models/gb_model.pkl'
        complex_feature_path = 'processed_data/feature_names.txt'

        # Load models and features
        print("Loading models and features...")
        simple_rf, simple_features = load_model_and_features(simple_rf_model_path, simple_feature_path)
        simple_gb, _ = load_model_and_features(simple_gb_model_path, simple_feature_path)

        complex_rf, complex_features = load_model_and_features(complex_rf_model_path, complex_feature_path)
        complex_gb, _ = load_model_and_features(complex_gb_model_path, complex_feature_path)

        # Check if we have valid models and features
        if simple_features is None:
            print(f"Could not load features from {simple_feature_path}. Checking for alternative...")
            # Try to find feature names in the model directory
            alt_feature_path = 'a_project/all_models/random_forest_features.txt'
            if os.path.exists(alt_feature_path):
                with open(alt_feature_path, 'r') as f:
                    simple_features = [line.strip() for line in f.readlines()]
                print(f"Loaded features from alternative path: {alt_feature_path}")

        if complex_features is None:
            print(f"Could not load features from {complex_feature_path}. Checking for alternative...")
            # Try to find feature names in the model directory
            alt_feature_path = 'models/random_forest_features.txt'
            if os.path.exists(alt_feature_path):
                with open(alt_feature_path, 'r') as f:
                    complex_features = [line.strip() for line in f.readlines()]
                print(f"Loaded features from alternative path: {alt_feature_path}")

        # Extract feature importance
        print("Extracting feature importance...")
        simple_rf_importance = extract_feature_importance(simple_rf, simple_features)
        simple_gb_importance = extract_feature_importance(simple_gb, simple_features)

        complex_rf_importance = extract_feature_importance(complex_rf, complex_features)
        complex_gb_importance = extract_feature_importance(complex_gb, complex_features)

        # Create comparison plots
        print("Creating comparison plots...")

        # Random Forest comparison
        if simple_rf_importance is not None and complex_rf_importance is not None:
            fig_rf = plot_feature_importance_comparison(simple_rf_importance, complex_rf_importance)
            fig_rf.savefig('comparison_plots/rf_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            print("Random Forest comparison plot saved.")

            fig_rf_common = plot_common_features(simple_rf_importance, complex_rf_importance)
            if fig_rf_common is not None:
                fig_rf_common.savefig('comparison_plots/rf_common_features_comparison.png', dpi=300, bbox_inches='tight')
                print("Random Forest common features plot saved.")
        else:
            print("Could not create Random Forest comparison plots due to missing data.")

        # Gradient Boosting comparison
        if simple_gb_importance is not None and complex_gb_importance is not None:
            fig_gb = plot_feature_importance_comparison(simple_gb_importance, complex_gb_importance)
            fig_gb.savefig('comparison_plots/gb_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            print("Gradient Boosting comparison plot saved.")

            fig_gb_common = plot_common_features(simple_gb_importance, complex_gb_importance)
            if fig_gb_common is not None:
                fig_gb_common.savefig('comparison_plots/gb_common_features_comparison.png', dpi=300, bbox_inches='tight')
                print("Gradient Boosting common features plot saved.")
        else:
            print("Could not create Gradient Boosting comparison plots due to missing data.")

        print("Feature importance comparison completed.")

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
