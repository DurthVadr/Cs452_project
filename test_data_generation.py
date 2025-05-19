"""
Test script for the enhanced data generation module.

This script loads the training data and applies the enhanced data generation
techniques to demonstrate their effectiveness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import project configuration
import config
from feature_config import get_feature_set
from data_generation import (
    apply_advanced_smote,
    generate_feature_aware_samples,
    generate_game_specific_samples,
    bootstrap_minority_class,
    generate_enhanced_training_data
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def load_data():
    """Load and prepare data for testing."""
    logger.info("Loading data for testing")
    
    try:
        # Load processed data
        train_data = pd.read_csv(config.TRAIN_DATA_FILE)
        
        # Get feature list
        features = get_feature_set('default')
        logger.info(f"Using {len(features)} features")
        
        # Create X and y for training
        X_train = train_data[features]
        y_train = train_data['result']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        logger.info(f"Data loaded successfully. Training set: {X_train.shape}")
        return X_train_scaled, y_train, train_data, features
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def visualize_data_distribution(X_original, y_original, X_enhanced, y_enhanced, features, method_name):
    """
    Visualize the distribution of original and enhanced data using PCA.
    
    Args:
        X_original: Original feature matrix
        y_original: Original labels
        X_enhanced: Enhanced feature matrix
        y_enhanced: Enhanced labels
        features: List of feature names
        method_name: Name of the enhancement method
    """
    logger.info(f"Visualizing data distribution for {method_name}")
    
    # Apply PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_original_pca = pca.fit_transform(X_original)
    X_enhanced_pca = pca.transform(X_enhanced)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot original data
    plt.scatter(
        X_original_pca[:, 0], 
        X_original_pca[:, 1], 
        c=y_original, 
        cmap='coolwarm',
        alpha=0.7, 
        marker='o',
        edgecolors='k',
        s=100,
        label='Original'
    )
    
    # Plot enhanced data
    plt.scatter(
        X_enhanced_pca[:, 0], 
        X_enhanced_pca[:, 1], 
        c=y_enhanced, 
        cmap='coolwarm',
        alpha=0.3, 
        marker='x',
        s=50,
        label='Enhanced'
    )
    
    # Add labels and legend
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA Visualization of {method_name}')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.join(config.PLOTS_DIR, 'data_generation'), exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, 'data_generation', f'{method_name.lower().replace(" ", "_")}_pca.png'))
    
    logger.info(f"PCA visualization saved for {method_name}")

def test_data_generation_methods():
    """Test different data generation methods and visualize results."""
    logger.info("Testing data generation methods")
    
    # Load data
    X_train, y_train, train_data, features = load_data()
    
    # Calculate feature correlations by class
    feature_correlations = {}
    for class_label in np.unique(y_train):
        X_class = X_train[y_train == class_label]
        if len(X_class) > 1:
            feature_correlations[class_label] = np.corrcoef(X_class, rowvar=False)
    
    # Test and visualize each method
    
    # 1. Standard SMOTE
    X_smote, y_smote, _, _ = apply_advanced_smote(X_train, y_train, method='smote')
    visualize_data_distribution(X_train, y_train, X_smote, y_smote, features, "Standard SMOTE")
    
    # 2. Borderline SMOTE
    X_borderline, y_borderline, _, _ = apply_advanced_smote(X_train, y_train, method='borderline')
    visualize_data_distribution(X_train, y_train, X_borderline, y_borderline, features, "Borderline SMOTE")
    
    # 3. Feature-aware generation
    X_feature_aware, y_feature_aware = generate_feature_aware_samples(
        X_train, y_train, feature_correlations, n_samples=100
    )
    if len(X_feature_aware) > 0:
        visualize_data_distribution(X_train, y_train, X_feature_aware, y_feature_aware, features, "Feature-Aware Generation")
    
    # 4. Game-specific generation
    X_game_specific, y_game_specific = generate_game_specific_samples(
        train_data, features, n_samples=100
    )
    if len(X_game_specific) > 0:
        visualize_data_distribution(X_train, y_train, X_game_specific, y_game_specific, features, "Game-Specific Generation")
    
    # 5. Bootstrap minority class
    X_bootstrap, y_bootstrap = bootstrap_minority_class(X_train, y_train)
    visualize_data_distribution(X_train, y_train, X_bootstrap, y_bootstrap, features, "Bootstrap Minority Class")
    
    # 6. Enhanced training data (combined approach)
    X_enhanced, y_enhanced = generate_enhanced_training_data(X_train, y_train, train_data, features)
    visualize_data_distribution(X_train, y_train, X_enhanced, y_enhanced, features, "Enhanced Training Data")
    
    # Print summary statistics
    print("\nData Generation Summary:")
    print(f"Original data shape: {X_train.shape}")
    print(f"SMOTE data shape: {X_smote.shape}")
    print(f"Borderline SMOTE data shape: {X_borderline.shape}")
    print(f"Feature-aware data shape: {X_feature_aware.shape}")
    print(f"Game-specific data shape: {X_game_specific.shape}")
    print(f"Bootstrap data shape: {X_bootstrap.shape}")
    print(f"Enhanced training data shape: {X_enhanced.shape}")
    
    # Class distribution
    print("\nClass Distribution:")
    print(f"Original: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    print(f"SMOTE: {pd.Series(y_smote).value_counts(normalize=True).to_dict()}")
    print(f"Borderline SMOTE: {pd.Series(y_borderline).value_counts(normalize=True).to_dict()}")
    print(f"Feature-aware: {pd.Series(y_feature_aware).value_counts(normalize=True).to_dict() if len(y_feature_aware) > 0 else 'N/A'}")
    print(f"Game-specific: {pd.Series(y_game_specific).value_counts(normalize=True).to_dict() if len(y_game_specific) > 0 else 'N/A'}")
    print(f"Bootstrap: {pd.Series(y_bootstrap).value_counts(normalize=True).to_dict()}")
    print(f"Enhanced: {pd.Series(y_enhanced).value_counts(normalize=True).to_dict()}")
    
    logger.info("Data generation testing completed")

if __name__ == "__main__":
    test_data_generation_methods()
