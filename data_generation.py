"""
Advanced data generation module for the NBA prediction project.

This module implements various data generation and augmentation techniques
to enhance the training data beyond basic SMOTE, including:
1. Advanced SMOTE variants
2. Feature-aware data generation
3. Game-specific augmentation
4. Statistical bootstrapping
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.neighbors import NearestNeighbors
import logging
import joblib
import os
from scipy import stats

# Import project configuration
import config
from feature_config import get_feature_set

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

def apply_advanced_smote(X_train, y_train, method='smote'):
    """
    Apply advanced SMOTE variants to balance the training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: SMOTE variant to use ('smote', 'borderline', 'svm', 'adasyn')
        
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled labels
        class_dist_before: Class distribution before resampling
        class_dist_after: Class distribution after resampling
    """
    logger.info(f"Applying {method.upper()} to balance training data")
    
    # Check class distribution before resampling
    class_dist_before = pd.Series(y_train).value_counts(normalize=True)
    logger.info(f"Class distribution before resampling: {class_dist_before.to_dict()}")
    
    # Select resampling method
    if method.lower() == 'smote':
        resampler = SMOTE(**config.SMOTE_PARAMS)
    elif method.lower() == 'borderline':
        resampler = BorderlineSMOTE(**config.BORDERLINE_SMOTE_PARAMS)
    elif method.lower() == 'svm':
        resampler = SVMSMOTE(**config.SVM_SMOTE_PARAMS)
    elif method.lower() == 'adasyn':
        resampler = ADASYN(**config.ADASYN_PARAMS)
    else:
        logger.warning(f"Unknown method {method}, falling back to standard SMOTE")
        resampler = SMOTE(**config.SMOTE_PARAMS)
    
    # Apply resampling
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    
    # Check class distribution after resampling
    class_dist_after = pd.Series(y_resampled).value_counts(normalize=True)
    logger.info(f"Class distribution after resampling: {class_dist_after.to_dict()}")
    logger.info(f"Training data shape after resampling: {X_resampled.shape}")
    
    return X_resampled, y_resampled, class_dist_before, class_dist_after

def generate_feature_aware_samples(X, y, feature_correlations, n_samples=100):
    """
    Generate synthetic samples that respect feature correlations.
    
    Args:
        X: Original feature matrix
        y: Original labels
        feature_correlations: Dictionary of feature correlation matrices by class
        n_samples: Number of samples to generate per class
        
    Returns:
        X_new: Generated feature matrix
        y_new: Generated labels
    """
    logger.info(f"Generating {n_samples} feature-aware samples per class")
    
    X_new_list = []
    y_new_list = []
    
    # Generate samples for each class
    for class_label in np.unique(y):
        # Get samples for this class
        X_class = X[y == class_label]
        
        # Skip if not enough samples
        if len(X_class) < 5:
            logger.warning(f"Not enough samples for class {class_label}, skipping")
            continue
            
        # Get correlation matrix for this class
        if class_label in feature_correlations:
            corr_matrix = feature_correlations[class_label]
        else:
            # Calculate correlation matrix if not provided
            corr_matrix = np.corrcoef(X_class, rowvar=False)
            
        # Generate multivariate normal samples
        try:
            # Calculate mean and covariance
            mean_vector = np.mean(X_class, axis=0)
            cov_matrix = np.cov(X_class, rowvar=False)
            
            # Generate samples
            new_samples = np.random.multivariate_normal(
                mean_vector, 
                cov_matrix, 
                size=n_samples
            )
            
            # Add to lists
            X_new_list.append(new_samples)
            y_new_list.append(np.full(n_samples, class_label))
            
        except np.linalg.LinAlgError:
            logger.warning(f"Covariance matrix not positive definite for class {class_label}")
            # Fall back to independent sampling
            new_samples = np.zeros((n_samples, X_class.shape[1]))
            for j in range(X_class.shape[1]):
                # Sample from the empirical distribution
                new_samples[:, j] = np.random.choice(X_class[:, j], size=n_samples)
            
            X_new_list.append(new_samples)
            y_new_list.append(np.full(n_samples, class_label))
    
    # Combine all generated samples
    if X_new_list:
        X_new = np.vstack(X_new_list)
        y_new = np.concatenate(y_new_list)
        logger.info(f"Generated {len(X_new)} feature-aware samples")
        return X_new, y_new
    else:
        logger.warning("No feature-aware samples generated")
        return np.empty((0, X.shape[1])), np.empty(0)

def generate_game_specific_samples(train_data, features, n_samples=100):
    """
    Generate synthetic samples based on game-specific patterns.
    
    This function creates synthetic NBA games by:
    1. Sampling team pairs that have played against each other
    2. Adjusting their statistics based on historical patterns
    3. Creating new synthetic matchups
    
    Args:
        train_data: Original training DataFrame with all game data
        features: List of feature names
        n_samples: Number of samples to generate
        
    Returns:
        X_new: Generated feature matrix
        y_new: Generated labels
    """
    logger.info(f"Generating {n_samples} game-specific samples")
    
    # Check if we have team identifiers in the data
    if 'home_team' not in train_data.columns or 'away_team' not in train_data.columns:
        logger.warning("Team identifiers not found in data, cannot generate game-specific samples")
        return np.empty((0, len(features))), np.empty(0)
    
    # Get unique team pairs
    team_pairs = train_data[['home_team', 'away_team']].drop_duplicates()
    
    # Initialize arrays for new samples
    X_new = np.zeros((n_samples, len(features)))
    y_new = np.zeros(n_samples)
    
    # Generate samples
    for i in range(n_samples):
        # Randomly select a team pair
        pair_idx = np.random.randint(0, len(team_pairs))
        home_team = team_pairs.iloc[pair_idx]['home_team']
        away_team = team_pairs.iloc[pair_idx]['away_team']
        
        # Get historical games between these teams
        historical_games = train_data[
            (train_data['home_team'] == home_team) & 
            (train_data['away_team'] == away_team)
        ]
        
        # If no historical games, sample a random game
        if len(historical_games) == 0:
            sample_idx = np.random.randint(0, len(train_data))
            sample_game = train_data.iloc[sample_idx]
        else:
            # Sample a random historical game
            sample_idx = np.random.randint(0, len(historical_games))
            sample_game = historical_games.iloc[sample_idx]
        
        # Extract features and label
        X_sample = sample_game[features].values
        y_sample = sample_game['result']
        
        # Add random noise to features (±10%)
        noise = np.random.uniform(0.9, 1.1, size=len(features))
        X_new[i] = X_sample * noise
        
        # Determine outcome (80% same as original, 20% flipped)
        if np.random.random() < 0.8:
            y_new[i] = y_sample
        else:
            y_new[i] = 1 - y_sample  # Flip the outcome
    
    logger.info(f"Generated {n_samples} game-specific samples")
    return X_new, y_new

def bootstrap_minority_class(X, y, n_samples=None):
    """
    Apply statistical bootstrapping to the minority class.
    
    Args:
        X: Feature matrix
        y: Labels
        n_samples: Number of samples to generate (defaults to match majority class)
        
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled labels
    """
    logger.info("Applying statistical bootstrapping to minority class")
    
    # Identify majority and minority classes
    class_counts = pd.Series(y).value_counts()
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]
    
    # Set number of samples if not specified
    if n_samples is None:
        n_samples = class_counts[majority_class] - class_counts[minority_class]
    
    # Get minority class samples
    X_minority = X[y == minority_class]
    
    # Bootstrap minority class
    bootstrap_indices = np.random.choice(
        range(len(X_minority)), 
        size=n_samples, 
        replace=True
    )
    X_bootstrap = X_minority[bootstrap_indices]
    y_bootstrap = np.full(n_samples, minority_class)
    
    # Combine with original data
    X_resampled = np.vstack([X, X_bootstrap])
    y_resampled = np.concatenate([y, y_bootstrap])
    
    logger.info(f"Generated {n_samples} bootstrapped samples")
    return X_resampled, y_resampled

def generate_enhanced_training_data(X_train, y_train, train_data, features):
    """
    Generate enhanced training data using multiple techniques.
    
    Args:
        X_train: Original training features
        y_train: Original training labels
        train_data: Original training DataFrame
        features: List of feature names
        
    Returns:
        X_enhanced: Enhanced training features
        y_enhanced: Enhanced training labels
    """
    logger.info("Generating enhanced training data")
    
    # Apply advanced SMOTE
    X_smote, y_smote, _, _ = apply_advanced_smote(
        X_train, y_train, method=config.SMOTE_METHOD
    )
    
    # Calculate feature correlations by class
    feature_correlations = {}
    for class_label in np.unique(y_train):
        X_class = X_train[y_train == class_label]
        if len(X_class) > 1:  # Need at least 2 samples to calculate correlation
            feature_correlations[class_label] = np.corrcoef(X_class, rowvar=False)
    
    # Generate feature-aware samples
    X_feature_aware, y_feature_aware = generate_feature_aware_samples(
        X_train, y_train, feature_correlations, n_samples=config.FEATURE_AWARE_SAMPLES
    )
    
    # Generate game-specific samples
    X_game_specific, y_game_specific = generate_game_specific_samples(
        train_data, features, n_samples=config.GAME_SPECIFIC_SAMPLES
    )
    
    # Combine all generated data
    X_combined = []
    y_combined = []
    
    # Add SMOTE samples
    X_combined.append(X_smote)
    y_combined.append(y_smote)
    
    # Add feature-aware samples if any were generated
    if len(X_feature_aware) > 0:
        X_combined.append(X_feature_aware)
        y_combined.append(y_feature_aware)
    
    # Add game-specific samples if any were generated
    if len(X_game_specific) > 0:
        X_combined.append(X_game_specific)
        y_combined.append(y_game_specific)
    
    # Combine all data
    X_enhanced = np.vstack(X_combined)
    y_enhanced = np.concatenate(y_combined)
    
    logger.info(f"Enhanced training data shape: {X_enhanced.shape}")
    return X_enhanced, y_enhanced
