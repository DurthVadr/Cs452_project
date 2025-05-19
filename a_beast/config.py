"""
Central configuration for NBA game prediction project.
Contains paths and feature definitions.
"""
import os
from pathlib import Path

# Directory Structure
BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "all_models")
PLOTS_DIR = os.path.join(BASE_DIR, "all_plots")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Plot subdirectories
CONFUSION_MATRICES_DIR = os.path.join(PLOTS_DIR, "confusion_matrices")
FEATURE_IMPORTANCE_DIR = os.path.join(PLOTS_DIR, "feature_importance")
ROC_CURVES_DIR = os.path.join(PLOTS_DIR, "roc_curves")
PERFORMANCE_METRICS_DIR = os.path.join(PLOTS_DIR, "performance_metrics")
TEAM_ANALYSIS_DIR = os.path.join(PLOTS_DIR, "team_analysis")
FOUR_FACTORS_DIR = os.path.join(PLOTS_DIR, "four_factors")
PERFORMANCE_ANALYSIS_DIR = os.path.join(PLOTS_DIR, "performance_analysis")

# Create all plot subdirectories
for directory in [
    CONFUSION_MATRICES_DIR, 
    FEATURE_IMPORTANCE_DIR,
    ROC_CURVES_DIR, 
    PERFORMANCE_METRICS_DIR,
    TEAM_ANALYSIS_DIR,
    FOUR_FACTORS_DIR,
    PERFORMANCE_ANALYSIS_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Data file paths
RAW_DATA_FILES = {
    "game_info": os.path.join(RAW_DATA_DIR, "game_info.csv"),
    "team_stats": os.path.join(RAW_DATA_DIR, "team_stats.csv"),
    "team_factor_10": os.path.join(RAW_DATA_DIR, "team_factor_10.csv"),
    "team_factor_20": os.path.join(RAW_DATA_DIR, "team_factor_20.csv"),
    "team_factor_30": os.path.join(RAW_DATA_DIR, "team_factor_30.csv")
}

# Processed data files
PROCESSED_DATA_FILES = {
    "combined_data": os.path.join(PROCESSED_DATA_DIR, "combined_data.csv"),
    "train_data": os.path.join(PROCESSED_DATA_DIR, "train_data.csv"),
    "test_data": os.path.join(PROCESSED_DATA_DIR, "test_data.csv"),
    "feature_names": os.path.join(PROCESSED_DATA_DIR, "feature_names.txt"),
    "X_train": os.path.join(PROCESSED_DATA_DIR, "X_train.npy"),
    "X_test": os.path.join(PROCESSED_DATA_DIR, "X_test.npy"),
    "y_train": os.path.join(PROCESSED_DATA_DIR, "y_train.npy"),
    "y_test": os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
}

# Enhanced Feature Sets
BASE_FEATURES = [
    'elo_diff',
    'away_last_n_win_pct', 'home_last_n_win_pct',
    'away_back_to_back', 'home_back_to_back',
    'away_vs_home_win_pct'
]

MOMENTUM_FEATURES = [
    'away_last_5_win_pct', 'home_last_5_win_pct',
    'away_last_15_win_pct', 'home_last_15_win_pct',
    'away_streak', 'home_streak'
]

FOUR_FACTORS_FEATURES = [
    'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
    'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp'
]

FOUR_FACTORS_ROLLING = [
    'a_eFGp_10', 'h_eFGp_10',
    'a_TOVp_10', 'h_TOVp_10',
    'a_ORBp_10', 'h_ORBp_10',
    'a_FTr_10', 'h_FTr_10'
]

DIFFERENTIAL_FEATURES = [
    'eFGp_diff', 'FTr_diff', 'ORBp_diff', 'TOVp_diff',
    'eFGp_roll_diff', 'FTr_roll_diff', 
    'ORBp_roll_diff', 'TOVp_roll_diff'
]

INTERACTION_FEATURES = [
    'elo_diff_back_to_back',  # Interaction between ELO diff and back-to-back
    'streak_vs_opp_streak',   # Interaction between team streaks
    'momentum_factor'         # Combined momentum indicator
]

# Combined feature set
ALL_FEATURES = (
    BASE_FEATURES + 
    MOMENTUM_FEATURES + 
    FOUR_FACTORS_FEATURES + 
    FOUR_FACTORS_ROLLING + 
    DIFFERENTIAL_FEATURES +
    INTERACTION_FEATURES
)

# Model configurations
MODEL_CONFIGS = {
    'base_models': ['logistic', 'random_forest', 'gradient_boosting'],
    'ensemble_types': ['voting', 'stacking', 'specialized_upset']
}

# Training parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

# Analysis parameters
ROLLING_WINDOWS = [5, 10, 15]  # Different windows for rolling statistics
UPSET_THRESHOLD = 0.2  # ELO difference threshold for upset definition