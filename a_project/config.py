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

# Create plot subdirectories
for directory in [CONFUSION_MATRICES_DIR, FEATURE_IMPORTANCE_DIR, 
                 ROC_CURVES_DIR, PERFORMANCE_METRICS_DIR]:
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
    # Add numpy array paths
    "X_train": os.path.join(PROCESSED_DATA_DIR, "X_train.npy"),
    "X_test": os.path.join(PROCESSED_DATA_DIR, "X_test.npy"),
    "y_train": os.path.join(PROCESSED_DATA_DIR, "y_train.npy"),
    "y_test": os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
}

# Feature sets
BASE_FEATURES = [
    'elo_diff',
    'away_last_n_win_pct', 'home_last_n_win_pct',
    'away_back_to_back', 'home_back_to_back',
    'away_vs_home_win_pct'
]

FOUR_FACTORS_FEATURES = [
    'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
    'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp'
]

DIFFERENTIAL_FEATURES = [
    'eFGp_diff', 'FTr_diff', 'ORBp_diff', 'TOVp_diff'
]

# Combined feature set
ALL_FEATURES = BASE_FEATURES + FOUR_FACTORS_FEATURES + DIFFERENTIAL_FEATURES

# Training parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42