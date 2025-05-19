"""
Central configuration for the NBA game prediction project.
Contains parameters, file paths, and feature definitions.
"""
import os
from pathlib import Path
import logging

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Raw data paths
RAW_DATA = {
    "game_info": os.path.join(DATA_DIR, "game_info.csv"),
    "team_stats": os.path.join(DATA_DIR, "team_stats.csv"),
    "team_factor_10": os.path.join(DATA_DIR, "team_factor_10.csv"),
    "team_factor_20": os.path.join(DATA_DIR, "team_factor_20.csv"),
    "team_factor_30": os.path.join(DATA_DIR, "team_factor_30.csv"),
    "team_full_10": os.path.join(DATA_DIR, "team_full_10.csv"),
    "team_full_20": os.path.join(DATA_DIR, "team_full_20.csv"),
    "team_full_30": os.path.join(DATA_DIR, "team_full_30.csv")
}

# Processed data paths
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

PROCESSED_DATA = {
    "combined_data": os.path.join(PROCESSED_DATA_DIR, "combined_data.csv"),
    "train_data": os.path.join(PROCESSED_DATA_DIR, "train_data.csv"),
    "test_data": os.path.join(PROCESSED_DATA_DIR, "test_data.csv"),
    "X_train": os.path.join(PROCESSED_DATA_DIR, "X_train.npy"),
    "X_test": os.path.join(PROCESSED_DATA_DIR, "X_test.npy"),
    "y_train": os.path.join(PROCESSED_DATA_DIR, "y_train.npy"),
    "y_test": os.path.join(PROCESSED_DATA_DIR, "y_test.npy"),
    "feature_names": os.path.join(PROCESSED_DATA_DIR, "feature_names.txt")
}

# Season config
TARGET_SEASON = 1819  # 2018-2019 season
TRAIN_TEST_SPLIT_RATIO = 0.8

# Feature definitions
BASE_FEATURES = [
    'elo_diff',
    'away_last_n_win_pct', 'home_last_n_win_pct',
    'away_back_to_back', 'home_back_to_back',
    'away_vs_home_win_pct',
    'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
    'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp',
]

DIFFERENTIAL_FEATURES = [
    'eFGp_diff', 'FTr_diff', 'ORBp_diff', 'TOVp_diff',
]

INTERACTION_FEATURES = [
    'h_eFGp_x_TOVp', 'a_eFGp_x_TOVp',
    'h_eFGp_x_ORBp', 'a_eFGp_x_ORBp',
]

MOMENTUM_FEATURES = [
    'away_streak', 'home_streak',
    'away_weighted_win_pct', 'home_weighted_win_pct'
]

UPSET_FEATURES = [
    'elo_diff_abs', 'favorite_back_to_back', 'underdog_back_to_back'
]

# Feature sets for different models
ALL_FEATURES = BASE_FEATURES + DIFFERENTIAL_FEATURES + INTERACTION_FEATURES + MOMENTUM_FEATURES
UPSET_MODEL_FEATURES = BASE_FEATURES + ['elo_diff_abs', 'favorite_back_to_back', 'underdog_back_to_back']

# ELO configuration
ELO_PARAMS = {
    "k_factor_range": [10, 15, 20, 25, 30, 35, 40],
    "home_advantage_range": [50, 75, 100, 125, 150, 175, 200],
    "initial_rating": 1500
}

# Model parameters
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 0.001,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42
    }
}

# Ensemble parameters
ENSEMBLE_PARAMS = {
    "voting_weights": {
        "logistic_regression": 1,
        "random_forest": 1,
        "gradient_boosting": 2
    },
    "upset_threshold": 0.4,
    "stacking_cv": 5
}

# Logging configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}