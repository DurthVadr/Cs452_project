"""
Main configuration file for the NBA prediction project.

This file centralizes all configuration settings for the project,
including file paths, model parameters, and training settings.
"""

import os
import datetime

# Project directories
DATA_DIR = 'data'
PROCESSED_DATA_DIR = 'processed_data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots'
HTML_REPORTS_DIR = 'html_reports'
LOGS_DIR = 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, HTML_REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create subdirectories for better organization
PLOT_SUBDIRS = ['confusion_matrices', 'feature_importance', 'roc_curves', 'performance_metrics']
for subdir in PLOT_SUBDIRS:
    os.makedirs(os.path.join(PLOTS_DIR, subdir), exist_ok=True)

# Data files
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
COMBINED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'combined_data.csv')

# Model files
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')
LOGISTIC_MODEL_FILE = os.path.join(MODELS_DIR, 'logistic_model.pkl')
RF_MODEL_FILE = os.path.join(MODELS_DIR, 'rf_model.pkl')
GB_MODEL_FILE = os.path.join(MODELS_DIR, 'gb_model.pkl')
ENSEMBLE_MODEL_FILE = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
UPSET_MODEL_FILE = os.path.join(MODELS_DIR, 'upset_model.pkl')

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model hyperparameters
LOGISTIC_PARAMS = {
    'C': 0.1,
    'penalty': 'l2',
    'solver': 'liblinear',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
}

GB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

# SMOTE parameters
SMOTE_PARAMS = {
    'random_state': RANDOM_STATE,
    'k_neighbors': 5
}

# Advanced SMOTE variants parameters
BORDERLINE_SMOTE_PARAMS = {
    'random_state': RANDOM_STATE,
    'k_neighbors': 5,
    'kind': 'borderline-1'
}

SVM_SMOTE_PARAMS = {
    'random_state': RANDOM_STATE,
    'k_neighbors': 5,
    'm_neighbors': 10
}

ADASYN_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_neighbors': 5
}

# Data generation parameters
SMOTE_METHOD = 'smote'  # Options: 'smote', 'borderline', 'svm', 'adasyn'
FEATURE_AWARE_SAMPLES = 100  # Number of feature-aware samples to generate per class
GAME_SPECIFIC_SAMPLES = 200  # Number of game-specific samples to generate
USE_ENHANCED_DATA_GENERATION = True  # Whether to use enhanced data generation

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'logistic': 1,
    'random_forest': 2,
    'gradient_boosting': 2
}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(LOGS_DIR, f'nba_prediction_{datetime.datetime.now().strftime("%Y%m%d")}.log')

# Report configuration
REPORT_TITLE = 'NBA Game Prediction - Enhanced Model Report'
REPORT_FILE = os.path.join(HTML_REPORTS_DIR, f'prediction_report_{datetime.datetime.now().strftime("%Y%m%d")}.html')

# Import feature sets from feature_config
from feature_config import get_feature_set, FEATURE_SETS, DEFAULT_FEATURE_SET
