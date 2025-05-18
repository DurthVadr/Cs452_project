"""
Data preparation pipeline for NBA game prediction.

This script:
1. Loads raw data
2. Processes and cleans the data
3. Generates features
4. Splits into training and test sets
5. Saves processed datasets
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.data.loader import load_raw_dataset, load_all_raw_datasets
from nba_prediction.data.processor import process_team_data
from nba_prediction.data.feature_engineering import engineer_all_features

logger = get_logger('data_preparation')

def main():
    """Run the data preparation pipeline."""
    logger.info("Starting data preparation pipeline")
    
    # Create output directories
    ensure_directory_exists(config.PROCESSED_DATA_DIR)
    
    # Step 1: Load raw data
    logger.info("Loading raw data")
    try:
        game_info = load_raw_dataset('game_info')
        logger.info(f"Loaded game_info with shape {game_info.shape}")
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        sys.exit(1)
    
    # Step 2: Filter by target season
    logger.info(f"Filtering data for target season {config.TARGET_SEASON}")
    season_data = game_info[game_info['season'] == config.TARGET_SEASON]
    if len(season_data) == 0:
        logger.error(f"No data found for season {config.TARGET_SEASON}")
        available_seasons = sorted(game_info['season'].unique())
        logger.info(f"Available seasons: {available_seasons}")
        logger.info(f"Using most recent season: {available_seasons[-1]}")
        season_data = game_info[game_info['season'] == available_seasons[-1]]
    
    logger.info(f"Selected season data shape: {season_data.shape}")
    
    # Step 3: Process team data
    logger.info("Processing team data")
    processed_team_data = process_team_data(season=config.TARGET_SEASON)
    
    # Step 4: Engineer features
    logger.info("Engineering features")
    combined_data = engineer_all_features(season_data)
    logger.info(f"Combined data with features shape: {combined_data.shape}")
    
    # Save combined data
    combined_data.to_csv(config.PROCESSED_DATA['combined_data'])
    
    # Step 5: Split data into training and test sets
    logger.info("Splitting data into training and test sets")
    
    # Sort by date for time-based split
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    combined_data = combined_data.sort_values('date')
    
    # Split based on configuration ratio
    train_size = int(len(combined_data) * config.TRAIN_TEST_SPLIT_RATIO)
    train_data = combined_data.iloc[:train_size]
    test_data = combined_data.iloc[train_size:]
    
    logger.info(f"Training set: {train_data.shape}, Test set: {test_data.shape}")
    
    # Save train and test sets
    train_data.to_csv(config.PROCESSED_DATA['train_data'])
    test_data.to_csv(config.PROCESSED_DATA['test_data'])
    
    # Step 6: Prepare feature matrices for modeling
    logger.info("Preparing feature matrices for modeling")
    
    # Use the features defined in config
    features = config.ALL_FEATURES
    
    # Create X and y for training and testing
    X_train = train_data[features].values
    y_train = train_data['result'].values
    X_test = test_data[features].values
    y_test = test_data['result'].values
    
    # Save feature matrices and targets
    np.save(config.PROCESSED_DATA['X_train'], X_train)
    np.save(config.PROCESSED_DATA['y_train'], y_train)
    np.save(config.PROCESSED_DATA['X_test'], X_test)
    np.save(config.PROCESSED_DATA['y_test'], y_test)
    
    # Save feature names for reference
    with open(config.PROCESSED_DATA['feature_names'], 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    
    logger.info("Data preparation completed successfully")
    logger.info(f"Processed data saved to {config.PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()