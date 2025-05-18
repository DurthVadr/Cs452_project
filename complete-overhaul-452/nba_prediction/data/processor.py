"""
Functions for processing and transforming NBA game data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.data.loader import load_raw_dataset

logger = get_logger(__name__)

def fill_missing_factor_data(df):
    """
    Fill missing values in factor data with team means or overall means.
    
    Args:
        df: DataFrame with factor data
        
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    # Fill missing values with mean for each team
    for col in ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']:
        if col in df.columns:
            # For away team stats
            if col.startswith('a_'):
                team_col = 'away_team'
            # For home team stats
            else:
                team_col = 'home_team'

            # Group by team and fill missing values with team mean
            team_means = df.groupby(team_col)[col].transform('mean')
            df[col] = df[col].fillna(team_means)

            # If still missing (new teams), fill with overall mean
            df[col] = df[col].fillna(df[col].mean())

    return df

def fill_missing_full_data(df):
    """
    Fill missing values in full stats data.
    
    Args:
        df: DataFrame with team full stats data
        
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    # Get all stat columns (excluding game info columns)
    stat_cols = [col for col in df.columns if col.startswith('a_') or col.startswith('h_')]

    # Fill missing values with mean for each team
    for col in stat_cols:
        # For away team stats
        if col.startswith('a_'):
            team_col = 'away_team'
        # For home team stats
        else:
            team_col = 'home_team'

        # Group by team and fill missing values with team mean
        team_means = df.groupby(team_col)[col].transform('mean')
        df[col] = df[col].fillna(team_means)

        # If still missing (new teams), fill with overall mean
        df[col] = df[col].fillna(df[col].mean())

    return df

def process_team_data(season=None):
    """
    Process team factor and full data, handling missing values.
    
    Args:
        season: Season identifier (defaults to TARGET_SEASON in config)
        
    Returns:
        Dictionary containing processed DataFrames
    """
    if season is None:
        season = config.TARGET_SEASON
        
    logger.info(f"Processing team data for season {season}")
    
    # Load raw data
    team_factor_10 = load_raw_dataset('team_factor_10')
    team_factor_20 = load_raw_dataset('team_factor_20')
    team_factor_30 = load_raw_dataset('team_factor_30')
    team_full_10 = load_raw_dataset('team_full_10')
    team_full_20 = load_raw_dataset('team_full_20')
    team_full_30 = load_raw_dataset('team_full_30')
    
    # Filter for specified season
    team_factor_10_season = team_factor_10[team_factor_10['season'] == season]
    team_factor_20_season = team_factor_20[team_factor_20['season'] == season]
    team_factor_30_season = team_factor_30[team_factor_30['season'] == season]
    team_full_10_season = team_full_10[team_full_10['season'] == season]
    team_full_20_season = team_full_20[team_full_20['season'] == season]
    team_full_30_season = team_full_30[team_full_30['season'] == season]
    
    # Process factor data
    logger.info("Processing factor data")
    team_factor_10_processed = fill_missing_factor_data(team_factor_10_season)
    team_factor_20_processed = fill_missing_factor_data(team_factor_20_season)
    team_factor_30_processed = fill_missing_factor_data(team_factor_30_season)
    
    # Process full data
    logger.info("Processing full data")
    team_full_10_processed = fill_missing_full_data(team_full_10_season)
    team_full_20_processed = fill_missing_full_data(team_full_20_season)
    team_full_30_processed = fill_missing_full_data(team_full_30_season)
    
    # Save processed data
    ensure_directory_exists(config.PROCESSED_DATA_DIR)
    team_factor_10_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_factor_10_processed.csv')
    team_factor_20_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_factor_20_processed.csv')
    team_factor_30_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_factor_30_processed.csv')
    team_full_10_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_full_10_processed.csv')
    team_full_20_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_full_20_processed.csv')
    team_full_30_processed.to_csv(Path(config.PROCESSED_DATA_DIR) / 'team_full_30_processed.csv')
    
    return {
        'team_factor_10': team_factor_10_processed,
        'team_factor_20': team_factor_20_processed,
        'team_factor_30': team_factor_30_processed,
        'team_full_10': team_full_10_processed,
        'team_full_20': team_full_20_processed,
        'team_full_30': team_full_30_processed
    }