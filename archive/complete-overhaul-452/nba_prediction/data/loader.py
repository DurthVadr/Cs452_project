"""
Functions for loading raw and processed data.
"""
import pandas as pd
import numpy as np
import os

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)

def load_raw_dataset(dataset_name):
    """
    Load a raw dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        DataFrame with the loaded data
    """
    file_path = config.RAW_DATA.get(dataset_name)
    if not file_path:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    logger.info(f"Loading {dataset_name} from {file_path}")
    try:
        return pd.read_csv(file_path, index_col=0)
    except Exception as e:
        logger.error(f"Error loading data file {file_path}: {e}")
        raise

def load_all_raw_datasets():
    """
    Load all raw datasets defined in the configuration.
    
    Returns:
        Dictionary of DataFrames with dataset names as keys
    """
    datasets = {}
    for name in config.RAW_DATA.keys():
        datasets[name] = load_raw_dataset(name)
    
    logger.info(f"Loaded {len(datasets)} raw datasets")
    return datasets

def load_processed_dataset(dataset_name):
    """
    Load a processed dataset by name.
    
    Args:
        dataset_name: Name of the processed dataset to load
        
    Returns:
        DataFrame or numpy array with the loaded data
    """
    file_path = config.PROCESSED_DATA.get(dataset_name)
    if not file_path:
        raise ValueError(f"Unknown processed dataset name: {dataset_name}")
        
    logger.info(f"Loading {dataset_name} from {file_path}")
    try:
        if str(file_path).endswith('.npy'):
            return np.load(file_path)
        else:
            return pd.read_csv(file_path, index_col=0)
    except Exception as e:
        logger.error(f"Error loading processed data file {file_path}: {e}")
        raise
        
def load_feature_names():
    """
    Load feature names from the feature names file.
    
    Returns:
        List of feature names
    """
    feature_path = config.PROCESSED_DATA.get("feature_names")
    
    try:
        with open(feature_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        logger.error(f"Error loading feature names from {feature_path}: {e}")
        # If feature names file is not available, return default features
        return config.ALL_FEATURES