"""
Common utility functions for the NBA prediction project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the standard processed data files.
    
    Returns:
        Tuple containing (combined_data, train_data, test_data)
    """
    combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
    train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
    test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)
    
    return combined_data, train_data, test_data

def load_feature_names() -> List[str]:
    """
    Load feature names from the feature_names.txt file.
    
    Returns:
        List of feature names
    """
    with open('processed_data/feature_names.txt', 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    
    return features

def load_model_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load the prepared model data (X_train, y_train, X_test, y_test, feature_names).
    
    Returns:
        Tuple containing (X_train_scaled, y_train, X_test_scaled, y_test, feature_names)
    """
    X_train_scaled = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test_scaled = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
    feature_names = load_feature_names()
    
    return X_train_scaled, y_train, X_test_scaled, y_test, feature_names

def set_plot_style() -> None:
    """
    Set the default plot style for consistency across all scripts.
    """
    plt.style.use('ggplot')
    sns.set_theme(font_scale=1.2)
    
def format_percentage(value: float) -> str:
    """
    Format a value as a percentage string.
    
    Args:
        value: Value to format (0-1 range)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"
