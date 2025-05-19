"""
Common utility functions for the NBA prediction project.

This module provides centralized utility functions used across multiple scripts
in the NBA prediction project, ensuring consistency and reducing code duplication.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime

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

def set_plot_style(font_scale: float = 1.2, figure_size: Tuple[int, int] = (12, 8)) -> None:
    """
    Set the default plot style for consistency across all scripts.

    Args:
        font_scale: Scale factor for font sizes
        figure_size: Default figure size (width, height) in inches
    """
    plt.style.use('ggplot')
    sns.set_theme(font_scale=font_scale)
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['figure.dpi'] = 100

def format_percentage(value: float) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Value to format (0-1 range)

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"

def save_figure(fig: plt.Figure, filename: str, directories: List[str] = None, formats: List[str] = None) -> None:
    """
    Save a matplotlib figure to multiple directories and formats.

    Args:
        fig: Matplotlib figure to save
        filename: Base filename without extension
        directories: List of directories to save to (default: ['plots'])
        formats: List of formats to save in (default: ['png'])
    """
    if directories is None:
        directories = ['plots']

    if formats is None:
        formats = ['png']

    for directory in directories:
        ensure_directory_exists(directory)
        for fmt in formats:
            output_path = os.path.join(directory, f"{filename}.{fmt}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

def load_model(model_path: str, default_model=None):
    """
    Load a model from disk, with fallback to a default model if the file doesn't exist.

    Args:
        model_path: Path to the model file
        default_model: Default model to return if the file doesn't exist

    Returns:
        Loaded model or default model
    """
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return default_model
    else:
        return default_model

def get_timestamp() -> str:
    """
    Get a formatted timestamp string for use in filenames.

    Returns:
        Formatted timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_confusion_matrix_plot(y_true, y_pred, title: str = 'Confusion Matrix'):
    """
    Create a confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()

    return fig
