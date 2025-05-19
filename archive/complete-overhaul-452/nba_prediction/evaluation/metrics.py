"""
Evaluation metrics for NBA prediction models.
"""
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.config import config

from nba_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)

def calculate_basic_metrics(y_true, y_pred, prefix=None):
    """
    Calculate basic classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    # Handle prefix
    prefix = f"{prefix}_" if prefix else ""
    
    # Calculate metrics
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

def calculate_upset_metrics(y_true, y_pred, favorites):
    """
    Calculate metrics specifically for upset predictions.
    
    Args:
        y_true: Array of true outcomes
        y_pred: Array of predicted outcomes
        favorites: Array indicating the favorite team (same format as y_true/y_pred)
        
    Returns:
        Dictionary of upset metrics
    """
    import numpy as np
    
    # An upset occurs when the underdog (non-favorite) wins
    actual_upsets = (favorites != y_true)
    predicted_upsets = (favorites != y_pred)
    
    # Calculate upset metrics
    actual_upset_count = actual_upsets.sum()
    pred_upset_count = predicted_upsets.sum()
    
    # Correct upset predictions
    correct_upset_preds = (predicted_upsets & actual_upsets)
    correct_upset_count = correct_upset_preds.sum()
    
    # Calculate metrics (handling division by zero)
    upset_accuracy = (y_pred == y_true)[actual_upsets].mean() if actual_upset_count > 0 else 0
    upset_precision = correct_upset_count / pred_upset_count if pred_upset_count > 0 else 0
    upset_recall = correct_upset_count / actual_upset_count if actual_upset_count > 0 else 0
    upset_f1 = 2 * (upset_precision * upset_recall) / (upset_precision + upset_recall) if (upset_precision + upset_recall) > 0 else 0
    
    logger.info(f"Upset statistics - Actual: {actual_upset_count}, Predicted: {pred_upset_count}, Correct: {correct_upset_count}")
    
    return {
        'upset_accuracy': float(upset_accuracy),
        'upset_precision': float(upset_precision),
        'upset_recall': float(upset_recall),
        'upset_f1': float(upset_f1)
    }

def evaluate_model_comprehensive(model, X, y, full_data=None, prefix=None):
    """
    Evaluate model with comprehensive metrics including upset prediction.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        full_data: Complete dataframe with favorites information
        prefix: Prefix for metric names
        
    Returns:
        metrics, predictions, probabilities
    """
    
    prefix = prefix + "_" if prefix else ""
    
    # Make predictions
    predictions = model.predict(X)
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except:
        probabilities = None
    
    # Calculate basic metrics
    metrics = {
        f'{prefix}accuracy': accuracy_score(y, predictions),
        f'{prefix}precision': precision_score(y, predictions, average='binary', zero_division=0),
        f'{prefix}recall': recall_score(y, predictions, average='binary', zero_division=0),
        f'{prefix}f1': f1_score(y, predictions, average='binary', zero_division=0)
    }
    
    # Add ROC AUC if probabilities are available
    if probabilities is not None:
        metrics[f'{prefix}roc_auc'] = roc_auc_score(y, probabilities)
    
    # Add upset metrics if favorites information is available
    if full_data is not None and 'favorite' in full_data.columns:
        upset_metrics = calculate_upset_metrics(y, predictions, full_data['favorite'].values)
        
        # Add prefix to upset metrics
        metrics.update({f"{prefix}{k}": v for k, v in upset_metrics.items()})
    
    # Instead of storing large arrays in the metrics dictionary,
    # save them as separate files and store only the paths
    predictions_dir = os.path.join(config.OUTPUT_DIR, "predictions")
    ensure_directory_exists(predictions_dir)
    
    # Save predictions to file
    if prefix:
        predictions_file = os.path.join(predictions_dir, f"{prefix.rstrip('_')}_predictions.npy")
        np.save(predictions_file, predictions)
        metrics[f'{prefix}predictions_path'] = predictions_file
    else:
        predictions_file = os.path.join(predictions_dir, "model_predictions.npy")
        np.save(predictions_file, predictions)
        metrics['predictions_path'] = predictions_file
    
    # Save probabilities if available
    if probabilities is not None:
        if prefix:
            proba_file = os.path.join(predictions_dir, f"{prefix.rstrip('_')}_probabilities.npy")
            np.save(proba_file, probabilities)
            metrics[f'{prefix}probabilities_path'] = proba_file
        else:
            proba_file = os.path.join(predictions_dir, "model_probabilities.npy")
            np.save(proba_file, probabilities)
            metrics['probabilities_path'] = proba_file
    
    return metrics, predictions, probabilities