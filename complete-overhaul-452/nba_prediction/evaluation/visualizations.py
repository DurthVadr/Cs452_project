"""
Visualization functions for model evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import save_figure

logger = get_logger(__name__)

def plot_confusion_matrix(y_true, y_pred, class_names=None, title=None, ax=None):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes (default: ['Away Win', 'Home Win'])
        title: Title for the plot
        ax: Matplotlib axis for the plot
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
        
    # Default class names
    if class_names is None:
        class_names = ['Away Win', 'Home Win']
        
    # Default title
    if title is None:
        title = 'Confusion Matrix'
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    return fig, ax

def plot_roc_curve(y_true, y_proba, model_name=None, ax=None):
    """
    Plot a ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        ax: Matplotlib axis for the plot
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    label = f'{model_name} (AUC = {roc_auc:.3f})' if model_name else f'AUC = {roc_auc:.3f}'
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    return fig, ax

def plot_precision_recall_curve(y_true, y_proba, model_name=None, ax=None):
    """
    Plot a precision-recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        ax: Matplotlib axis for the plot
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    label = f'{model_name} (AUC = {pr_auc:.3f})' if model_name else f'AUC = {pr_auc:.3f}'
    ax.plot(recall, precision, lw=2, label=label)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    
    return fig, ax

def plot_feature_importance(model, feature_names, top_n=20, title=None, ax=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Title for the plot
        ax: Matplotlib axis for the plot
        
    Returns:
        Figure and axis objects
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't have feature_importances_ attribute")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, top_n * 0.5))
    else:
        fig = ax.figure
        
    # Default title
    if title is None:
        title = 'Feature Importance'
        
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Plot feature importance
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax

def plot_accuracy_by_category(metrics_df, category_col, metric_col='accuracy', title=None, ax=None):
    """
    Plot accuracy by category.
    
    Args:
        metrics_df: DataFrame with metrics by category
        category_col: Column name for categories
        metric_col: Column name for the metric to plot
        title: Title for the plot
        ax: Matplotlib axis for the plot
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
        
    # Default title
    if title is None:
        title = f'{metric_col.capitalize()} by {category_col.capitalize()}'
    
    # Plot accuracy by category
    sns.barplot(x=category_col, y=metric_col, data=metrics_df, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(metric_col.capitalize())
    ax.set_xlabel(category_col.capitalize())
    
    # Add value labels on bars
    for i, v in enumerate(metrics_df[metric_col]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    if category_col == 'elo_diff_bin':
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
    plt.tight_layout()
    return fig, ax