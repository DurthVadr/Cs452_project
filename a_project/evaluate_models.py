"""
Model evaluation script for NBA game prediction project.
Focuses on detailed performance analysis including upset prediction capability.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import joblib
from config import *

def load_data_and_models():
    """Load test data and trained models."""
    print("Loading data and models...")
    
    # Load test data
    X_test = np.load(PROCESSED_DATA_FILES['X_test'])
    y_test = np.load(PROCESSED_DATA_FILES['y_test'])
    test_data = pd.read_csv(PROCESSED_DATA_FILES['test_data'])
    
    # Load feature names
    with open(PROCESSED_DATA_FILES['feature_names'], 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Load all models
    models = {}
    for model_name in ['logistic', 'random_forest', 'gradient_boosting', 'ensemble']:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        models[model_name] = joblib.load(model_path)
    
    return X_test, y_test, test_data, feature_names, models

def evaluate_basic_metrics(model, X_test, y_test, is_upset):
    """Calculate basic classification metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # Calculate separate metrics for upsets
    if is_upset is not None:
        upset_mask = is_upset == 1
        metrics.update({
            'upset_accuracy': accuracy_score(y_test[upset_mask], y_pred[upset_mask]),
            'upset_precision': precision_score(y_test[upset_mask], y_pred[upset_mask], zero_division=0),
            'upset_recall': recall_score(y_test[upset_mask], y_pred[upset_mask], zero_division=0),
            'upset_f1': f1_score(y_test[upset_mask], y_pred[upset_mask], zero_division=0)
        })
    
    return metrics, y_pred

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(models, X_test, y_test, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, title, save_path):
    """Plot feature importance if available."""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def analyze_upset_predictions(model, X_test, y_test, is_upset, model_name):
    """Analyze model's performance on upset predictions."""
    y_pred = model.predict(X_test)
    
    # Separate upset games
    upset_mask = is_upset == 1
    X_test_upsets = X_test[upset_mask]
    y_test_upsets = y_test[upset_mask]
    y_pred_upsets = y_pred[upset_mask]
    
    # Calculate upset-specific metrics
    upset_metrics = {
        'total_upsets': upset_mask.sum(),
        'predicted_upsets': (y_pred != (is_upset == 0)).sum(),
        'correct_upset_predictions': np.sum((y_pred != (is_upset == 0)) & upset_mask),
        'upset_accuracy': accuracy_score(y_test_upsets, y_pred_upsets),
        'upset_precision': precision_score(y_test_upsets, y_pred_upsets, zero_division=0),
        'upset_recall': recall_score(y_test_upsets, y_pred_upsets, zero_division=0),
        'upset_f1': f1_score(y_test_upsets, y_pred_upsets, zero_division=0)
    }
    
    return upset_metrics

def analyze_nonupset_predictions(model, X_test, y_test, is_upset, model_name):
    """Analyze model's performance on non-upset predictions (favorite team wins)."""
    y_pred = model.predict(X_test)
    
    # Separate non-upset games
    nonupset_mask = is_upset == 0
    X_test_nonupsets = X_test[nonupset_mask]
    y_test_nonupsets = y_test[nonupset_mask]
    y_pred_nonupsets = y_pred[nonupset_mask]
    
    # Calculate non-upset-specific metrics
    nonupset_metrics = {
        'total_nonupsets': nonupset_mask.sum(),
        'predicted_nonupsets': (y_pred == (is_upset == 0)).sum(),
        'correct_nonupset_predictions': np.sum((y_pred == (is_upset == 0)) & nonupset_mask),
        'nonupset_accuracy': accuracy_score(y_test_nonupsets, y_pred_nonupsets),
        'nonupset_precision': precision_score(y_test_nonupsets, y_pred_nonupsets, zero_division=0),
        'nonupset_recall': recall_score(y_test_nonupsets, y_pred_nonupsets, zero_division=0),
        'nonupset_f1': f1_score(y_test_nonupsets, y_pred_nonupsets, zero_division=0)
    }
    
    return nonupset_metrics

def main():
    """Main execution function."""
    try:
        # Load data and models
        X_test, y_test, test_data, feature_names, models = load_data_and_models()
        
        # Prepare results storage
        all_metrics = {}
        
        # Evaluate each model
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Calculate basic metrics
            metrics, y_pred = evaluate_basic_metrics(
                model, X_test, y_test, test_data['is_upset'].values
            )
            all_metrics[name] = metrics
            
            # Plot confusion matrix
            plot_confusion_matrix(
                y_test, y_pred,
                f'Confusion Matrix - {name.title()}',
                os.path.join(CONFUSION_MATRICES_DIR, f'{name}_cm.png')
            )
            
            # Plot feature importance
            if name in ['random_forest', 'gradient_boosting']:
                plot_feature_importance(
                    model, feature_names,
                    f'Feature Importance - {name.title()}',
                    os.path.join(FEATURE_IMPORTANCE_DIR, f'{name}_importance.png')
                )
            
            # Analyze both upset and non-upset predictions
            upset_metrics = analyze_upset_predictions(
                model, X_test, y_test, 
                test_data['is_upset'].values, name
            )
            nonupset_metrics = analyze_nonupset_predictions(
                model, X_test, y_test,
                test_data['is_upset'].values, name
            )
            
            # Update metrics dictionary
            all_metrics[name].update(upset_metrics)
            all_metrics[name].update(nonupset_metrics)
        
        # Plot ROC curves for all models
        plot_roc_curve(
            models, X_test, y_test,
            os.path.join(ROC_CURVES_DIR, 'all_models_roc.png')
        )
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(all_metrics).round(4)
        metrics_df.to_csv(os.path.join(PERFORMANCE_METRICS_DIR, 'model_metrics.csv'))
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved in: {PLOTS_DIR}")
        
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    main()