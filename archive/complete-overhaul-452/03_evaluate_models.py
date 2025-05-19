"""
Model evaluation pipeline for NBA game prediction.

This script:
1. Loads trained models
2. Loads test data
3. Evaluates each model
4. Generates and saves evaluation metrics and visualizations
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists, save_figure
from nba_prediction.data.loader import load_processed_dataset, load_feature_names
from nba_prediction.models.registry import ModelRegistry
from nba_prediction.evaluation.metrics import evaluate_model_comprehensive, calculate_upset_metrics
from nba_prediction.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_accuracy_by_category
)

logger = get_logger('model_evaluation')

def main():
    """Run the model evaluation pipeline."""
    logger.info("Starting model evaluation pipeline")
    
    # Step 1: Prepare output directories
    evaluation_dir = os.path.join(config.OUTPUT_DIR, "evaluation")
    plots_dir = os.path.join(evaluation_dir, "plots")
    metrics_dir = os.path.join(evaluation_dir, "metrics")
    
    ensure_directory_exists(evaluation_dir)
    ensure_directory_exists(plots_dir)
    ensure_directory_exists(metrics_dir)
    
    # Step 2: Load test data
    logger.info("Loading test data")
    try:
        X_test = load_processed_dataset('X_test')
        y_test = load_processed_dataset('y_test')
        test_data = load_processed_dataset('test_data')
        feature_names = load_feature_names()
        
        logger.info(f"Loaded test data with shape: {X_test.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)
    
    # Step 3: Load trained models
    logger.info("Loading trained models")
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        logger.error("No trained models found in registry")
        sys.exit(1)
    
    logger.info(f"Found models: {', '.join(models)}")
    
    # Step 4: Evaluate each model
    logger.info("Evaluating models")
    evaluation_results = {}
    
    for model_name in models:
        logger.info(f"Evaluating {model_name}")
        try:
            # Load model
            model = registry.load_model(model_name)
            
            # Special handling for ELO model
            if model_name == 'elo':
                # For ELO, we need to evaluate using its own evaluation method
                from nba_prediction.models.elo import evaluate_elo_system
                
                accuracy, preds = evaluate_elo_system(model, test_data)
                
                # Create simple metrics
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': preds
                }

                # Add upset metrics for ELO
                if 'favorite' in test_data.columns:
                    upset_metrics = calculate_upset_metrics(y_test, preds, test_data['favorite'].values)
                    evaluation_results[model_name].update(upset_metrics)
                
                # Confusion matrix for ELO
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_confusion_matrix(y_test, preds, title=f"ELO Confusion Matrix", ax=ax)
                save_figure(fig, f"{model_name}_confusion_matrix", plots_dir)
                plt.close(fig)
                
            else:
                # Load scaler if available
                try:
                    scaler = registry.load_artifact(model_name, 'scaler')
                    X_test_scaled = scaler.transform(X_test)
                except:
                    logger.warning(f"No scaler found for {model_name}, using raw features")
                    X_test_scaled = X_test
                
                # Evaluate model
                metrics, predictions, probabilities = evaluate_model_comprehensive(model, X_test_scaled, y_test, test_data, prefix=model_name)
                evaluation_results[model_name] = metrics
                
                # Generate visualizations
                
                # Confusion matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_confusion_matrix(y_test, predictions, title=f"{model_name} Confusion Matrix", ax=ax)
                save_figure(fig, f"{model_name}_confusion_matrix", plots_dir)
                plt.close(fig)
                
                # ROC curve (if probabilities available)
                if probabilities is not None:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_roc_curve(y_test, probabilities, model_name=model_name, ax=ax)
                    save_figure(fig, f"{model_name}_roc_curve", plots_dir)
                    plt.close(fig)
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    plot_feature_importance(model, feature_names, ax=ax, title=f"{model_name} Feature Importance")
                    save_figure(fig, f"{model_name}_feature_importance", plots_dir)
                    plt.close(fig)
            
            logger.info(f"Completed evaluation for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    # Step 5: Save evaluation results
    logger.info("Saving evaluation results")
    
    # Save as JSON
    evaluation_file = os.path.join(metrics_dir, "evaluation_results.json")
    
    # Convert any numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    serializable_results = {}
    for model_name, metrics in evaluation_results.items():
        serializable_results[model_name] = {k: convert_numpy(v) for k, v in metrics.items() 
                                           if not isinstance(v, np.ndarray)}
    
    with open(evaluation_file, 'w') as f:
        json.dump(serializable_results, f, default=convert_numpy, indent=2)
    
    # Step 6: Generate comparison visualizations
    logger.info("Generating comparison visualizations")
    
    # Compare accuracies
    accuracies = {model: results.get('accuracy', 0) for model, results in evaluation_results.items()}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(accuracies.keys(), accuracies.values())
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0.5, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    save_figure(fig, "model_accuracy_comparison", plots_dir)
    plt.close(fig)
    
    # Compare upset prediction performance (if available)
    upset_metrics = {}
    for model, results in evaluation_results.items():
        if 'upset_accuracy' in results:
            upset_metrics[model] = {
                'accuracy': results.get('upset_accuracy', 0),
                'precision': results.get('upset_precision', 0),
                'recall': results.get('upset_recall', 0),
                'f1': results.get('upset_f1', 0)
            }
    
    if upset_metrics:
        # Create comparison dataframe
        upset_df = pd.DataFrame(upset_metrics)
        
        # Plot upset metrics
        fig, ax = plt.subplots(figsize=(14, 8))
        upset_df.T.plot(kind='bar', ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Upset Prediction Performance Comparison')
        ax.legend(title='Metric')
        plt.tight_layout()
        save_figure(fig, "upset_prediction_comparison", plots_dir)
        plt.close(fig)
    
    logger.info("Model evaluation completed")
    logger.info(f"Evaluation results saved to {evaluation_dir}")

if __name__ == "__main__":
    main()