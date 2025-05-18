"""
Report generation pipeline for NBA game prediction.

This script:
1. Loads evaluation results
2. Generates final reports with visualizations
3. Creates summary tables and plots
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists, save_figure
from nba_prediction.models.registry import ModelRegistry
from nba_prediction.reporting.report_generator import (
    generate_model_evaluation_report,
    generate_html_report,
    generate_model_summary_table,
    generate_feature_importance_report
)
from nba_prediction.data.loader import load_feature_names

logger = get_logger('report_generation')

def main():
    """Run the report generation pipeline."""
    logger.info("Starting report generation pipeline")
    
    # Step 1: Create output directories
    reports_dir = config.REPORTS_DIR
    ensure_directory_exists(reports_dir)
    
    # Step 2: Load evaluation results
    logger.info("Loading evaluation results")
    
    # Try loading from evaluation directory
    eval_file = os.path.join(config.OUTPUT_DIR, "evaluation", "metrics", "evaluation_results.json")
    ensemble_eval_file = os.path.join(config.OUTPUT_DIR, "ensemble", "evaluation_results.json")
    
    evaluation_results = {}
    
    # Load base model evaluation results
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                evaluation_results.update(json.load(f))
            logger.info(f"Loaded evaluation results from {eval_file}")
        except Exception as e:
            logger.warning(f"Failed to load evaluation results from {eval_file}: {e}")
    
    # Load ensemble evaluation results
    if os.path.exists(ensemble_eval_file):
        try:
            with open(ensemble_eval_file, 'r') as f:
                ensemble_results = json.load(f)
                # Only add results not already in base evaluation results
                for model, metrics in ensemble_results.items():
                    if model not in evaluation_results:
                        evaluation_results[model] = metrics
            logger.info(f"Loaded ensemble evaluation results from {ensemble_eval_file}")
        except Exception as e:
            logger.warning(f"Failed to load ensemble evaluation results from {ensemble_eval_file}: {e}")
    
    # If no evaluation results, try fetching from model registry
    if not evaluation_results:
        logger.warning("No evaluation results found in files, attempting to fetch from model registry")
        registry = ModelRegistry()
        for model_name in registry.list_models():
            try:
                model_info = registry.get_model_info(model_name)
                if "accuracy" in model_info:
                    evaluation_results[model_name] = {"accuracy": model_info["accuracy"]}
                    # Add any other metrics if available
                    for key, value in model_info.items():
                        if key not in ["version", "path", "created_at", "artifacts"] and isinstance(value, (int, float)):
                            evaluation_results[model_name][key] = value
            except Exception as e:
                logger.warning(f"Failed to get info for {model_name}: {e}")
    
    if not evaluation_results:
        logger.error("No evaluation results found, cannot generate reports")
        sys.exit(1)
    
    # Step 3: Generate evaluation report
    logger.info("Generating evaluation report")
    try:
        report_path = generate_model_evaluation_report(evaluation_results, report_path=reports_dir)
        logger.info(f"Generated evaluation report: {report_path}")
        
        # Convert to HTML if possible
        html_path = generate_html_report(report_path)
        if html_path:
            logger.info(f"Generated HTML report: {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {e}")
    
    # Step 4: Generate feature importance visualizations
    logger.info("Generating feature importance visualizations")
    feature_names = load_feature_names()
    
    # Load models that might have feature importances
    registry = ModelRegistry()
    feature_models = {}
    
    for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
        try:
            if model_name in registry.list_models():
                feature_models[model_name] = registry.load_model(model_name)
        except Exception as e:
            logger.warning(f"Failed to load {model_name} for feature importance: {e}")
    
    if feature_models:
        try:
            feature_plots = generate_feature_importance_report(
                feature_models, feature_names, 
                output_path=os.path.join(reports_dir, 'feature_importance')
            )
            logger.info(f"Generated feature importance visualizations for {len(feature_plots)} models")
        except Exception as e:
            logger.error(f"Failed to generate feature importance visualizations: {e}")
    
    # Step 5: Generate final model comparison visualization
    logger.info("Generating final model comparison")
    
    try:
        # Extract accuracies
        accuracies = {model: results.get('accuracy', 0) for model, results in evaluation_results.items()}
        
        # Create DataFrame for better sorting
        acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['accuracy'])
        acc_df = acc_df.sort_values('accuracy', ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define colors for different model types
        colors = []
        for model in acc_df.index:
            if model == 'elo':
                colors.append('lightcoral')
            elif model.startswith('voting') or model.startswith('stacking'):
                colors.append('gold')
            elif model == 'upset_specialized':
                colors.append('green')
            else:
                colors.append('royalblue')
        
        # Plot bars
        bars = ax.bar(acc_df.index, acc_df['accuracy'], color=colors)
        
        # Add labels and titles
        ax.set_title('Final Model Accuracy Comparison', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_ylim(0.5, 1.0)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, "final_model_comparison", reports_dir)
        plt.close(fig)
        
        logger.info("Generated final model comparison visualization")
    except Exception as e:
        logger.error(f"Failed to generate final model comparison: {e}")
    
    # Step 6: Generate executive summary
    logger.info("Generating executive summary")
    
    # Find best model
    if evaluation_results:
        best_model = max(evaluation_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
        best_accuracy = evaluation_results[best_model].get('accuracy', 0)
    else:
        best_model = "Unknown"
        best_accuracy = 0
    
    # Create executive summary
    summary_path = os.path.join(reports_dir, "executive_summary.md")
    try:
        with open(summary_path, 'w') as f:
            f.write("# NBA Game Prediction: Executive Summary\n\n")
            f.write("## Project Overview\n\n")
            f.write("This project developed a machine learning system for predicting NBA game outcomes ")
            f.write("with a special focus on improving predictions of upsets (where underdogs win).\n\n")
            
            f.write("## Key Results\n\n")
            f.write(f"- **Best Model:** {best_model} with {best_accuracy:.4f} accuracy\n")
            f.write(f"- **Models Evaluated:** {len(evaluation_results)}\n")
            
            # Add base vs ensemble comparison if both are available
            base_models = [m for m in evaluation_results.keys() if m in ['logistic_regression', 'random_forest', 'gradient_boosting', 'elo']]
            ensemble_models = [m for m in evaluation_results.keys() if m not in base_models]
            
            if base_models and ensemble_models:
                base_avg = sum(evaluation_results[m].get('accuracy', 0) for m in base_models) / len(base_models)
                ensemble_avg = sum(evaluation_results[m].get('accuracy', 0) for m in ensemble_models) / len(ensemble_models)
                improvement = (ensemble_avg - base_avg) * 100
                
                f.write(f"- **Average Ensemble Improvement:** {improvement:.2f}% over base models\n\n")
            
            f.write("## Key Features\n\n")
            f.write("The most predictive features for NBA game outcomes include:\n\n")
            f.write("- Team ELO rating differences\n")
            f.write("- Recent team performance metrics\n")
            f.write("- Four Factors statistics (shooting efficiency, rebounding, etc.)\n")
            f.write("- Back-to-back game indicators\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. Use the ensemble approach for the most accurate predictions\n")
            f.write("2. Pay special attention to fatigue factors (back-to-back games) for upset prediction\n")
            f.write("3. Continue collecting more detailed in-game statistics for future modeling\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Deploy model as a real-time prediction service\n")
            f.write("2. Develop player-level factors to account for injuries and lineup changes\n")
            f.write("3. Incorporate live betting odds for comparative analysis\n\n")
            
            f.write("*For detailed model evaluations and visualizations, see the full report.*\n")
        
        # Generate HTML version if possible
        html_summary = generate_html_report(summary_path)
        if html_summary:
            logger.info(f"Generated HTML executive summary: {html_summary}")
        
        logger.info(f"Generated executive summary: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to generate executive summary: {e}")
    
    logger.info("Report generation completed")
    logger.info(f"All reports saved to {reports_dir}")

if __name__ == "__main__":
    main()