"""
Functions to generate reports for model evaluation and analysis.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import markdown
import json
from datetime import datetime

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists, save_figure
from nba_prediction.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_accuracy_by_category
)

logger = get_logger(__name__)

def generate_model_summary_table(evaluation_results, output_format='markdown'):
    """
    Generate a summary table of model performance.
    
    Args:
        evaluation_results: Dictionary of evaluation results by model
        output_format: Format for the output ('markdown' or 'html')
        
    Returns:
        Table in the specified format
    """
    # Create DataFrame for model metrics
    models = list(evaluation_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    if not models:
        logger.warning("No models found in evaluation results")
        return "No models evaluated"
    
    # Get all metrics that are common across models
    common_metrics = set()
    for model_metrics in evaluation_results.values():
        common_metrics.update(model_metrics.keys())
    
    # Filter to standard metrics + model-specific ones
    all_metrics = [m for m in metrics if any(m in model_metrics for model_metrics in evaluation_results.values())]
    all_metrics += [m for m in common_metrics if m not in metrics and not m.endswith('_pred')]
    
    # Create DataFrame
    results_df = pd.DataFrame(index=models, columns=all_metrics)
    
    # Fill DataFrame with results
    for model, model_metrics in evaluation_results.items():
        for metric in all_metrics:
            if metric in model_metrics:
                results_df.loc[model, metric] = model_metrics[metric]
    
    # Sort by accuracy
    if 'accuracy' in results_df.columns:
        results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Format values
    formatted_df = results_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # Output in specified format
    if output_format == 'markdown':
        return formatted_df.to_markdown()
    elif output_format == 'html':
        return formatted_df.to_html(classes='table table-striped')
    else:
        return formatted_df

def generate_feature_importance_report(models, feature_names, output_path=None):
    """
    Generate and save feature importance plots for multiple models.
    
    Args:
        models: Dictionary of trained models
        feature_names: List of feature names
        output_path: Path to save the plots
        
    Returns:
        Dictionary mapping model names to feature importance plots
    """
    output_path = output_path or os.path.join(config.REPORTS_DIR, 'feature_importance')
    ensure_directory_exists(output_path)
    
    feature_plots = {}
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_') or (hasattr(model, 'coef_') and hasattr(model, 'classes_')):
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
                    
                # Sort features by importance
                indices = importances.argsort()[::-1]
                top_n = min(20, len(feature_names))  # Show top 20 features or all if less
                
                # Plot feature importances
                plt.barh(range(top_n), importances[indices][:top_n])
                plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
                plt.xlabel('Importance')
                plt.title(f'Feature Importance for {name}')
                
                # Save figure
                file_name = f"{name.replace(' ', '_').lower()}_importance"
                save_figure(fig, file_name, output_path)
                plt.close(fig)
                
                feature_plots[name] = os.path.join(output_path, f"{file_name}.png")
                
            except Exception as e:
                logger.error(f"Error generating feature importance plot for {name}: {e}")
    
    return feature_plots

def generate_model_evaluation_report(evaluation_results, report_path=None, report_name=None):
    """
    Generate a comprehensive evaluation report for models.
    
    Args:
        evaluation_results: Dictionary of evaluation results by model
        report_path: Path to save the report
        report_name: Name for the report file
        
    Returns:
        Path to the generated report
    """
    report_path = report_path or config.REPORTS_DIR
    ensure_directory_exists(report_path)
    
    report_name = report_name or f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_file = os.path.join(report_path, f"{report_name}.md")
    
    logger.info(f"Generating evaluation report at {report_file}")
    
    with open(report_file, 'w') as f:
        # Report header
        f.write("# NBA Game Prediction - Model Evaluation Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Model summary table
        f.write("## Model Performance Summary\n\n")
        f.write(generate_model_summary_table(evaluation_results))
        f.write("\n\n")
        
        # Individual model evaluations
        f.write("## Individual Model Evaluations\n\n")
        
        for model_name, metrics in evaluation_results.items():
            f.write(f"### {model_name}\n\n")
            
            # Basic metrics
            f.write("#### Performance Metrics\n\n")
            basic_metrics = {k: v for k, v in metrics.items() 
                            if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] and not isinstance(v, (list, dict))}
            
            if basic_metrics:
                metrics_df = pd.DataFrame(basic_metrics, index=['Value']).T
                f.write(metrics_df.to_markdown())
                f.write("\n\n")
            
            # Check for upset-specific metrics
            upset_metrics = {k: v for k, v in metrics.items() if k.startswith('upset_') and not isinstance(v, (list, dict))}
            
            if upset_metrics:
                f.write("#### Upset Prediction Performance\n\n")
                upset_df = pd.DataFrame(upset_metrics, index=['Value']).T
                f.write(upset_df.to_markdown())
                f.write("\n\n")
                
            # Add reference to visualizations if they exist
            f.write(f"*For detailed visualizations, see the visualization directory.*\n\n")
            
        # Conclusions
        f.write("## Conclusions\n\n")
        
        if evaluation_results:
            # Determine the best model based on accuracy
            best_model = max(evaluation_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
            best_accuracy = evaluation_results[best_model].get('accuracy', 0)
            
            f.write(f"The best performing model is **{best_model}** with an accuracy of {best_accuracy:.4f}.\n\n")
            
            # Look for best model for upset prediction
            if any('upset_accuracy' in metrics for metrics in evaluation_results.values()):
                best_upset_model = max(evaluation_results.items(), key=lambda x: x[1].get('upset_accuracy', 0))[0]
                best_upset_accuracy = evaluation_results[best_upset_model].get('upset_accuracy', 0)
                f.write(f"For upset prediction, the best model is **{best_upset_model}** with an upset accuracy of {best_upset_accuracy:.4f}.\n\n")
    
    logger.info(f"Evaluation report generated: {report_file}")
    return report_file

def generate_html_report(md_report_path):
    """
    Convert a Markdown report to HTML.
    
    Args:
        md_report_path: Path to the Markdown report
        
    Returns:
        Path to the generated HTML report
    """
    try:
        import markdown
        
        # Read Markdown file
        with open(md_report_path, 'r') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        # Add some basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NBA Game Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                .container {{ margin: 0 auto; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        html_path = md_report_path.replace('.md', '.html')
        with open(html_path, 'w') as f:
            f.write(styled_html)
        
        logger.info(f"HTML report generated: {html_path}")
        return html_path
        
    except ImportError:
        logger.warning("markdown module not available. Unable to generate HTML report.")
        return None