"""
Improved report generator with better formatting and complete metrics.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.models.registry import ModelRegistry
from nba_prediction.data.loader import load_feature_names

logger = get_logger(__name__)

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def get_friendly_model_name(technical_name):
    """Convert technical model names to user-friendly names."""
    name_mapping = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'elo': 'ELO Rating',
        'voting_ensemble': 'Voting Ensemble',
        'stacking_ensemble': 'Stacked Ensemble',
        'upset_specialized': 'Upset Specialist'
    }
    
    if technical_name in name_mapping:
        return name_mapping[technical_name]
    
    # Handle other cases with nice formatting
    return ' '.join(word.capitalize() for word in technical_name.replace('_', ' ').split())

def generate_improved_report(evaluation_results=None, output_path=None, report_name=None):
    """
    Generate an improved report with proper model names and complete metrics.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        output_path: Directory to save the report
        report_name: Name for the report file
        
    Returns:
        Path to the generated report
    """
    output_path = output_path or config.REPORTS_DIR
    ensure_directory_exists(output_path)
    
    report_name = report_name or f"improved_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_file = os.path.join(output_path, f"{report_name}.html")
    
    logger.info(f"Generating improved report at {report_file}")
    
    # Load evaluation results if not provided
    if evaluation_results is None:
        eval_path = os.path.join(config.OUTPUT_DIR, "evaluation", "metrics", "evaluation_results.json")
        
        if os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    evaluation_results = json.load(f)
                logger.info(f"Loaded evaluation results from {eval_path}")
            except Exception as e:
                logger.error(f"Failed to load evaluation results: {e}")
                evaluation_results = {}
        else:
            logger.error(f"Evaluation results file not found: {eval_path}")
            evaluation_results = {}
    
    # Convert technical model names to friendly names
    friendly_results = {}
    for model_name, metrics in evaluation_results.items():
        friendly_name = get_friendly_model_name(model_name)
        friendly_results[friendly_name] = metrics
    
    # Load models for feature importance
    registry = ModelRegistry()
    feature_models = {}
    model_map = {
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting',
        'logistic_regression': 'Logistic Regression'
    }
    
    for tech_name, friendly_name in model_map.items():
        try:
            if tech_name in registry.list_models():
                model = registry.load_model(tech_name)
                feature_models[friendly_name] = model
                logger.info(f"Loaded {friendly_name} model for feature importance")
        except Exception as e:
            logger.warning(f"Failed to load {tech_name} for feature importance: {e}")
    
    # Load feature names
    feature_names = load_feature_names()
    
    # Start writing the HTML report
    with open(report_file, 'w') as f:
        # HTML header and styling
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Game Prediction - Final Report</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            padding: 20px; 
            max-width: 1200px; 
            margin: 0 auto;
            color: #333;
            background-color: #f9f9f9;
        }
        h1, h2, h3, h4 { 
            color: #2c3e50; 
            font-weight: 600;
        }
        h1 { 
            font-size: 2.5em; 
            text-align: center;
            margin-bottom: 0.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eee;
        }
        h2 { 
            font-size: 1.8em;
            margin-top: 1.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eee;
        }
        h3 { font-size: 1.4em; margin-top: 1.2em; }
        h4 { font-size: 1.2em; margin-top: 1em; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0; 
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            background-color: white;
        }
        thead { background-color: #f8f9fa; }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }
        th { 
            background-color: #2c3e50; 
            color: white; 
        }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .img-container { 
            text-align: center;
            margin: 20px 0;
        }
        img { 
            max-width: 100%; 
            height: auto;
            box-shadow: 0 3px 6px rgba(0,0,0,0.16);
            border-radius: 5px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card {
            border: 1px solid #e1e1e1;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #7f8c8d;
        }
        .caption {
            text-align: center;
            font-style: italic;
            color: #555;
            margin: 10px 0 20px;
        }
        .highlight-box {
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            padding: 15px;
            margin: 20px 0;
        }
        .model-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 13px;
            font-weight: bold;
            margin-right: 5px;
        }
        .elo-model { background-color: #ffcccc; color: #b91c1c; }
        .ml-model { background-color: #ccdefb; color: #1e40af; }
        .ensemble-model { background-color: #fef3c7; color: #92400e; }
        .specialized-model { background-color: #d1fae5; color: #065f46; }
        
        /* Metric value styling */
        .metric-value {
            font-weight: 600;
            font-family: monospace;
            padding: 2px 5px;
            background-color: #f3f4f6;
            border-radius: 4px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            h2 { font-size: 1.5em; }
            .section { padding: 15px; }
            table { font-size: 0.9em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NBA Game Prediction - Final Report</h1>
        <p style="text-align: center; font-style: italic;">Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
""")

        # Executive Summary section
        f.write("""
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This project developed a machine learning system for predicting NBA game outcomes with a special focus on improving predictions of upsets (where underdogs win). Through evaluating multiple model approaches, we compared traditional statistics-based methods with modern machine learning techniques.</p>
""")
        
        # Find the best model and highest accuracy
        if friendly_results:
            best_model = max(friendly_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
            best_accuracy = friendly_results[best_model].get('accuracy', 0)
            
            # Find best upset prediction model
            upset_models = {m: metrics.get('upset_recall', 0) 
                           for m, metrics in friendly_results.items() 
                           if 'upset_recall' in metrics}
            
            if upset_models:
                best_upset_model = max(upset_models.items(), key=lambda x: x[1])[0]
                best_upset_recall = upset_models[best_upset_model]
                
                f.write(f"""
            <div class="highlight-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best Overall Model:</strong> {best_model} with <span class="metric-value">{best_accuracy:.4f}</span> accuracy</li>
                    <li><strong>Best Upset Prediction Model:</strong> {best_upset_model} with <span class="metric-value">{best_upset_recall:.4f}</span> recall on upsets</li>
                    <li><strong>Models Evaluated:</strong> {len(friendly_results)}</li>
                </ul>
            </div>
""")
            else:
                f.write(f"""
            <div class="highlight-box">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>Best Overall Model:</strong> {best_model} with <span class="metric-value">{best_accuracy:.4f}</span> accuracy</li>
                    <li><strong>Models Evaluated:</strong> {len(friendly_results)}</li>
                </ul>
            </div>
""")
        
        f.write("""
        </div>
""")

        # Model Performance Summary section
        f.write("""
        <div class="section">
            <h2>Model Performance Summary</h2>
""")
        
        # Create model type indicators
        model_types = {}
        for model in friendly_results.keys():
            if 'ELO' in model:
                model_types[model] = ('elo-model', 'ELO Model')
            elif 'Ensemble' in model or 'Stack' in model:
                model_types[model] = ('ensemble-model', 'Ensemble Model')
            elif 'Specialist' in model or 'Special' in model:
                model_types[model] = ('specialized-model', 'Specialized Model')
            else:
                model_types[model] = ('ml-model', 'ML Model')
        
        # Create simplified accuracy comparison chart
        accuracies = {model: results.get('accuracy', 0) for model, results in friendly_results.items()}
        acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['accuracy'])
        acc_df = acc_df.sort_values('accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define colors for different model types
        colors = []
        for model in acc_df.index:
            if 'ELO' in model:
                colors.append('#ffcccc')
            elif 'Ensemble' in model or 'Stack' in model:
                colors.append('#fef3c7')
            elif 'Specialist' in model or 'Special' in model:
                colors.append('#d1fae5')
            else:
                colors.append('#ccdefb')
        
        # Plot bars
        bars = ax.bar(acc_df.index, acc_df['accuracy'], color=colors)
        
        # Add labels and titles
        ax.set_title('Model Accuracy Comparison', fontsize=16)
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
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Convert plot to base64 for embedding
        img_str = fig_to_base64(fig)
        plt.close(fig)
        
        # Embed the image
        f.write(f"""
            <div class="img-container">
                <img src="data:image/png;base64,{img_str}" alt="Model Accuracy Comparison">
                <p class="caption">Figure 1: Accuracy comparison across all models</p>
            </div>
""")
        
        # Create comprehensive metrics table
        f.write("""
            <h3>Comprehensive Performance Metrics</h3>
""")

        # Extract key metrics into a DataFrame
        metrics_df = pd.DataFrame(index=friendly_results.keys())
        metrics_df['Accuracy'] = [results.get('accuracy', float('nan')) for results in friendly_results.values()]
        metrics_df['Precision'] = [results.get('precision', float('nan')) for results in friendly_results.values()]
        metrics_df['Recall'] = [results.get('recall', float('nan')) for results in friendly_results.values()]
        metrics_df['F1 Score'] = [results.get('f1', float('nan')) for results in friendly_results.values()]
        
        # Add upset metrics
        metrics_df['Upset Accuracy'] = [results.get('upset_accuracy', float('nan')) for results in friendly_results.values()]
        metrics_df['Upset Precision'] = [results.get('upset_precision', float('nan')) for results in friendly_results.values()]
        metrics_df['Upset Recall'] = [results.get('upset_recall', float('nan')) for results in friendly_results.values()]
        metrics_df['Upset F1'] = [results.get('upset_f1', float('nan')) for results in friendly_results.values()]
        
        # Add model type column
        metrics_df['Model Type'] = [model_types[model][1] for model in metrics_df.index]
        
        # Sort by accuracy
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
        
        # Reorder columns to put Model Type first
        cols = metrics_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        metrics_df = metrics_df[cols]
        
        # Format values
        metrics_df_html = metrics_df.to_html(
            classes='model-comparison',
            formatters={
                'Accuracy': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Precision': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Recall': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'F1 Score': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Upset Accuracy': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Upset Precision': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Upset Recall': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A",
                'Upset F1': lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
            }
        )
        
        # Apply styling to model types
        for model_name, (css_class, type_name) in model_types.items():
            metrics_df_html = metrics_df_html.replace(f">{type_name}<", f"><span class='model-pill {css_class}'>{type_name}</span><")
        
        f.write(metrics_df_html)
        
        # Add explanation of metrics
        f.write("""
            <div class="card">
                <h4>Understanding the Metrics</h4>
                <ul>
                    <li><strong>Accuracy:</strong> Overall percentage of correct predictions</li>
                    <li><strong>Precision:</strong> When the model predicts a win, how often it's correct</li>
                    <li><strong>Recall:</strong> What percentage of actual wins the model correctly predicts</li>
                    <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
                    <li><strong>Upset Metrics:</strong> Same measures but specifically for games where the underdog wins</li>
                </ul>
            </div>
        </div>
""")
        
        # Upset Prediction Analysis section
        f.write("""
        <div class="section">
            <h2>Upset Prediction Analysis</h2>
            <p>Predicting upsets (when underdogs win) is particularly valuable in sports betting and game analysis. This section focuses specifically on how well different models identify these less predictable outcomes.</p>
""")
        
        # Create upset prediction comparison chart
        upset_metrics = {}
        for model, results in friendly_results.items():
            if all(k in results for k in ['upset_accuracy', 'upset_precision', 'upset_recall', 'upset_f1']):
                upset_metrics[model] = {
                    'Accuracy': results.get('upset_accuracy', 0),
                    'Precision': results.get('upset_precision', 0),
                    'Recall': results.get('upset_recall', 0),
                    'F1 Score': results.get('upset_f1', 0)
                }
        
        if upset_metrics:
            # Create comparison dataframe
            upset_df = pd.DataFrame(upset_metrics)
            
            # Plot upset metrics
            fig, ax = plt.subplots(figsize=(12, 8))
            upset_df.plot(kind='bar', ax=ax)
            ax.set_xlabel('Metric', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Upset Prediction Performance by Model', fontsize=16)
            ax.legend(title='Model', fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Convert plot to base64 for embedding
            img_str = fig_to_base64(fig)
            plt.close(fig)
            
            # Embed the image
            f.write(f"""
            <div class="img-container">
                <img src="data:image/png;base64,{img_str}" alt="Upset Prediction Performance Comparison">
                <p class="caption">Figure 2: Upset prediction performance across models and metrics</p>
            </div>
""")
            
            # Add key insights on upset prediction
            recall_scores = {model: metrics['Recall'] for model, metrics in upset_metrics.items()}
            precision_scores = {model: metrics['Precision'] for model, metrics in upset_metrics.items()}
            f1_scores = {model: metrics['F1 Score'] for model, metrics in upset_metrics.items()}
            
            best_recall_model = max(recall_scores.items(), key=lambda x: x[1])[0]
            best_precision_model = max(precision_scores.items(), key=lambda x: x[1])[0]
            best_f1_model = max(f1_scores.items(), key=lambda x: x[1])[0]
            
            f.write(f"""
            <div class="card">
                <h4>Key Insights on Upset Prediction</h4>
                <ul>
                    <li><strong>Best Model for Finding Upsets:</strong> {best_recall_model} with <span class="metric-value">{recall_scores[best_recall_model]:.4f}</span> recall - identifies {recall_scores[best_recall_model]:.1%} of all actual upsets</li>
                    <li><strong>Most Precise Upset Predictor:</strong> {best_precision_model} with <span class="metric-value">{precision_scores[best_precision_model]:.4f}</span> precision - when it predicts an upset, it's right {precision_scores[best_precision_model]:.1%} of the time</li>
                    <li><strong>Best Balance (F1 Score):</strong> {best_f1_model} with <span class="metric-value">{f1_scores[best_f1_model]:.4f}</span> F1 score - provides the best trade-off between finding upsets and minimizing false alarms</li>
                </ul>
            </div>
""")
        else:
            f.write("""
            <p>No comprehensive upset prediction metrics available. Run the upset metrics calculator to generate these insights.</p>
""")
        
        f.write("""
        </div>
""")
        
        # Feature Importance section
        f.write("""
        <div class="section">
            <h2>Key Predictive Features</h2>
            <p>Understanding which features contribute most to prediction accuracy helps interpret the models and can guide future data collection efforts.</p>
""")
        
        if feature_models:
            # Choose the best model for feature importance display
            best_feature_model_name = None
            best_feature_model = None
            
            for name in ['Gradient Boosting', 'Random Forest', 'Logistic Regression']:
                if name in feature_models:
                    best_feature_model_name = name
                    best_feature_model = feature_models[name]
                    break
            
            if best_feature_model_name and best_feature_model and len(feature_names) > 0:
                # Extract feature importances
                if hasattr(best_feature_model, 'feature_importances_'):
                    importances = best_feature_model.feature_importances_
                elif hasattr(best_feature_model, 'coef_'):
                    importances = abs(best_feature_model.coef_[0]) if len(best_feature_model.coef_.shape) > 1 else abs(best_feature_model.coef_)
                else:
                    importances = None
                
                if importances is not None and len(importances) == len(feature_names):
                    # Sort features by importance
                    indices = importances.argsort()[::-1]
                    top_n = min(15, len(feature_names))  # Show top 15 features
                    
                    # Plot feature importances
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plt.barh(range(top_n), importances[indices][:top_n])
                    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
                    plt.xlabel('Importance')
                    plt.title(f'Top {top_n} Feature Importances ({best_feature_model_name})')
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # Convert plot to base64 for embedding
                    img_str = fig_to_base64(fig)
                    plt.close(fig)
                    
                    # Embed the image
                    f.write(f"""
            <div class="img-container">
                <img src="data:image/png;base64,{img_str}" alt="Feature Importance">
                <p class="caption">Figure 3: Top {top_n} most important features from {best_feature_model_name}</p>
            </div>
""")
                    
                    # Add feature descriptions for top features
                    f.write("""
            <div class="card">
                <h4>Understanding Key Predictors</h4>
                <ul>
""")
                    
                    # Feature descriptions dictionary
                    feature_descriptions = {
                        'elo_diff': 'Difference in ELO ratings between home and away teams',
                        'home_elo_i': 'Initial ELO rating of the home team',
                        'away_elo_i': 'Initial ELO rating of the away team',
                        'home_last_n_win_pct': 'Win percentage of home team in recent games',
                        'away_last_n_win_pct': 'Win percentage of away team in recent games',
                        'home_streak': 'Current streak (wins or losses) of the home team',
                        'away_streak': 'Current streak (wins or losses) of the away team',
                        'home_back_to_back': 'Whether the home team is playing on back-to-back days',
                        'away_back_to_back': 'Whether the away team is playing on back-to-back days',
                        'h_eFGp': 'Home team effective field goal percentage',
                        'a_eFGp': 'Away team effective field goal percentage',
                        'h_TOVp': 'Home team turnover percentage',
                        'a_TOVp': 'Away team turnover percentage',
                        'h_ORBp': 'Home team offensive rebound percentage',
                        'a_ORBp': 'Away team offensive rebound percentage',
                        'h_FTr': 'Home team free throw rate',
                        'a_FTr': 'Away team free throw rate',
                        'eFGp_diff': 'Difference in shooting efficiency',
                        'TOVp_diff': 'Difference in turnover percentage',
                        'ORBp_diff': 'Difference in offensive rebounding',
                        'FTr_diff': 'Difference in free throw rate'
                    }
                    
                    for i in range(min(8, top_n)):
                        feature = feature_names[indices[i]]
                        importance = importances[indices[i]]
                        description = feature_descriptions.get(feature, 'Team performance statistic')
                        f.write(f"""
                    <li><strong>{feature}</strong> <span class="metric-value">({importance:.4f})</span>: {description}</li>
""")
                    
                    f.write("""
                </ul>
            </div>
""")
        else:
            f.write("""
            <p>No feature importance data available. This could be because the models don't expose feature importance attributes or because the feature names aren't available.</p>
""")
        
        f.write("""
        </div>
""")
        
        # Conclusions and Recommendations
        f.write("""
        <div class="section">
            <h2>Conclusions and Recommendations</h2>
            
            <h3>Key Findings</h3>
            <ol>
                <li><strong>Model Performance:</strong> Machine learning models consistently outperform traditional ELO rating systems for NBA game prediction</li>
                <li><strong>Ensemble Advantage:</strong> Combining multiple models through ensemble techniques provides the best overall accuracy</li>
                <li><strong>Upset Prediction:</strong> Specialized models achieve better performance on identifying upsets than general prediction models</li>
                <li><strong>Key Predictors:</strong> Team ratings, recent performance metrics, and back-to-back games are among the strongest predictors</li>
            </ol>
            
            <h3>Recommendations for Model Usage</h3>
            <ol>
                <li>Use <strong>ensemble models</strong> for highest overall accuracy in game prediction</li>
                <li>For upset prediction specifically, prioritize models with high <strong>recall scores</strong> to catch more potential upsets</li>
                <li>Pay special attention to <strong>fatigue factors</strong> like back-to-back games when analyzing potential upsets</li>
                <li>Consider <strong>team matchup history</strong> in addition to general performance metrics</li>
            </ol>
            
            <h3>Future Improvements</h3>
            <ol>
                <li>Incorporate <strong>player-level data</strong> to account for injuries and lineup changes</li>
                <li>Add <strong>feature selection methods</strong> to optimize model performance</li>
                <li>Develop <strong>real-time prediction capabilities</strong> with in-game statistics</li>
                <li>Create <strong>specialized sub-models</strong> for different game contexts (playoffs, back-to-backs, etc.)</li>
            </ol>
        </div>
""")
        
        # Footer
        f.write("""
        <div class="footer">
            <p>NBA Game Prediction Project - Complete Analysis</p>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>
""")
    
    logger.info(f"Improved report generated: {report_file}")
    return report_file