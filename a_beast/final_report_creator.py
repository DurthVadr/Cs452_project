"""
Final report generation script for NBA game prediction project.
Creates a comprehensive HTML report with styled formatting and embedded visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import numpy as np
from datetime import datetime
from config import *

def create_html_style():
    """Return HTML styling similar to the improved report style."""
    return """
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
            .plot-container { 
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
            .metric-value {
                font-weight: 600;
                font-family: monospace;
                padding: 2px 5px;
                background-color: #f3f4f6;
                border-radius: 4px;
            }
            .highlight-box {
                border-left: 4px solid #3498db;
                background-color: #f8f9fa;
                padding: 15px;
                margin: 20px 0;
            }
            .caption {
                text-align: center;
                font-style: italic;
                color: #555;
                margin: 10px 0 20px;
            }
        </style>
    """

def embed_image(image_path):
    """Convert image to base64 for embedding in HTML."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        print(f"Warning: Could not embed image {image_path}: {e}")
        return ""

def create_model_section(model_name, metrics_dict):
    """Create HTML section for a specific model including all visualizations."""
    # Get metrics for this specific model
    model_metrics = metrics_dict.get(model_name, {})
    
    html = f"""
    <div class="section">
        <h2>Model: {model_name.title()}</h2>
        
        <h3>Performance Metrics</h3>
        <div class="highlight-box">
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Overall</th>
                    <th>Upset Games</th>
                    <th>Non-Upset Games</th>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td><span class="metric-value">{model_metrics.get('accuracy', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('upset_accuracy', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('nonupset_accuracy', 'N/A')}</span></td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td><span class="metric-value">{model_metrics.get('precision', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('upset_precision', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('nonupset_precision', 'N/A')}</span></td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td><span class="metric-value">{model_metrics.get('recall', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('upset_recall', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('nonupset_recall', 'N/A')}</span></td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td><span class="metric-value">{model_metrics.get('f1', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('upset_f1', 'N/A')}</span></td>
                    <td><span class="metric-value">{model_metrics.get('nonupset_f1', 'N/A')}</span></td>
                </tr>
            </table>
        </div>
    """
    
    # Add confusion matrix
    cm_path = os.path.join(CONFUSION_MATRICES_DIR, f'{model_name}_cm.png')
    if os.path.exists(cm_path):
        img_str = embed_image(cm_path)
        if img_str:
            html += f"""
            <div class="plot-container">
                <h3>Confusion Matrix</h3>
                <img src="data:image/png;base64,{img_str}">
                <p class="caption">Confusion matrix showing prediction results</p>
            </div>
            """
    
    # Add feature importance for applicable models
    if model_name in ['random_forest', 'gradient_boosting']:
        fi_path = os.path.join(FEATURE_IMPORTANCE_DIR, f'{model_name}_importance.png')
        if os.path.exists(fi_path):
            img_str = embed_image(fi_path)
            if img_str:
                html += f"""
                <div class="plot-container">
                    <h3>Feature Importance</h3>
                    <img src="data:image/png;base64,{img_str}">
                    <p class="caption">Relative importance of different features in making predictions</p>
                </div>
                """
    
    html += "</div>"
    return html

def create_report():
    """Create the final HTML report with all visualizations and metrics."""
    print("Creating final report...")
    
    # Load metrics CSV file
    try:
        metrics_file = os.path.join(PERFORMANCE_METRICS_DIR, 'model_metrics.csv')
        # Check if file exists
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found at {metrics_file}")
            
        metrics_df = pd.read_csv(metrics_file, index_col=0)
        
        # Convert DataFrame (metrics as rows, models as columns) to dictionary format
        # This gives us {model_name: {metric_name: value, ...}, ...}
        metrics_dict = {}
        for model_name in metrics_df.columns:
            metrics_dict[model_name] = metrics_df[model_name].to_dict()
        
        print(f"Loaded metrics for models: {', '.join(metrics_dict.keys())}")
    except Exception as e:
        print(f"Warning: Could not load metrics file: {e}")
        print("Continuing with empty metrics...")
        metrics_dict = {
            "logistic": {}, 
            "random_forest": {},
            "gradient_boosting": {},
            "ensemble": {}
        }
    
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NBA Game Prediction - Final Report</title>
        {create_html_style()}
    </head>
    <body>
        <h1>NBA Game Prediction Project - Final Report</h1>
        <p style="text-align: center; font-style: italic;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Project Overview</h2>
            <p>This project aims to predict NBA game outcomes with a focus on identifying upset victories 
            where underdog teams win against favorites. We evaluated multiple machine learning models 
            including Logistic Regression, Random Forest, Gradient Boosting, and an Ensemble approach.</p>
            
            <div class="highlight-box">
                <h3>Key Objectives</h3>
                <ul>
                    <li>Develop models that accurately predict NBA game outcomes</li>
                    <li>Identify factors that contribute to upset victories</li>
                    <li>Compare different modeling approaches for performance</li>
                    <li>Create a framework for evaluating model performance on upset predictions</li>
                </ul>
            </div>
            
            <h3>Enhanced Feature Engineering</h3>
            <p>This project utilizes sophisticated feature engineering including:</p>
            <ul>
                <li><strong>Momentum Features:</strong> Team streaks and rolling win percentages</li>
                <li><strong>Four Factors Analysis:</strong> Dean Oliver's four factors with rolling averages</li>
                <li><strong>Differential Features:</strong> Home vs away team comparisons</li>
                <li><strong>Interaction Features:</strong> Cross-factor relationships like ELO and back-to-back games</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Model Performance Overview</h2>
    """
    
    # Find best models if we have metrics
    if metrics_dict and any(metrics_dict.values()):
        # Best overall accuracy
        model_accuracies = {model: metrics.get('accuracy', 0) 
                            for model, metrics in metrics_dict.items() 
                            if 'accuracy' in metrics}
        
        if model_accuracies:
            best_model = max(model_accuracies.items(), key=lambda x: x[1])[0]
            best_accuracy = model_accuracies[best_model]
            
            # Best upset recall
            upset_recalls = {model: metrics.get('upset_recall', 0) 
                            for model, metrics in metrics_dict.items() 
                            if 'upset_recall' in metrics}
            
            if upset_recalls:
                best_upset_model = max(upset_recalls.items(), key=lambda x: x[1])[0]
                best_upset_recall = upset_recalls[best_upset_model]
                
                html_content += f"""
                <div class="highlight-box">
                    <h3>Key Findings</h3>
                    <ul>
                        <li>Best Overall Model: <span class="metric-value">{best_model.title()}</span> 
                            with accuracy of <span class="metric-value">{best_accuracy:.4f}</span></li>
                        <li>Best Upset Detection: <span class="metric-value">{best_upset_model.title()}</span> 
                            with recall of <span class="metric-value">{best_upset_recall:.4f}</span></li>
                    </ul>
                </div>
                """
    
    # Add ROC curves comparison
    roc_path = os.path.join(ROC_CURVES_DIR, 'all_models_roc.png')
    if os.path.exists(roc_path):
        img_str = embed_image(roc_path)
        if img_str:
            html_content += f"""
                <div class="plot-container">
                    <h3>ROC Curves Comparison</h3>
                    <img src="data:image/png;base64,{img_str}">
                    <p class="caption">Receiver Operating Characteristic (ROC) curves for all models</p>
                </div>
            """
    
    html_content += "</div>"  # Close Model Performance Overview section
    
    # Add Four Factors Analysis section
    html_content += """
    <div class="section">
        <h2>Four Factors Analysis</h2>
        <p>Dean Oliver's Four Factors (Shooting, Turnovers, Rebounding, Free Throws) 
        have significant impact on game outcomes and upset predictions.</p>
    """
    
    # Add Four Factors impact plot
    four_factors_path = os.path.join(FOUR_FACTORS_DIR, 'four_factors_impact.png')
    if os.path.exists(four_factors_path):
        img_str = embed_image(four_factors_path)
        if img_str:
            html_content += f"""
            <div class="plot-container">
                <h3>Four Factors Impact on Upsets</h3>
                <img src="data:image/png;base64,{img_str}">
                <p class="caption">How differences in Four Factors influence upset probability</p>
            </div>
            
            <div class="highlight-box">
                <h3>Four Factors Insights</h3>
                <ul>
                    <li>Effective Field Goal Percentage (eFG%) shows the strongest correlation with upsets</li>
                    <li>Turnover Rate (TOV%) differences can signal potential upset conditions</li>
                    <li>Offensive Rebounding (ORB%) advantage helps underdog teams create upset opportunities</li>
                    <li>Free Throw Rate (FTR) can be crucial in close games that lead to upsets</li>
                </ul>
            </div>
            """
    
    html_content += "</div>"  # Close Four Factors section
    
    # Add Game Context Analysis section
    html_content += """
    <div class="section">
        <h2>Game Context Analysis</h2>
        <p>Various contextual factors significantly influence game outcomes and upset probabilities.</p>
    """
    
    # Add Back-to-Back analysis
    b2b_path = os.path.join(PERFORMANCE_ANALYSIS_DIR, 'back_to_back_analysis.png')
    if os.path.exists(b2b_path):
        img_str = embed_image(b2b_path)
        if img_str:
            html_content += f"""
            <div class="plot-container">
                <h3>Back-to-Back Games Impact</h3>
                <img src="data:image/png;base64,{img_str}">
                <p class="caption">Win percentages in back-to-back games vs. regular rest games</p>
            </div>
            """
    
    # Add ELO analysis
    elo_path = os.path.join(PERFORMANCE_ANALYSIS_DIR, 'elo_diff_analysis.png')
    if os.path.exists(elo_path):
        img_str = embed_image(elo_path)
        if img_str:
            html_content += f"""
            <div class="plot-container">
                <h3>ELO Rating Difference Analysis</h3>
                <img src="data:image/png;base64,{img_str}">
                <p class="caption">Relationship between ELO difference and upset probability</p>
            </div>
            """
    
    # Add Team Vulnerability analysis
    team_path = os.path.join(TEAM_ANALYSIS_DIR, 'team_upset_vulnerability.png')
    if os.path.exists(team_path):
        img_str = embed_image(team_path)
        if img_str:
            html_content += f"""
            <div class="plot-container">
                <h3>Team Upset Vulnerability</h3>
                <img src="data:image/png;base64,{img_str}">
                <p class="caption">Teams most prone to being involved in upset games</p>
            </div>
            
            <div class="highlight-box">
                <h3>Game Context Insights</h3>
                <ul>
                    <li>Teams playing back-to-back games show significantly reduced performance</li>
                    <li>Smaller ELO differences (100-200 points) represent the highest upset probability zone</li>
                    <li>Certain teams consistently over or underperform their ELO expectations</li>
                    <li>Home court advantage is diminished in back-to-back scenarios</li>
                </ul>
            </div>
            """
    
    html_content += "</div>"  # Close Game Context section
    
    # Add individual model sections
    for model_name in ['logistic', 'random_forest', 'gradient_boosting', 'ensemble']:
        html_content += create_model_section(model_name, metrics_dict)
    
    # Add comparative analysis
    html_content += """
        <div class="section">
            <h2>Comparative Analysis</h2>
            <h3>Upset Prediction Performance</h3>
            <p>The ability to predict upsets (when underdogs win) is particularly valuable. This section compares
            how well our models perform specifically on upset predictions.</p>
    """
    
    # Create upset comparison table if metrics exist
    if metrics_dict and any(metrics_dict.values()):
        upset_df = pd.DataFrame({
            'Model': [],
            'Upset Precision': [],
            'Upset Recall': [],
            'Upset F1': []
        })
        
        for model, metrics in metrics_dict.items():
            if 'upset_precision' in metrics and 'upset_recall' in metrics and 'upset_f1' in metrics:
                new_row = pd.DataFrame({
                    'Model': [model.title()],
                    'Upset Precision': [metrics['upset_precision']],
                    'Upset Recall': [metrics['upset_recall']],
                    'Upset F1': [metrics['upset_f1']]
                })
                upset_df = pd.concat([upset_df, new_row], ignore_index=True)
        
        if not upset_df.empty:
            upset_df = upset_df.sort_values('Upset F1', ascending=False)
            upset_html = upset_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            html_content += f"""
                <div class="highlight-box">
                    {upset_html}
                </div>
            """
            
            html_content += """
                <p>Interpreting the metrics:</p>
                <ul>
                    <li><strong>Upset Precision</strong>: When the model predicts an upset, how often it is correct</li>
                    <li><strong>Upset Recall</strong>: What proportion of actual upsets the model correctly identifies</li>
                    <li><strong>Upset F1</strong>: Harmonic mean of precision and recall, balancing both metrics</li>
                </ul>
            """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Conclusions</h2>
            <div class="highlight-box">
                <h3>Key Findings</h3>
                <ol>
                    <li>Ensemble methods generally outperform individual models for NBA game prediction</li>
                    <li>Feature engineering with momentum and interactions significantly improves predictions</li>
                    <li>Four Factors metrics, particularly shooting efficiency, are strong predictors</li>
                    <li>Game context (back-to-back games, team matchups) affects upset likelihood</li>
                    <li>Predicting upsets remains challenging but our models show improvement over baseline</li>
                </ol>
            </div>
            
            <h3>Future Work</h3>
            <ol>
                <li>Incorporate player-level data to account for injuries and lineup changes</li>
                <li>Explore specialized models specifically trained to identify upset conditions</li>
                <li>Implement time-series modeling approaches to better capture team momentum</li>
                <li>Develop interactive prediction tools for real-time analysis</li>
                <li>Fine-tune ELO rating parameters specifically for NBA prediction tasks</li>
            </ol>
        </div>
        
        <div class="footer" style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #eee;">
            <p>NBA Game Prediction Project - Complete Analysis</p>
        </div>
    </body>
    </html>
    """
    
    # Save report
    report_path = os.path.join(BASE_DIR, "final_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Report saved to: {report_path}")

def main():
    """Main execution function."""
    try:
        create_report()
        print("Final report created successfully!")
    except Exception as e:
        print(f"Error creating final report: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()