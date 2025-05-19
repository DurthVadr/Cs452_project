"""
Optimized model building module for the NBA prediction project.

This module implements an optimized approach to building and evaluating
machine learning models for NBA game prediction, with a focus on:
1. Simplified but effective model architecture
2. Better handling of class imbalance
3. Explicit modeling of upsets
4. Improved evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from datetime import datetime
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

# Import project configuration
import config
from feature_config import get_feature_set
from data_generation import apply_advanced_smote, generate_enhanced_training_data

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def load_data(feature_set_name='default'):
    """
    Load and prepare data for model building.

    Args:
        feature_set_name: Name of the feature set to use

    Returns:
        X_train_scaled: Scaled training features
        y_train: Training labels
        X_test_scaled: Scaled test features
        y_test: Test labels
        features: List of feature names
        scaler: Fitted StandardScaler
        train_data: Full training DataFrame
        test_data: Full test DataFrame
    """
    logger.info(f"Loading data with feature set '{feature_set_name}'")

    try:
        # Load processed data
        train_data = pd.read_csv(config.TRAIN_DATA_FILE)
        test_data = pd.read_csv(config.TEST_DATA_FILE)

        # Get feature list
        features = get_feature_set(feature_set_name)
        logger.info(f"Using {len(features)} features: {', '.join(features)}")

        # Create X and y for training and testing
        X_train = train_data[features]
        y_train = train_data['result']
        X_test = test_data[features]
        y_test = test_data['result']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        joblib.dump(scaler, config.SCALER_FILE)

        logger.info(f"Data loaded successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train_scaled, y_train, X_test_scaled, y_test, features, scaler, train_data, test_data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def apply_smote(X_train_scaled, y_train):
    """Apply SMOTE to balance the training data."""
    logger.info("Applying SMOTE to balance training data")

    # Check class distribution before SMOTE
    class_dist_before = pd.Series(y_train).value_counts(normalize=True)
    logger.info(f"Class distribution before SMOTE: {class_dist_before.to_dict()}")

    # Apply SMOTE
    smote = SMOTE(**config.SMOTE_PARAMS)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Check class distribution after SMOTE
    class_dist_after = pd.Series(y_train_resampled).value_counts(normalize=True)
    logger.info(f"Class distribution after SMOTE: {class_dist_after.to_dict()}")
    logger.info(f"Training data shape after SMOTE: {X_train_resampled.shape}")

    return X_train_resampled, y_train_resampled, class_dist_before, class_dist_after

def build_optimized_models(X_train, y_train, features):
    """
    Build optimized models using a focused approach.

    Args:
        X_train: Training features
        y_train: Training labels
        features: List of feature names

    Returns:
        Dictionary of trained models
    """
    logger.info("Building optimized models")

    # Initialize models with optimized parameters
    logistic = LogisticRegression(**config.LOGISTIC_PARAMS)
    rf = RandomForestClassifier(**config.RF_PARAMS)
    gb = GradientBoostingClassifier(**config.GB_PARAMS)

    # Train base models
    logger.info("Training logistic regression model")
    logistic.fit(X_train, y_train)

    logger.info("Training random forest model")
    rf.fit(X_train, y_train)

    logger.info("Training gradient boosting model")
    gb.fit(X_train, y_train)

    # Create weighted voting ensemble
    logger.info("Creating weighted voting ensemble")
    ensemble = VotingClassifier(
        estimators=[
            ('logistic', logistic),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft',
        weights=[
            config.ENSEMBLE_WEIGHTS['logistic'],
            config.ENSEMBLE_WEIGHTS['random_forest'],
            config.ENSEMBLE_WEIGHTS['gradient_boosting']
        ]
    )
    ensemble.fit(X_train, y_train)

    # Feature importance analysis (for Random Forest)
    if hasattr(rf, 'feature_importances_'):
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(min(20, len(features))),
                importances[indices[:20]],
                align='center')
        plt.yticks(range(min(20, len(features))), [features[i] for i in indices[:20]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, 'feature_importance', 'rf_feature_importance.png'))

        # Log top 10 features
        logger.info("Top 10 features by importance:")
        for i in range(min(10, len(features))):
            logger.info(f"{i+1}. {features[indices[i]]} ({importances[indices[i]]:.4f})")

    # Save models
    logger.info("Saving models")
    joblib.dump(logistic, config.LOGISTIC_MODEL_FILE)
    joblib.dump(rf, config.RF_MODEL_FILE)
    joblib.dump(gb, config.GB_MODEL_FILE)
    joblib.dump(ensemble, config.ENSEMBLE_MODEL_FILE)

    return {
        'logistic': logistic,
        'random_forest': rf,
        'gradient_boosting': gb,
        'ensemble': ensemble
    }

def evaluate_models(models, X_train, y_train, X_test, y_test, train_data, test_data):
    """
    Evaluate models with comprehensive metrics.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        train_data: Full training DataFrame
        test_data: Full test DataFrame

    Returns:
        Dictionary of evaluation results
        DataFrame of model accuracies
    """
    logger.info("Evaluating models")

    # Identify upsets in test data
    test_data['favorite'] = np.where(test_data['home_elo_i'] > test_data['away_elo_i'], 1, 0)
    test_data['upset'] = np.where(test_data['favorite'] != test_data['result'], 1, 0)

    # Get upset indices
    upset_indices = test_data[test_data['upset'] == 1].index
    upset_mask = test_data.index.isin(upset_indices)

    # Calculate ELO prediction accuracy
    elo_predictions = np.where(test_data['home_elo_i'] > test_data['away_elo_i'], 1, 0)
    elo_accuracy = accuracy_score(test_data['result'], elo_predictions)
    logger.info(f"ELO prediction accuracy: {elo_accuracy:.4f}")

    results = {'elo': {'accuracy': elo_accuracy}}
    accuracies = {'elo': elo_accuracy}

    for name, model in models.items():
        logger.info(f"Evaluating {name} model")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"{name} test accuracy: {accuracy:.4f}")
        logger.info(f"{name} test precision: {precision:.4f}")
        logger.info(f"{name} test recall: {recall:.4f}")
        logger.info(f"{name} test F1 score: {f1:.4f}")

        # Calculate upset-specific metrics
        if len(upset_indices) > 0:
            # Get predictions for upset games
            upset_y = y_test[upset_mask]
            upset_pred = y_pred[upset_mask]

            # Calculate upset-specific metrics
            upset_accuracy = accuracy_score(upset_y, upset_pred)

            # For upsets (when favorite loses)
            upset_detection = (test_data['favorite'] != pd.Series(y_pred, index=test_data.index))
            upset_precision = precision_score(test_data['upset'], upset_detection)
            upset_recall = recall_score(test_data['upset'], upset_detection)
            upset_f1 = f1_score(test_data['upset'], upset_detection)

            logger.info(f"{name} upset accuracy: {upset_accuracy:.4f}")
            logger.info(f"{name} upset precision: {upset_precision:.4f}")
            logger.info(f"{name} upset recall: {upset_recall:.4f}")
            logger.info(f"{name} upset F1 score: {upset_f1:.4f}")

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Away Win', 'Home Win'],
                   yticklabels=['Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIR, 'confusion_matrices', f'{name}_confusion_matrix.png'))

        # Create ROC curve if probabilities are available
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(config.PLOTS_DIR, 'roc_curves', f'{name}_roc_curve.png'))

        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'upset_accuracy': upset_accuracy if len(upset_indices) > 0 else None,
            'upset_precision': upset_precision if len(upset_indices) > 0 else None,
            'upset_recall': upset_recall if len(upset_indices) > 0 else None,
            'upset_f1': upset_f1 if len(upset_indices) > 0 else None
        }

        accuracies[name] = accuracy

    # Convert accuracies to DataFrame
    accuracies_df = pd.DataFrame({'accuracy': accuracies}).sort_values('accuracy', ascending=False)

    # Save accuracies
    accuracies_df.to_csv(os.path.join(config.PLOTS_DIR, 'performance_metrics', 'model_accuracies.csv'))

    return results, accuracies_df

def create_html_report(results, accuracies_df, class_dist_before, class_dist_after, features):
    """Create an HTML report summarizing the model results."""
    logger.info("Creating HTML report")

    # Get best model
    best_model = accuracies_df.index[0]
    best_accuracy = accuracies_df.iloc[0, 0]

    # Calculate improvement over ELO
    elo_accuracy = results['elo']['accuracy']
    improvement = best_accuracy - elo_accuracy

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{config.REPORT_TITLE}</title>

        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                color: #333;
                background-color: #f9f9f9;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
                font-weight: 600;
            }}
            h1 {{
                font-size: 2.5em;
                text-align: center;
                margin-bottom: 0.5em;
                padding-bottom: 0.3em;
                border-bottom: 1px solid #eee;
            }}
            h2 {{
                font-size: 1.8em;
                margin-top: 1.5em;
                padding-bottom: 0.3em;
                border-bottom: 1px solid #eee;
            }}
            h3 {{ font-size: 1.4em; margin-top: 1.2em; }}
            h4 {{ font-size: 1.2em; margin-top: 1em; }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 2px 3px rgba(0,0,0,0.1);
                background-color: white;
            }}
            thead {{ background-color: #f8f9fa; }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #2c3e50;
                color: white;
            }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 3px 6px rgba(0,0,0,0.16);
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-weight: 600;
                font-family: monospace;
                padding: 2px 5px;
                background-color: #f3f4f6;
                border-radius: 4px;
            }}
            .highlight-box {{
                border-left: 4px solid #3498db;
                background-color: #f8f9fa;
                padding: 15px;
                margin: 20px 0;
            }}
            .caption {{
                text-align: center;
                font-style: italic;
                color: #555;
                margin: 10px 0 20px;
            }}
        </style>

    </head>
    <body>
        <h1>{config.REPORT_TITLE}</h1>
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
        </div>

        <div class="section">
            <h2>Model Performance Overview</h2>

                <div class="highlight-box">
                    <h3>Key Findings</h3>
                    <ul>
                        <li>Best Overall Model: <span class="metric-value">{best_model}</span>
                            with accuracy of <span class="metric-value">{best_accuracy:.4f}</span></li>
                        <li>Improvement over ELO baseline: <span class="metric-value">{improvement:.4f}</span>
                            ({improvement*100:.2f}%)</li>
                    </ul>
                </div>

                <div class="plot-container">
                    <h3>Model Accuracy Comparison</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Improvement over ELO</th>
                            </tr>
                        </thead>
                        <tbody>
    """

    # Add rows for each model
    for model, accuracy in accuracies_df.iterrows():
        model_improvement = accuracy['accuracy'] - elo_accuracy
        html_content += f"""
                            <tr>
                                <td>{model}</td>
                                <td>{accuracy['accuracy']:.4f}</td>
                                <td>{model_improvement:.4f} ({model_improvement*100:.2f}%)</td>
                            </tr>
        """

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <h2>Data Enhancement</h2>
                <p>We applied advanced data generation techniques to enhance the training data
                and improve model performance, especially for predicting upset games.</p>

                <div class="highlight-box">
                    <h3>Data Generation Techniques Used</h3>
                    <ul>
                        <li><strong>Advanced SMOTE:</strong> {config.SMOTE_METHOD.upper()} variant for balanced class distribution</li>
                        <li><strong>Feature-Aware Generation:</strong> {config.FEATURE_AWARE_SAMPLES} samples per class that respect feature correlations</li>
                        <li><strong>Game-Specific Augmentation:</strong> {config.GAME_SPECIFIC_SAMPLES} samples based on team matchup patterns</li>
                    </ul>
                </div>

                <h3>Class Distribution Before and After Data Enhancement</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Before Enhancement</th>
                            <th>After Enhancement</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    # Add rows for each class
    for cls in sorted(class_dist_before.index):
        before = class_dist_before.get(cls, 0)
        after = class_dist_after.get(cls, 0)
        change = after - before
        change_pct = (change / before) * 100 if before > 0 else float('inf')

        html_content += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{before:.4f}</td>
                            <td>{after:.4f}</td>
                            <td>{change_pct:+.2f}%</td>
                        </tr>
        """

    html_content += """
                    </tbody>
                </table>

                <p>Our enhanced data generation approach creates synthetic samples that better preserve the complex
                relationships between features, leading to more robust model training, especially for predicting rare events.</p>
            </div>

            <div class="section">
                <h2>Feature Importance</h2>
                <p>Understanding which features contribute most to prediction accuracy helps us interpret
                the model and focus on the most relevant factors for NBA game outcomes.</p>

                <div class="plot-container">
                    <img src="../plots/feature_importance/rf_feature_importance.png" alt="Feature Importance">
                    <p class="caption">Feature importance from Random Forest model</p>
                </div>
            </div>

            <div class="section">
                <h2>Confusion Matrices</h2>
                <p>Confusion matrices help us understand where our models are making correct predictions
                and where they're making mistakes.</p>

                <div class="plot-container">
                    <img src="../plots/confusion_matrices/ensemble_confusion_matrix.png" alt="Ensemble Confusion Matrix">
                    <p class="caption">Confusion matrix for the Ensemble model</p>
                </div>
            </div>

            <div class="section">
                <h2>Conclusion</h2>
                <p>The best performing model is the <span class="metric-value">{best_model}</span> with a test accuracy of
                <span class="metric-value">{best_accuracy:.4f}</span>, which represents an improvement of
                <span class="metric-value">{improvement:.4f}</span> ({improvement*100:.2f}%) over the baseline ELO model.</p>

                <p>Our approach of using SMOTE to balance the training data, combined with a focused feature set
                and optimized model parameters, has resulted in improved prediction accuracy, especially for upset games.</p>
            </div>
        </body>
        </html>
    """

    # Write HTML to file
    with open(config.REPORT_FILE, 'w') as f:
        f.write(html_content)

    logger.info(f"HTML report created: {config.REPORT_FILE}")

def main():
    """Main function to run the optimized model building process."""
    start_time = time.time()
    logger.info("Starting optimized model building process")

    try:
        # Load data
        X_train, y_train, X_test, y_test, features, scaler, train_data, test_data = load_data()

        # Apply data generation techniques
        if config.USE_ENHANCED_DATA_GENERATION:
            logger.info("Using enhanced data generation techniques")
            X_train_resampled, y_train_resampled = generate_enhanced_training_data(
                X_train, y_train, train_data, features
            )
            # Calculate class distribution before and after
            class_dist_before = pd.Series(y_train).value_counts(normalize=True)
            class_dist_after = pd.Series(y_train_resampled).value_counts(normalize=True)
        else:
            # Apply standard SMOTE to balance training data
            logger.info("Using standard SMOTE for data generation")
            X_train_resampled, y_train_resampled, class_dist_before, class_dist_after = apply_smote(X_train, y_train)

        # Build optimized models
        models = build_optimized_models(X_train_resampled, y_train_resampled, features)

        # Evaluate models
        results, accuracies_df = evaluate_models(
            models, X_train_resampled, y_train_resampled, X_test, y_test, train_data, test_data
        )

        # Create HTML report
        create_html_report(results, accuracies_df, class_dist_before, class_dist_after, features)

        logger.info(f"Optimized model building completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in optimized model building: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
