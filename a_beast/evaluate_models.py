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

def plot_four_factors_impact(test_data, save_dir):
    """Analyze and visualize Four Factors impact on game outcomes."""
    factors = ['eFGp', 'TOVp', 'ORBp', 'FTr']
    
    plt.figure(figsize=(15, 10))
    for i, factor in enumerate(factors, 1):
        plt.subplot(2, 2, i)
        
        sns.boxplot(data=test_data, x='is_upset', y=f'{factor}_diff')
        plt.title(f'{factor} Differential Impact on Upsets')
        plt.xlabel('Upset Occurred')
        plt.ylabel(f'{factor} Differential (Home - Away)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'four_factors_impact.png'))
    plt.close()

def plot_back_to_back_analysis(test_data, save_dir):
    """Analyze impact of back-to-back games."""
    # Back-to-back win rates
    plt.figure(figsize=(10, 6))
    b2b_data = pd.DataFrame({
        'Game Type': ['Regular Rest', 'Back-to-Back'],
        'Home Win %': [
            test_data[~test_data['home_back_to_back']]['result'].mean() * 100,
            test_data[test_data['home_back_to_back']]['result'].mean() * 100
        ],
        'Away Win %': [
            (1 - test_data[~test_data['away_back_to_back']]['result'].mean()) * 100,
            (1 - test_data[test_data['away_back_to_back']]['result'].mean()) * 100
        ]
    })
    
    b2b_data.plot(x='Game Type', kind='bar', rot=0)
    plt.title('Win Percentages: Back-to-Back vs Regular Rest')
    plt.ylabel('Win Percentage')
    plt.legend(title='Team')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'back_to_back_analysis.png'))
    plt.close()

def plot_elo_diff_analysis(test_data, save_dir):
    """Analyze prediction accuracy across different ELO differentials."""
    plt.figure(figsize=(12, 6))
    
    # Create ELO difference bins
    test_data['elo_diff_bin'] = pd.qcut(test_data['elo_diff'], q=10)
    
    # Calculate accuracy for each bin
    elo_accuracy = test_data.groupby('elo_diff_bin')['is_upset'].mean()
    
    plt.plot(range(len(elo_accuracy)), elo_accuracy.values, marker='o')
    plt.title('Upset Probability vs ELO Difference')
    plt.xlabel('ELO Difference Decile')
    plt.ylabel('Upset Probability')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'elo_diff_analysis.png'))
    plt.close()

def analyze_team_upset_vulnerability(test_data, save_dir):
    """Analyze and visualize team-specific upset patterns."""
    # Calculate upset rates for each team
    home_upsets = test_data.groupby('home_team')['is_upset'].agg(['mean', 'count'])
    away_upsets = test_data.groupby('away_team')['is_upset'].agg(['mean', 'count'])
    
    # Combine home and away stats
    team_upsets = pd.DataFrame({
        'Home_Upset_Rate': home_upsets['mean'],
        'Away_Upset_Rate': away_upsets['mean'],
        'Total_Games': home_upsets['count'] + away_upsets['count']
    })
    
    # Plot team vulnerability
    plt.figure(figsize=(15, 8))
    team_upsets['Overall_Upset_Rate'] = (
        (home_upsets['mean'] * home_upsets['count'] + 
         away_upsets['mean'] * away_upsets['count']) / 
        team_upsets['Total_Games']
    )
    
    team_upsets = team_upsets.sort_values('Overall_Upset_Rate', ascending=False)
    
    plt.bar(range(len(team_upsets)), team_upsets['Overall_Upset_Rate'])
    plt.xticks(range(len(team_upsets)), team_upsets.index, rotation=45, ha='right')
    plt.title('Team Upset Vulnerability')
    plt.ylabel('Upset Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'team_upset_vulnerability.png'))
    plt.close()

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
            
            # Basic metrics and standard plots
            metrics, y_pred = evaluate_basic_metrics(
                model, X_test, y_test, test_data['is_upset'].values
            )
            all_metrics[name] = metrics
            
            # Standard plots
            plot_confusion_matrix(
                y_test, y_pred,
                f'Confusion Matrix - {name.title()}',
                os.path.join(CONFUSION_MATRICES_DIR, f'{name}_cm.png')
            )
            
            if name in ['random_forest', 'gradient_boosting']:
                plot_feature_importance(
                    model, feature_names,
                    f'Feature Importance - {name.title()}',
                    os.path.join(FEATURE_IMPORTANCE_DIR, f'{name}_importance.png')
                )
        
        # Plot ROC curves for all models
        plot_roc_curve(
            models, X_test, y_test,
            os.path.join(ROC_CURVES_DIR, 'all_models_roc.png')
        )
        
        # Additional analysis plots
        print("\nGenerating additional analysis plots...")
        
        # Four Factors analysis
        plot_four_factors_impact(test_data, FOUR_FACTORS_DIR)
        
        # Back-to-back game analysis
        plot_back_to_back_analysis(test_data, PERFORMANCE_ANALYSIS_DIR)
        
        # ELO difference analysis
        plot_elo_diff_analysis(test_data, PERFORMANCE_ANALYSIS_DIR)
        
        # Team upset vulnerability analysis
        analyze_team_upset_vulnerability(test_data, TEAM_ANALYSIS_DIR)
        
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