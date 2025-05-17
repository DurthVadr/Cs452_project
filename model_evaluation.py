import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Create output directory for model evaluation
if not os.path.exists('evaluation'):
    os.makedirs('evaluation')

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Define datasets
datasets = ['Original', 'ROS', 'SMOTE']

# Load feature names
with open('processed_data/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"Using {len(feature_names)} features for model evaluation.")

# Create X and y data for test
X_test = test_data[feature_names].values
y_test = test_data['result'].values

# Define model types
model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']

# Load all models
print("Loading models...")
loaded_models = {}

for dataset in datasets:
    loaded_models[dataset] = {}
    
    for model_type in model_types:
        model_path = f"models/{dataset.lower()}/{model_type}_model.pkl"
        scaler_path = f"models/{dataset.lower()}/scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Transform test data with the appropriate scaler
                X_test_scaled = scaler.transform(X_test)
                
                # Store model, scaler and scaled test data
                loaded_models[dataset][model_type] = {
                    'model': model,
                    'scaler': scaler,
                    'X_test_scaled': X_test_scaled,
                    'display_name': model_type.replace('_', ' ').title()
                }
                
                print(f"Loaded {model_type} model for {dataset} dataset.")
            except Exception as e:
                print(f"Error loading {model_type} model for {dataset} dataset: {e}")
        else:
            print(f"Model {model_path} or scaler {scaler_path} not found.")

# Evaluate models
print("Evaluating models...")
model_results = {}

for dataset in datasets:
    if dataset not in model_results:
        model_results[dataset] = {}
    
    for model_type, model_data in loaded_models[dataset].items():
        model = model_data['model']
        X_test_scaled = model_data['X_test_scaled']
        display_name = model_data['display_name']
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC curve and AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
        else:
            fpr, tpr, roc_auc = None, None, None
            precision, recall, pr_auc = None, None, None
        
        # Store results
        model_results[dataset][display_name] = {
            'accuracy': accuracy,
            'report': report,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'pr_auc': pr_auc
        }
        
        # Create output directory for this dataset
        dataset_dir = f"evaluation/{dataset.lower()}"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Away Win', 'Home Win'],
                    yticklabels=['Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{display_name} Confusion Matrix ({dataset} Dataset)')
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/{model_type}_confusion_matrix.png")
        
        # Plot ROC curve if available
        if fpr is not None and tpr is not None:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{display_name} ROC Curve ({dataset} Dataset)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f"{dataset_dir}/{model_type}_roc_curve.png")
        
        # Plot precision-recall curve if available
        if precision is not None and recall is not None:
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{display_name} Precision-Recall Curve ({dataset} Dataset)')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(f"{dataset_dir}/{model_type}_pr_curve.png")

# Compare model accuracies across all datasets
all_accuracies = {}
for dataset in datasets:
    for model_name, results in model_results[dataset].items():
        all_accuracies[f"{dataset} - {model_name}"] = results['accuracy']

# Add ELO accuracy
elo_accuracy = (test_data['elo_pred'] == test_data['result']).mean()
all_accuracies['ELO'] = elo_accuracy

# Convert to DataFrame
accuracies_df = pd.DataFrame.from_dict(all_accuracies, orient='index', columns=['accuracy'])
accuracies_df = accuracies_df.sort_values('accuracy', ascending=False)
accuracies_df.to_csv('evaluation/model_accuracies.csv')

# Visualize model accuracies
plt.figure(figsize=(14, 8))
sns.barplot(x=accuracies_df.index, y='accuracy', data=accuracies_df)
plt.title('Model Test Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 0.8)
for i, v in enumerate(accuracies_df['accuracy']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('evaluation/model_accuracies.png')

# Analyze prediction errors for each dataset's best model
print("Analyzing prediction errors...")

for dataset in datasets:
    # Get best model for this dataset
    dataset_accuracies = {name: results['accuracy'] for name, results in model_results[dataset].items()}
    best_model_name = max(dataset_accuracies, key=dataset_accuracies.get)
    print(f"Best model for {dataset} dataset: {best_model_name}")
    
    # Make predictions with best model
    model_data = loaded_models[dataset][best_model_name.lower().replace(' ', '_')]
    model = model_data['model']
    X_test_scaled = model_data['X_test_scaled']
    y_pred_best = model.predict(X_test_scaled)
    
    # Create DataFrame with test data and predictions
    test_with_preds = test_data.copy()
    test_with_preds['predicted'] = y_pred_best
    test_with_preds['correct'] = test_with_preds['predicted'] == test_with_preds['result']
    
    # Create directory for this dataset's analysis
    dataset_dir = f"evaluation/{dataset.lower()}"
    
    # Analyze errors by ELO difference
    test_with_preds['elo_diff_abs'] = abs(test_with_preds['home_elo_i'] - test_with_preds['away_elo_i'])
    test_with_preds['elo_diff_bin'] = pd.cut(test_with_preds['elo_diff_abs'],
                                            bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 1000],
                                            labels=['0-50', '50-100', '100-150', '150-200',
                                                   '200-250', '250-300', '300-350', '350-400', '400+'])
    
    accuracy_by_elo_diff = test_with_preds.groupby('elo_diff_bin')['correct'].mean()
    accuracy_by_elo_diff = pd.DataFrame(accuracy_by_elo_diff)
    accuracy_by_elo_diff.columns = ['accuracy']
    accuracy_by_elo_diff['count'] = test_with_preds.groupby('elo_diff_bin').size()
    accuracy_by_elo_diff.to_csv(f"{dataset_dir}/accuracy_by_elo_diff.csv")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=accuracy_by_elo_diff.index, y='accuracy', data=accuracy_by_elo_diff)
    plt.title(f'Prediction Accuracy by ELO Difference ({dataset} Dataset)')
    plt.xlabel('ELO Difference')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(accuracy_by_elo_diff['accuracy']):
        plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_elo_diff['count'].iloc[i]})", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/accuracy_by_elo_diff.png")
    
    # Analyze errors by home/away
    accuracy_by_location = pd.DataFrame({
        'Home Win': test_with_preds[test_with_preds['result'] == 1]['correct'].mean(),
        'Away Win': test_with_preds[test_with_preds['result'] == 0]['correct'].mean()
    }, index=['accuracy'])
    accuracy_by_location['Overall'] = test_with_preds['correct'].mean()
    accuracy_by_location = accuracy_by_location.transpose()
    accuracy_by_location['count'] = [
        len(test_with_preds[test_with_preds['result'] == 1]),
        len(test_with_preds[test_with_preds['result'] == 0]),
        len(test_with_preds)
    ]
    accuracy_by_location.to_csv(f"{dataset_dir}/accuracy_by_location.csv")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracy_by_location.index, y='accuracy', data=accuracy_by_location)
    plt.title(f'Prediction Accuracy by Game Outcome ({dataset} Dataset)')
    plt.xlabel('Actual Outcome')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(accuracy_by_location['accuracy']):
        plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_location['count'].iloc[i]})", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/accuracy_by_location.png")
    
    # Analyze errors by back-to-back games
    accuracy_by_b2b = pd.DataFrame({
        'No B2B': test_with_preds[(test_with_preds['away_back_to_back'] == 0) &
                                  (test_with_preds['home_back_to_back'] == 0)]['correct'].mean(),
        'Away B2B': test_with_preds[(test_with_preds['away_back_to_back'] == 1) &
                                   (test_with_preds['home_back_to_back'] == 0)]['correct'].mean(),
        'Home B2B': test_with_preds[(test_with_preds['away_back_to_back'] == 0) &
                                   (test_with_preds['home_back_to_back'] == 1)]['correct'].mean(),
        'Both B2B': test_with_preds[(test_with_preds['away_back_to_back'] == 1) &
                                   (test_with_preds['home_back_to_back'] == 1)]['correct'].mean()
    }, index=['accuracy'])
    accuracy_by_b2b = accuracy_by_b2b.transpose()
    accuracy_by_b2b['count'] = [
        len(test_with_preds[(test_with_preds['away_back_to_back'] == 0) & (test_with_preds['home_back_to_back'] == 0)]),
        len(test_with_preds[(test_with_preds['away_back_to_back'] == 1) & (test_with_preds['home_back_to_back'] == 0)]),
        len(test_with_preds[(test_with_preds['away_back_to_back'] == 0) & (test_with_preds['home_back_to_back'] == 1)]),
        len(test_with_preds[(test_with_preds['away_back_to_back'] == 1) & (test_with_preds['home_back_to_back'] == 1)])
    ]
    accuracy_by_b2b.to_csv(f"{dataset_dir}/accuracy_by_b2b.csv")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracy_by_b2b.index, y='accuracy', data=accuracy_by_b2b)
    plt.title(f'Prediction Accuracy by Back-to-Back Status ({dataset} Dataset)')
    plt.xlabel('Back-to-Back Status')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)
    for i, v in enumerate(accuracy_by_b2b['accuracy']):
        plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_b2b['count'].iloc[i]})", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/accuracy_by_b2b.png")
    
    # Analyze upset prediction performance
    print(f"Analyzing upset prediction performance for {dataset} dataset...")
    
    # Define upsets based on ELO ratings
    test_with_preds['favorite'] = np.where(test_with_preds['home_elo_i'] > test_with_preds['away_elo_i'], 1, 0)
    test_with_preds['upset'] = np.where(test_with_preds['favorite'] != test_with_preds['result'], 1, 0)
    
    # Calculate accuracy for upset and non-upset games
    accuracy_by_upset = pd.DataFrame({
        'Non-Upset': test_with_preds[test_with_preds['upset'] == 0]['correct'].mean(),
        'Upset': test_with_preds[test_with_preds['upset'] == 1]['correct'].mean()
    }, index=['accuracy'])
    accuracy_by_upset = accuracy_by_upset.transpose()
    accuracy_by_upset['count'] = [
        len(test_with_preds[test_with_preds['upset'] == 0]),
        len(test_with_preds[test_with_preds['upset'] == 1])
    ]
    accuracy_by_upset.to_csv(f"{dataset_dir}/accuracy_by_upset.csv")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracy_by_upset.index, y='accuracy', data=accuracy_by_upset)
    plt.title(f'Prediction Accuracy by Upset Status ({dataset} Dataset)')
    plt.xlabel('Upset Status')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(accuracy_by_upset['accuracy']):
        plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_upset['count'].iloc[i]})", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/accuracy_by_upset.png")
    
    # Calculate upset rate
    upset_count = len(test_with_preds[test_with_preds['upset'] == 1])
    total_count = len(test_with_preds)
    upset_rate = (upset_count / total_count) * 100
    print(f"Upset rate in test data for {dataset} dataset: {upset_rate:.1f}%")
    
    # Compare with ELO predictions
    elo_accuracy = (test_with_preds['elo_pred'] == test_with_preds['result']).mean()
    elo_accuracy_non_upset = test_with_preds[test_with_preds['upset'] == 0].apply(
        lambda x: x['elo_pred'] == x['result'], axis=1).mean()
    elo_accuracy_upset = test_with_preds[test_with_preds['upset'] == 1].apply(
        lambda x: x['elo_pred'] == x['result'], axis=1).mean()
    
    model_accuracy = test_with_preds['correct'].mean()
    model_accuracy_non_upset = test_with_preds[test_with_preds['upset'] == 0]['correct'].mean()
    model_accuracy_upset = test_with_preds[test_with_preds['upset'] == 1]['correct'].mean()
    
    comparison_data = pd.DataFrame({
        'ELO': [elo_accuracy, elo_accuracy_non_upset, elo_accuracy_upset],
        best_model_name: [model_accuracy, model_accuracy_non_upset, model_accuracy_upset]
    }, index=['Overall', 'Non-Upset', 'Upset'])
    comparison_data.to_csv(f"{dataset_dir}/elo_vs_model_comparison.csv")
    
    plt.figure(figsize=(12, 8))
    comparison_data.plot(kind='bar')
    plt.title(f'ELO vs Model Accuracy ({dataset} Dataset)')
    plt.xlabel('Game Type')
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.legend(title='Prediction Method')
    for i, v in enumerate(comparison_data.values.flatten()):
        plt.text(i % 3 + (i // 3) * 0.25 - 0.1, v + 0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/elo_vs_model_comparison.png")

# Create overall summary report
with open('evaluation/model_evaluation_summary.md', 'w') as f:
    f.write("# NBA Game Prediction Model Evaluation\n\n")
    
    f.write("## Model Accuracy Across All Approaches\n\n")
    f.write("| Model | Test Accuracy |\n")
    f.write("|-------|---------------|\n")
    for name, accuracy in accuracies_df.iterrows():
        f.write(f"| {name} | {accuracy['accuracy']:.4f} |\n")
    
    f.write("\n## Best Model Overall\n\n")
    best_overall = accuracies_df.index[0]
    best_accuracy = accuracies_df.iloc[0, 0]
    f.write(f"The best overall model is **{best_overall}** with a test accuracy of {best_accuracy:.4f}.\n\n")
    
    f.write("\n## Dataset-Specific Results\n\n")
    for dataset in datasets:
        f.write(f"### {dataset} Dataset\n\n")
        
        # Get best model for this dataset
        dataset_accuracies = {name: results['accuracy'] for name, results in model_results[dataset].items()}
        best_model_name = max(dataset_accuracies, key=dataset_accuracies.get)
        best_accuracy = dataset_accuracies[best_model_name]
        
        f.write(f"The best model for the {dataset} dataset is **{best_model_name}** with a test accuracy of {best_accuracy:.4f}.\n\n")
        
        # Add comparison with ELO
        comparison_data = pd.read_csv(f"evaluation/{dataset.lower()}/elo_vs_model_comparison.csv", index_col=0)
        
        f.write("#### ELO vs Model Comparison\n\n")
        f.write("| Game Type | ELO | Model |\n")
        f.write("|-----------|-----|-------|\n")
        for idx, row in comparison_data.iterrows():
            f.write(f"| {idx} | {row['ELO']:.4f} | {row[best_model_name]:.4f} |\n")
        
        f.write("\n#### Upset Prediction Performance\n\n")
        accuracy_by_upset = pd.read_csv(f"evaluation/{dataset.lower()}/accuracy_by_upset.csv", index_col=0)
        
        f.write("| Upset Status | Accuracy | Count |\n")
        f.write("|-------------|----------|-------|\n")
        for idx, row in accuracy_by_upset.iterrows():
            f.write(f"| {idx} | {row['accuracy']:.4f} | {int(row['count'])} |\n")
        
        f.write("\n")

print("Model evaluation completed. Results saved to evaluation/")