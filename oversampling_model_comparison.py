import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Create output directory for oversampling comparison
if not os.path.exists('oversampling_comparison'):
    os.makedirs('oversampling_comparison')

# Load the processed data
print("Loading processed data...")
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)
train_data_original = pd.read_csv('processed_data/train_data_original.csv', index_col=0)
train_data_ros = pd.read_csv('processed_data/train_data_ros.csv', index_col=0)
train_data_smote = pd.read_csv('processed_data/train_data_smote.csv', index_col=0)

# Load feature names 
with open('processed_data/feature_names.txt', 'r') as f:
    features = [line.strip() for line in f.readlines() if line.strip()]

print(f"Using {len(features)} features for model building.")

# Display class distributions
def print_class_distribution(dataset, name):
    upset_count = sum(dataset['upset'] == 1)
    non_upset_count = sum(dataset['upset'] == 0)
    total = upset_count + non_upset_count
    print(f"{name} dataset:")
    print(f"  Upset games: {upset_count} ({upset_count/total*100:.1f}%)")
    print(f"  Non-upset games: {non_upset_count} ({non_upset_count/total*100:.1f}%)")
    print(f"  Total samples: {total}")

print("\n--- Dataset Class Distributions ---")
print_class_distribution(train_data_original, "Original training")
print_class_distribution(train_data_ros, "Random Oversampling")
print_class_distribution(train_data_smote, "SMOTE")
print_class_distribution(test_data, "Test")

# Prepare data for each approach
datasets = {
    'Original': train_data_original,
    'Random Oversampling': train_data_ros,
    'SMOTE': train_data_smote
}

X_test = test_data[features]
y_test = test_data['result']

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

# Train and evaluate models for each dataset
results = {}
feature_importances = {}
confusion_matrices = {}

for dataset_name, train_data in datasets.items():
    print(f"\n--- Training models on {dataset_name} dataset ---")
    
    X_train = train_data[features]
    y_train = train_data['result']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate upset-specific metrics
        test_with_preds = test_data.copy()
        test_with_preds['predicted'] = y_pred
        test_with_preds['correct'] = test_with_preds['predicted'] == test_with_preds['result']
        
        # Analyze upset prediction performance
        upset_samples = test_with_preds[test_with_preds['upset'] == 1]
        non_upset_samples = test_with_preds[test_with_preds['upset'] == 0]
        
        upset_accuracy = upset_samples['correct'].mean() if len(upset_samples) > 0 else 0
        non_upset_accuracy = non_upset_samples['correct'].mean() if len(non_upset_samples) > 0 else 0
        
        # Calculate recall for predicting upsets correctly
        # For each actual upset (favorite != result), check if we correctly predicted the result
        upset_recall = upset_samples['correct'].mean() if len(upset_samples) > 0 else 0
        
        # Calculate precision for upset predictions
        # First, identify predictions that imply an upset
        test_with_preds['predicted_upset'] = (test_with_preds['favorite'] != test_with_preds['predicted']).astype(int)
        predicted_upset_samples = test_with_preds[test_with_preds['predicted_upset'] == 1]
        upset_precision = predicted_upset_samples['correct'].mean() if len(predicted_upset_samples) > 0 else 0
        
        # Count actual upsets correctly identified
        correctly_predicted_upsets = len(upset_samples[upset_samples['correct'] == True])
        total_upsets = len(upset_samples)
        upset_predicted_count = len(predicted_upset_samples)
        upset_correct_predictions = len(predicted_upset_samples[predicted_upset_samples['correct'] == True])
        
        # Store results
        if dataset_name not in results:
            results[dataset_name] = {}
        
        results[dataset_name][model_name] = {
            'accuracy': accuracy,
            'non_upset_accuracy': non_upset_accuracy,
            'upset_accuracy': upset_accuracy,
            'upset_recall': upset_recall,
            'upset_precision': upset_precision,
            'correct_upset_count': correctly_predicted_upsets,
            'total_upsets': total_upsets,
            'predicted_upset_count': upset_predicted_count,
            'correct_predictions_of_upsets': upset_correct_predictions
        }
        
        # Store confusion matrix
        cm = confusion_matrix(test_with_preds['upset'], test_with_preds['predicted_upset'])
        if dataset_name not in confusion_matrices:
            confusion_matrices[dataset_name] = {}
        confusion_matrices[dataset_name][model_name] = cm
        
        # Save confusion matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Upset Predicted', 'Upset Predicted'],
                   yticklabels=['No Actual Upset', 'Actual Upset'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Upset Confusion Matrix - {dataset_name} - {model_name}')
        plt.tight_layout()
        plt.savefig(f'oversampling_comparison/confusion_matrix_{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}.png')
        
        # Extract feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            if dataset_name not in feature_importances:
                feature_importances[dataset_name] = {}
            feature_importances[dataset_name][model_name] = {features[i]: importances[indices][i] for i in range(len(features))}
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances - {dataset_name} - {model_name}')
            plt.bar(range(len(features)), importances[indices], align='center')
            plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'oversampling_comparison/feature_importance_{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}.png')
        
        # Save the model
        joblib.dump(model, f'oversampling_comparison/{dataset_name.lower().replace(" ", "_")}_{model_name.lower().replace(" ", "_")}_model.pkl')
        joblib.dump(scaler, f'oversampling_comparison/{dataset_name.lower().replace(" ", "_")}_scaler.pkl')

# Create comparison tables
print("\n--- Creating comparison tables ---")

# Prepare result DataFrames
accuracy_df = pd.DataFrame(index=models.keys(), columns=datasets.keys())
upset_recall_df = pd.DataFrame(index=models.keys(), columns=datasets.keys())
upset_precision_df = pd.DataFrame(index=models.keys(), columns=datasets.keys())
correct_upset_count_df = pd.DataFrame(index=models.keys(), columns=datasets.keys())

# Fill in the DataFrames
for dataset_name in datasets.keys():
    for model_name in models.keys():
        accuracy_df.loc[model_name, dataset_name] = results[dataset_name][model_name]['accuracy']
        upset_recall_df.loc[model_name, dataset_name] = results[dataset_name][model_name]['upset_recall']
        upset_precision_df.loc[model_name, dataset_name] = results[dataset_name][model_name]['upset_precision']
        correct_upset_count_df.loc[model_name, dataset_name] = f"{results[dataset_name][model_name]['correct_upset_count']}/{results[dataset_name][model_name]['total_upsets']}"

# Save comparison tables
accuracy_df.to_csv('oversampling_comparison/accuracy_comparison.csv')
upset_recall_df.to_csv('oversampling_comparison/upset_recall_comparison.csv')
upset_precision_df.to_csv('oversampling_comparison/upset_precision_comparison.csv')
correct_upset_count_df.to_csv('oversampling_comparison/correct_upset_count_comparison.csv')

# Create visualizations
plt.figure(figsize=(14, 8))
ax = sns.heatmap(accuracy_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('Overall Accuracy Comparison')
plt.tight_layout()
plt.savefig('oversampling_comparison/accuracy_comparison.png')

plt.figure(figsize=(14, 8))
ax = sns.heatmap(upset_recall_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('Upset Recall Comparison')
plt.tight_layout()
plt.savefig('oversampling_comparison/upset_recall_comparison.png')

plt.figure(figsize=(14, 8))
ax = sns.heatmap(upset_precision_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('Upset Precision Comparison')
plt.tight_layout()
plt.savefig('oversampling_comparison/upset_precision_comparison.png')

# Bar chart for upset recall comparison
plt.figure(figsize=(14, 8))
upset_recall_df.plot(kind='bar', figsize=(14, 8))
plt.title('Upset Recall by Model and Sampling Method')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.ylim(0, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('oversampling_comparison/upset_recall_bar_chart.png')

# Bar chart for precision comparison
plt.figure(figsize=(14, 8))
upset_precision_df.plot(kind='bar', figsize=(14, 8))
plt.title('Upset Precision by Model and Sampling Method')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.ylim(0, 1.0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('oversampling_comparison/upset_precision_bar_chart.png')

# Create comprehensive summary report
with open('oversampling_comparison/oversampling_comparison_report.md', 'w') as f:
    f.write("# Oversampling Methods Comparison for NBA Game Prediction\n\n")
    
    f.write("## Dataset Statistics\n\n")
    f.write("| Dataset | Total Samples | Upset Games | Non-Upset Games | Upset Percentage |\n")
    f.write("|---------|---------------|-------------|-----------------|------------------|\n")
    
    for name, data in datasets.items():
        upset = sum(data['upset'] == 1)
        non_upset = sum(data['upset'] == 0)
        total = upset + non_upset
        f.write(f"| {name} | {total} | {upset} | {non_upset} | {upset/total*100:.1f}% |\n")
    
    upset_test = sum(test_data['upset'] == 1)
    non_upset_test = sum(test_data['upset'] == 0)
    total_test = upset_test + non_upset_test
    f.write(f"| Test | {total_test} | {upset_test} | {non_upset_test} | {upset_test/total_test*100:.1f}% |\n\n")
    
    f.write("## Overall Accuracy\n\n")
    f.write("![Overall Accuracy Comparison](accuracy_comparison.png)\n\n")
    f.write("| Model | " + " | ".join(datasets.keys()) + " |\n")
    f.write("|" + "---|" * (len(datasets) + 1) + "\n")
    
    for model_name in models.keys():
        line = f"| {model_name} |"
        for dataset_name in datasets.keys():
            line += f" {results[dataset_name][model_name]['accuracy']:.4f} |"
        f.write(line + "\n")
    
    f.write("\n## Upset Prediction Performance\n\n")
    f.write("### Upset Recall (Correctly Identified Upsets / Total Actual Upsets)\n\n")
    f.write("![Upset Recall Comparison](upset_recall_comparison.png)\n\n")
    f.write("![Upset Recall Bar Chart](upset_recall_bar_chart.png)\n\n")
    f.write("| Model | " + " | ".join(datasets.keys()) + " |\n")
    f.write("|" + "---|" * (len(datasets) + 1) + "\n")
    
    for model_name in models.keys():
        line = f"| {model_name} |"
        for dataset_name in datasets.keys():
            correct = results[dataset_name][model_name]['correct_upset_count']
            total = results[dataset_name][model_name]['total_upsets']
            line += f" {results[dataset_name][model_name]['upset_recall']:.4f} ({correct}/{total}) |"
        f.write(line + "\n")
    
    f.write("\n### Upset Precision (Correctly Predicted Upsets / Total Predicted Upsets)\n\n")
    f.write("![Upset Precision Comparison](upset_precision_comparison.png)\n\n")
    f.write("![Upset Precision Bar Chart](upset_precision_bar_chart.png)\n\n")
    f.write("| Model | " + " | ".join(datasets.keys()) + " |\n")
    f.write("|" + "---|" * (len(datasets) + 1) + "\n")
    
    for model_name in models.keys():
        line = f"| {model_name} |"
        for dataset_name in datasets.keys():
            correct = results[dataset_name][model_name]['correct_predictions_of_upsets']
            total = results[dataset_name][model_name]['predicted_upset_count']
            ratio = results[dataset_name][model_name]['upset_precision']
            line += f" {ratio:.4f} ({correct}/{total}) |"
        f.write(line + "\n")
    
    f.write("\n## Confusion Matrices\n\n")
    
    for dataset_name in datasets.keys():
        f.write(f"### {dataset_name}\n\n")
        
        for model_name in models.keys():
            f.write(f"#### {model_name}\n\n")
            f.write(f"![Confusion Matrix](confusion_matrix_{dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.png)\n\n")
    
    f.write("\n## Conclusion\n\n")
    
    # Find best model for upset recall
    best_recall = 0
    best_dataset = ""
    best_model = ""
    
    for dataset_name in datasets.keys():
        for model_name in models.keys():
            recall = results[dataset_name][model_name]['upset_recall']
            if recall > best_recall:
                best_recall = recall
                best_dataset = dataset_name
                best_model = model_name
    
    f.write(f"The best model for predicting upsets is **{best_model}** trained with **{best_dataset}**, ")
    f.write(f"achieving an upset recall of **{best_recall:.4f}**.\n\n")
    
    # Compare with original performance
    original_best_recall = max([results['Original'][model]['upset_recall'] for model in models.keys()])
    improvement = (best_recall / original_best_recall - 1) * 100
    
    f.write(f"This represents a **{improvement:.1f}%** improvement in upset recall compared to ")
    f.write(f"the best model trained on the original imbalanced dataset (recall: {original_best_recall:.4f}).\n\n")
    
    # Comment on trade-offs
    f.write("### Trade-offs\n\n")
    f.write("While oversampling improves upset recall, there are trade-offs to consider:\n\n")
    
    best_overall_acc = 0
    best_acc_dataset = ""
    best_acc_model = ""
    
    for dataset_name in datasets.keys():
        for model_name in models.keys():
            acc = results[dataset_name][model_name]['accuracy']
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_acc_dataset = dataset_name
                best_acc_model = model_name
    
    original_best_acc = max([results['Original'][model]['accuracy'] for model in models.keys()])
    acc_change = (best_overall_acc / original_best_acc - 1) * 100
    
    if acc_change >= 0:
        f.write(f"1. **Overall Accuracy**: The best overall accuracy ({best_overall_acc:.4f}) was achieved by ")
        f.write(f"**{best_acc_model}** trained on **{best_acc_dataset}**, representing a ")
        f.write(f"**{acc_change:.1f}%** improvement over the best original model.\n\n")
    else:
        f.write(f"1. **Overall Accuracy Trade-off**: The best overall accuracy ({best_overall_acc:.4f}) was achieved by ")
        f.write(f"**{best_acc_model}** trained on **{best_acc_dataset}**, representing a ")
        f.write(f"**{-acc_change:.1f}%** decrease compared to the best original model ({original_best_acc:.4f}).\n\n")
    
    best_precision = 0
    best_prec_dataset = ""
    best_prec_model = ""
    
    for dataset_name in datasets.keys():
        for model_name in models.keys():
            prec = results[dataset_name][model_name]['upset_precision']
            if prec > best_precision:
                best_precision = prec
                best_prec_dataset = dataset_name
                best_prec_model = model_name
    
    original_best_prec = max([results['Original'][model]['upset_precision'] for model in models.keys()])
    prec_change = (best_precision / original_best_prec - 1) * 100 if original_best_prec > 0 else float('inf')
    
    if original_best_prec == 0:
        f.write(f"2. **Upset Precision**: The best upset precision ({best_precision:.4f}) was achieved by ")
        f.write(f"**{best_prec_model}** trained on **{best_prec_dataset}**, compared to no correct upset ")
        f.write(f"predictions with the original models.\n")
    elif prec_change >= 0:
        f.write(f"2. **Upset Precision**: The best upset precision ({best_precision:.4f}) was achieved by ")
        f.write(f"**{best_prec_model}** trained on **{best_prec_dataset}**, representing a ")
        f.write(f"**{prec_change:.1f}%** improvement over the best original model.\n")
    else:
        f.write(f"2. **Upset Precision Trade-off**: The best upset precision ({best_precision:.4f}) was achieved by ")
        f.write(f"**{best_prec_model}** trained on **{best_prec_dataset}**, representing a ")
        f.write(f"**{-prec_change:.1f}%** decrease compared to the best original model ({original_best_prec:.4f}).\n")

print("\nOversampling comparison analysis completed. Results saved to oversampling_comparison/ directory.")
