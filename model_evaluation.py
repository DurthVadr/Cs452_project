import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
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
train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Load feature names
with open('processed_data/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]  # Skip empty lines

# Create X and y data from the train and test dataframes
X_train = train_data[feature_names].values
y_train = train_data['result'].values
X_test = test_data[feature_names].values
y_test = test_data['result'].values

# Save the new X and y data
np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/X_test.npy', X_test)
np.save('processed_data/y_test.npy', y_test)

print(f"Using {len(feature_names)} features for model evaluation.")

# Load or train models
print("Loading models...")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Function to check if model needs retraining
def check_model_features(model, expected_features):
    """Check if model was trained with the expected number of features"""
    if hasattr(model, 'n_features_in_'):
        return model.n_features_in_ == expected_features
    elif hasattr(model, 'coef_') and len(model.coef_.shape) > 1:
        return model.coef_.shape[1] == expected_features
    else:
        # If we can't determine, assume it needs retraining
        return False

# Scale features for training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

try:
    # Try to load the best models from the full model building script
    best_models = {}
    models_to_retrain = []

    for model_name in ['random_forest_best.pkl', 'logistic_regression_best.pkl', 'gradient_boosting_best.pkl']:
        if os.path.exists(f'models/{model_name}'):
            model = joblib.load(f'models/{model_name}')
            display_name = model_name.replace('_best.pkl', '').replace('_', ' ').title()

            # Check if model needs retraining due to feature count mismatch
            if check_model_features(model, len(feature_names)):
                best_models[display_name] = model
                print(f"Loaded {display_name} model successfully.")
            else:
                print(f"{display_name} model has feature count mismatch. Will retrain.")
                models_to_retrain.append(display_name)

    # Retrain models if needed
    if models_to_retrain:
        print(f"Retraining models with new features: {', '.join(models_to_retrain)}")

        if 'Random Forest' in models_to_retrain:
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            best_models['Random Forest'] = rf_model
            # Save the retrained model with a different name to avoid overwriting
            joblib.dump(rf_model, 'models/random_forest_retrained.pkl')
            print("Retrained Random Forest model.")

        if 'Logistic Regression' in models_to_retrain:
            lr_model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            best_models['Logistic Regression'] = lr_model
            joblib.dump(lr_model, 'models/logistic_regression_retrained.pkl')
            print("Retrained Logistic Regression model.")

        if 'Gradient Boosting' in models_to_retrain:
            gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            gb_model.fit(X_train_scaled, y_train)
            best_models['Gradient Boosting'] = gb_model
            joblib.dump(gb_model, 'models/gradient_boosting_retrained.pkl')
            print("Retrained Gradient Boosting model.")

    # Load upset models (we'll retrain these if needed in a separate step)
    if os.path.exists('models/home_upset_model.pkl'):
        home_upset_model = joblib.load('models/home_upset_model.pkl')
        away_upset_model = joblib.load('models/away_upset_model.pkl')
        home_upset_scaler = joblib.load('models/home_upset_scaler.pkl')
        away_upset_scaler = joblib.load('models/away_upset_scaler.pkl')
    else:
        home_upset_model = None
        away_upset_model = None
        home_upset_scaler = None
        away_upset_scaler = None
except Exception as e:
    print(f"Error loading models: {e}")
    # If models from full script aren't available, use simplified models
    best_models = {}
    models_to_retrain = []

    for model_name in ['logistic_regression_model.pkl', 'random_forest_model.pkl', 'gradient_boosting_model.pkl']:
        if os.path.exists(f'models/{model_name}'):
            model = joblib.load(f'models/{model_name}')
            display_name = model_name.replace('_model.pkl', '').replace('_', ' ').title()

            # Check if model needs retraining due to feature count mismatch
            if check_model_features(model, len(feature_names)):
                best_models[display_name] = model
                print(f"Loaded {display_name} model successfully.")
            else:
                print(f"{display_name} model has feature count mismatch. Will retrain.")
                models_to_retrain.append(display_name)

    # Retrain models if needed
    if models_to_retrain:
        print(f"Retraining models with new features: {', '.join(models_to_retrain)}")

        if 'Random Forest' in models_to_retrain:
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            best_models['Random Forest'] = rf_model
            joblib.dump(rf_model, 'models/random_forest_retrained.pkl')
            print("Retrained Random Forest model.")

        if 'Logistic Regression' in models_to_retrain:
            lr_model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            best_models['Logistic Regression'] = lr_model
            joblib.dump(lr_model, 'models/logistic_regression_retrained.pkl')
            print("Retrained Logistic Regression model.")

        if 'Gradient Boosting' in models_to_retrain:
            gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            gb_model.fit(X_train_scaled, y_train)
            best_models['Gradient Boosting'] = gb_model
            joblib.dump(gb_model, 'models/gradient_boosting_retrained.pkl')
            print("Retrained Gradient Boosting model.")

    # Load upset model
    if os.path.exists('models/upset_model.pkl'):
        upset_model = joblib.load('models/upset_model.pkl')
        upset_scaler = joblib.load('models/upset_scaler.pkl')
    else:
        upset_model = None
        upset_scaler = None

# If no models were loaded or retrained, train simple models
if not best_models:
    print("No pre-trained models found. Training simple models...")

    best_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
        'Random Forest': RandomForestClassifier(random_state=42).fit(X_train_scaled, y_train),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42).fit(X_train_scaled, y_train)
    }

    # Save the newly trained models
    for name, model in best_models.items():
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_new.pkl')

    print("Trained and saved new models.")

# Evaluate models
print("Evaluating models...")
model_results = {}

for name, model in best_models.items():
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Calculate ROC curve and AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
    else:
        fpr, tpr, roc_auc = None, None, None
        precision, recall, pr_auc = None, None, None

    # Store results
    model_results[name] = {
        'accuracy': accuracy,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    }

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
    plt.savefig(f'evaluation/{name.replace(" ", "_").lower()}_confusion_matrix.png')

    # Plot ROC curve if available
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'evaluation/{name.replace(" ", "_").lower()}_roc_curve.png')

    # Plot precision-recall curve if available
    if precision is not None and recall is not None:
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f'evaluation/{name.replace(" ", "_").lower()}_pr_curve.png')

# Compare model accuracies
accuracies = {name: results['accuracy'] for name, results in model_results.items()}
accuracies_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['accuracy'])
accuracies_df = accuracies_df.sort_values('accuracy', ascending=False)
accuracies_df.to_csv('evaluation/model_accuracies.csv')

# Visualize model accuracies
plt.figure(figsize=(12, 6))
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

# Analyze prediction errors
print("Analyzing prediction errors...")

# Get best model
best_model_name = accuracies_df.index[0]
best_model = best_models[best_model_name]

# Make predictions with best model
y_pred_best = best_model.predict(X_test)

# Create DataFrame with test data and predictions
test_with_preds = test_data.copy()
test_with_preds['predicted'] = y_pred_best
test_with_preds['correct'] = test_with_preds['predicted'] == test_with_preds['result']

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
accuracy_by_elo_diff.to_csv('evaluation/accuracy_by_elo_diff.csv')

plt.figure(figsize=(12, 6))
sns.barplot(x=accuracy_by_elo_diff.index, y='accuracy', data=accuracy_by_elo_diff)
plt.title('Prediction Accuracy by ELO Difference')
plt.xlabel('ELO Difference')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 1.0)
for i, v in enumerate(accuracy_by_elo_diff['accuracy']):
    plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_elo_diff['count'].iloc[i]})", ha='center')
plt.tight_layout()
plt.savefig('evaluation/accuracy_by_elo_diff.png')

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
accuracy_by_location.to_csv('evaluation/accuracy_by_location.csv')

plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_by_location.index, y='accuracy', data=accuracy_by_location)
plt.title('Prediction Accuracy by Game Outcome')
plt.xlabel('Actual Outcome')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
for i, v in enumerate(accuracy_by_location['accuracy']):
    plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_location['count'].iloc[i]})", ha='center')
plt.tight_layout()
plt.savefig('evaluation/accuracy_by_location.png')

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
    len(test_with_preds[(test_with_preds['away_back_to_back'] == 0) &
                        (test_with_preds['home_back_to_back'] == 0)]),
    len(test_with_preds[(test_with_preds['away_back_to_back'] == 1) &
                        (test_with_preds['home_back_to_back'] == 0)]),
    len(test_with_preds[(test_with_preds['away_back_to_back'] == 0) &
                        (test_with_preds['home_back_to_back'] == 1)]),
    len(test_with_preds[(test_with_preds['away_back_to_back'] == 1) &
                        (test_with_preds['home_back_to_back'] == 1)])
]
accuracy_by_b2b.to_csv('evaluation/accuracy_by_b2b.csv')

plt.figure(figsize=(12, 6))
sns.barplot(x=accuracy_by_b2b.index, y='accuracy', data=accuracy_by_b2b)
plt.title('Prediction Accuracy by Back-to-Back Games')
plt.xlabel('Back-to-Back Status')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
for i, v in enumerate(accuracy_by_b2b['accuracy']):
    plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_b2b['count'].iloc[i]})", ha='center')
plt.tight_layout()
plt.savefig('evaluation/accuracy_by_b2b.png')

# Analyze upset prediction performance
print("Analyzing upset prediction performance...")

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
accuracy_by_upset.to_csv('evaluation/accuracy_by_upset.csv')

plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_by_upset.index, y='accuracy', data=accuracy_by_upset)
plt.title('Prediction Accuracy by Upset Status')
plt.xlabel('Game Type')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
for i, v in enumerate(accuracy_by_upset['accuracy']):
    plt.text(i, v + 0.01, f"{v:.3f} (n={accuracy_by_upset['count'].iloc[i]})", ha='center')
plt.tight_layout()
plt.savefig('evaluation/accuracy_by_upset.png')

# Calculate upset rate
upset_rate = test_with_preds['upset'].mean() * 100
print(f"Upset rate in test data: {upset_rate:.1f}%")

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
comparison_data.to_csv('evaluation/elo_vs_model_comparison.csv')

plt.figure(figsize=(12, 8))
comparison_data.plot(kind='bar')
plt.title('ELO vs Model Accuracy')
plt.xlabel('Game Type')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
plt.legend(title='Prediction Method')
for i, v in enumerate(comparison_data.values.flatten()):
    plt.text(i % 3 + (i // 3) * 0.25 - 0.1, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig('evaluation/elo_vs_model_comparison.png')

# Create summary report
with open('evaluation/model_evaluation_summary.md', 'w') as f:
    f.write("# NBA Game Prediction Model Evaluation\n\n")

    f.write("## Model Accuracy\n\n")
    f.write("| Model | Test Accuracy |\n")
    f.write("|-------|---------------|\n")
    for name, accuracy in accuracies_df.iterrows():
        f.write(f"| {name} | {accuracy['accuracy']:.4f} |\n")

    f.write("\n## Best Model Performance\n\n")
    f.write(f"The best performing model is **{best_model_name}** with a test accuracy of {accuracies_df.iloc[0, 0]:.4f}.\n\n")

    # Add classification report
    report = model_results[best_model_name]['report']
    f.write
    f.write("### Classification Report\n\n")
    f.write("| Class | Precision | Recall | F1-Score | Support |\n")
    f.write("|-------|-----------|--------|----------|--------|\n")
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            f.write(f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']} |\n")

