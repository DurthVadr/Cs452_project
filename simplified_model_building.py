import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Create output directory for model building
if not os.path.exists('models'):
    os.makedirs('models')

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Load different training datasets
print("Loading different sampling method datasets...")
datasets = {
    'Original': pd.read_csv('processed_data/train_data_original.csv', index_col=0),
    'ROS': pd.read_csv('processed_data/train_data_ros.csv', index_col=0),
    'SMOTE': pd.read_csv('processed_data/train_data_smote.csv', index_col=0)
}

# Load feature names
with open('processed_data/feature_names.txt', 'r') as f:
    features = [line.strip() for line in f.readlines() if line.strip()]  # Skip empty lines

print(f"Using {len(features)} features for model building.")

# Define X and y for test data
X_test = test_data[features]
y_test = test_data['result']

# Scale test features (will scale train features for each dataset separately)
test_scaler = StandardScaler()
X_test_scaled = test_scaler.fit_transform(X_test)

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

# Dictionary to store results for all datasets
all_results = {}

# Train and evaluate models for each dataset
for dataset_name, train_data in datasets.items():
    print(f"\n{'-'*50}")
    print(f"Processing {dataset_name} dataset...")
    print(f"{'-'*50}")
    
    # Create directory for this dataset's models
    dataset_dir = f"models/{dataset_name.lower()}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Create X and y for training data
    X_train = train_data[features]
    y_train = train_data['result']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled_for_dataset = scaler.transform(X_test)  # Re-scale test data using this dataset's scaler
    
    # Save scaler
    joblib.dump(scaler, f"{dataset_dir}/scaler.pkl")
    
    # Evaluate each model with cross-validation
    cv_results = {}
    for name, model in models.items():
        print(f"Cross-validating {name} on {dataset_name} dataset...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
    
    # Convert CV results to DataFrame
    cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index')
    cv_results_df = cv_results_df.sort_values('mean_cv_score', ascending=False)
    cv_results_df.to_csv(f"{dataset_dir}/baseline_cv_results.csv")
    
    # Visualize cross-validation results
    plt.figure(figsize=(12, 6))
    cv_plot = sns.barplot(x=cv_results_df.index, y='mean_cv_score', data=cv_results_df)
    plt.title(f'Model Cross-Validation Accuracy ({dataset_name} Dataset)')
    plt.xlabel('Model')
    plt.ylabel('Mean CV Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 0.8)
    for i, v in enumerate(cv_results_df['mean_cv_score']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/baseline_cv_results.png")
    
    # Train and evaluate models
    test_results = {}
    for name, model in models.items():
        print(f"Training and evaluating {name} on {dataset_name} dataset...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled_for_dataset)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        test_results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Away Win', 'Home Win'],
                   yticklabels=['Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix ({dataset_name} Dataset)')
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/{name.replace(' ', '_').lower()}_confusion_matrix.png")
        
        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({name} - {dataset_name} Dataset)')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{dataset_dir}/{name.replace(' ', '_').lower()}_feature_importance.png")
        
        elif name == 'Logistic Regression':
            # For logistic regression, use coefficients
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({name} - {dataset_name} Dataset)')
            plt.bar(range(X_train.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{dataset_dir}/{name.replace(' ', '_').lower()}_feature_importance.png")
        
        # Save model
        joblib.dump(model, f"{dataset_dir}/{name.replace(' ', '_').lower()}_model.pkl")
    
    # Compare test results
    test_accuracies = {name: results['accuracy'] for name, results in test_results.items()}
    test_accuracies_df = pd.DataFrame.from_dict(test_accuracies, orient='index', columns=['accuracy'])
    test_accuracies_df = test_accuracies_df.sort_values('accuracy', ascending=False)
    test_accuracies_df.to_csv(f"{dataset_dir}/test_accuracies.csv")
    
    # Visualize test accuracies
    plt.figure(figsize=(10, 6))
    sns.barplot(x=test_accuracies_df.index, y='accuracy', data=test_accuracies_df)
    plt.title(f'Model Test Accuracy ({dataset_name} Dataset)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 0.8)
    for i, v in enumerate(test_accuracies_df['accuracy']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/test_accuracies.png")
    
    # Select best model for this dataset
    best_model_name = test_accuracies_df.index[0]
    best_model = models[best_model_name]
    print(f"Best model for {dataset_name} dataset: {best_model_name} with test accuracy: {test_accuracies_df.iloc[0, 0]:.4f}")
    
    # Store results for this dataset
    all_results[dataset_name] = {
        'cv_results': cv_results_df,
        'test_results': test_results,
        'best_model_name': best_model_name,
        'best_model_accuracy': test_accuracies_df.iloc[0, 0]
    }
    
    # Develop specialized upset prediction model for this dataset
    print(f"Developing specialized upset prediction model for {dataset_name} dataset...")
    
    # Prepare data for upset prediction
    # Define upsets based on ELO ratings
    train_data['favorite'] = np.where(train_data['home_elo_i'] > train_data['away_elo_i'], 1, 0)
    train_data['upset'] = np.where(train_data['favorite'] != train_data['result'], 1, 0)
    
    # Filter for games where home team is favorite
    home_favorite_games = train_data[train_data['favorite'] == 1]
    X_home_favorite = home_favorite_games[features].values
    y_home_favorite = home_favorite_games['upset'].values
    
    test_data['favorite'] = np.where(test_data['home_elo_i'] > test_data['away_elo_i'], 1, 0)
    test_data['upset'] = np.where(test_data['favorite'] != test_data['result'], 1, 0)
    test_home_favorite = test_data[test_data['favorite'] == 1]
    X_test_home = test_home_favorite[features].values
    y_test_home = test_home_favorite['upset'].values
    
    # Scale features for upset prediction
    scaler_home = StandardScaler()
    X_home_favorite_scaled = scaler_home.fit_transform(X_home_favorite)
    X_test_home_scaled = scaler_home.transform(X_test_home)
    
    # Train upset model using same algorithm as best overall model
    if best_model_name == 'Logistic Regression':
        upset_model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42)
    elif best_model_name == 'Random Forest':
        upset_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    elif best_model_name == 'Gradient Boosting':
        upset_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    else:
        # Default to Random Forest
        upset_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    
    # Train upset model
    upset_model.fit(X_home_favorite_scaled, y_home_favorite)
    upset_pred = upset_model.predict(X_test_home_scaled)
    upset_accuracy = accuracy_score(y_test_home, upset_pred)
    print(f"Upset model accuracy for {dataset_name} dataset: {upset_accuracy:.4f}")
    
    # Save upset model and scaler
    joblib.dump(upset_model, f"{dataset_dir}/upset_model.pkl")
    joblib.dump(scaler_home, f"{dataset_dir}/upset_scaler.pkl")
    
    # Create confusion matrix for upset model
    cm_upset = confusion_matrix(y_test_home, upset_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_upset, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Upset', 'Upset'],
                yticklabels=['No Upset', 'Upset'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Upset Model Confusion Matrix ({dataset_name} Dataset)')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/upset_confusion_matrix.png")
    
    # Store upset model results
    all_results[dataset_name]['upset_accuracy'] = upset_accuracy

# Compare with ELO predictions
elo_accuracy = (test_data['elo_pred'] == test_data['result']).mean()
print(f"ELO prediction accuracy: {elo_accuracy:.4f}")

# Compare all models across datasets
all_best_accuracies = {
    'ELO': elo_accuracy
}

for dataset_name, results in all_results.items():
    model_name = results['best_model_name']
    accuracy = results['best_model_accuracy']
    all_best_accuracies[f"{dataset_name} - {model_name}"] = accuracy

# Convert to DataFrame
all_accuracies_df = pd.DataFrame.from_dict(all_best_accuracies, orient='index', columns=['accuracy'])
all_accuracies_df = all_accuracies_df.sort_values('accuracy', ascending=False)
all_accuracies_df.to_csv('models/all_model_accuracies.csv')

# Visualize all model accuracies
plt.figure(figsize=(12, 8))
sns.barplot(x=all_accuracies_df.index, y='accuracy', data=all_accuracies_df)
plt.title('All Model Test Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 0.8)
for i, v in enumerate(all_accuracies_df['accuracy']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('models/all_model_accuracies.png')

# Create summary report
with open('models/model_summary.md', 'w') as f:
    f.write("# NBA Game Prediction Model Summary\n\n")
    
    for dataset_name, results in all_results.items():
        f.write(f"## {dataset_name} Dataset\n\n")
        
        f.write("### Baseline Models\n\n")
        f.write("Cross-validation results for baseline models:\n\n")
        f.write("| Model | Mean CV Accuracy | Std CV Accuracy |\n")
        f.write("|-------|-----------------|----------------|\n")
        for name, row in results['cv_results'].iterrows():
            f.write(f"| {name} | {row['mean_cv_score']:.4f} | {row['std_cv_score']:.4f} |\n")
        
        f.write("\n### Test Results\n\n")
        f.write("| Model | Test Accuracy |\n")
        f.write("|-------|---------------|\n")
        for name, res in results['test_results'].items():
            f.write(f"| {name} | {res['accuracy']:.4f} |\n")
        
        f.write("\n### Upset Prediction Model\n\n")
        f.write(f"Upset model accuracy: {results['upset_accuracy']:.4f}\n\n")
    
    f.write("\n## Overall Model Comparison\n\n")
    f.write("| Model | Test Accuracy |\n")
    f.write("|-------|---------------|\n")
    for name, accuracy in all_accuracies_df.iterrows():
        f.write(f"| {name} | {accuracy['accuracy']:.4f} |\n")
    
    f.write("\n## Conclusion\n\n")
    best_model_overall = all_accuracies_df.index[0]
    best_accuracy_overall = all_accuracies_df.iloc[0, 0]
    f.write(f"The best performing model is **{best_model_overall}** with a test accuracy of {best_accuracy_overall:.4f}.\n")
    
    # Compare to ELO baseline
    improvement = best_accuracy_overall - elo_accuracy
    f.write(f"\nThis represents an improvement of {improvement:.4f} ({improvement*100:.2f}%) over the baseline ELO model.\n")

print("Model building completed. Results saved to models/")
