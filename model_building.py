import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
X_train = np.load('processed_data/X_train.npy')
y_train = np.load('processed_data/y_train.npy')
X_test = np.load('processed_data/X_test.npy')
y_test = np.load('processed_data/y_test.npy')

# Load feature names
with open('processed_data/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Load combined data for additional analysis
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Build baseline prediction model
print("Building baseline prediction model...")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

# Evaluate each model with cross-validation
cv_results = {}
for name, model in models.items():
    print(f"Cross-validating {name}...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std()
    }

# Convert results to DataFrame
cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index')
cv_results_df = cv_results_df.sort_values('mean_cv_score', ascending=False)
cv_results_df.to_csv('models/baseline_cv_results.csv')

# Visualize cross-validation results
plt.figure(figsize=(12, 6))
cv_plot = sns.barplot(x=cv_results_df.index, y='mean_cv_score', data=cv_results_df)
plt.title('Model Cross-Validation Accuracy')
plt.xlabel('Model')
plt.ylabel('Mean CV Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 0.8)  # Adjust as needed based on results
for i, v in enumerate(cv_results_df['mean_cv_score']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('models/baseline_cv_results.png')

# Select top 3 models for hyperparameter tuning
top_models = cv_results_df.head(3).index.tolist()
print(f"Top 3 models selected for hyperparameter tuning: {top_models}")

# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    },
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
}

# Perform grid search for top models
best_models = {}
for model_name in top_models:
    print(f"Performing grid search for {model_name}...")
    model = models[model_name]
    param_grid = param_grids[model_name]

    # Use a subset of parameters if grid is too large
    if model_name == 'Logistic Regression':
        # Simplify grid for Logistic Regression
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', None],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save best model
    best_models[model_name] = grid_search.best_estimator_

    # Save grid search results
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv(f'models/{model_name.replace(" ", "_").lower()}_grid_search.csv')

    # Print best parameters and score
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")

    # Save best model
    joblib.dump(grid_search.best_estimator_,
                f'models/{model_name.replace(" ", "_").lower()}_best.pkl')

# Evaluate best models on test set
test_results = {}
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
    else:
        fpr, tpr, roc_auc = None, None, None

    # Store results
    test_results[name] = {
        'accuracy': accuracy,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'models/{name.replace(" ", "_").lower()}_confusion_matrix.png')

    # Plot ROC curve if available
    if fpr is not None and tpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'models/{name.replace(" ", "_").lower()}_roc_curve.png')

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({name})')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'models/{name.replace(" ", "_").lower()}_feature_importance.png')

    elif name == 'Logistic Regression':
        # For logistic regression, use coefficients
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({name})')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'models/{name.replace(" ", "_").lower()}_feature_importance.png')

# Compare test results
test_accuracies = {name: results['accuracy'] for name, results in test_results.items()}
test_accuracies_df = pd.DataFrame.from_dict(test_accuracies, orient='index', columns=['accuracy'])
test_accuracies_df = test_accuracies_df.sort_values('accuracy', ascending=False)
test_accuracies_df.to_csv('models/test_accuracies.csv')

# Visualize test accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=test_accuracies_df.index, y='accuracy', data=test_accuracies_df)
plt.title('Model Test Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 0.8)  # Adjust as needed based on results
for i, v in enumerate(test_accuracies_df['accuracy']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('models/test_accuracies.png')

# Select best overall model
best_model_name = test_accuracies_df.index[0]
best_model = best_models[best_model_name]
print(f"Best overall model: {best_model_name} with test accuracy: {test_accuracies_df.iloc[0, 0]:.4f}")

# Develop specialized upset prediction model
print("Developing specialized upset prediction model...")

# Prepare data for upset prediction
# Define upsets based on ELO ratings
combined_data['favorite'] = np.where(combined_data['home_elo_i'] > combined_data['away_elo_i'], 1, 0)
combined_data['upset'] = np.where(combined_data['favorite'] != combined_data['result'], 1, 0)

# Filter for games where home team is favorite
home_favorite_games = combined_data[combined_data['favorite'] == 1]
X_home_favorite = home_favorite_games[feature_names].values
y_home_favorite = home_favorite_games['upset'].values

# Filter for games where away team is favorite
away_favorite_games = combined_data[combined_data['favorite'] == 0]
X_away_favorite = away_favorite_games[feature_names].values
y_away_favorite = away_favorite_games['upset'].values

# Split data for home favorite games
X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(
    X_home_favorite, y_home_favorite, test_size=0.2, random_state=42
)

# Split data for away favorite games
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(
    X_away_favorite, y_away_favorite, test_size=0.2, random_state=42
)

# Scale features
scaler_home = StandardScaler()
X_train_home_scaled = scaler_home.fit_transform(X_train_home)
X_test_home_scaled = scaler_home.transform(X_test_home)

scaler_away = StandardScaler()
X_train_away_scaled = scaler_away.fit_transform(X_train_away)
X_test_away_scaled = scaler_away.transform(X_test_away)

# Train upset prediction models
# Use the same model type as the best overall model
if best_model_name == 'Logistic Regression':
    home_upset_model = LogisticRegression(random_state=42)
    away_upset_model = LogisticRegression(random_state=42)
elif best_model_name == 'Random Forest':
    home_upset_model = RandomForestClassifier(random_state=42)
    away_upset_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Gradient Boosting':
    home_upset_model = GradientBoostingClassifier(random_state=42)
    away_upset_model = GradientBoostingClassifier(random_state=42)
else:
    # Default to Random Forest if best model is not one of the above
    home_upset_model = RandomForestClassifier(random_state=42)
    away_upset_model = RandomForestClassifier(random_state=42)

# Train home upset model
home_upset_model.fit(X_train_home_scaled, y_train_home)
home_upset_pred = home_upset_model.predict(X_test_home_scaled)
home_upset_accuracy = accuracy_score(y_test_home, home_upset_pred)
print(f"Home upset model accuracy: {home_upset_accuracy:.4f}")

# Train away upset model
away_upset_model.fit(X_train_away_scaled, y_train_away)
away_upset_pred = away_upset_model.predict(X_test_away_scaled)
away_upset_accuracy = accuracy_score(y_test_away, away_upset_pred)
print(f"Away upset model accuracy: {away_upset_accuracy:.4f}")

# Save upset models
joblib.dump(home_upset_model, 'models/home_upset_model.pkl')
joblib.dump(away_upset_model, 'models/away_upset_model.pkl')
joblib.dump(scaler_home, 'models/home_upset_scaler.pkl')
joblib.dump(scaler_away, 'models/away_upset_scaler.pkl')

# Create confusion matrices for upset models
# Home upset model
cm_home = confusion_matrix(y_test_home, home_upset_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_home, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Upset', 'Upset'],
            yticklabels=['No Upset', 'Upset'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Home Favorite Upset Model Confusion Matrix')
plt.tight_layout()
plt.savefig('models/home_upset_confusion_matrix.png')

# Away upset model
cm_away = confusion_matrix(y_test_away, away_upset_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_away, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Upset', 'Upset'],
            yticklabels=['No Upset', 'Upset'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Away Favorite Upset Model Confusion Matrix')
plt.tight_layout()
plt.savefig('models/away_upset_confusion_matrix.png')

# Integrate models for final predictions
print("Integrating models for final predictions...")

# Function to make integrated predictions
def integrated_prediction(X, favorite_indicator, main_model, home_upset_model, away_upset_model,
                         home_scaler, away_scaler):
    """
    Make predictions using the integrated model approach.

    Parameters:
    -----------
    X : array-like
        Features for prediction
    favorite_indicator : array-like
        Binary indicator of whether home team is favorite (1) or away team is favorite (0)
    main_model : model
        Main prediction model
    home_upset_model : model
        Model for predicting upsets when home team is favorite
    away_upset_model : model
        Model for predicting upsets when away team is favorite
    home_scaler : scaler
        Scaler for home upset model features
    away_scaler : scaler
        Scaler for away upset model features

    Returns:
    --------
    predictions : array-like
        Final predictions (0 for away win, 1 for home win)
    """
    # Initialize predictions
    predictions = np.zeros(len(X))

    # Make main model predictions
    main_predictions = main_model.predict(X)

    # For each game
    for i in range(len(X)):
        if favorite_indicator[i] == 1:  # Home team is favorite
            # Scale features for home upset model
            X_home_scaled = home_scaler.transform(X[i].reshape(1, -1))

            # Predict if this is an upset
            is_upset = home_upset_model.predict(X_home_scaled)[0]

            # If predicted to be an upset, predict away win
            if is_upset:
                predictions[i] = 0
            else:
                predictions[i] = main_predictions[i]

        else:  # Away team is favorite
            # Scale features for away upset model
            X_away_scaled = away_scaler.transform(X[i].reshape(1, -1))

            # Predict if this is an upset
            is_upset = away_upset_model.predict(X_away_scaled)[0]

            # If predicted to be an upset, predict home win
            if is_upset:
                predictions[i] = 1
            else:
                predictions[i] = main_predictions[i]

    return predictions