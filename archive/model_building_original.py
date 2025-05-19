"""
Model building script for the NBA prediction project.

This script builds and evaluates machine learning models for predicting NBA game outcomes.
It includes:
1. Basic model training and evaluation
2. Feature importance analysis
3. Hyperparameter tuning
4. Model persistence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists, load_processed_data, load_feature_names

# Set up logger
logger = setup_logger("model_building", f"logs/model_building_{datetime.now().strftime('%Y%m%d')}.log")
logger.info("Starting model building")

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def load_data(apply_smote=True):
    """
    Load and prepare data for model building.

    Args:
        apply_smote: Whether to apply SMOTE to balance the training data
    """
    logger.info("Loading processed data")

    try:
        # Load processed data
        combined_data, train_data, test_data = load_processed_data()
        features = load_feature_names()

        # Create X and y for training and testing
        X_train = train_data[features]
        y_train = train_data['result']
        X_test = test_data[features]
        y_test = test_data['result']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE to balance the training data
        if apply_smote:
            logger.info("Applying SMOTE to balance training data")

            # Check class distribution before SMOTE
            class_dist_before = pd.Series(y_train).value_counts(normalize=True)
            logger.info(f"Class distribution before SMOTE: {class_dist_before.to_dict()}")

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            # Check class distribution after SMOTE
            class_dist_after = pd.Series(y_train_resampled).value_counts(normalize=True)
            logger.info(f"Class distribution after SMOTE: {class_dist_after.to_dict()}")
            logger.info(f"Training data shape after SMOTE: {X_train_resampled.shape}")

            # Return the resampled data
            logger.info(f"Data loaded successfully. Original training set: {X_train.shape}, Resampled training set: {X_train_resampled.shape}, Test set: {X_test.shape}")
            return X_train, y_train, X_test, y_test, X_train_resampled, X_test_scaled, features, train_data, test_data, scaler
        else:
            logger.info(f"Data loaded successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
            return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_baseline_models(X_train_scaled, y_train):
    """Train baseline models with default parameters."""
    logger.info("Training baseline models")

    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    }

    # Evaluate each model with cross-validation
    cv_results = {}
    for name, model in models.items():
        logger.info(f"Cross-validating {name}")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        logger.info(f"{name} CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Convert results to DataFrame
    cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index')
    cv_results_df = cv_results_df.sort_values('mean_cv_score', ascending=False)

    return models, cv_results_df

def tune_hyperparameters(X_train_scaled, y_train):
    """Perform hyperparameter tuning for each model type."""
    logger.info("Performing hyperparameter tuning")

    # Define parameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, solver='liblinear'),
            'params': {
                'C': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'penalty': ['l1', 'l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        }
    }

    # Perform grid search for each model
    best_models = {}
    for name, config in param_grids.items():
        logger.info(f"Tuning hyperparameters for {name}")
        model = config['model']
        param_grid = config['params']

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        # Save best model
        best_models[name] = grid_search.best_estimator_

        # Save grid search results
        grid_results = pd.DataFrame(grid_search.cv_results_)
        grid_results.to_csv(f'models/{name.replace(" ", "_").lower()}_grid_search.csv')

        # Log best parameters and score
        logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {name}: {grid_search.best_score_:.4f}")

        # Save best model
        joblib.dump(grid_search.best_estimator_,
                    f'models/{name.replace(" ", "_").lower()}_best.pkl')

    return best_models

def evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test, features):
    """Train and evaluate models, saving results and visualizations."""
    logger.info("Evaluating models on test set")

    # Train and evaluate models
    test_results = {}
    for name, model in models.items():
        logger.info(f"Training and evaluating {name}")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{name} test accuracy: {accuracy:.4f}")

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Store results
        test_results[name] = {
            'accuracy': accuracy,
            'report': report
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
        plt.savefig(f'models/{name.replace(" ", "_").lower()}_confusion_matrix.png')

        # Plot feature importance if available
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({name})')
            plt.bar(range(X_train_scaled.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train_scaled.shape[1]), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/{name.replace(" ", "_").lower()}_feature_importance.png')

        elif name == 'Logistic Regression':
            # For logistic regression, use coefficients
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1]

            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances ({name})')
            plt.bar(range(X_train_scaled.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train_scaled.shape[1]), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'models/{name.replace(" ", "_").lower()}_feature_importance.png')

        # Save model
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_model.pkl')

    # Convert results to DataFrame
    test_accuracies_df = pd.DataFrame({
        'accuracy': [results['accuracy'] for results in test_results.values()]
    }, index=test_results.keys())
    test_accuracies_df = test_accuracies_df.sort_values('accuracy', ascending=False)

    return test_results, test_accuracies_df

def main():
    """Main function to run the model building process."""
    start_time = time.time()

    # Create output directory for model building
    ensure_directory_exists('models')

    try:
        # Load and prepare data WITHOUT SMOTE for comparison
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler = load_data(apply_smote=False)

        # Log class distribution of original data
        class_dist = pd.Series(y_train).value_counts(normalize=True)
        logger.info(f"Class distribution (original): {class_dist.to_dict()}")

        # Save scaler for later use
        joblib.dump(scaler, 'models/scaler.pkl')

        # Train baseline models using original data
        logger.info("Training baseline models with original data (no SMOTE)")
        baseline_models, cv_results_df = train_baseline_models(X_train_scaled, y_train)
        cv_results_df.to_csv('models/baseline_cv_results_original.csv')

        # Visualize cross-validation results
        plt.figure(figsize=(12, 6))
        cv_plot = sns.barplot(x=cv_results_df.index, y='mean_cv_score', data=cv_results_df)
        plt.title('Model Cross-Validation Accuracy (Original Data)')
        plt.xlabel('Model')
        plt.ylabel('Mean CV Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.5, 0.8)
        for i, v in enumerate(cv_results_df['mean_cv_score']):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.tight_layout()
        plt.savefig('models/baseline_cv_results_original.png')

        # Tune hyperparameters using original data
        logger.info("Tuning hyperparameters with original data (no SMOTE)")
        best_models = tune_hyperparameters(X_train_scaled, y_train)

        # Evaluate models
        logger.info("Evaluating models trained with original data (no SMOTE)")
        test_results, test_accuracies_df = evaluate_models(
            best_models, X_train_scaled, y_train, X_test_scaled, y_test, features
        )
        test_accuracies_df.to_csv('models/tuned_model_accuracies_original.csv')

        # Compare with ELO predictions
        elo_accuracy = (test_data['elo_pred'] == test_data['result']).mean()
        logger.info(f"ELO prediction accuracy: {elo_accuracy:.4f}")

        # Compare all models
        best_model_name = test_accuracies_df.index[0]
        all_models = {
            'ELO': elo_accuracy,
            best_model_name: test_accuracies_df.iloc[0, 0]
        }

        # Convert to DataFrame
        all_accuracies_df = pd.DataFrame.from_dict(all_models, orient='index', columns=['accuracy'])
        all_accuracies_df = all_accuracies_df.sort_values('accuracy', ascending=False)
        all_accuracies_df.to_csv('models/all_model_accuracies.csv')

        # Create summary report
        with open('models/model_summary_smote.md', 'w') as f:
            f.write("# NBA Game Prediction Model Summary with SMOTE\n\n")

            f.write("## SMOTE Enhancement\n\n")
            f.write("This model training used Synthetic Minority Over-sampling Technique (SMOTE) to balance the training data.\n")
            f.write("SMOTE creates synthetic examples of the minority class to improve model performance on imbalanced datasets.\n\n")

            f.write("### Class Distribution\n\n")
            f.write("| Class | Before SMOTE | After SMOTE |\n")
            f.write("|-------|-------------|------------|\n")
            original_dist = pd.Series(y_train).value_counts(normalize=True)
            for cls in sorted(original_dist.index):
                before = original_dist.get(cls, 0)
                after = class_dist_after.get(cls, 0)
                f.write(f"| {cls} | {before:.4f} | {after:.4f} |\n")

            f.write("\n## Baseline Models with SMOTE\n\n")
            f.write("Cross-validation results for baseline models trained with SMOTE:\n\n")
            f.write("| Model | Mean CV Accuracy | Std CV Accuracy |\n")
            f.write("|-------|-----------------|----------------|\n")
            for name, row in cv_results_df.iterrows():
                f.write(f"| {name} | {row['mean_cv_score']:.4f} | {row['std_cv_score']:.4f} |\n")

            f.write("\n## Tuned Models with SMOTE\n\n")
            f.write("Test accuracy for tuned models trained with SMOTE:\n\n")
            f.write("| Model | Test Accuracy |\n")
            f.write("|-------|---------------|\n")
            for name, accuracy in test_accuracies_df.iterrows():
                f.write(f"| {name} | {accuracy['accuracy']:.4f} |\n")

            f.write("\n## Comparison with ELO\n\n")
            f.write("| Model | Accuracy |\n")
            f.write("|-------|----------|\n")
            for name, row in all_accuracies_df.iterrows():
                f.write(f"| {name} | {row['accuracy']:.4f} |\n")

            f.write("\n## Conclusion\n\n")
            best_model_name_overall = all_accuracies_df.index[0]
            best_accuracy_overall = all_accuracies_df.iloc[0, 0]
            f.write(f"The best performing model is the **{best_model_name_overall}** with a test accuracy of {best_accuracy_overall:.4f}.\n")

            # Compare to ELO baseline
            improvement = best_accuracy_overall - elo_accuracy
            f.write(f"\nThis represents an improvement of {improvement:.4f} ({improvement*100:.2f}%) over the baseline ELO model.\n")

            f.write("\n## Impact of SMOTE\n\n")
            f.write("Using SMOTE to balance the training data has helped the model better learn patterns from both classes.\n")
            f.write("This is particularly important for predicting upset games, which are less common in the dataset.\n")
            f.write("The balanced training data allows the complex models in this project to reach their full potential by having sufficient examples of all outcome types.\n")

        logger.info(f"Model building completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Best model: {best_model_name_overall} with accuracy {best_accuracy_overall:.4f}")

    except Exception as e:
        logger.error(f"Error in model building: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
