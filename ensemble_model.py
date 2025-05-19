"""
Ensemble model development script for the NBA prediction project.

This script builds and evaluates ensemble models for predicting NBA game outcomes.
It includes:
1. Voting ensemble
2. Stacking ensemble
3. Specialized upset prediction models
4. Comparison of different ensemble approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import joblib
import os
import time
from datetime import datetime

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists, load_processed_data, load_feature_names

# Set up logger
logger = setup_logger("ensemble_model", f"logs/ensemble_model_{datetime.now().strftime('%Y%m%d')}.log")
logger.info("Starting ensemble model development")

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def load_data():
    """Load and prepare data for ensemble model building."""
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

        # Define upsets based on ELO ratings
        logger.info("Identifying upsets based on ELO ratings")
        train_data['favorite'] = np.where(train_data['home_elo_i'] > train_data['away_elo_i'], 1, 0)
        train_data['upset'] = np.where(train_data['favorite'] != train_data['result'], 1, 0)

        test_data['favorite'] = np.where(test_data['home_elo_i'] > test_data['away_elo_i'], 1, 0)
        test_data['upset'] = np.where(test_data['favorite'] != test_data['result'], 1, 0)

        # Calculate upset rate
        upset_rate = test_data['upset'].mean() * 100
        logger.info(f"Upset rate in the test dataset: {upset_rate:.1f}%")

        logger.info(f"Data loaded successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_base_models():
    """Load the best models from the model building phase."""
    logger.info("Loading base models")

    try:
        # Try to load the best models from the model building script
        base_models = {}
        model_files = {
            'Gradient Boosting': 'gradient_boosting_best.pkl',
            'Random Forest': 'random_forest_best.pkl',
            'Logistic Regression': 'logistic_regression_best.pkl'
        }

        for name, file in model_files.items():
            if os.path.exists(f'models/{file}'):
                model = joblib.load(f'models/{file}')
                base_models[name] = model
                logger.info(f"Loaded {name} model from {file}")
            else:
                logger.warning(f"Could not find {file}, will train a new model")

        # If no models were loaded, create new ones
        if not base_models:
            logger.info("No pre-trained models found, creating new base models")
            base_models = {
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
                'Logistic Regression': LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42)
            }

        return base_models

    except Exception as e:
        logger.error(f"Error loading base models: {e}")
        raise

def create_voting_ensemble(base_models, X_train_scaled, y_train):
    """Create and train a voting ensemble model."""
    logger.info("Creating voting ensemble")

    # Create estimator list for VotingClassifier
    estimators = [(name.replace(' ', '_').lower(), model) for name, model in base_models.items()]

    # Create voting ensemble with soft voting (using predicted probabilities)
    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 1, 1]  # Give more weight to Gradient Boosting (typically best performer)
    )

    # Train the ensemble
    voting_ensemble.fit(X_train_scaled, y_train)

    # Save the model
    joblib.dump(voting_ensemble, 'ensemble_model/voting_ensemble.pkl')

    return voting_ensemble

def create_stacking_ensemble(base_models, X_train_scaled, y_train, X_test_scaled, y_test):
    """Create and train a stacking ensemble model."""
    logger.info("Creating stacking ensemble")

    # Generate predictions from base models using k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features_train = np.zeros((X_train_scaled.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test_scaled.shape[0], len(base_models)))

    for i, (name, model) in enumerate(base_models.items()):
        logger.info(f"Generating meta-features for {name}")

        # Train the model on the full training set and predict on test set
        model.fit(X_train_scaled, y_train)
        meta_features_test[:, i] = model.predict_proba(X_test_scaled)[:, 1]

        # Use cross-validation to generate out-of-fold predictions for training meta-model
        for train_idx, val_idx in kf.split(X_train_scaled):
            # Split data
            X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train model on training fold
            model.fit(X_train_fold, y_train_fold)

            # Predict on validation fold
            meta_features_train[val_idx, i] = model.predict_proba(X_val_fold)[:, 1]

    # Add original features to meta-features
    meta_features_train_with_orig = np.hstack((meta_features_train, X_train_scaled))
    meta_features_test_with_orig = np.hstack((meta_features_test, X_test_scaled))

    # Train meta-model
    meta_model = LogisticRegression(C=0.1, random_state=42)
    meta_model.fit(meta_features_train_with_orig, y_train)

    # Save meta-model and base models
    joblib.dump(meta_model, 'ensemble_model/meta_model.pkl')
    for i, (name, model) in enumerate(base_models.items()):
        joblib.dump(model, f'ensemble_model/base_model_{i}.pkl')

    # Evaluate meta-model
    meta_predictions = meta_model.predict(meta_features_test_with_orig)
    stacking_accuracy = accuracy_score(y_test, meta_predictions)
    logger.info(f"Stacking ensemble accuracy: {stacking_accuracy:.4f}")

    return meta_model, stacking_accuracy, meta_features_train_with_orig, meta_features_test_with_orig

def create_upset_prediction_model(X_train, y_train, train_data, X_test, test_data, features):
    """Create a specialized model for predicting upsets."""
    logger.info("Creating upset prediction model")

    # Create additional features specifically for upset detection
    upset_features = features + ['elo_diff_abs', 'favorite_back_to_back', 'underdog_back_to_back']

    # Add upset-specific features
    train_data['elo_diff_abs'] = abs(train_data['home_elo_i'] - train_data['away_elo_i'])
    train_data['favorite_back_to_back'] = np.where(
        train_data['favorite'] == 1,
        train_data['home_back_to_back'],
        train_data['away_back_to_back']
    )
    train_data['underdog_back_to_back'] = np.where(
        train_data['favorite'] == 0,
        train_data['home_back_to_back'],
        train_data['away_back_to_back']
    )

    test_data['elo_diff_abs'] = abs(test_data['home_elo_i'] - test_data['away_elo_i'])
    test_data['favorite_back_to_back'] = np.where(
        test_data['favorite'] == 1,
        test_data['home_back_to_back'],
        test_data['away_back_to_back']
    )
    test_data['underdog_back_to_back'] = np.where(
        test_data['favorite'] == 0,
        test_data['home_back_to_back'],
        test_data['away_back_to_back']
    )

    # Prepare data for upset prediction model
    X_train_upset = train_data[upset_features]
    y_train_upset = train_data['upset']

    X_test_upset = test_data[upset_features]
    y_test_upset = test_data['upset']

    # Scale features
    scaler_upset = StandardScaler()
    X_train_upset_scaled = scaler_upset.fit_transform(X_train_upset)
    X_test_upset_scaled = scaler_upset.transform(X_test_upset)

    # Train upset prediction model
    upset_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        min_samples_split=5,
        random_state=42
    )
    upset_model.fit(X_train_upset_scaled, y_train_upset)

    # Train regular outcome prediction model
    regular_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=3,
        min_samples_split=5,
        random_state=42
    )
    regular_model.fit(X_train, y_train)

    # Save models
    joblib.dump(upset_model, 'ensemble_model/upset_model.pkl')
    joblib.dump(regular_model, 'ensemble_model/regular_model.pkl')
    joblib.dump(scaler_upset, 'ensemble_model/scaler_upset.pkl')

    # Predict upsets with enhanced model
    upset_proba = upset_model.predict_proba(X_test_upset_scaled)[:, 1]

    # Predict outcomes with regular model
    regular_proba = regular_model.predict_proba(X_test)[:, 1]

    # Adaptive combination based on upset probability
    adaptive_predictions = []
    for i in range(len(y_test_upset)):
        upset_probability = upset_proba[i]

        if upset_probability > 0.5:  # High chance of upset
            # If favorite is home team (1), predict away team wins (0)
            # If favorite is away team (0), predict home team wins (1)
            adaptive_predictions.append(1 - test_data.iloc[i]['favorite'])
        else:
            # Use regular model prediction
            adaptive_predictions.append(1 if regular_proba[i] > 0.5 else 0)

    # Calculate accuracy
    upset_model_accuracy = accuracy_score(y_test_upset, upset_model.predict(X_test_upset_scaled))
    adaptive_accuracy = accuracy_score(test_data['result'], adaptive_predictions)

    logger.info(f"Upset prediction model accuracy: {upset_model_accuracy:.4f}")
    logger.info(f"Adaptive ensemble accuracy: {adaptive_accuracy:.4f}")

    return upset_model, regular_model, adaptive_accuracy

def plot_feature_importance(model, feature_names, title, output_path, top_n=20):
    """
    Plot feature importance for a model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        title: Title for the plot
        output_path: Path to save the plot
        top_n: Number of top features to display
    """
    plt.figure(figsize=(12, 10))

    # Get feature importances based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Model {type(model).__name__} doesn't have feature_importances_ or coef_ attribute")
        return

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:min(top_n, len(feature_names))]

    # Plot horizontal bar chart
    plt.barh(range(len(top_indices)), importances[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")

    # Log top 10 features
    logger.info(f"Top 10 features for {title}:")
    for i in range(min(10, len(feature_names))):
        logger.info(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def plot_meta_model_importance(meta_model, base_model_names, features, output_path):
    """
    Plot feature importance for the meta-model in stacking ensemble.

    Args:
        meta_model: Trained meta-model
        base_model_names: Names of base models
        features: Original feature names
        output_path: Path to save the plot
    """
    if not hasattr(meta_model, 'coef_'):
        logger.warning("Meta-model doesn't have coef_ attribute")
        return

    plt.figure(figsize=(12, 8))

    # Get coefficients
    coefs = np.abs(meta_model.coef_[0])

    # Create feature names for meta-features (base model predictions + original features)
    meta_feature_names = list(base_model_names) + features

    # Sort by importance
    indices = np.argsort(coefs)[::-1]
    top_n = min(20, len(meta_feature_names))

    # Plot
    plt.barh(range(top_n), coefs[indices[:top_n]])
    plt.yticks(range(top_n), [meta_feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Coefficient Magnitude')
    plt.title('Meta-Model Feature Importance')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Meta-model feature importance plot saved to {output_path}")

    # Log importance of base model predictions
    logger.info("Importance of base model predictions in meta-model:")
    for i, name in enumerate(base_model_names):
        logger.info(f"{name}: {coefs[i]:.4f}")

def main():
    """Main function to run the ensemble model development process."""
    start_time = time.time()

    # Create output directory for ensemble model
    ensure_directory_exists('ensemble_model')
    ensure_directory_exists('ensemble_model/feature_importance')

    try:
        # Load and prepare data
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler = load_data()

        # Save scaler for later use
        joblib.dump(scaler, 'ensemble_model/scaler.pkl')

        # Load base models
        base_models = load_base_models()

        # Train base models if needed
        for name, model in base_models.items():
            if not hasattr(model, 'classes_'):  # Model hasn't been trained yet
                logger.info(f"Training {name} model")
                model.fit(X_train_scaled, y_train)

        # Create voting ensemble
        voting_ensemble = create_voting_ensemble(base_models, X_train_scaled, y_train)
        voting_predictions = voting_ensemble.predict(X_test_scaled)
        voting_accuracy = accuracy_score(y_test, voting_predictions)
        logger.info(f"Voting ensemble accuracy: {voting_accuracy:.4f}")

        # Create stacking ensemble
        meta_model, stacking_accuracy, meta_features_train, meta_features_test = create_stacking_ensemble(
            base_models, X_train_scaled, y_train, X_test_scaled, y_test
        )

        # Create upset prediction model
        upset_model, regular_model, adaptive_accuracy = create_upset_prediction_model(
            X_train_scaled, y_train, train_data, X_test_scaled, test_data, features
        )

        # Compare all approaches
        approaches = {
            "Voting Ensemble": voting_accuracy,
            "Stacking Ensemble": stacking_accuracy,
            "Adaptive Upset Ensemble": adaptive_accuracy
        }

        # Add individual base model accuracies
        for name, model in base_models.items():
            model_predictions = model.predict(X_test_scaled)
            model_accuracy = accuracy_score(y_test, model_predictions)
            approaches[name] = model_accuracy

        # Find the best approach
        best_approach = max(approaches.items(), key=lambda x: x[1])
        logger.info(f"Best approach: {best_approach[0]} with accuracy {best_approach[1]:.4f}")

        # Compare with original best model
        try:
            original_accuracy = joblib.load('models/all_model_accuracies.csv')
            original_accuracy = original_accuracy.iloc[0, 0]
        except:
            # Use a default value if the file doesn't exist
            original_accuracy = max([approaches[name] for name in base_models.keys()])

        improvement = (best_approach[1] - original_accuracy) * 100
        logger.info(f"Improvement over original model: {improvement:.2f}%")

        # Create visualization of approach comparison
        plt.figure(figsize=(12, 8))
        approaches_df = pd.DataFrame({
            'Accuracy': approaches.values()
        }, index=approaches.keys())
        approaches_df = approaches_df.sort_values('Accuracy', ascending=False)

        sns.barplot(x='Accuracy', y=approaches_df.index, data=approaches_df)
        plt.title('Comparison of Ensemble Approaches', fontsize=16)
        plt.xlabel('Accuracy', fontsize=14)
        plt.xlim(0.5, 0.8)
        for i, v in enumerate(approaches_df['Accuracy']):
            plt.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig('ensemble_model/ensemble_comparison.png')

        # Save results to CSV
        approaches_df.to_csv('ensemble_model/ensemble_results.csv')

        # Generate feature importance plots for base models
        logger.info("Generating feature importance plots for base models")
        for name, model in base_models.items():
            plot_feature_importance(
                model,
                features,
                f"{name} Feature Importance",
                f"ensemble_model/feature_importance/{name.lower().replace(' ', '_')}_importance.png"
            )

        # Generate feature importance plot for upset prediction model
        logger.info("Generating feature importance plot for upset prediction model")
        # Create upset features list (same as in create_upset_prediction_model function)
        upset_features = features + ['elo_diff_abs', 'favorite_back_to_back', 'underdog_back_to_back']
        plot_feature_importance(
            upset_model,
            upset_features,
            "Upset Prediction Model Feature Importance",
            "ensemble_model/feature_importance/upset_model_importance.png"
        )

        # Generate feature importance plot for regular prediction model
        logger.info("Generating feature importance plot for regular prediction model")
        plot_feature_importance(
            regular_model,
            features,
            "Regular Prediction Model Feature Importance",
            "ensemble_model/feature_importance/regular_model_importance.png"
        )

        # Generate feature importance plot for meta-model in stacking ensemble
        logger.info("Generating feature importance plot for meta-model")
        plot_meta_model_importance(
            meta_model,
            list(base_models.keys()),
            features,
            "ensemble_model/feature_importance/meta_model_importance.png"
        )

        # Create summary report
        with open('ensemble_model/ensemble_summary.md', 'w') as f:
            f.write("# NBA Game Prediction Ensemble Model Summary\n\n")

            f.write("## Ensemble Approaches\n\n")
            f.write("| Approach | Accuracy |\n")
            f.write("|----------|----------|\n")
            for name, accuracy in approaches_df.iterrows():
                f.write(f"| {name} | {accuracy['Accuracy']:.4f} |\n")

            f.write("\n## Best Approach\n\n")
            f.write(f"The best performing approach is **{best_approach[0]}** with a test accuracy of {best_approach[1]:.4f}.\n")

            f.write("\n## Improvement\n\n")
            f.write(f"This represents an improvement of {improvement:.2f}% over the original best model.\n")

            f.write("\n## Approach Descriptions\n\n")
            f.write("1. **Voting Ensemble**: Combines predictions from multiple models using weighted voting.\n")
            f.write("2. **Stacking Ensemble**: Uses predictions from base models as features for a meta-model.\n")
            f.write("3. **Adaptive Upset Ensemble**: Specializes in predicting upset games using additional features.\n")

            f.write("\n## Feature Importance Analysis\n\n")
            f.write("Feature importance plots have been generated for all models to provide insights into which features are most influential in making predictions.\n\n")

            f.write("### Base Models\n\n")
            for name in base_models.keys():
                f.write(f"- [{name} Feature Importance](feature_importance/{name.lower().replace(' ', '_')}_importance.png)\n")

            f.write("\n### Ensemble Models\n\n")
            f.write("- [Meta-Model Feature Importance](feature_importance/meta_model_importance.png) - Shows the importance of base model predictions and original features in the stacking ensemble\n")
            f.write("- [Upset Prediction Model Feature Importance](feature_importance/upset_model_importance.png) - Shows the importance of features in predicting upset games\n")
            f.write("- [Regular Prediction Model Feature Importance](feature_importance/regular_model_importance.png) - Shows the importance of features in the regular prediction model\n")

        logger.info(f"Ensemble model development completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in ensemble model development: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
