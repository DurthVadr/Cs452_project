"""
Model building and training script for NBA game prediction project.
Implements multiple models with focus on upset prediction capability.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import joblib
from config import *

def load_training_data():
    """Load prepared training and test data."""
    print("Loading prepared data...")
    # Define file paths
    data_dir = 'data/processed'

    # Load numpy arrays
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')

    # Load feature names
    with open(f'{data_dir}/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    return X_train, X_test, y_train, y_test, feature_names

def create_base_models():
    """Create base models without class weights."""
    models = {
        'logistic': LogisticRegression(
            C=0.001,
            penalty='l2',
            solver='liblinear',
            random_state=RANDOM_STATE
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        )
    }
    return models

def create_ensemble_model(base_models):
    """Create voting ensemble from base models."""
    estimators = [(name, model) for name, model in base_models.items()]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability predictions
        n_jobs=-1  # Use all available cores
    )
    return ensemble

def train_and_save_models(X_train, y_train, feature_names):
    """Train all models and save them."""
    print("Training models...")

    # Create and train models with balanced class weights
    base_models = create_base_models()
    ensemble_model = create_ensemble_model(base_models)
    trained_models = {}

    # Train base models
    for name, model in base_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

    # Train ensemble
    print("Training ensemble...")
    ensemble_model.fit(X_train, y_train)
    trained_models['ensemble'] = ensemble_model
    joblib.dump(ensemble_model, os.path.join(MODELS_DIR, "ensemble.pkl"))
    # Save feature names with each model for validation
    for name in trained_models:
        feat_file = os.path.join(MODELS_DIR, f"{name}_features.txt")
        with open(feat_file, 'w') as f:
            f.write('\n'.join(feature_names))
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Basic evaluation of trained models."""
    print("\nPreliminary Model Evaluation:")
    print("-" * 50)
    results = {}
    for name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # For upset analysis, we need probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        print(f"{name.title()} Accuracy: {accuracy:.4f}")
    return results

def main():
    """Main execution function."""
    try:
        # Load data
        X_train, X_test, y_train, y_test, feature_names = load_training_data()
        # Train and save models
        trained_models = train_and_save_models(X_train, y_train, feature_names)
        # Quick evaluation
        results = evaluate_models(trained_models, X_test, y_test)
        print("\nModel training completed successfully!")
        print("Models saved in:", MODELS_DIR)
    except Exception as e:
        print(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    main()