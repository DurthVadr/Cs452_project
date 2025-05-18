"""
Ensemble model building pipeline for NBA game prediction.

This script:
1. Loads trained base models
2. Creates different ensemble models
3. Trains and evaluates ensembles
4. Compares performance with base models
5. Registers best ensemble
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.data.loader import load_processed_dataset, load_feature_names
from nba_prediction.models.registry import ModelRegistry
from nba_prediction.models.ensemble import (
    create_voting_ensemble,
    create_stacking_ensemble,
    create_upset_specialized_ensemble,
    predict_with_upset_ensemble
)
from nba_prediction.evaluation.metrics import evaluate_model_comprehensive
from sklearn.preprocessing import StandardScaler

logger = get_logger('ensemble_building')

def main():
    """Run the ensemble building pipeline."""
    logger.info("Starting ensemble building pipeline")
    
    # Step 1: Load processed data
    logger.info("Loading processed data")
    try:
        X_train = load_processed_dataset('X_train')
        y_train = load_processed_dataset('y_train')
        X_test = load_processed_dataset('X_test')
        y_test = load_processed_dataset('y_test')
        train_data = load_processed_dataset('train_data')
        test_data = load_processed_dataset('test_data')
        feature_names = load_feature_names()
        
        logger.info(f"Loaded training data with shape: {X_train.shape}")
        logger.info(f"Loaded test data with shape: {X_test.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        sys.exit(1)
    
    # Step 2: Load base models
    logger.info("Loading base models")
    registry = ModelRegistry()
    base_model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    base_models = {}
    for model_type in base_model_types:
        try:
            base_models[model_type] = registry.load_model(model_type)
            logger.info(f"Loaded {model_type} model")
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model: {e}")
    
    if len(base_models) == 0:
        logger.error("No base models found, cannot build ensembles")
        sys.exit(1)
    
    # Step 3: Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Build voting ensemble
    logger.info("Building voting ensemble")
    try:
        voting_ensemble = create_voting_ensemble(base_models=base_models)
        voting_ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate voting ensemble
        metrics, predictions, probabilities = evaluate_model_comprehensive(
            voting_ensemble, X_test_scaled, y_test, test_data, prefix="voting_ensemble"
        )
        
        # Register voting ensemble
        registry.register_model(
            model_name="voting_ensemble",
            model=voting_ensemble,
            metadata={
                "base_models": list(base_models.keys()),
                "accuracy": metrics.get("voting_ensemble_accuracy", 0),
                "artifacts": {
                    "scaler": scaler
                },
                **metrics
            }
        )
        
        logger.info(f"Registered voting ensemble with accuracy: {metrics.get('voting_ensemble_accuracy', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to build voting ensemble: {e}")
    
    # Step 5: Build stacking ensemble
    logger.info("Building stacking ensemble")
    try:
        stacking_ensemble = create_stacking_ensemble(X_train_scaled, y_train, base_models=base_models)
        
        # Evaluate stacking ensemble
        metrics, predictions, probabilities = evaluate_model_comprehensive(
            stacking_ensemble, X_test_scaled, y_test, test_data, prefix="stacking_ensemble"
        )
        
        # Register stacking ensemble
        registry.register_model(
            model_name="stacking_ensemble",
            model=stacking_ensemble,
            metadata={
                "base_models": list(base_models.keys()),
                "accuracy": metrics.get("stacking_ensemble_accuracy", 0),
                "artifacts": {
                    "scaler": scaler
                },
                **metrics
            }
        )
        
        logger.info(f"Registered stacking ensemble with accuracy: {metrics.get('stacking_ensemble_accuracy', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Failed to build stacking ensemble: {e}")
    
    # Step 6: Build upset specialized ensemble
    logger.info("Building upset specialized ensemble")
    try:
        main_features = config.ALL_FEATURES
        upset_features = config.UPSET_MODEL_FEATURES
        
        main_model, upset_model, main_scaler, upset_scaler, upset_threshold = create_upset_specialized_ensemble(
            train_data, main_features, upset_features
        )
        
        # Evaluate on test data
        X_main_test = test_data[main_features]
        X_upset_test = test_data[upset_features]
        favorite = test_data['favorite'].values
        
        # Scale test data
        X_main_test_scaled = main_scaler.transform(X_main_test)
        X_upset_test_scaled = upset_scaler.transform(X_upset_test)
        
        # Make predictions using upset specialized ensemble
        upset_preds = predict_with_upset_ensemble(
            X_main_test, X_upset_test, favorite,
            main_model, upset_model, 
            main_scaler, upset_scaler,
            upset_threshold
        )
        
        # Calculate accuracy
        accuracy = (upset_preds == y_test).mean()
        
        # Register upset specialized ensemble
        registry.register_model(
            model_name="upset_specialized",
            model=main_model,  # Register main model as primary
            metadata={
                "accuracy": accuracy,
                "upset_threshold": upset_threshold,
                "artifacts": {
                    "upset_model": upset_model,
                    "main_scaler": main_scaler,
                    "upset_scaler": upset_scaler
                }
            }
        )
        
        logger.info(f"Registered upset specialized ensemble with accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to build upset specialized ensemble: {e}")
    
    # Step 7: Compare all models
    logger.info("Comparing all models")
    
    # Load evaluation results from previous step if available
    eval_file = os.path.join(config.OUTPUT_DIR, "evaluation", "metrics", "evaluation_results.json")
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                base_model_results = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load previous evaluation results: {e}")
            base_model_results = {}
    else:
        base_model_results = {}
    
    # Add ensemble results
    all_models = registry.list_models()
    ensemble_models = [m for m in all_models if m not in base_model_types and m != 'elo']
    
    ensemble_results = {}
    for model_name in ensemble_models:
        try:
            model_info = registry.get_model_info(model_name)
            ensemble_results[model_name] = {"accuracy": model_info.get("accuracy", 0)}
        except Exception as e:
            logger.warning(f"Failed to get info for {model_name}: {e}")
    
    # Combine results
    all_results = {**base_model_results, **ensemble_results}
    
    # Find best model
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
        best_accuracy = all_results[best_model].get('accuracy', 0)
        logger.info(f"Best model: {best_model} with accuracy {best_accuracy:.4f}")
    
    # Save updated results
    ensemble_eval_file = os.path.join(config.OUTPUT_DIR, "ensemble", "evaluation_results.json")
    ensure_directory_exists(os.path.dirname(ensemble_eval_file))
    
    with open(ensemble_eval_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Ensemble building completed")
    logger.info(f"Built ensembles: {', '.join(ensemble_models)}")

if __name__ == "__main__":
    main()