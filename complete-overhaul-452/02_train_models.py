"""
Model training pipeline for NBA game prediction.

This script:
1. Loads processed data
2. Trains base models (logistic regression, random forest, gradient boosting)
3. Optimizes and trains ELO model
4. Registers trained models
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.utils.common import ensure_directory_exists
from nba_prediction.data.loader import load_processed_dataset, load_feature_names
from nba_prediction.models.base_models import train_model
from nba_prediction.models.elo import find_optimal_elo_parameters, train_elo_system
from nba_prediction.models.registry import ModelRegistry

logger = get_logger('model_training')

def main():
    """Run the model training pipeline."""
    logger.info("Starting model training pipeline")
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # Step 1: Load processed data
    logger.info("Loading processed data")
    try:
        X_train = load_processed_dataset('X_train')
        y_train = load_processed_dataset('y_train')
        X_test = load_processed_dataset('X_test')
        y_test = load_processed_dataset('y_test')
        feature_names = load_feature_names()
        
        # Also load full datasets for ELO training
        train_data = load_processed_dataset('train_data')
        test_data = load_processed_dataset('test_data')
        
        logger.info(f"Loaded training data with shape: {X_train.shape}")
        logger.info(f"Loaded test data with shape: {X_test.shape}")
        logger.info(f"Loaded {len(feature_names)} features")
        
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        sys.exit(1)
    
    # Step 2: Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Train base models
    logger.info("Training base models")
    base_model_types = ['logistic_regression', 'random_forest', 'gradient_boosting']
    trained_models = {}
    
    for model_type in base_model_types:
        logger.info(f"Training {model_type} model")
        try:
            model = train_model(X_train_scaled, y_train, model_type=model_type)
            trained_models[model_type] = model
            
            # Register model
            registry.register_model(
                model_name=model_type,
                model=model,
                metadata={
                    "accuracy": (model.predict(X_test_scaled) == y_test).mean(),
                    "feature_count": len(feature_names),
                    "samples": len(y_train),
                    "artifacts": {
                        "scaler": scaler
                    }
                }
            )
            
            logger.info(f"Successfully trained and registered {model_type} model")
            
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
    
    # Step 4: Train ELO model
    logger.info("Training ELO model")
    try:
        # Find optimal ELO parameters on training data
        best_k, best_ha, best_elo, best_accuracy = find_optimal_elo_parameters(train_data)
        logger.info(f"Found optimal ELO parameters: k={best_k}, home_advantage={best_ha}, accuracy={best_accuracy:.4f}")
        
        # Register ELO model
        registry.register_model(
            model_name="elo",
            model=best_elo,
            metadata={
                "k_factor": best_k,
                "home_advantage": best_ha,
                "accuracy": best_accuracy,
                "parameters": {
                    "k_factor": best_k,
                    "home_advantage": best_ha,
                    "initial_rating": config.ELO_PARAMS["initial_rating"]
                }
            }
        )
        
        logger.info("Successfully trained and registered ELO model")
        
    except Exception as e:
        logger.error(f"Failed to train ELO model: {e}")
    
    # Log completion
    logger.info("Model training completed")
    logger.info(f"Trained models: {', '.join(registry.list_models())}")

if __name__ == "__main__":
    main()