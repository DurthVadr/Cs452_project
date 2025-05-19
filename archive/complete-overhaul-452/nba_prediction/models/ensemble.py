"""
Ensemble model implementations for NBA game prediction.
"""
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.models.base_models import create_logistic_regression, create_random_forest, create_gradient_boosting

logger = get_logger(__name__)

def create_voting_ensemble(base_models=None, voting='soft', weights=None):
    """
    Create a voting ensemble from base models.
    
    Args:
        base_models: Dictionary of {name: model} pairs (default: create standard models)
        voting: Voting strategy ('hard' or 'soft')
        weights: Dictionary of {name: weight} pairs (default: from config)
        
    Returns:
        VotingClassifier ensemble model
    """
    # Use default models if none provided
    if base_models is None:
        base_models = {
            'logistic_regression': create_logistic_regression(),
            'random_forest': create_random_forest(),
            'gradient_boosting': create_gradient_boosting()
        }
        
    # Use default weights if none provided
    if weights is None:
        weights = config.ENSEMBLE_PARAMS['voting_weights']
    
    # Create estimators list for VotingClassifier
    estimators = []
    weight_list = []
    
    for name, model in base_models.items():
        estimators.append((name, model))
        weight_list.append(weights.get(name, 1))
        
    # Create and return ensemble
    logger.info(f"Creating voting ensemble with {len(estimators)} base models and weights {weight_list}")
    return VotingClassifier(estimators=estimators, voting=voting, weights=weight_list)

def create_stacking_ensemble(X_train, y_train, base_models=None, cv=None):
    """
    Create a stacking ensemble using a meta-model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        base_models: Dictionary of {name: model} pairs (default: create standard models)
        cv: Cross-validation strategy (default: from config)
        
    Returns:
        Trained stacking ensemble model
    """
    from sklearn.ensemble import StackingClassifier
    
    # Use default models if none provided
    if base_models is None:
        base_models = {
            'logistic_regression': create_logistic_regression(),
            'random_forest': create_random_forest(),
            'gradient_boosting': create_gradient_boosting()
        }
    
    # Use default CV if none provided
    if cv is None:
        cv = config.ENSEMBLE_PARAMS['stacking_cv']
        
    # Create estimators list for StackingClassifier
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Create meta-model
    meta_model = LogisticRegression(C=0.1, random_state=42)
    
    # Create and train stacking ensemble
    logger.info(f"Creating stacking ensemble with {len(estimators)} base models")
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv,
        passthrough=True  # Include original features
    )
    
    # Train the ensemble
    stack.fit(X_train, y_train)
    
    return stack

def create_upset_specialized_ensemble(train_data, features, upset_features):
    """
    Create an ensemble specialized for upset prediction.
    
    Args:
        train_data: Training DataFrame with features and targets
        features: List of features for the main model
        upset_features: List of features for the upset model
        
    Returns:
        Tuple of (trained main model, trained upset model, upset threshold)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Create main model
    X_train_main = train_data[features]
    y_train = train_data['result']
    
    # Scale features
    scaler_main = StandardScaler()
    X_train_main_scaled = scaler_main.fit_transform(X_train_main)
    
    # Train main model
    main_model = create_gradient_boosting()
    main_model.fit(X_train_main_scaled, y_train)
    
    # Create upset prediction model
    X_train_upset = train_data[upset_features]
    y_train_upset = train_data['upset']
    
    # Scale features
    scaler_upset = StandardScaler()
    X_train_upset_scaled = scaler_upset.fit_transform(X_train_upset)
    
    # Train upset model
    upset_model = create_gradient_boosting()
    upset_model.fit(X_train_upset_scaled, y_train_upset)
    
    # Use default upset threshold
    upset_threshold = config.ENSEMBLE_PARAMS['upset_threshold']
    
    logger.info(f"Created specialized upset ensemble with threshold {upset_threshold}")
    
    return main_model, upset_model, scaler_main, scaler_upset, upset_threshold

def predict_with_upset_ensemble(X_main, X_upset, favorite, main_model, upset_model, main_scaler, upset_scaler, upset_threshold):
    """
    Make predictions using the upset specialized ensemble.
    
    Args:
        X_main: Features for the main model
        X_upset: Features for the upset model
        favorite: Array indicating the favorite team (0=away, 1=home)
        main_model: Trained main model
        upset_model: Trained upset model
        main_scaler: Scaler for main features
        upset_scaler: Scaler for upset features
        upset_threshold: Threshold for considering a prediction an upset
        
    Returns:
        Array of predictions
    """
    # Scale features
    X_main_scaled = main_scaler.transform(X_main)
    X_upset_scaled = upset_scaler.transform(X_upset)
    
    # Get upset probabilities
    upset_proba = upset_model.predict_proba(X_upset_scaled)[:, 1]
    
    # Get main predictions
    main_predictions = main_model.predict(X_main_scaled)
    
    # Final predictions
    final_predictions = np.copy(main_predictions)
    
    # For games with high upset probability, predict an upset
    for i, upset_prob in enumerate(upset_proba):
        if upset_prob > upset_threshold:
            # Predict the opposite of the favorite
            final_predictions[i] = 1 - favorite[i]
            
    return final_predictions