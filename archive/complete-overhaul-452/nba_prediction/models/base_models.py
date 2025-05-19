"""
Base model definitions for NBA game prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_logistic_regression(params=None):
    """
    Create a logistic regression model with specified parameters.
    
    Args:
        params: Dictionary of parameters (default: from config)
        
    Returns:
        LogisticRegression model
    """
    params = params or config.MODEL_PARAMS["logistic_regression"]
    logger.info(f"Creating LogisticRegression with params: {params}")
    return LogisticRegression(**params)

def create_random_forest(params=None):
    """
    Create a random forest model with specified parameters.
    
    Args:
        params: Dictionary of parameters (default: from config)
        
    Returns:
        RandomForestClassifier model
    """
    params = params or config.MODEL_PARAMS["random_forest"]
    logger.info(f"Creating RandomForestClassifier with params: {params}")
    return RandomForestClassifier(**params)

def create_gradient_boosting(params=None):
    """
    Create a gradient boosting model with specified parameters.
    
    Args:
        params: Dictionary of parameters (default: from config)
        
    Returns:
        GradientBoostingClassifier model
    """
    params = params or config.MODEL_PARAMS["gradient_boosting"]
    logger.info(f"Creating GradientBoostingClassifier with params: {params}")
    return GradientBoostingClassifier(**params)

def get_model_factory(model_type):
    """
    Get the factory function for the specified model type.
    
    Args:
        model_type: Type of model ('logistic_regression', 'random_forest', or 'gradient_boosting')
        
    Returns:
        Function that creates a model of the specified type
    """
    factories = {
        'logistic_regression': create_logistic_regression,
        'random_forest': create_random_forest,
        'gradient_boosting': create_gradient_boosting
    }
    
    if model_type not in factories:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return factories[model_type]

def train_model(X_train, y_train, model_type=None, params=None):
    """
    Train a model on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to train (default: 'gradient_boosting')
        params: Dictionary of parameters for the model
        
    Returns:
        Trained model
    """
    model_type = model_type or 'gradient_boosting'
    
    # Create the model
    factory = get_model_factory(model_type)
    model = factory(params)
    
    # Train the model
    logger.info(f"Training {model_type} model on {X_train.shape[0]} samples")
    model.fit(X_train, y_train)
    
    return model