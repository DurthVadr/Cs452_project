"""
Logging configuration for the project.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

from nba_prediction.config import config

def get_logger(name):
    """
    Get a logger with the specified name and configuration.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Set level
        logger.setLevel(config.LOGGING_CONFIG["level"])
        
        # Create formatters
        formatter = logging.Formatter(config.LOGGING_CONFIG["format"])
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler for logging to file
        log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.split('.')[-1]}.log"
        file_handler = logging.FileHandler(os.path.join(config.LOGS_DIR, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_script_logger(script_file):
    """
    Get a logger for a script file, using the script's filename as the logger name.
    
    Args:
        script_file: __file__ from the script
        
    Returns:
        Configured logger
    """
    script_name = Path(script_file).stem
    return get_logger(f"nba_prediction.{script_name}")