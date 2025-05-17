"""
Logging configuration for the NBA prediction project.
This module provides a centralized logging configuration for all scripts.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional, Union, Literal

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Union[int, Literal["debug", "info", "warning", "error", "critical"]] = "info",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file. If None, a default path will be used
        level: Log level (debug, info, warning, error, critical)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        
    Returns:
        Configured logger
    """
    # Convert string level to logging level if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output:
        # Use default log file if none provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"logs/{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_script_logger(script_name: str) -> logging.Logger:
    """
    Get a logger for a script with standard configuration.
    
    Args:
        script_name: Name of the script (without .py extension)
        
    Returns:
        Configured logger
    """
    # Remove .py extension if present
    if script_name.endswith('.py'):
        script_name = script_name[:-3]
    
    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"logs/{script_name}_{timestamp}.log"
    
    return setup_logger(script_name, log_file, level="info")
