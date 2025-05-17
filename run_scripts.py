"""
Main script to run all NBA prediction project scripts in sequence.
This script ensures all required directories exist and executes each script in order.
"""

import subprocess
import time
import os
from datetime import datetime

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists

# Set up logger
logger = setup_logger("run_scripts", f"logs/run_scripts_{datetime.now().strftime('%Y%m%d')}.log")
logger.info("Starting NBA prediction pipeline")

def check_required_files():
    """Check if critical files exist before running dependent scripts"""
    critical_files = [
        'processed_data/train_data_original.csv',
        'processed_data/test_data.csv',
        'processed_data/combined_data.csv'
    ]
    
    # Optional files - pipeline can continue if these are missing
    optional_files = [
        'processed_data/train_data_ros.csv',
        'processed_data/train_data_smote.csv',
    ]
    
    missing_critical = [f for f in critical_files if not os.path.exists(f)]
    missing_optional = [f for f in optional_files if not os.path.exists(f)]
    
    if missing_critical:
        logger.error(f"Missing critical files: {missing_critical}")
        logger.error("Data preparation may have failed. Please fix errors in data_preparation.py and run again.")
        return False
        
    if missing_optional:
        logger.warning(f"Missing optional files: {missing_optional}")
        logger.warning("Some sampling methods may not be available for analysis.")
    
    return True

# List of scripts to run in order
scripts = [
    "data_exploration.py",
    "data_preparation.py",
    "elo_analysis.py",
    "team_performance_analysis.py",
    "simplified_model_building.py",
    "model_evaluation.py",
    "ensemble_model_development.py",
    "create_final_report.py"
]

# Define directories
required_dirs = [
    "processed_data", "plots", "elo_analysis",
    "team_analysis", "models", "evaluation",
    "final_report", "ensemble_model", "logs",
    "html_reports", "html_reports/images",
    "models/original", "models/ros", "models/smote"  # Added model subdirectories
]

# Create directories if they don't exist
for directory in required_dirs:
    ensure_directory_exists(directory)
    logger.debug(f"Ensured directory exists: {directory}")

# Run each script in sequence
total_start_time = time.time()

for script in scripts:
    logger.info(f"Running {script}")
    start_time = time.time()

    try:
        result = subprocess.run(["python", script], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully completed {script} in {time.time() - start_time:.2f} seconds")
            
            # Check data preparation results before proceeding
            if script == "data_preparation.py":
                if not check_required_files():
                    logger.error("Critical files missing after data preparation. Stopping pipeline execution.")
                    break
                logger.info("Data preparation completed and all required files were created successfully.")
                
        else:
            logger.error(f"Error running {script}. Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            
            # Stop execution if data preparation fails as other scripts depend on it
            if script == "data_preparation.py":
                logger.error("Data preparation failed. Stopping pipeline execution.")
                break
            
            logger.warning("Continuing with next script despite error")
            
    except Exception as e:
        logger.error(f"Exception while running {script}: {e}")
        
        # Stop execution if data preparation fails
        if script == "data_preparation.py":
            logger.error("Data preparation failed. Stopping pipeline execution.")
            break
            
        logger.warning("Continuing with next script despite error")

total_execution_time = time.time() - total_start_time
logger.info(f"All scripts have been executed. Total execution time: {total_execution_time:.2f} seconds")

print(f"\nNBA prediction pipeline completed in {total_execution_time:.2f} seconds")
print("Check the logs directory for detailed execution logs")
