"""
Main script to run all NBA prediction project scripts in sequence.
This script ensures all required directories exist and executes each script in order.
"""

import subprocess
import time
from datetime import datetime

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists

# Set up logger
logger = setup_logger("run_scripts", f"logs/run_scripts_{datetime.now().strftime('%Y%m%d')}.log")
logger.info("Starting NBA prediction pipeline")

# List of scripts to run in order
scripts = [
    "data_exploration.py",
    "data_preparation.py",
    "elo_analysis.py",
    "team_performance_analysis.py",
    "simplified_model_building.py",
    "model_evaluation.py",
    "create_final_report.py",
    "ensemble_model_development.py"
]

# Create necessary directories
required_dirs = [
    "processed_data", "plots", "elo_analysis",
    "team_analysis", "models", "evaluation",
    "final_report", "ensemble_model", "logs",
    "html_reports", "html_reports/images"
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
        else:
            logger.error(f"Error running {script}. Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            logger.warning("Continuing with next script despite error")
    except Exception as e:
        logger.error(f"Exception while running {script}: {e}")
        logger.warning("Continuing with next script despite error")

total_execution_time = time.time() - total_start_time
logger.info(f"All scripts have been executed. Total execution time: {total_execution_time:.2f} seconds")