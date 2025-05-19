"""
Main script to run the NBA prediction pipeline.

This script provides a clean interface to run the entire NBA prediction pipeline
or individual components as needed. It ensures all required directories exist
and executes each script in the correct order.
"""

import subprocess
import time
import os
import argparse
from datetime import datetime
import sys

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists

# Set up logger
logger = setup_logger("run_pipeline", f"logs/run_pipeline_{datetime.now().strftime('%Y%m%d')}.log")

# Define the pipeline stages and corresponding scripts
PIPELINE_STAGES = {
    'data_exploration': 'data_exploration.py',
    'data_preparation': 'data_preparation.py',
    'elo_analysis': 'elo_analysis.py',
    'team_analysis': 'team_performance_analysis.py',
    'model_building': 'optimized_model_building.py',  # Updated to use optimized model building
    'model_evaluation': 'model_evaluation.py',
    'ensemble_model': 'ensemble_model.py',
    'report': 'create_report.py'
}

# Define required directories
REQUIRED_DIRS = [
    "processed_data", "plots", "elo_analysis",
    "team_analysis", "models", "evaluation",
    "final_report", "ensemble_model", "logs",
    "html_reports", "html_reports/images"
]

def run_script(script_path, continue_on_error=True):
    """
    Run a Python script and return its exit code.

    Args:
        script_path: Path to the script to run
        continue_on_error: Whether to continue execution if the script fails

    Returns:
        Exit code of the script (0 for success)
    """
    logger.info(f"Running {script_path}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"Successfully completed {script_path} in {time.time() - start_time:.2f} seconds")
            return 0
        else:
            logger.error(f"Error running {script_path}. Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")

            if continue_on_error:
                logger.warning("Continuing with next script despite error")
                return result.returncode
            else:
                logger.error("Stopping pipeline due to error")
                sys.exit(result.returncode)
    except Exception as e:
        logger.error(f"Exception while running {script_path}: {e}")

        if continue_on_error:
            logger.warning("Continuing with next script despite error")
            return 1
        else:
            logger.error("Stopping pipeline due to error")
            sys.exit(1)

def create_directories():
    """Create all required directories if they don't exist."""
    for directory in REQUIRED_DIRS:
        ensure_directory_exists(directory)
        logger.debug(f"Ensured directory exists: {directory}")

def run_pipeline(stages=None, continue_on_error=True):
    """
    Run the specified stages of the pipeline.

    Args:
        stages: List of stages to run. If None, run all stages.
        continue_on_error: Whether to continue execution if a script fails
    """
    # Create necessary directories
    create_directories()

    # Determine which stages to run
    if stages is None:
        stages_to_run = list(PIPELINE_STAGES.keys())
    else:
        stages_to_run = [stage for stage in stages if stage in PIPELINE_STAGES]
        if not stages_to_run:
            logger.error(f"No valid stages specified. Valid stages are: {', '.join(PIPELINE_STAGES.keys())}")
            return

    logger.info(f"Starting NBA prediction pipeline with stages: {', '.join(stages_to_run)}")

    # Run each stage in sequence
    total_start_time = time.time()
    exit_codes = {}

    for stage in stages_to_run:
        script = PIPELINE_STAGES[stage]
        logger.info(f"=== Starting {stage.upper()} stage ===")
        exit_code = run_script(script, continue_on_error)
        exit_codes[stage] = exit_code

        if exit_code != 0 and not continue_on_error:
            break

        logger.info(f"=== Completed {stage.upper()} stage ===")

    # Summarize results
    total_execution_time = time.time() - total_start_time
    logger.info(f"All scripts have been executed. Total execution time: {total_execution_time:.2f} seconds")

    # Report any failures
    failures = {stage: code for stage, code in exit_codes.items() if code != 0}
    if failures:
        logger.warning(f"The following stages had errors: {', '.join(failures.keys())}")
    else:
        logger.info("All stages completed successfully!")

    return len(failures) == 0

def main():
    """Parse command line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the NBA prediction pipeline")
    parser.add_argument('--stages', type=str, nargs='+',
                        choices=list(PIPELINE_STAGES.keys()) + ['all'],
                        default=['all'],
                        help="Pipeline stages to run")
    parser.add_argument('--stop-on-error', action='store_true',
                        help="Stop pipeline execution if a script fails")

    args = parser.parse_args()

    # Determine which stages to run
    if 'all' in args.stages:
        stages = None  # Run all stages
    else:
        stages = args.stages

    # Run the pipeline
    success = run_pipeline(stages, not args.stop_on_error)

    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
