"""
Run the complete NBA game prediction pipeline.

This script executes all stages of the pipeline in sequence:
1. Data preparation
2. Model training
3. Model evaluation
4. Ensemble building
5. Report generation
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.utils.logging_config import get_logger

logger = get_logger('pipeline_runner')

def run_script(script_path):
    """Run a Python script and return its exit code."""
    logger.info(f"Running {script_path}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        logger.info(f"Completed {script_path} in {elapsed:.1f} seconds")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return e.returncode

def main():
    """Run the complete pipeline or selected stages."""
    parser = argparse.ArgumentParser(description="Run the NBA prediction pipeline")
    parser.add_argument('--stages', type=str, nargs='+', choices=['data', 'train', 'evaluate', 'ensemble', 'report', 'all'],
                        default=['all'], help="Pipeline stages to run")
    
    args = parser.parse_args()
    
    # Define the pipeline stages
    pipeline = {
        'data': '01_prepare_data.py',
        'train': '02_train_models.py',
        'evaluate': '03_evaluate_models.py',
        'ensemble': '04_build_ensembles.py',
        'report': '05_generate_reports.py'
    }
    
    # Determine which stages to run
    stages_to_run = list(pipeline.keys()) if 'all' in args.stages else [s for s in args.stages if s != 'all']
    
    logger.info(f"Starting NBA prediction pipeline with stages: {', '.join(stages_to_run)}")
    
    # Run each stage in sequence
    successful = True
    for stage in stages_to_run:
        script_path = pipeline[stage]
        logger.info(f"=== Starting {stage.upper()} stage ===")
        exit_code = run_script(script_path)
        
        if exit_code != 0:
            logger.error(f"{stage.upper()} stage failed with exit code {exit_code}")
            successful = False
            break
        
        logger.info(f"=== Completed {stage.upper()} stage successfully ===")
    
    if successful:
        logger.info("NBA prediction pipeline completed successfully!")
    else:
        logger.error("NBA prediction pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()