"""
Simple cleanup script to delete contents of output directories.
"""

import os
import shutil
import time
from datetime import datetime

# Import utility modules if available, otherwise use basic logging
try:
    from utils.logging_config import setup_logger
    logger = setup_logger("cleanup", f"logs/cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("cleanup")

# List of directories to clean
required_dirs = [
    "processed_data", "plots", "elo_analysis",
    "team_analysis", "models", "evaluation",
    "final_report", "ensemble_model", "logs",
    "html_reports", "html_reports/images"
]

def clean_directory(directory):
    """Remove all files and subdirectories within a directory but keep the directory itself"""
    if not os.path.exists(directory):
        logger.info(f"Directory {directory} does not exist. Skipping.")
        return
    
    logger.info(f"Cleaning directory: {directory}")
    try:
        # Skip logs directory to preserve current logs
        if directory == "logs":
            logger.info("Skipping logs directory to preserve current logs")
            return
            
        # Remove all contents
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                    logger.debug(f"Removed file: {item_path}")
                elif os.path.isdir(item_path):
                    # Skip cleaning subdirectories that are in our required_dirs list
                    if item_path not in required_dirs:
                        shutil.rmtree(item_path)
                        logger.debug(f"Removed directory: {item_path}")
            except Exception as e:
                logger.error(f"Failed to remove {item_path}: {e}")
                
        logger.info(f"Finished cleaning {directory}")
    except Exception as e:
        logger.error(f"Error while cleaning {directory}: {e}")

def main():
    print("Starting cleanup...")
    start_time = time.time()
    
    # Clean each directory
    for directory in required_dirs:
        clean_directory(directory)
    
    duration = time.time() - start_time
    print(f"Cleanup completed in {duration:.2f} seconds")
    logger.info(f"Cleanup completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()