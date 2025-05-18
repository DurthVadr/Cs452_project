"""
Generate a final comprehensive report for the NBA prediction project.
This version ensures proper model names and includes all upset metrics.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger
from nba_prediction.reporting.improved_report import generate_improved_report

logger = get_logger('final_report_generator')

def main():
    """Generate the final report with proper model names and complete metrics."""
    logger.info("Starting final report generation")
    
    try:
        # First, calculate upset metrics for all models
        logger.info("Calculating upset metrics for all models")
        os.system(f"{sys.executable} calculate_upset_metrics.py")
        
        # Now generate the improved report
        report_path = generate_improved_report(report_name="final_report")
        
        logger.info(f"Final report successfully generated at: {report_path}")
        logger.info("Open this HTML file in your web browser to view the complete report")
        
    except Exception as e:
        logger.error(f"Failed to generate final report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()