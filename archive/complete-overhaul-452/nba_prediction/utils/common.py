"""
Common utility functions used across the project.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def ensure_directory_exists(directory_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
def set_plot_style():
    """Set the default plotting style for the project."""
    plt.style.use('ggplot')
    sns.set_theme(font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100

def save_figure(fig, filename, dir_path=None, formats=None):
    """
    Save a matplotlib figure to disk in specified formats.
    
    Args:
        fig: Matplotlib figure object
        filename: Filename without extension
        dir_path: Directory path (default: config.PLOTS_DIR)
        formats: List of formats to save in (default: ['png', 'pdf'])
    """
    from nba_prediction.config import config
    
    dir_path = dir_path or config.PLOTS_DIR
    formats = formats or ['png', 'pdf']
    
    ensure_directory_exists(dir_path)
    
    for fmt in formats:
        output_path = os.path.join(dir_path, f"{filename}.{fmt}")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return fig