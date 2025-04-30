import os
import subprocess

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
    "final_report", "ensemble_model"
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Run each script in sequence
for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script])
    print(f"Completed {script}")

print("All scripts have been executed.")