import os
import shutil

# Folders to clean
folders = [
    "processed_data",
    "plots",
    "elo_analysis",
    "team_analysis", 
    "models",
    "evaluation",
    "final_report",
    "ensemble_model"
]

for folder in folders:
    if os.path.exists(folder):
        print(f"Cleaning {folder}...")
        # Remove all contents but keep the folder
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error with {item_path}: {e}")

print("All folder contents cleaned.")