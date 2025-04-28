# NBA Game Prediction Project (2018-2019 Season)

This project uses data science techniques to predict the outcomes of NBA games for the 2018-2019 season. It includes data exploration, feature engineering, ELO rating system optimization, team performance analysis, model building (including specialized upset prediction), and ensemble modeling.

## Data Requirements

To run this project locally, you need the following data files placed in a directory accessible by the scripts (e.g., `/home/ubuntu/upload/` or a similar path you configure in the scripts):

- `game_info.csv`: Contains basic information about each game.
- `team_stats.csv`: Contains team statistics for each game.
- `nbaallelo.csv`: Contains historical ELO ratings for NBA teams.
- `team_full_10.csv`, `team_full_20.csv`, `team_full_30.csv`: Contain team boxscore data averaged over the last 10, 20, and 30 games.
- `team_factor_10.csv`, `team_factor_20.csv`, `team_factor_30.csv`: Contain team Four Factors data averaged over the last 10, 20, and 30 games.

**Note:** The scripts currently assume these files are in `/home/ubuntu/upload/`. You may need to modify the file paths in the Python scripts if you place them elsewhere.

## Setup Instructions

1.  **Python Environment**: Ensure you have Python 3.10 or later installed.
2.  **Create Project Directory**: Create a main directory for the project (e.g., `nba_prediction`).
3.  **Install Libraries**: Install the required Python libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
4.  **Place Scripts**: Place all the Python scripts (`data_exploration.py`, `data_preparation.py`, `elo_analysis.py`, `team_performance_analysis.py`, `simplified_model_building.py`, `model_evaluation.py`, `create_final_report.py`, `ensemble_model_development.py`) into the project directory.

## Execution Order

Run the Python scripts in the following order from the project directory (` `):

1.  **Data Exploration and Preparation**:
    ```bash
    python3 data_exploration.py
    python3 data_preparation.py
    ```
    *   **Input**: Raw CSV files from `/home/ubuntu/upload/`.
    *   **Output**: 
        *   `data_summary.md` (in project root)
        *   Processed data files in ` processed_data/`
        *   Exploratory visualizations in ` exploration_plots/`

2.  **ELO Rating System Analysis**:
    ```bash
    python3 elo_analysis.py
    ```
    *   **Input**: `nbaallelo.csv`, `game_info.csv`.
    *   **Output**: ELO analysis results and visualizations in ` elo_analysis/`.

3.  **Team Performance Analysis**:
    ```bash
    python3 team_performance_analysis.py
    ```
    *   **Input**: Processed data from ` processed_data/`.
    *   **Output**: Team analysis results and visualizations in ` team_analysis/`.

4.  **Model Building (Simplified Version)**:
    ```bash
    python3 simplified_model_building.py 
    ```
    *   **Input**: Processed data from ` processed_data/`.
    *   **Output**: Trained models, evaluation results, and visualizations in ` models/`.

5.  **Model Evaluation**:
    ```bash
    python3 model_evaluation.py
    ```
    *   **Input**: Processed data and trained models from previous steps.
    *   **Output**: Detailed model evaluation results and visualizations in ` evaluation/`.

6.  **Final Report Generation**:
    ```bash
    python3 create_final_report.py
    ```
    *   **Input**: Results from all previous steps.
    *   **Output**: Final report in Markdown and HTML formats, along with necessary images, in ` final_report/`.

7.  **Ensemble Model Development**:
    ```bash
    python3 ensemble_model_development.py
    ```
    *   **Input**: Processed data and trained models.
    *   **Output**: Ensemble models, evaluation results, and visualizations in ` ensemble_model/`.

## Output Description

After running all the scripts, the project directory will contain the following subdirectories with results:

- `processed_data/`: Cleaned and prepared data used for modeling.
- `exploration_plots/`: Visualizations from the initial data exploration.
- `elo_analysis/`: Results and plots related to the ELO rating system.
- `team_analysis/`: Results and plots from the team performance analysis.
- `models/`: Trained machine learning models, evaluation results, and feature importance plots from the simplified model building step.
- `evaluation/`: Detailed model evaluation metrics, confusion matrices, ROC curves, and error analysis plots.
- `final_report/`: The final comprehensive project report in Markdown (`nba_prediction_report.md`) and HTML (`improved_nba_prediction_report.html`) formats, including all generated visualizations in the `images/` subdirectory.
- `ensemble_model/`: Trained ensemble models, comparison plots, and summary report for the ensemble modeling approaches.

By following these steps, you should be able to reproduce the analysis and prediction results presented in the project reports.
