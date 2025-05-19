# NBA Game Prediction Project (2018-2019 Season)

This project uses data science techniques to predict the outcomes of NBA games for the 2018-2019 season. It includes data exploration, feature engineering, ELO rating system optimization, team performance analysis, model building (including specialized upset prediction), and ensemble modeling.

## Recent Improvements

The project has been enhanced with the following improvements:

1. **Optimized Model Building Approach**:
   - Implemented a centralized configuration system with feature_config.py and config.py
   - Added SMOTE (Synthetic Minority Over-sampling Technique) for better handling of class imbalance
   - Created a focused feature selection approach based on basketball analytics principles
   - Improved model architecture with weighted ensemble techniques
   - Enhanced upset prediction capabilities with specialized evaluation metrics

2. **Code Organization and Structure**:
   - Added a centralized logging system for consistent logging across all scripts
   - Created utility modules for common functions to reduce code duplication
   - Improved error handling with try-except blocks for better robustness
   - Organized project structure with clear separation of configuration, data preparation, and modeling

3. **Advanced Feature Engineering**:
   - Added differential features between home and away teams for Four Factors (eFGp, FTr, ORBp, TOVp)
   - Created interaction features to capture combined effects of shooting efficiency with turnover prevention and offensive rebounding
   - Implemented a structured approach to feature organization (base features, four factors, advanced metrics)
   - These new features help the model better understand team matchup dynamics

4. **Momentum Features**:
   - Added streak indicators (positive for winning streaks, negative for losing streaks)
   - Implemented weighted recent performance metrics that give more importance to recent games
   - These features capture a team's current form and momentum going into each game

5. **Comprehensive Logging and Reporting**:
   - Implemented a structured logging system with different log levels (debug, info, warning, error)
   - Added detailed logging for model training, evaluation, and performance metrics
   - Created enhanced HTML reports with interactive visualizations
   - Log files are organized by script and date for easy troubleshooting

## Data Requirements

To run this project locally, you need the following data files placed in a `data` directory within your project folder:

- `game_info.csv`: Contains basic information about each game.
- `team_stats.csv`: Contains team statistics for each game.
- `team_full_10.csv`, `team_full_20.csv`, `team_full_30.csv`: Contain team boxscore data averaged over the last 10, 20, and 30 games.
- `team_factor_10.csv`, `team_factor_20.csv`, `team_factor_30.csv`: Contain team Four Factors data averaged over the last 10, 20, and 30 games.

**Note:** The scripts currently assume these files are in a `data` directory within your project folder. You may need to modify the file paths in the Python scripts if you place them elsewhere.

## Project Structure

The project is organized into the following components:

- **Main Scripts**: Core analysis and modeling scripts that are executed in sequence
- **Utility Modules**: Reusable functions and configuration in the `utils/` directory
- **Data Directory**: Contains input data files
- **Output Directories**: Various directories for storing results and visualizations

### Directory Structure

```
nba_prediction/
├── data/                  # Input data files
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── logging_config.py  # Centralized logging configuration
│   └── common.py          # Common utility functions
├── logs/                  # Log files generated during execution
├── processed_data/        # Processed data files
├── plots/                 # Exploratory visualizations
├── elo_analysis/          # ELO system analysis results
├── team_analysis/         # Team performance analysis results
├── models/                # Trained models and feature importance
├── evaluation/            # Model evaluation results
├── final_report/          # Final report and visualizations
├── html_reports/          # HTML reports optimized for web viewing
│   └── images/            # Images for HTML reports
├── ensemble_model/        # Ensemble model results
└── archive/               # Archived files not currently in use
```

## Setup Instructions

1.  **Python Environment**: Ensure you have Python 3.10 or later installed.
2.  **Clone Repository**: Clone this repository or download the source code.
3.  **Install Libraries**: Install the required Python libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
4.  **Prepare Data Directory**: Create a `data` directory and place the required data files (see Data Requirements section).

## Running the Pipeline

You can run the entire pipeline or specific stages using the `run_pipeline.py` script:

```bash
# Run the entire pipeline
python run_pipeline.py

# Run specific stages
python run_pipeline.py --stages data_exploration data_preparation

# Stop on first error
python run_pipeline.py --stop-on-error
```

Available stages:
- `data_exploration`: Explore and visualize the raw data
- `data_preparation`: Clean and prepare data for modeling
- `elo_analysis`: Analyze ELO rating system performance
- `team_analysis`: Analyze team performance metrics
- `model_building`: Build and tune prediction models
- `model_evaluation`: Evaluate model performance
- `ensemble_model`: Develop ensemble models
- `report`: Generate final reports

Alternatively, you can run each script individually in the following order:

1.  **Data Exploration and Preparation**:
    ```bash
    python data_exploration.py
    python data_preparation.py
    ```
    *   **Input**: Raw CSV files from the `data/` directory.
    *   **Output**:
        *   `data_summary.md` (in project root)
        *   Processed data files in `processed_data/`
        *   Exploratory visualizations in `plots/`
    *   **Features**: The `data_preparation.py` creates:
        *   Differential and interaction features between home and away teams
        *   Streak indicators to capture winning and losing streaks
        *   Weighted recent performance metrics that prioritize more recent games

2.  **ELO Rating System Analysis**:
    ```bash
    python elo_analysis.py
    ```
    *   **Input**: `game_info.csv` from the `data/` directory.
    *   **Output**: ELO analysis results and visualizations in `elo_analysis/`.

3.  **Team Performance Analysis**:
    ```bash
    python team_performance_analysis.py
    ```
    *   **Input**: Processed data from `processed_data/`.
    *   **Output**: Team analysis results and visualizations in `team_analysis/`.

4.  **Model Building**:
    ```bash
    python optimized_model_building.py
    ```
    *   **Input**: Processed data from `processed_data/`.
    *   **Output**: Trained models, evaluation results, and visualizations in `models/` and `plots/`.
    *   **Features**: The optimized model building script includes:
        * Centralized configuration with feature_config.py and config.py
        * SMOTE for handling class imbalance
        * Focused feature selection based on basketball analytics
        * Weighted ensemble techniques
        * Comprehensive upset prediction evaluation
        * Enhanced HTML reporting

5.  **Model Evaluation**:
    ```bash
    python model_evaluation.py
    ```
    *   **Input**: Processed data and trained models from previous steps.
    *   **Output**: Detailed model evaluation results and visualizations in `evaluation/`.

6.  **Ensemble Model Development**:
    ```bash
    python ensemble_model.py
    ```
    *   **Input**: Processed data and trained models.
    *   **Output**: Ensemble models, evaluation results, and visualizations in `ensemble_model/`.

7.  **Report Generation**:
    ```bash
    python create_report.py
    ```
    *   **Input**: Results from all previous steps.
    *   **Output**: Final report in Markdown and HTML formats, along with necessary images, in `final_report/` and `html_reports/`.

## Output Description

After running all the scripts, the project directory will contain the following subdirectories with results:

- `processed_data/`: Cleaned and prepared data used for modeling.
- `plots/`: Visualizations from the initial data exploration.
- `elo_analysis/`: Results and plots related to the ELO rating system.
- `team_analysis/`: Results and plots from the team performance analysis.
- `models/`: Trained machine learning models, evaluation results, and feature importance plots.
- `evaluation/`: Detailed model evaluation metrics, confusion matrices, ROC curves, and error analysis plots.
- `final_report/`: The final comprehensive project report in Markdown and HTML formats, including all generated visualizations.
- `ensemble_model/`: Trained ensemble models, comparison plots, and summary report for the ensemble modeling approaches.
- `logs/`: Log files from each script execution, organized by script name and date.
- `html_reports/`: HTML reports with visualizations, optimized for web viewing.

## Logging System

The project includes a comprehensive logging system that:

1. **Creates Structured Logs**: Each script generates detailed logs with timestamps and log levels.
2. **Supports Multiple Log Levels**: Debug, info, warning, error, and critical levels for different types of messages.
3. **Organizes Logs by Script**: Each script has its own log file named with the script name and date.
4. **Provides Console Output**: Important messages are displayed in the console during execution.
5. **Facilitates Debugging**: Detailed error messages and stack traces are captured for troubleshooting.

Log files are stored in the `logs/` directory with filenames in the format `script_name_YYYYMMDD.log`.

## Extending the Project

To add new features or models to the project:

1. Follow the established code structure and logging patterns.
2. Use the utility modules in the `utils/` directory for common functionality.
3. Add appropriate logging statements to track execution and performance.
4. Update the README.md file to document your changes.

By following these guidelines, you should be able to reproduce the analysis and prediction results presented in the project reports and extend the project with your own improvements.
