# NBA Game Prediction Model

## Overview
This project implements various machine learning models to predict NBA game outcomes using historical game data and team statistics. The models analyze "Four Factors" of basketball success along with other performance metrics to make predictions.

## Data Sources
The project uses several datasets from NBA games including:
- Game information and results
- Team statistics
- Four Factors data (10, 20, and 30 game averages)
- Complete boxscore data (10, 20, and 30 game averages)
- Historical NBA Elo ratings

## Models Implemented
The project evaluates multiple classification models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Elo Rating System

## Key Features
- Custom `ModelEval` class for streamlined model evaluation
- Automated cross-validation and grid search
- Performance metrics tracking
- Confusion matrix visualization
- Residuals analysis
- Elo rating system implementation with home court advantage

## Results
The models achieved varying degrees of success in predicting game outcomes. Top performing models include:
- Random Forest Classifier
- Logistic Regression with grid search optimization
- Custom Elo rating system

## Project Structure
```
├── data/                  # Data files and CSV exports
├── nba_modeling.ipynb     # Main Jupyter notebook with analysis
└── README.md             # Project documentation
```

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn

## Usage
1. Clone the repository
2. Install required dependencies
3. Run the Jupyter notebook `nba_modeling.ipynb`
4. Data will be automatically downloaded and cached in the `data/` directory

## Data Features
The Four Factors analyzed in this project are:
- Effective Field Goal Percentage (eFG%)
- Turnover Rate (TOV%)
- Offensive Rebound Rate (ORB%)
- Free Throw Rate (FTR)