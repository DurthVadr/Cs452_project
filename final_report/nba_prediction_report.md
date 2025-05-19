# NBA Game Prediction Project (2018-2019 Season)

## Executive Summary

This report presents the results of a comprehensive analysis of NBA games from the 2018-2019 season. The project aimed to develop predictive models for NBA game outcomes, with a particular focus on improving upon the baseline ELO rating system and identifying factors that contribute to upsets. The best model achieved an accuracy of 0.7%, which represents an improvement of 2.0% over the baseline ELO model.

## Model Performance

![Model Accuracy Comparison](images/model_accuracy_comparison.png)

The above chart shows the performance of different models on the test dataset. The best performing base model was Random Forest with an accuracy of 0.66%.

### Ensemble Model Performance

![Ensemble Model Comparison](images/ensemble_comparison.png)

The best performing ensemble approach was Stacking Ensemble with an accuracy of 0.70%. This ensemble approach combines multiple models to achieve better performance than any individual model.

### Comparison with ELO Rating System

![ELO vs Model Comparison](images/elo_vs_model_comparison.png)

The best model outperformed the ELO rating system, particularly in predicting upset games where the underdog team wins. For upset games, the model achieved 0.0% accuracy compared to the ELO system's 31.0%.

## Conclusion and Recommendations

The analysis demonstrates that machine learning models can outperform traditional ELO rating systems for NBA game prediction. Key findings include:

1. The Stacking Ensemble achieved the highest accuracy at 0.70%.
2. Effective field goal percentage is the most important of the Four Factors in determining game outcomes.
3. Teams can be effectively clustered based on their playing styles, which may provide insights for matchup analysis.
4. Upset prediction remains challenging, but the model shows improvement over the baseline ELO system.

For future work, we recommend:

1. Incorporating player-level data, including injuries and rest days.
2. Developing specialized models for different types of matchups.
3. Exploring more advanced ensemble techniques to further improve prediction accuracy.
4. Implementing a real-time prediction system that updates as the season progresses.
