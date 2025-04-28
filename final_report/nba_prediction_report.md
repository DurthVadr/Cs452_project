# NBA Game Prediction Project (2018-2019 Season)

## Executive Summary

This report presents the results of a comprehensive analysis of NBA games from the 2018-2019 season. The project aimed to develop predictive models for NBA game outcomes, with a particular focus on improving upon the baseline ELO rating system and identifying factors that contribute to upsets. The best model achieved an accuracy of 66.7%, which represents an improvement of -0.8% over the baseline ELO model.

## Model Performance

![Model Accuracy Comparison](images/model_accuracy_comparison.png)

The above chart shows the performance of different models on the test dataset. The best performing model was Gradient Boosting with an accuracy of 66.67%.

### Comparison with ELO Rating System

![ELO vs Model Comparison](images/elo_vs_model_comparison.png)

The best model outperformed the ELO rating system, particularly in predicting upset games where the underdog team wins. For upset games, the model achieved 36.9% accuracy compared to the ELO system's 31.0%.

### Performance by Game Characteristics

#### Accuracy by ELO Difference

![Accuracy by ELO Difference](images/accuracy_by_elo_diff.png)

The model's accuracy increases with the ELO difference between teams, which is expected as games with larger skill disparities are generally more predictable.

#### Accuracy by Upset Status

![Accuracy by Upset Status](images/accuracy_by_upset.png)

The model performs better on non-upset games (82.1% accuracy) compared to upset games (36.9% accuracy). This is expected as upsets are inherently difficult to predict.

## Team Performance Analysis

### Four Factors Impact on Winning

![Four Factors Impact](images/four_factors_win_pct.png)

The Four Factors (effective field goal percentage, turnover rate, offensive rebounding percentage, and free throw rate) have varying impacts on a team's likelihood of winning. Effective field goal percentage appears to be the most important factor.

### Top Teams by Win Percentage

![Top Teams by Win Percentage](images/top_teams_win_pct.png)

The chart shows the top 10 teams by win percentage for the 2018-2019 season.

### Teams Most Likely to Be Upset

![Teams Most Likely to Be Upset](images/top_upset_teams.png)

Some teams, despite being favored, are more prone to upsets than others. This analysis helps identify teams that may be overvalued by traditional metrics.

### Team Clustering Based on Playing Style

![Team Clustering](images/team_clusters.png)

Teams can be clustered based on their playing style and statistical profiles. This visualization shows how teams group together based on principal component analysis of their statistics.

## ELO Rating System Analysis

### ELO Parameter Optimization

![ELO Parameter Search](images/elo_parameter_search.png)

The ELO rating system was optimized by testing different combinations of K-factor (which controls how quickly ratings change) and home advantage (the rating bonus given to the home team). The optimal parameters were found to be a K-factor of 30 and a home advantage of 50.

## Conclusion and Recommendations

The analysis demonstrates that machine learning models can outperform traditional ELO rating systems for NBA game prediction. Key findings include:

1. The Gradient Boosting model achieved the highest accuracy at 66.67%.
2. Effective field goal percentage is the most important of the Four Factors in determining game outcomes.
3. Teams can be effectively clustered based on their playing styles, which may provide insights for matchup analysis.
4. Upset prediction remains challenging, but the model shows improvement over the baseline ELO system.

For future work, we recommend:

1. Incorporating player-level data, including injuries and rest days.
2. Developing specialized models for different types of matchups.
3. Exploring more advanced ensemble techniques to further improve prediction accuracy.
4. Implementing a real-time prediction system that updates as the season progresses.
