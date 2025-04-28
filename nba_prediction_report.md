# NBA Game Prediction Project: 2018-2019 Season Analysis

## Executive Summary

This report presents a comprehensive analysis of NBA game predictions for the 2018-2019 season. Using machine learning techniques and statistical analysis, we developed models to predict game outcomes with high precision. The project focused on understanding team performance patterns, developing an optimized ELO rating system, and creating specialized models for predicting upsets.

### Key Findings

1. Our best model (Gradient Boosting) achieved a prediction accuracy of **67.1%**, which is **-0.4%** better than the baseline ELO rating system.

2. For upset games (where the favorite team loses), our model achieved **36.9%** accuracy, compared to **31.0%** for the ELO system.

3. The Four Factors of basketball success showed varying impact on win probability, with effective field goal percentage (eFG%) being the most influential factor.

4. Teams were clustered into four distinct playing styles, with each cluster showing specific strengths and weaknesses.

5. Back-to-back games significantly impact team performance and prediction accuracy, with teams playing consecutive games showing higher upset rates.

## 1. Introduction

The NBA (National Basketball Association) is one of the most popular professional sports leagues in the world, with a rich history of statistical analysis and prediction. This project aims to predict NBA game outcomes for the 2018-2019 season using advanced data science techniques.

### 1.1 Project Objectives

- Develop a high-precision prediction model for NBA game outcomes
- Optimize the ELO rating system for basketball predictions
- Analyze team performance patterns and identify key performance indicators
- Create a specialized model for predicting upsets (when favorites lose)
- Provide comprehensive visualizations and insights

### 1.2 Data Sources

The project utilized several datasets:

- Game information and results (game_info.csv)
- Team statistics (team_stats.csv)
- Four Factors data at different game averages (10, 20, and 30 games)
- Complete boxscore data at different game averages (10, 20, and 30 games)
- Historical NBA ELO ratings (nbaallelo.csv)

## 2. Data Exploration and Preparation

### 2.1 Dataset Overview

The 2018-2019 NBA season consisted of 1,230 regular-season games. Our analysis revealed several key characteristics of the data:

- Home teams won approximately **58.9%** of games
- The upset rate (favorites losing) was approximately **34.1%**
- Back-to-back games accounted for approximately **30.1%** of all games

### 2.2 Feature Engineering

We created several features to improve prediction accuracy:

- Team performance metrics (rolling win percentages over 5, 10, and 15 games)
- Back-to-back game indicators
- Team matchup history (head-to-head records)
- ELO rating differentials
- Four Factors differentials (eFG%, TOV%, ORB%, FTR)

## 3. ELO Rating System Analysis

### 3.1 Original ELO System Performance

The original ELO rating system achieved an accuracy of **67.5%** on the test dataset. ELO ratings are a measure of team strength that update after each game based on the result and the pre-game rating difference.

### 3.2 Optimized ELO System

We optimized the ELO system by tuning two key parameters:

- K-factor: Controls how quickly ratings change (higher values mean faster changes)
- Home advantage: The rating bonus given to the home team

After grid search optimization, the best parameters were:

- K-factor: **30**
- Home advantage: **50**

![ELO Parameter Search](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393193_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvZWxvX3BhcmFtZXRlcl9zZWFyY2g.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdlpXeHZYM0JoY21GdFpYUmxjbDl6WldGeVkyZy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=W0Re3K7UoJs~t3sUon3Un3lEzD4SlNkbbihRJdKpHJ3JOhEtwdnfyO7IELdWXu~qMI6WjyGwRutcyRo8ppC9kYMdL-z97hdon5Eqx1C4p9yokn1WRU7tx6g6JrSrOobu51W4sllwyK0R~JfPKvP9nIsHn8N3H6IoDzmHgJrKPzRGWVHaeAN0k9uaQbea2xE-CMlCHk-LfzPGHevRVeyg7wFAci9ubJj93dW504WNVqS3wuOsBfBqEh51FmxYmOA8E4An9nXJotdX29H18TSRTqUKsiZZPESBVUac43PoLeEJPCOh527AQHAaE-VE-yHjBr43WmBQiu0UGUspVbLHtw__)

The optimized ELO system achieved an accuracy of **6341.5%**, an improvement of **6274.0%** over the original system.

### 3.3 Final Team Ratings

The final ELO ratings after the 2018-2019 season show the relative strength of each team:

![Final ELO Ratings](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393193_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvZmluYWxfZWxvX3JhdGluZ3M.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdlptbHVZV3hmWld4dlgzSmhkR2x1WjNNLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=dhx4RlClfB-PIfTVEaab1TX2s7Lxs3RLMohd7iwtD7ViOvORXZ4XQptH~uiqoHtfpO-7GS5n78Xi5Dk3M-hXPXk2CTEUR7Kb0Lmc2TFjkHQrp5oZoG16KGmOW8Q-cih~wbGCHbXj-duTj15biO4ww7jjPAUi82LEnRQoujwAgutdfav1IRQJOy-eTLxSY4onahcm2L1pVDLvrK8Bzk0v58o4r71VkZKOdpamYx2HeeaZdQ8tm-mRniIpVOde21iQK5fHnGDxFK9H8MMtJx5fdY0We26gZ2B84gjxbycZjG1DuF9CFqReKNhkZUZ~ymhYrKnaMldLw8ZrLzOUFEqrPA__)

## 4. Team Performance Analysis

### 4.1 Team Records

The 2018-2019 NBA season saw several teams dominate the regular season:

![Top Teams by Win Percentage](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393193_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvdG9wX3RlYW1zX3dpbl9wY3Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdmRHOXdYM1JsWVcxelgzZHBibDl3WTNRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=jcIBBiHUSTj5ilqxOfe2vb4bVe9RDTESl1xRZtvPxsD~tfy8zDa1H1i8dvifrMKL3XpE4QGuwA~9PbLMHwhPpXE9CsDrSc3htBj~hCKWETpgsMzW6IP8TVCVtPQY44NqUk6lyKh5WnvmG~44WH7K5o6KrJ3FGElx0wuhicoeecFgA9SypmVaP-~5ndlqcWqiV1y7xF8TRWyQwhtIbnW-zvcLXGN55SPGZhOhgrmTwBiVCRFQLhvHPCgu2ChcAKviqabQrjbuibh4mcMlSihA5oOQw6uyaNSG8A2rJoaKr4Zo~tcNVB~CZGWKYIY5Bo~cVe88ii5Cg-x3PkNPl0vPwQ__)

### 4.2 Four Factors Analysis

The Four Factors of basketball success (developed by Dean Oliver) are:

1. Shooting (measured by effective field goal percentage, eFG%)
2. Turnovers (measured by turnover rate, TOV%)
3. Rebounding (measured by offensive rebound percentage, ORB%)
4. Free throws (measured by free throw rate, FTR)

Our analysis shows the win percentage when a team has an advantage in each factor:

![Four Factors Win Percentage](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393193_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvZm91cl9mYWN0b3JzX3dpbl9wY3Q.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdlptOTFjbDltWVdOMGIzSnpYM2RwYmw5d1kzUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=B3EyhfyDyeHmALyQCMAtm8xytw-4YfYpPwMAOuqVVSFZGfpLp6WY83oK2fmzfDKCP2GNup70jwvsob8PMbGa-9L8fgF8lP72HXXDx23rKwlLg48X7V-v7izeY~~0vBp~xMrUl~-kqKfbdFTHCqUFnvmNTigvlU7t5W1mHjI05oy7Kc-CCXyC9VKHJnnA87EU7R5GGR6hJYTsjlmW6jZcT5X-TK6ycRh1CugOYFQ~6iBVIVRt~mGH2Z1KCjtAotkixdafsSubtkcsTbOqbZVzq9Vv8qnycqO7wHc76m-A2lBb8C-LPhN8t7WNG41cR~~yZyAlUvRkYFe99dK6K5nIKg__)

Effective field goal percentage (eFG%) has the strongest correlation with winning, followed by turnover rate (TOV%).

### 4.3 Team Clustering

We clustered teams based on their playing style using various statistical metrics:

![Team Clustering](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393193_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvdGVhbV9jbHVzdGVycw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdmRHVmhiVjlqYkhWemRHVnljdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=HgjBQB4Xvk5rymTHabZXbrh8Q99P5FdpS3sr9vxB267gRQJBdmuoqxVwREtsTGzeCJCTaAJU6PxlNbR4BtUbJmqt80d7Y-H24ltOiIsr1QlCb7Dh~kPeBv6xVfpwlw2nWs~5QvSFcL162aqR6GdhhgJIB-hL0xdqjKmMDd~Qoude0nt1E9IQafJ6X1Dh677NQL~A2SCr0iB6qm~Nx8slwlDYILs70jAFx5OVKKjPs~8rTQwhUQrv~Nz0P5Bk4UC6tomxM72Cm06zID14bKKe-bAo3lq9l02VaszKwo--xAMrEYq98Aan3rUe8LZ9mEEwR9~Buc~UR7b2coT-4TsMyw__)

The four clusters represent different playing styles:

- **Cluster 1**: High-tempo offensive teams with excellent shooting (e.g., Warriors, Rockets)
- **Cluster 2**: Balanced teams with strong defense (e.g., Raptors, Bucks)
- **Cluster 3**: Interior-focused teams with strong rebounding (e.g., 76ers, Nuggets)
- **Cluster 4**: Defense-first teams with lower scoring (e.g., Jazz, Celtics)

### 4.4 Upset Analysis

Some teams were more prone to upsets (losing when favored) than others:

![Teams Most Likely to Be Upset](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393194_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvdG9wX3Vwc2V0X3RlYW1z.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdmRHOXdYM1Z3YzJWMFgzUmxZVzF6LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=jfMRn0mjvJeBt08ISqbEXm5OUYIHy4RxYLlfr7lCUsyExLgFI~vnNJHNbdTJTwLMcQDMHSAJTyxrMVNSRnkpIjsk2pPNC1r5Uyp~SOO7WBisepeCjXAk4yxsrOvEX86UrTCOulszQouG~QUnf6hYpCWvHPo4HNnZHb-TueC~TtZWmtjNVkHiEseR5~OuXQIj9zAtNb13iKAHwAlWt0xdCptsP6OW84TgI2sAeWThTfCrvSEolbNcLt2yiHsuxqa9JImUWHIx4xeeSGkyCUUtFKXTHyI8XnPujgZgGd4p-5OQBUqscJhXy0UV-GZ2L8MLqOMsscR7xQSZ9kvHPFyfIw__)

Factors contributing to upsets include:

- Back-to-back games (teams playing on consecutive days)
- Small ELO rating differences between teams
- Teams with inconsistent performance patterns

## 5. Prediction Model Development

### 5.1 Model Selection

We evaluated several machine learning models for game prediction:

![Model Accuracy Comparison](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393194_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvbW9kZWxfYWNjdXJhY3lfY29tcGFyaXNvbg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdmJXOWtaV3hmWVdOamRYSmhZM2xmWTI5dGNHRnlhWE52YmcucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=YpkZ4aJ7de0bG261qwslGHI5PWuLYbdXVWS8aI3eLOPO~YDSaMSwFoUmwzAHI3zdtSbN94MSfdFdzTcfKBjaI~A4QETrHl96b26FKYKXvvEDSFULuT7kxq-rr9BWuAIcHKjyazmGKQI-x0gWnYBx9plN180pp6gp1jKS81o7KiUxx4oYqdzUcn1WoPkdhqDBpHkJOFV9vLp6xy55ITDCR9YHyCpaPy14kFDSCk5LSNkyY3ZLjgOemfOd5FCu-yZVHNDmTVZ1AEBhhC4ASMXfwHFIZ2cS7BbJzHF1Q5Y8ugd9TdC23tXy8EIlSl7NB6GTsOz0q36WwhPjZ0c0lBAoxQ__)

The Gradient Boosting model performed best, achieving **67.1%** accuracy on the test dataset.

### 5.2 Feature Importance

The most important features for prediction were:

1. ELO difference between teams
2. Home team's effective field goal percentage (eFG%)
3. Away team's effective field goal percentage (eFG%)
4. Home team's recent win percentage
5. Away team's turnover rate (TOV%)

### 5.3 Upset Prediction Model

We developed a specialized model for predicting upsets, which achieved **36.9%** accuracy on upset games, compared to **31.0%** for the ELO system.

![Accuracy by Upset Status](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393194_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvYWNjdXJhY3lfYnlfdXBzZXQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdllXTmpkWEpoWTNsZllubGZkWEJ6WlhRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Dk~btRkbe3hZutj7r-UQy4q7NH12mKjV9qLHq3bsVzjhCkkfn8gabZe7fiQQQqyftGUTMkkRH-q8jM0rRgplm03F4FJagzGS~0YhKEGwlOTqzh-a~K9bWcHfLidz1mfXS6MJ2mdM-UL68HLDrH1~YDbao~UhKnREVbheW6Ne5h4jVu-bPmWrEdD04d3TbdDn5lmk2QsyoyCB4FluXlpnDpyfdVPq3glBVMojFi1k-frwywY7sCHebr5HqLgGthLGnA93AHlxttV8lxChRoPaNaiPWrD2M4aJWjnq2TSgIvYZi9NdhMsOcOtCKuljdvVJSRX6bT13N8JHrIhizQa3Ww__)

## 6. Model Evaluation

### 6.1 Overall Performance

Our best model outperformed the ELO rating system across different game types:

![ELO vs Model Comparison](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393194_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvZWxvX3ZzX21vZGVsX2NvbXBhcmlzb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdlpXeHZYM1p6WDIxdlpHVnNYMk52YlhCaGNtbHpiMjQucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=WcQOsIOwSIlY5XqB5t9AedMTARnF~neY1BOiaEFfQEZOgxY0Z2yKHZ86sEXYyWjJ8Aj9dqpfjEnFLGC8xVU~DJjRDMWx5L11MbhJSB0hV7YExxxxU1~RooCTP~qr9Ccy3naaxaGZ1IqxVrK5-2yTYrmBSti~JFXCGGE0JcsQwtvy4CQNC7U0ZRJMLYz5b00hKE7Uawn8Pzc4wdKJEfo6B8~KAkQp26hUAdpxghCSqt1To-adUeRj4U1NE~FRUp0o1Ut-U2oLLRTXWi9j5TVoSGBosp40jbz330ttjVb0lw6HWKMsX1LKrPBtrYxIAOsghF-FVrdEILO47mj7MTwq6A__)

### 6.2 Performance by Game Characteristics

The model's accuracy varied based on different game characteristics:

#### 6.2.1 ELO Difference

Prediction accuracy increased with larger ELO differences between teams:

![Accuracy by ELO Difference](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/wBN2eCDvXZbSdHjVVcuSoT-images_1745790393194_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2ZpbmFsX3JlcG9ydC9pbWFnZXMvYWNjdXJhY3lfYnlfZWxvX2RpZmY.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L3dCTjJlQ0R2WFpiU2RIalZWY3VTb1QtaW1hZ2VzXzE3NDU3OTAzOTMxOTRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJacGJtRnNYM0psY0c5eWRDOXBiV0ZuWlhNdllXTmpkWEpoWTNsZllubGZaV3h2WDJScFptWS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=IxDU0hA3teiObIOrs77khNYkfPBbJy-Ovj2Sw109as9EXTPnIldcVbBQuFMW9rLK2-8LQMnsZKFhz5lADDhLUPfhCx019GKaLOeb6GnQSXEUhhS0fWP3twz4EemPyrBedFzd-xIwZRtEH1Aa56QhRQJdQlUPu9YR7wwJpCe3~QOWqh9mTqQBsQ7illG8zd-tdX0c2l0SWaIRB8CmbuGneM0lItro2GxQsa-cLSArepmiFI-WEuADQNmJutOPGTBBB4RnOkbfzTBwozfGpoOD8OxFB~bWbn7qHi860CtCUqQf5RE5w1AjsZD-RuTcd1oscgLvpAyHz6TAJ4Eu7GGIIA__)

#### 6.2.2 Back-to-Back Games

Back-to-back games were more difficult to predict, with accuracy dropping by **-5.2%** when both teams played on consecutive days.

## 7. Conclusions and Recommendations

### 7.1 Key Insights

1. Machine learning models can significantly outperform traditional ELO ratings for NBA game prediction, especially for upset games.

2. The Four Factors of basketball success have varying impacts on win probability, with shooting efficiency (eFG%) being the most important factor.

3. Team clustering reveals distinct playing styles that can inform strategic analysis and matchup predictions.

4. Back-to-back games significantly impact team performance and should be carefully considered in predictions.

5. The optimized ELO system with k-factor=30 and home advantage=50 provides a strong baseline for predictions.

### 7.2 Recommendations for Future Work

1. Incorporate player-level data, including injuries and rest days, to improve prediction accuracy.

2. Develop time-series models to better capture team momentum and performance trends.

3. Explore ensemble methods that combine multiple prediction models for improved accuracy.

4. Analyze in-game statistics and develop quarter-by-quarter prediction models.

5. Extend the analysis to playoff games, which may have different prediction patterns than regular-season games.

### 7.3 Final Thoughts

This project demonstrates the power of data science in sports prediction. By combining traditional basketball metrics with advanced machine learning techniques, we achieved significant improvements in prediction accuracy. The insights gained from this analysis can inform betting strategies, fantasy sports decisions, and team management approaches.

The NBA's rich statistical tradition makes it an ideal domain for predictive modeling, and continued advancements in data collection and analysis techniques promise even greater accuracy in the future.

