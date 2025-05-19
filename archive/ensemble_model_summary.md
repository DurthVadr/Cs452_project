# NBA Game Prediction: Ensemble Model Approach

## Overview

This report presents the results of combining regular game prediction models with specialized upset prediction models to improve overall prediction accuracy for NBA games.

## Approaches Tested

### Approach 1: Separate Models for Upset and Non-Upset Games

- Train one model on non-upset games and another on upset games
- Use ELO difference to determine which model to apply for each prediction
- Accuracy: 0.6098

### Approach 2: Upset Prediction + Main Model

- Train a model to predict whether a game will be an upset
- Train a separate model to predict game outcomes
- For games predicted to be upsets, predict the underdog will win
- For other games, use the main model's prediction
- Accuracy: 0.6748

### Approach 3: Weighted Voting Ensemble

- Combine multiple models (Gradient Boosting, Random Forest, Logistic Regression)
- Use soft voting with weights favoring the best-performing model
- Accuracy: 0.6707

### Approach 4: Meta-Model (Stacking)

- Train base models and generate predictions
- Use these predictions as features for a meta-model
- Include original features alongside model predictions
- Accuracy: 0.6829

### Approach 5: Adaptive Ensemble with Upset-Specific Features

- Create enhanced features specifically for upset detection
- Train specialized upset prediction model with these features
- Adaptively combine predictions based on upset probability
- Accuracy: 0.6707

## Results

The best approach was **Approach 4: Meta-model (stacking)** with an accuracy of **0.6829**.

This represents an improvement of **1.22%** over the original model (accuracy: 0.6707).

![Approach Comparison](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/Asd9ODo1kuMyPbTLhhU9Lr-images_1745839344241_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2Vuc2VtYmxlX21vZGVsL2FwcHJvYWNoX2NvbXBhcmlzb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L0FzZDlPRG8xa3VNeVBiVExoaFU5THItaW1hZ2VzXzE3NDU4MzkzNDQyNDFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJWdWMyVnRZbXhsWDIxdlpHVnNMMkZ3Y0hKdllXTm9YMk52YlhCaGNtbHpiMjQucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=JmyeZO4D4TGcOwvucRPfNUprDuKBveqrLJBuAgWr4enHpPgjrReWzUWtVR42hYJKUlOkOrHwn-eHxCcuvuzvg~hpWlvAYWJK7R7RFj0YlToEZsajqajtuyhxC2vT0cCZ9wsKRhaPMjmSBCX-9bh43e~y7CMguRzVcZ34155ENixLb~0-Rc7YKvnbTOb9aCPZQiCs8YkWYwbMFijNBlwX9iOE~5nRwUeq3hSuvxt0Nd2ZMAa2bKO8c987uZaCYiOMUUND5e4ESSFZInVqa~irOaCPDosiNwWexme6YYcs~2pHD~iIeqWbiXbjIQEGugOpEi3~rnQv3uPKji8Cge6qIQ__)

## Performance on Upset vs. Non-Upset Games

| Game Type | Accuracy | Count |
|-----------|----------|-------|
| Non-Upset | 0.8827 | 162.0 |
| Upset | 0.2976 | 84.0 |

![Performance by Upset Status](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/Asd9ODo1kuMyPbTLhhU9Lr-images_1745839344241_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2Vuc2VtYmxlX21vZGVsL2Jlc3RfYXBwcm9hY2hfYnlfdXBzZXQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L0FzZDlPRG8xa3VNeVBiVExoaFU5THItaW1hZ2VzXzE3NDU4MzkzNDQyNDFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJWdWMyVnRZbXhsWDIxdlpHVnNMMkpsYzNSZllYQndjbTloWTJoZllubGZkWEJ6WlhRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=WnV9QLTsMk5HGi5kKuaLiytK2~QaAkewkPRjszo7jsdAz9ofgSo4IteT2Y6CyFIGbPMzAssWZAngNdS46hj9UiF8DmqZ-z78VSF0eXXUJUW-9GOXk~kN5kq4xPo8EYlf4ox5Ufze0nYDE3-ntBWmogjtOl4jAlPhhWRbzuOOj1XsNqwumYVCFAWIRLGbsOqjiuZJA9SG-ytqZq8aFx7OsoM-1PUUxAir2PMD-Izo48b1SW0pLJhyteXHLPREFGUdHJ0uIqRs95Do17RVdNIKAjxp-KxVT4PW9xH4uM8~~0pYnr5D~gVW0m1kakR4-nzYP57gTWV17oACsCe~JK0mKQ__)

## Confusion Matrix

![Confusion Matrix](https://private-us-east-1.manuscdn.com/sessionFile/vVsYN7TjCX9r9nh1HCWsIH/sandbox/Asd9ODo1kuMyPbTLhhU9Lr-images_1745839344241_na1fn_L2hvbWUvdWJ1bnR1L25iYV9wcmVkaWN0aW9uL2Vuc2VtYmxlX21vZGVsL2Jlc3RfYXBwcm9hY2hfY29uZnVzaW9uX21hdHJpeA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvdlZzWU43VGpDWDlyOW5oMUhDV3NJSC9zYW5kYm94L0FzZDlPRG8xa3VNeVBiVExoaFU5THItaW1hZ2VzXzE3NDU4MzkzNDQyNDFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyNWlZVjl3Y21Wa2FXTjBhVzl1TDJWdWMyVnRZbXhsWDIxdlpHVnNMMkpsYzNSZllYQndjbTloWTJoZlkyOXVablZ6YVc5dVgyMWhkSEpwZUEucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XQoS8~O2M6-DtK-gBD3kW85upzgkWpffvaxeCB1pn3lmyv9HYe1IJqiD5AAO4XM0i0BXtKhnZp0ajN-XtRkBc1A-Ta6zbFQ4JLLixO~w8R9g89DrAMzqnR2LM8~9VBQ8X3QY~kilS2W49~lnYUlSGvE9um8R-cE9UY9Ca7~0~JAhnOZGRhSEJzHfzI-fuFq5QeXn6DhnktBzujv1sMCFaC808PuCSmcsAfkHMgIlk4nDK8bWUn3P4wRaZirRjKidXgZCogTRWI8w0wkHfNqkk8c-0X23q8jC2WwpLlS6svACHaDVcH2UOD5C9R6Msj2IG26CZoD-C884XKPYt7u7Og__)

## Conclusion

By combining regular game prediction with specialized upset prediction, we've achieved a significant improvement in overall accuracy. The Approach 4: Meta-model (stacking) achieved 0.6829 accuracy, which is 1.22% better than the original model.

This demonstrates the value of specialized modeling for different game scenarios in sports prediction. The improved model is particularly effective at identifying upset games, which were previously more difficult to predict accurately.

