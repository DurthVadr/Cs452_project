"""
Feature configuration for the NBA prediction project.

This file defines the feature sets used for model training and evaluation.
Features are organized into meaningful categories based on basketball analytics.
"""

# Base features - core predictors that are available for all games
BASE_FEATURES = [
    'home_elo_i', 'away_elo_i', 'elo_diff',
    'home_back_to_back', 'away_back_to_back'
]

# Four Factors features - based on Dean Oliver's "Four Factors of Basketball Success"
FOUR_FACTORS_FEATURES = [
    'h_eFGp', 'a_eFGp', 'eFGp_diff',  # Shooting efficiency
    'h_TOVp', 'a_TOVp', 'TOVp_diff',  # Turnovers
    'h_ORBp', 'a_ORBp', 'ORBp_diff',  # Rebounding
    'h_FTr', 'a_FTr', 'FTr_diff'      # Free throws
]

# Advanced metrics features - additional team comparison metrics
ADVANCED_FEATURES = [
    'point_diff'
]

# Recent form features - team performance in recent games
RECENT_FORM_FEATURES = [
    'home_last_n_win_pct', 'away_last_n_win_pct',
    'home_streak', 'away_streak',
    'home_weighted_win_pct', 'away_weighted_win_pct'
]

# Interaction features
INTERACTION_FEATURES = [
    'h_eFGp_x_TOVp', 'a_eFGp_x_TOVp',
    'h_eFGp_x_ORBp', 'a_eFGp_x_ORBp'
]

# Matchup features
MATCHUP_FEATURES = [
    'away_vs_home_win_pct'
]

# Feature sets for different models
FEATURE_SETS = {
    'base': BASE_FEATURES,
    'four_factors': BASE_FEATURES + FOUR_FACTORS_FEATURES,
    'advanced': BASE_FEATURES + FOUR_FACTORS_FEATURES + ADVANCED_FEATURES,
    'recent_form': BASE_FEATURES + FOUR_FACTORS_FEATURES + ADVANCED_FEATURES + RECENT_FORM_FEATURES,
    'interaction': BASE_FEATURES + FOUR_FACTORS_FEATURES + ADVANCED_FEATURES + RECENT_FORM_FEATURES + INTERACTION_FEATURES,
    'matchup': BASE_FEATURES + FOUR_FACTORS_FEATURES + ADVANCED_FEATURES + RECENT_FORM_FEATURES + MATCHUP_FEATURES,
    'full': BASE_FEATURES + FOUR_FACTORS_FEATURES + ADVANCED_FEATURES + RECENT_FORM_FEATURES + INTERACTION_FEATURES + MATCHUP_FEATURES
}

# Default feature set to use if not specified
DEFAULT_FEATURE_SET = 'four_factors'

# Features to use for upset prediction specifically
UPSET_FEATURES = BASE_FEATURES + FOUR_FACTORS_FEATURES + [
    'home_last_n_win_pct', 'away_last_n_win_pct',
    'away_vs_home_win_pct',
    'h_eFGp_x_TOVp', 'a_eFGp_x_TOVp'
]

# Differential features - all features that represent differences between teams
DIFFERENTIAL_FEATURES = [feat for feat in BASE_FEATURES + FOUR_FACTORS_FEATURES +
                         ADVANCED_FEATURES + RECENT_FORM_FEATURES +
                         INTERACTION_FEATURES + MATCHUP_FEATURES if 'diff' in feat]

# Function to get feature list by name
def get_feature_set(name='default'):
    """Get a feature set by name."""
    if name == 'default':
        return FEATURE_SETS[DEFAULT_FEATURE_SET]
    elif name == 'upset':
        return UPSET_FEATURES
    elif name == 'differential':
        return DIFFERENTIAL_FEATURES
    elif name in FEATURE_SETS:
        return FEATURE_SETS[name]
    else:
        raise ValueError(f"Feature set '{name}' not found. Available sets: {list(FEATURE_SETS.keys())}")
