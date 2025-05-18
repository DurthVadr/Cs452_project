"""
ELO rating system implementation for NBA game prediction.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class EloSystem:
    """
    ELO rating system for predicting NBA game outcomes.
    """
    k_factor: float = 20.0
    home_advantage: float = 100.0
    initial_rating: float = 1500.0
    team_ratings: Dict[str, float] = field(default_factory=dict)
    
    def get_rating(self, team: str) -> float:
        """
        Get the current rating for a team.
        
        Args:
            team: Team name
            
        Returns:
            Team's current ELO rating
        """
        if team not in self.team_ratings:
            self.team_ratings[team] = self.initial_rating
        return self.team_ratings[team]
    
    def update_rating(self, team: str, change: float) -> None:
        """
        Update a team's rating.
        
        Args:
            team: Team name
            change: Change in rating
        """
        self.team_ratings[team] = self.get_rating(team) + change
        
    def expected_result(self, team_a: str, team_b: str, team_a_home: bool = False) -> float:
        """
        Calculate the expected result for a game.
        
        Args:
            team_a: First team
            team_b: Second team
            team_a_home: Whether team_a is the home team
            
        Returns:
            Probability of team_a winning
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Apply home court advantage
        if team_a_home:
            rating_a += self.home_advantage
        else:
            rating_b += self.home_advantage
            
        # Calculate expected result using ELO formula
        exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return exp_a
        
    def update_ratings(self, team_a: str, team_b: str, result: float, team_a_home: bool = False) -> float:
        """
        Update ratings based on game result.
        
        Args:
            team_a: First team
            team_b: Second team
            result: Actual result (1 for team_a win, 0 for team_b win)
            team_a_home: Whether team_a is the home team
            
        Returns:
            Change in rating for team_a
        """
        expected_a = self.expected_result(team_a, team_b, team_a_home)
        
        # Update ratings
        rating_change = self.k_factor * (result - expected_a)
        self.update_rating(team_a, rating_change)
        self.update_rating(team_b, -rating_change)
        
        return rating_change
        
    def predict(self, team_a: str, team_b: str, team_a_home: bool = False) -> int:
        """
        Predict the outcome of a game.
        
        Args:
            team_a: First team
            team_b: Second team
            team_a_home: Whether team_a is the home team
            
        Returns:
            0 if team_a is predicted to win, 1 if team_b is predicted to win
        """
        expected_a = self.expected_result(team_a, team_b, team_a_home)
        return 0 if expected_a >= 0.5 else 1
    
    def reset_ratings(self) -> None:
        """Reset all team ratings to the initial value."""
        self.team_ratings = {}


def train_elo_system(games_df: pd.DataFrame, k_factor: float, home_advantage: float) -> Tuple[EloSystem, float]:
    """
    Train an ELO system on the provided games.
    
    Args:
        games_df: DataFrame with game data
        k_factor: K-factor for the ELO system
        home_advantage: Home court advantage
        
    Returns:
        Tuple of (trained ELO system, accuracy)
    """
    # Initialize ELO system
    elo_system = EloSystem(k_factor=k_factor, home_advantage=home_advantage)
    
    # Reset ratings
    elo_system.reset_ratings()
    
    # Make predictions and update ratings
    correct = 0
    
    for _, game in games_df.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']
        
        # Predict winner
        prediction = elo_system.predict(away_team, home_team, team_a_home=False)
        
        # Update accuracy
        if prediction == game['result']:
            correct += 1
            
        # Update ratings based on actual result
        actual_result = 1 if game['result'] == 0 else 0  # 0 for away win, 1 for home win
        elo_system.update_ratings(away_team, home_team, actual_result, team_a_home=False)
    
    # Calculate accuracy
    accuracy = correct / len(games_df)
    
    return elo_system, accuracy


def evaluate_elo_system(elo_system: EloSystem, games_df: pd.DataFrame) -> Tuple[float, List[int]]:
    """
    Evaluate an ELO system on the provided games.
    
    Args:
        elo_system: Trained ELO system
        games_df: DataFrame with game data
        
    Returns:
        Tuple of (accuracy, list of predictions)
    """
    predictions = []
    correct = 0
    
    for _, game in games_df.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']
        
        # Predict winner
        prediction = elo_system.predict(away_team, home_team, team_a_home=False)
        predictions.append(prediction)
        
        # Update accuracy
        if prediction == game['result']:
            correct += 1
            
        # Update ratings based on actual result
        actual_result = 1 if game['result'] == 0 else 0  # 0 for away win, 1 for home win
        elo_system.update_ratings(away_team, home_team, actual_result, team_a_home=False)
    
    # Calculate accuracy
    accuracy = correct / len(games_df)
    
    return accuracy, predictions


def find_optimal_elo_parameters(train_df: pd.DataFrame) -> Tuple[float, float, EloSystem, float]:
    """
    Find optimal ELO parameters using grid search.
    
    Args:
        train_df: Training data
        
    Returns:
        Tuple of (best k_factor, best home_advantage, best ELO system, best accuracy)
    """
    best_k = 0
    best_ha = 0
    best_accuracy = 0
    best_elo = None
    
    for k in config.ELO_PARAMS["k_factor_range"]:
        for ha in config.ELO_PARAMS["home_advantage_range"]:
            elo_system, accuracy = train_elo_system(train_df, k, ha)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_ha = ha
                best_elo = elo_system
                
    logger.info(f"Optimal ELO parameters: k_factor={best_k}, home_advantage={best_ha}, accuracy={best_accuracy:.4f}")
    return best_k, best_ha, best_elo, best_accuracy