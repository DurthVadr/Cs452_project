"""
Feature engineering for NBA game prediction.
"""
import pandas as pd
import numpy as np

from nba_prediction.config import config
from nba_prediction.utils.logging_config import get_logger

logger = get_logger(__name__)

def calculate_recent_performance(games_df, n_games=10):
    """
    Calculate team's recent performance (win percentage in last N games).
    
    Args:
        games_df: DataFrame with game data
        n_games: Number of games to consider for recent performance
        
    Returns:
        DataFrame with added recent performance columns
    """
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for recent performance
    df['away_last_n_win_pct'] = np.nan
    df['home_last_n_win_pct'] = np.nan

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # For each team, calculate win percentage in last N games
    for team in teams:
        # Get games where team played (either home or away)
        team_games = df[(df['away_team'] == team) | (df['home_team'] == team)].sort_values('date')

        # For each game, calculate win percentage in last N games
        for idx, game in team_games.iterrows():
            # Find the index of current game in the sorted team games
            game_idx = team_games.index.get_loc(idx)

            # If we have at least N previous games
            if game_idx >= n_games:
                # Get last N games
                last_n_games = team_games.iloc[game_idx-n_games:game_idx]

                # Calculate win percentage
                wins = 0
                for _, last_game in last_n_games.iterrows():
                    if (last_game['away_team'] == team and last_game['result'] == 0) or \
                       (last_game['home_team'] == team and last_game['result'] == 1):
                        wins += 1

                win_pct = wins / n_games

                # Update the appropriate column in the original dataframe
                if game['away_team'] == team:
                    df.at[idx, 'away_last_n_win_pct'] = win_pct
                else:
                    df.at[idx, 'home_last_n_win_pct'] = win_pct

    return df

def calculate_streak(games_df):
    """
    Calculate team's streak before the current game (positive for winning streak, negative for losing streak).
    
    Args:
        games_df: DataFrame with game data
        
    Returns:
        DataFrame with added streak columns
    """
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for streak
    df['away_streak'] = 0
    df['home_streak'] = 0

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # For each team, calculate streak before each game
    for team in teams:
        # Get games where team played (either home or away)
        team_games = df[(df['away_team'] == team) | (df['home_team'] == team)].sort_values('date')

        # Initialize streak
        current_streak = 0

        # Create a dictionary to store previous streak for each game
        prev_streaks = {}

        # First pass: calculate streaks after each game
        for idx, game in team_games.iterrows():
            # Store the streak before this game
            prev_streaks[idx] = current_streak

            # Update streak based on game result
            if (game['away_team'] == team and game['result'] == 0) or \
               (game['home_team'] == team and game['result'] == 1):
                # Team won
                if current_streak >= 0:
                    # Continue winning streak
                    current_streak += 1
                else:
                    # Start new winning streak
                    current_streak = 1
            else:
                # Team lost
                if current_streak <= 0:
                    # Continue losing streak
                    current_streak -= 1
                else:
                    # Start new losing streak
                    current_streak = -1

        # Second pass: update dataframe with streak before each game
        for idx, game in team_games.iterrows():
            # Update the appropriate column in the original dataframe
            if game['away_team'] == team:
                df.at[idx, 'away_streak'] = prev_streaks[idx]
            else:
                df.at[idx, 'home_streak'] = prev_streaks[idx]

    return df

def calculate_weighted_performance(games_df, windows=[3, 5, 10], weights=[0.5, 0.3, 0.2]):
    """
    Calculate weighted recent performance (more weight to recent games).
    
    Args:
        games_df: DataFrame with game data
        windows: List of window sizes for recent games
        weights: List of weights for each window
        
    Returns:
        DataFrame with added weighted performance columns
    """
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for weighted performance
    df['away_weighted_win_pct'] = np.nan
    df['home_weighted_win_pct'] = np.nan

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # For each team, calculate weighted win percentage
    for team in teams:
        # Get games where team played (either home or away)
        team_games = df[(df['away_team'] == team) | (df['home_team'] == team)].sort_values('date')

        # For each game, calculate weighted win percentage based on previous games
        for idx, game in team_games.iterrows():
            # Find the index of current game in the sorted team games
            game_idx = team_games.index.get_loc(idx)

            # If we have enough previous games for the largest window
            if game_idx >= max(windows):
                weighted_win_pct = 0

                # Calculate win percentage for each window and apply weights
                for i, window in enumerate(windows):
                    # Get games BEFORE this game for this window
                    window_games = team_games.iloc[game_idx-window:game_idx]

                    # Count wins
                    wins = 0
                    for _, window_game in window_games.iterrows():
                        if (window_game['away_team'] == team and window_game['result'] == 0) or \
                           (window_game['home_team'] == team and window_game['result'] == 1):
                            wins += 1

                    # Calculate win percentage for this window
                    window_win_pct = wins / window

                    # Add weighted contribution
                    weighted_win_pct += window_win_pct * weights[i]

                # Update the appropriate column in the original dataframe
                if game['away_team'] == team:
                    df.at[idx, 'away_weighted_win_pct'] = weighted_win_pct
                else:
                    df.at[idx, 'home_weighted_win_pct'] = weighted_win_pct

    return df

def identify_back_to_back(games_df):
    """
    Identify back-to-back games for teams.
    
    Args:
        games_df: DataFrame with game data
        
    Returns:
        DataFrame with added back-to-back columns
    """
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for back-to-back games
    df['away_back_to_back'] = 0
    df['home_back_to_back'] = 0

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # For each team, identify back-to-back games
    for team in teams:
        # Get games where team played (either home or away)
        team_games = df[(df['away_team'] == team) | (df['home_team'] == team)].sort_values('date')

        # For each game except the first one, check if it's a back-to-back
        for i in range(1, len(team_games)):
            current_game = team_games.iloc[i]
            previous_game = team_games.iloc[i-1]

            # Calculate days between games
            days_between = (current_game['date'] - previous_game['date']).days

            # If games are on consecutive days, mark as back-to-back
            if days_between <= 1:
                # Update the appropriate column in the original dataframe
                if current_game['away_team'] == team:
                    df.at[current_game.name, 'away_back_to_back'] = 1
                else:
                    df.at[current_game.name, 'home_back_to_back'] = 1

    return df

def calculate_head_to_head(games_df):
    """
    Calculate head-to-head record between teams.
    
    Args:
        games_df: DataFrame with game data
        
    Returns:
        DataFrame with added head-to-head columns
    """
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for head-to-head record
    df['away_vs_home_wins'] = 0
    df['away_vs_home_losses'] = 0
    df['away_vs_home_win_pct'] = 0.5  # Default to 0.5 when no history

    # For each game, calculate head-to-head record before that game
    for idx, game in df.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']
        game_date = game['date']

        # Get previous games between these teams
        previous_matchups = df[
            ((df['away_team'] == away_team) & (df['home_team'] == home_team) |
             (df['away_team'] == home_team) & (df['home_team'] == away_team)) &
            (df['date'] < game_date)
        ]

        if len(previous_matchups) > 0:
            # Count wins for away team
            away_wins = len(previous_matchups[
                ((previous_matchups['away_team'] == away_team) & (previous_matchups['result'] == 0)) |
                ((previous_matchups['home_team'] == away_team) & (previous_matchups['result'] == 1))
            ])

            # Update columns
            df.at[idx, 'away_vs_home_wins'] = away_wins
            df.at[idx, 'away_vs_home_losses'] = len(previous_matchups) - away_wins
            df.at[idx, 'away_vs_home_win_pct'] = away_wins / len(previous_matchups)

    return df

def create_differential_features(df):
    """
    Create differential features between home and away teams.
    
    Args:
        df: DataFrame with team statistics
        
    Returns:
        DataFrame with added differential features
    """
    df = df.copy()
    
    df['elo_diff'] = df['home_elo_i'] - df['away_elo_i']
    df['eFGp_diff'] = df['h_eFGp'] - df['a_eFGp']
    df['FTr_diff'] = df['h_FTr'] - df['a_FTr']
    df['ORBp_diff'] = df['h_ORBp'] - df['a_ORBp']
    df['TOVp_diff'] = df['h_TOVp'] - df['a_TOVp']
    
    return df

def create_interaction_features(df):
    """
    Create interaction features for Four Factors.
    
    Args:
        df: DataFrame with team statistics
        
    Returns:
        DataFrame with added interaction features
    """
    df = df.copy()
    
    df['h_eFGp_x_TOVp'] = df['h_eFGp'] * (1 - df['h_TOVp'])
    df['a_eFGp_x_TOVp'] = df['a_eFGp'] * (1 - df['a_TOVp'])
    df['h_eFGp_x_ORBp'] = df['h_eFGp'] * df['h_ORBp']
    df['a_eFGp_x_ORBp'] = df['a_eFGp'] * df['a_ORBp']
    
    return df

def create_upset_features(df):
    """
    Create features specific to upset prediction.
    
    Args:
        df: DataFrame with team statistics and favorite column
        
    Returns:
        DataFrame with added upset-specific features
    """
    df = df.copy()
    
    df['elo_diff_abs'] = abs(df['home_elo_i'] - df['away_elo_i'])
    
    # Create upset indicator
    df['favorite'] = np.where(df['home_elo_i'] > df['away_elo_i'], 1, 0)
    df['upset'] = np.where(df['favorite'] != df['result'], 1, 0)
    
    # Create favorite/underdog back-to-back features
    df['favorite_back_to_back'] = np.where(
        df['favorite'] == 1,
        df['home_back_to_back'],
        df['away_back_to_back']
    )
    df['underdog_back_to_back'] = np.where(
        df['favorite'] == 0,
        df['home_back_to_back'],
        df['away_back_to_back']
    )
    
    return df

def engineer_all_features(season_data):
    """
    Apply all feature engineering steps to create full feature set.
    
    Args:
        season_data: DataFrame with season's game data
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Engineering features for game data")
    
    # Convert date to datetime
    season_data['date'] = pd.to_datetime(season_data['date'])
    
    # Sort by date
    df = season_data.sort_values('date')
    
    # Calculate recent performance
    logger.info("Calculating recent performance")
    df = calculate_recent_performance(df, n_games=10)
    
    # Calculate streak
    logger.info("Calculating team streaks")
    df = calculate_streak(df)
    
    # Calculate weighted performance
    logger.info("Calculating weighted performance")
    df = calculate_weighted_performance(df)
    
    # Identify back-to-back games
    logger.info("Identifying back-to-back games")
    df = identify_back_to_back(df)
    
    # Calculate head-to-head record
    logger.info("Calculating head-to-head records")
    df = calculate_head_to_head(df)
    
    # Load and merge team factor data
    logger.info("Loading team factor data")
    from nba_prediction.data.loader import load_raw_dataset
    team_factor_10 = load_raw_dataset('team_factor_10')
    team_factor_10_season = team_factor_10[team_factor_10['season'] == df['season'].iloc[0]]
    
    # Essential Four Factors columns
    factor_cols = ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
    
    # Merge with team factor data
    logger.info("Merging team factor data")
    df = pd.merge(
        df,
        team_factor_10_season[['game_id'] + factor_cols],
        on='game_id',
        how='left'
    )
    
    # Fill in missing factor values with defaults (league averages)
    default_values = {
        'a_eFGp': 0.52, 'h_eFGp': 0.52,  # League average eFG%
        'a_TOVp': 0.14, 'h_TOVp': 0.14,  # League average TOV%
        'a_ORBp': 0.22, 'h_ORBp': 0.22,  # League average ORB%
        'a_FTr': 0.25, 'h_FTr': 0.25     # League average FT rate
    }
    
    for col in factor_cols:
        df[col] = df[col].fillna(default_values.get(col, 0))
    
    # Calculate ELO difference
    logger.info("Calculating ELO difference")
    df['elo_diff'] = df['home_elo_i'] - df['away_elo_i']
    
    # Create differential features
    logger.info("Creating differential features")
    df = create_differential_features(df)
    
    # Create interaction features
    logger.info("Creating interaction features")
    df = create_interaction_features(df)
    
    # Create upset features
    logger.info("Creating upset features")
    df = create_upset_features(df)
    
    # Fill missing values
    logger.info("Filling missing values")
    df = df.fillna({
        'away_last_n_win_pct': 0.5,
        'home_last_n_win_pct': 0.5,
        'away_back_to_back': 0,
        'home_back_to_back': 0,
        'away_vs_home_win_pct': 0.5,
        'away_streak': 0,
        'home_streak': 0,
        'away_weighted_win_pct': 0.5,
        'home_weighted_win_pct': 0.5
    })
    
    return df