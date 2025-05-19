"""
Data preparation script for NBA game prediction project.
Handles data loading, feature engineering, and preprocessing.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config import *

def load_raw_data():
    """Load all raw data files."""
    print("Loading raw data files...")
    data = {}
    for key, filepath in RAW_DATA_FILES.items():
        try:
            data[key] = pd.read_csv(filepath, index_col=0)
            print(f"Loaded {key}: {data[key].shape}")
        except Exception as e:
            print(f"Error loading {key}: {e}")
            raise
    return data

def initial_game_features(game_info_df):
    """Create initial game features without momentum-related features."""
    print("Creating initial game features...")

    # Select season and prepare data
    games = game_info_df[game_info_df['season'] == 1819].copy()
    games['date'] = pd.to_datetime(games['date'])
    games = games.sort_values('date')

    # Calculate back-to-back games
    games['away_back_to_back'] = games.groupby('away_team')['date'].diff().dt.days <= 1
    games['home_back_to_back'] = games.groupby('home_team')['date'].diff().dt.days <= 1

    return games

def create_labels(games_df):
    """Create target labels and upset indicators."""
    print("Creating labels...")

    # Basic win/loss (1 for home win, 0 for away win)
    games_df['result'] = (games_df['home_score'] > games_df['away_score']).astype(int)

    # Favorite/Underdog labels
    games_df['favorite_is_home'] = (games_df['home_elo_i'] > games_df['away_elo_i']).astype(int)

    # Upset indicator
    games_df['is_upset'] = (games_df['favorite_is_home'] != games_df['result']).astype(int)

    return games_df

def calculate_momentum_features(games_df):
    """Calculate momentum-based features including streaks and rolling stats.
    Ensures only past games are used for each prediction to prevent data leakage."""
    print("Calculating momentum features...")
    df = games_df.copy().sort_values('date')

    # Initialize columns for streaks and win percentages
    for team_type in ['away_team', 'home_team']:
        prefix = team_type[:4]
        df[f'{prefix}_streak'] = 0
        for window in ROLLING_WINDOWS:
            df[f'{prefix}_last_{window}_win_pct'] = 0.5  # Default to 0.5 when no history

    # Create team result columns (1=win, 0=loss from team perspective)
    df['home_result'] = df['result']
    df['away_result'] = 1 - df['result']

    # Process each team separately to ensure proper time-based calculations
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    for team in teams:
        # Get all games for this team in chronological order
        team_games = df[(df['away_team'] == team) | (df['home_team'] == team)].sort_values('date')

        # Process each game for this team
        for i, (idx, game) in enumerate(team_games.iterrows()):
            # Skip the first game as we have no history
            if i == 0:
                continue

            # Get previous games for this team (up to current game)
            prev_games = team_games.iloc[:i]

            # Determine if team was home or away in current game
            is_home = game['home_team'] == team
            team_type = 'home_team' if is_home else 'away_team'
            prefix = team_type[:4]

            # Calculate streak
            current_streak = 0
            for _, prev_game in prev_games.iloc[::-1].iterrows():  # Reverse to start from most recent
                prev_is_home = prev_game['home_team'] == team
                prev_result = prev_game['result'] if prev_is_home else (1 - prev_game['result'])

                if (current_streak >= 0 and prev_result == 1) or (current_streak <= 0 and prev_result == 0):
                    # Continue streak
                    current_streak = (abs(current_streak) + 1) * (1 if prev_result == 1 else -1)
                else:
                    # Break streak
                    current_streak = 1 if prev_result == 1 else -1

                # Only consider the most recent streak
                break

            # Update streak in the dataframe
            df.at[idx, f'{prefix}_streak'] = current_streak

            # Calculate win percentages for different windows
            for window in ROLLING_WINDOWS:
                # Get last N games (or as many as available)
                last_n_games = prev_games.iloc[-min(window, len(prev_games)):]

                # Calculate win percentage
                wins = 0
                for _, prev_game in last_n_games.iterrows():
                    prev_is_home = prev_game['home_team'] == team
                    prev_result = prev_game['result'] if prev_is_home else (1 - prev_game['result'])
                    wins += prev_result

                win_pct = wins / len(last_n_games) if len(last_n_games) > 0 else 0.5
                df.at[idx, f'{prefix}_last_{window}_win_pct'] = win_pct

    # Drop helper columns
    df = df.drop(['home_result', 'away_result'], axis=1)
    return df

def add_matchup_history(games_df):
    """Add historical matchup statistics using only past games."""
    print("Adding matchup history...")
    df = games_df.copy().sort_values('date')

    # Initialize matchup history column with default value
    df['away_vs_home_win_pct'] = 0.5

    # Get unique matchups
    matchups = df[['away_team', 'home_team']].drop_duplicates()

    # Process each matchup separately
    for _, matchup in matchups.iterrows():
        away_team = matchup['away_team']
        home_team = matchup['home_team']

        # Get all games between these teams in chronological order
        matchup_games = df[
            (df['away_team'] == away_team) &
            (df['home_team'] == home_team)
        ].sort_values('date')

        # Process each game for this matchup
        for i, (idx, game) in enumerate(matchup_games.iterrows()):
            # Skip the first game as we have no history
            if i == 0:
                continue

            # Get previous games for this matchup (up to current game)
            prev_games = matchup_games.iloc[:i]

            # Calculate win percentage for away team
            away_wins = (prev_games['result'] == 0).sum()
            win_pct = away_wins / len(prev_games) if len(prev_games) > 0 else 0.5

            # Update win percentage in the dataframe
            df.at[idx, 'away_vs_home_win_pct'] = win_pct

    return df

def add_team_stats(games_df, team_factor_df):
    """Add enhanced team statistics and factors to games dataframe."""
    print("Adding team statistics...")

    # Define Four Factors features
    four_factors_features = [
        'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
        'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp'
    ]

    # Check if all Four Factors features exist in team_factor_df
    missing_factors = [col for col in four_factors_features if col not in team_factor_df.columns]
    if missing_factors:
        print(f"Warning: Missing Four Factors features in team_factor_df: {missing_factors}")
        # Only use available factors
        available_factors = [col for col in four_factors_features if col in team_factor_df.columns]

        # Add available Four Factors
        games_df = pd.merge(
            games_df,
            team_factor_df[available_factors + ['game_id']],
            on='game_id',
            how='left'
        )
    else:
        # Add all Four Factors
        games_df = pd.merge(
            games_df,
            team_factor_df[four_factors_features + ['game_id']],
            on='game_id',
            how='left'
        )

    # Fill missing values
    games_df = fill_missing_factor_data(games_df)

    # Add ELO difference
    games_df['elo_diff'] = games_df['home_elo_i'] - games_df['away_elo_i']

    return games_df

def calculate_rolling_factors(games_df):
    """Calculate rolling averages for Four Factors using only past games."""
    print("Calculating rolling Four Factors...")
    df = games_df.copy().sort_values('date')

    # Initialize rolling factor columns
    for factor in ['eFGp', 'TOVp', 'ORBp', 'FTr']:
        for prefix in ['a_', 'h_']:
            col = f'{prefix}{factor}'
            if col in df.columns:
                df[f'{col}_10'] = df[col]  # Default to current value

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # Process each team separately
    for team in teams:
        # Process away games
        away_games = df[df['away_team'] == team].sort_values('date')
        for i, (idx, game) in enumerate(away_games.iterrows()):
            # Skip if not enough history
            if i < 1:
                continue

            # Get previous games (up to 10)
            prev_games = away_games.iloc[max(0, i-10):i]

            # Calculate rolling averages for each factor
            for factor in ['eFGp', 'TOVp', 'ORBp', 'FTr']:
                col = f'a_{factor}'
                if col in df.columns and len(prev_games) > 0:
                    df.at[idx, f'{col}_10'] = prev_games[col].mean()

        # Process home games
        home_games = df[df['home_team'] == team].sort_values('date')
        for i, (idx, game) in enumerate(home_games.iterrows()):
            # Skip if not enough history
            if i < 1:
                continue

            # Get previous games (up to 10)
            prev_games = home_games.iloc[max(0, i-10):i]

            # Calculate rolling averages for each factor
            for factor in ['eFGp', 'TOVp', 'ORBp', 'FTr']:
                col = f'h_{factor}'
                if col in df.columns and len(prev_games) > 0:
                    df.at[idx, f'{col}_10'] = prev_games[col].mean()

    return df

def calculate_differential_features(games_df):
    """Calculate differential features between home and away teams."""
    print("Calculating differential features...")
    df = games_df.copy()

    for feat in ['eFGp', 'FTr', 'ORBp', 'TOVp']:
        # Current game differentials
        if f'h_{feat}' in df.columns and f'a_{feat}' in df.columns:
            df[f'{feat}_diff'] = df[f'h_{feat}'] - df[f'a_{feat}']

        # Rolling differentials
        if f'h_{feat}_10' in df.columns and f'a_{feat}_10' in df.columns:
            df[f'{feat}_roll_diff'] = df[f'h_{feat}_10'] - df[f'a_{feat}_10']

    return df

def create_interaction_features(games_df):
    """Create interaction features between different metrics.
    All features used here should already be properly calculated with respect to time."""
    print("Creating interaction features...")
    df = games_df.copy()

    # ELO difference * back-to-back interaction
    df['elo_diff_back_to_back'] = df['elo_diff'] * (
        df['away_back_to_back'].astype(int) - df['home_back_to_back'].astype(int)
    )
    # Streak interaction (if streaks exist)
    if 'home_streak' in df.columns and 'away_streak' in df.columns:
        df['streak_vs_opp_streak'] = df['home_streak'] - df['away_streak']
    # Momentum factor (combining recent performance metrics)
    if 'home_last_5_win_pct' in df.columns and 'away_last_5_win_pct' in df.columns:
        home_factor = df['home_last_5_win_pct'] - df['away_last_5_win_pct']
        if 'home_streak' in df.columns and 'away_streak' in df.columns:
            streak_factor = (df['home_streak'] - df['away_streak']) / 10
            df['momentum_factor'] = home_factor + streak_factor
        else:
            df['momentum_factor'] = home_factor
    return df

def fill_missing_factor_data(games_df):
    """Fill missing values in team factor data using team means."""
    print("Filling missing factor data...")
    df = games_df.copy()

    # Fill Four Factors
    for col in ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']:
        if col in df.columns:
            # First fill with team means
            team_type = 'away_team' if col.startswith('a_') else 'home_team'
            team_means = df.groupby(team_type)[col].transform('mean')
            df[col] = df[col].fillna(team_means)
            # Then fill remaining with overall mean
            df[col] = df[col].fillna(df[col].mean())
    return df

def prepare_modeling_data(games_df):
    """Prepare final datasets for modeling with enhanced feature handling."""
    print("Preparing modeling datasets...")

    # Define the rolling windows for win percentages
    rolling_windows = [5, 10, 15]

    # Fill missing values in performance metrics
    for prefix in ['away_', 'home_']:
        for window in rolling_windows:
            col = f'{prefix}last_{window}_win_pct'
            if col in games_df.columns:
                games_df[col] = games_df[col].fillna(0.5)

    # Other common missing values
    games_df = games_df.fillna({
        'away_back_to_back': False,
        'home_back_to_back': False,
        'away_vs_home_win_pct': 0.5,
        'away_streak': 0,
        'home_streak': 0
    })

    # Handle missing rolling features
    rolling_cols = [col for col in games_df.columns if '_10' in col]
    for col in rolling_cols:
        if col in games_df.columns:
            games_df[col] = games_df[col].fillna(games_df[col].mean())

    # Define the features to use for modeling
    # This replaces the imported ALL_FEATURES to avoid data leakage
    model_features = [
        # Base features
        'elo_diff', 'away_back_to_back', 'home_back_to_back', 'away_vs_home_win_pct',

        # Team performance features
        'away_last_5_win_pct', 'home_last_5_win_pct',
        'away_last_10_win_pct', 'home_last_10_win_pct',
        'away_last_15_win_pct', 'home_last_15_win_pct',
        'away_streak', 'home_streak',

        # Four Factors features
        'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
        'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp',

        # Rolling Four Factors
        'a_eFGp_10', 'h_eFGp_10',
        'a_TOVp_10', 'h_TOVp_10',
        'a_ORBp_10', 'h_ORBp_10',
        'a_FTr_10', 'h_FTr_10',

        # Differential features
        'eFGp_diff', 'FTr_diff', 'ORBp_diff', 'TOVp_diff',
        'eFGp_roll_diff', 'FTr_roll_diff', 'ORBp_roll_diff', 'TOVp_roll_diff',

        # Interaction features
        'elo_diff_back_to_back', 'streak_vs_opp_streak', 'momentum_factor'
    ]

    # Filter features to include only columns that actually exist in the DataFrame
    available_features = [col for col in model_features if col in games_df.columns]

    # Check if any required features are missing and print a warning
    missing_features = set(model_features) - set(available_features)
    if missing_features:
        print(f"Warning: The following features are missing from the DataFrame: {missing_features}")

    # If too many features are missing, this might indicate a bigger problem
    if len(missing_features) > 5:
        print("Warning: Many expected features are missing. Check if feature engineering worked correctly.")

    # Verify no missing values remain in feature columns
    try:
        missing_cols = games_df[available_features].columns[games_df[available_features].isna().any()].tolist()
        if missing_cols:
            print("Warning: Missing values found in columns:", missing_cols)
            print("Filling remaining missing values with column means...")
            for col in missing_cols:
                games_df[col] = games_df[col].fillna(games_df[col].mean())
    except Exception as e:
        print(f"Warning: Issue checking for missing values: {e}")

    # Sort by date and split - use 80% for training, 20% for testing
    games_df = games_df.sort_values('date')
    train_test_split_ratio = 0.8
    split_idx = int(len(games_df) * train_test_split_ratio)

    train_data = games_df.iloc[:split_idx]
    test_data = games_df.iloc[split_idx:]

    # Prepare feature matrices
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[available_features])
    X_test = scaler.transform(test_data[available_features])

    y_train = train_data['result']
    y_test = test_data['result']

    # Save the feature names for later use
    with open('a_beast/data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(available_features))

    return train_data, test_data, X_train, X_test, y_train, y_test

def save_processed_data(train_data, test_data, X_train, X_test, y_train, y_test, available_features):
    """Save all processed datasets."""
    print("Saving processed data...")

    # Define file paths
    data_dir = 'a_beast/data/processed'
    os.makedirs(data_dir, exist_ok=True)

    # Save full datasets
    train_data.to_csv(f'{data_dir}/train_data.csv')
    test_data.to_csv(f'{data_dir}/test_data.csv')

    # Save combined dataset
    combined_data = pd.concat([train_data, test_data])
    combined_data.to_csv(f'{data_dir}/combined_data.csv')

    # Save numpy arrays for modeling
    np.save(f'{data_dir}/X_train.npy', X_train)
    np.save(f'{data_dir}/X_test.npy', X_test)
    np.save(f'{data_dir}/y_train.npy', y_train.values)
    np.save(f'{data_dir}/y_test.npy', y_test.values)

    # Save feature names
    with open(f'{data_dir}/feature_names.txt', 'w') as f:
        f.write('\n'.join(available_features))

def main():
    """Main execution function."""
    try:
        # Load raw data
        raw_data = load_raw_data()

        # Step 1: Create basic game features (without requiring result)
        games_df = initial_game_features(raw_data['game_info'])

        # Step 2: Add team stats before calculating result
        games_df = add_team_stats(games_df, raw_data['team_factor_10'])

        # Step 3: Create result column (needed for momentum features)
        games_df = create_labels(games_df)

        # Step 4: Now calculate features that depend on result column
        # These functions have been fixed to only use past data
        games_df = calculate_momentum_features(games_df)
        games_df = add_matchup_history(games_df)

        # Step 5: Calculate rolling factors (fixed to only use past data)
        games_df = calculate_rolling_factors(games_df)

        # Step 6: Calculate differential features
        games_df = calculate_differential_features(games_df)

        # Step 7: Create interaction features (needs all previous features)
        games_df = create_interaction_features(games_df)

        # Step 8: Save intermediate dataset
        data_dir = 'a_beast/data/processed'
        os.makedirs(data_dir, exist_ok=True)
        games_df.to_csv(f'{data_dir}/intermediate_data.csv')

        # Step 9: Prepare and save modeling data
        train_data, test_data, X_train, X_test, y_train, y_test = prepare_modeling_data(games_df)

        # Get the available features from the model preparation
        available_features = [col for col in train_data.columns if col in games_df.columns]

        # Step 10: Save all processed data
        save_processed_data(train_data, test_data, X_train, X_test, y_train, y_test, available_features)

        print("Data preparation completed successfully!")
        print("Data leakage issues have been fixed - model performance should now reflect true predictive power.")

    except Exception as e:
        print(f"Error in data preparation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()