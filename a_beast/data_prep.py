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
    """Calculate momentum-based features including streaks and rolling stats."""
    print("Calculating momentum features...")
    df = games_df.copy().sort_values('date')
    
    for team_type in ['away_team', 'home_team']:
        prefix = team_type[:4]
        is_home = team_type == 'home_team'
        
        # Create team result series (1=win, 0=loss from team perspective)
        if is_home:
            # For home team: result already is 1=win, 0=loss
            df[f'{prefix}_result'] = df['result']
        else:
            # For away team: invert result (1=loss, 0=win → 0=loss, 1=win)
            df[f'{prefix}_result'] = 1 - df['result']
        
        # Calculate win streaks using transform to maintain index alignment
        df[f'{prefix}_streak'] = df.sort_values('date').groupby(team_type)[f'{prefix}_result'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
        )
        
        # Calculate rolling win percentages for different windows
        for window in ROLLING_WINDOWS:
            df[f'{prefix}_last_{window}_win_pct'] = df.sort_values('date').groupby(team_type)[f'{prefix}_result'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
    
    # Drop helper columns
    df = df.drop(['home_result', 'away_result'], axis=1)
    return df

def add_matchup_history(games_df):
    """Add historical matchup statistics."""
    print("Adding matchup history...")
    df = games_df.copy().sort_values('date')
    
    # Calculate head-to-head history
    df['away_vs_home_win_pct'] = df.apply(
        lambda row: df[
            ((df['away_team'] == row['away_team']) & 
             (df['home_team'] == row['home_team']) & 
             (df['date'] < row['date']))
        ]['result'].mean() if len(df[
            ((df['away_team'] == row['away_team']) & 
             (df['home_team'] == row['home_team']) & 
             (df['date'] < row['date']))
        ]) > 0 else 0.5,
        axis=1
    )
    
    return df

def add_team_stats(games_df, team_factor_df):
    """Add enhanced team statistics and factors to games dataframe."""
    print("Adding team statistics...")
    
    # Add Four Factors
    games_df = pd.merge(
        games_df,
        team_factor_df[FOUR_FACTORS_FEATURES + ['game_id']],
        on='game_id',
        how='left'
    )

    # Fill missing values
    games_df = fill_missing_factor_data(games_df)

    # Add ELO difference
    games_df['elo_diff'] = games_df['home_elo_i'] - games_df['away_elo_i']
    
    return games_df

def calculate_rolling_factors(games_df):
    """Calculate rolling averages for Four Factors."""
    print("Calculating rolling Four Factors...")
    df = games_df.copy().sort_values('date')
    
    for factor in ['eFGp', 'TOVp', 'ORBp', 'FTr']:
        # 10-game rolling averages
        for prefix in ['a_', 'h_']:
            col = f'{prefix}{factor}'
            if col in df.columns:
                team_col = 'away_team' if prefix == 'a_' else 'home_team'
                
                df[f'{col}_10'] = df.groupby(team_col)[col].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                )
    
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
    """Create interaction features between different metrics."""
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
    
    # Fill missing values in performance metrics
    for prefix in ['away_', 'home_']:
        for window in ROLLING_WINDOWS:
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
    
    # Filter ALL_FEATURES to include only columns that actually exist in the DataFrame
    available_features = [col for col in ALL_FEATURES if col in games_df.columns]
    
    # Check if any required features are missing and print a warning
    missing_features = set(ALL_FEATURES) - set(available_features)
    if missing_features:
        print(f"Warning: The following features from ALL_FEATURES are missing from the DataFrame: {missing_features}")
    
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
    
    # Sort by date and split
    games_df = games_df.sort_values('date')
    split_idx = int(len(games_df) * TRAIN_TEST_SPLIT_RATIO)
    
    train_data = games_df.iloc[:split_idx]
    test_data = games_df.iloc[split_idx:]
    
    # Prepare feature matrices
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[available_features])
    X_test = scaler.transform(test_data[available_features])
    
    y_train = train_data['result']
    y_test = test_data['result']
    
    return train_data, test_data, X_train, X_test, y_train, y_test

def save_processed_data(train_data, test_data, X_train, X_test, y_train, y_test):
    """Save all processed datasets."""
    print("Saving processed data...")
    
    # Save full datasets
    train_data.to_csv(PROCESSED_DATA_FILES['train_data'])
    test_data.to_csv(PROCESSED_DATA_FILES['test_data'])
    
    # Save numpy arrays for modeling
    np.save(PROCESSED_DATA_FILES['X_train'], X_train)
    np.save(PROCESSED_DATA_FILES['X_test'], X_test)
    np.save(PROCESSED_DATA_FILES['y_train'], y_train.values)
    np.save(PROCESSED_DATA_FILES['y_test'], y_test.values)
    
    # Save feature names
    with open(PROCESSED_DATA_FILES['feature_names'], 'w') as f:
        f.write('\n'.join(ALL_FEATURES))

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
        games_df = calculate_momentum_features(games_df)
        games_df = add_matchup_history(games_df)
        
        # Step 5: Calculate rolling factors (needs result for context)
        games_df = calculate_rolling_factors(games_df)
        
        # Step 6: Calculate differential features
        games_df = calculate_differential_features(games_df)
        
        # Step 7: Create interaction features (needs all previous features)
        games_df = create_interaction_features(games_df)
        
        # Step 8: Save combined dataset
        games_df.to_csv(PROCESSED_DATA_FILES['combined_data'])
        
        # Step 9: Prepare and save modeling data
        train_data, test_data, X_train, X_test, y_train, y_test = prepare_modeling_data(games_df)
        save_processed_data(train_data, test_data, X_train, X_test, y_train, y_test)
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()