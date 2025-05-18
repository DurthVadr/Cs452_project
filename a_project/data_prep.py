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

def create_game_features(game_info_df, team_stats_df, team_factor_df):
    """Create features for each game."""
    print("Creating game features...")
    
    # Filter for 2018-2019 season and sort by date
    games = game_info_df[game_info_df['season'] == 1819].copy()
    games['date'] = pd.to_datetime(games['date'])
    games = games.sort_values('date')

    # Calculate back-to-back games
    games['away_back_to_back'] = games.groupby('away_team')['date'].diff().dt.days <= 1
    games['home_back_to_back'] = games.groupby('home_team')['date'].diff().dt.days <= 1
    
    # Calculate rolling win percentages (last 10 games)
    for team_type in ['away_team', 'home_team']:
        games[f'{team_type[:4]}_last_n_win_pct'] = games.groupby(team_type).rolling(10)[
            'result'].apply(lambda x: (x == (1 if team_type.startswith('home') else 0)).mean()
        ).reset_index(0, drop=True)

    # Add team matchup history
    games['away_vs_home_win_pct'] = games.apply(
        lambda row: games[
            ((games['away_team'] == row['away_team']) & 
             (games['home_team'] == row['home_team']) & 
             (games['date'] < row['date']))
        ]['result'].mean(),
        axis=1
    )

    return games

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

def add_team_stats(games_df, team_stats_df, team_factor_df):
    """Add team statistics and factors to games dataframe."""
    print("Adding team statistics...")
    
    # Add Four Factors
    games_df = pd.merge(
        games_df,
        team_factor_df[FOUR_FACTORS_FEATURES + ['game_id']],
        on='game_id',
        how='left'
    )

    # Fill missing Four Factors data
    games_df = fill_missing_factor_data(games_df)

    # Calculate differential features
    for feat in ['eFGp', 'FTr', 'ORBp', 'TOVp']:
        games_df[f'{feat}_diff'] = games_df[f'h_{feat}'] - games_df[f'a_{feat}']

    # Add ELO difference
    games_df['elo_diff'] = games_df['home_elo_i'] - games_df['away_elo_i']

    return games_df

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

def prepare_modeling_data(games_df):
    """Prepare final datasets for modeling."""
    print("Preparing modeling datasets...")
    
    # Fill missing values in performance metrics
    games_df = games_df.fillna({
        'away_last_n_win_pct': 0.5,
        'home_last_n_win_pct': 0.5,
        'away_back_to_back': 0,
        'home_back_to_back': 0,
        'away_vs_home_win_pct': 0.5
    })
    
    # Verify no missing values remain
    missing_cols = games_df[ALL_FEATURES].columns[games_df[ALL_FEATURES].isna().any()].tolist()
    if missing_cols:
        print("Warning: Missing values found in columns:", missing_cols)
        print("Filling remaining missing values with column means...")
        games_df[ALL_FEATURES] = games_df[ALL_FEATURES].fillna(games_df[ALL_FEATURES].mean())
    
    # Sort by date and split
    games_df = games_df.sort_values('date')
    split_idx = int(len(games_df) * 0.8)
    
    train_data = games_df.iloc[:split_idx]
    test_data = games_df.iloc[split_idx:]
    
    # Prepare feature matrices
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[ALL_FEATURES])
    X_test = scaler.transform(test_data[ALL_FEATURES])
    
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
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save feature names
    with open(PROCESSED_DATA_FILES['feature_names'], 'w') as f:
        f.write('\n'.join(ALL_FEATURES))

def main():
    """Main execution function."""
    try:
        # Load raw data
        raw_data = load_raw_data()
        
        # Create initial game features
        games_df = create_game_features(
            raw_data['game_info'],
            raw_data['team_stats'],
            raw_data['team_factor_10']
        )
        
        # Add team statistics
        games_df = add_team_stats(
            games_df,
            raw_data['team_stats'],
            raw_data['team_factor_10']
        )
        
        # Create labels
        games_df = create_labels(games_df)
        
        # Save combined dataset
        games_df.to_csv(PROCESSED_DATA_FILES['combined_data'])
        
        # Prepare and save modeling data
        train_data, test_data, X_train, X_test, y_train, y_test = prepare_modeling_data(games_df)
        save_processed_data(train_data, test_data, X_train, X_test, y_train, y_test)
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise

if __name__ == "__main__":
    main()