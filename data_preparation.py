import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Load the datasets
print("Loading datasets...")
game_info = pd.read_csv('data/game_info.csv', index_col=0)
team_stats = pd.read_csv('data/team_stats.csv', index_col=0)
team_factor_10 = pd.read_csv('data/team_factor_10.csv', index_col=0)
team_factor_20 = pd.read_csv('data/team_factor_20.csv', index_col=0)
team_factor_30 = pd.read_csv('data/team_factor_30.csv', index_col=0)
team_full_10 = pd.read_csv('data/team_full_10.csv', index_col=0)
team_full_20 = pd.read_csv('data/team_full_20.csv', index_col=0)
team_full_30 = pd.read_csv('data/team_full_30.csv', index_col=0)

# Filter for 2018-2019 season
season_2018_2019 = 1819
games_2018_2019 = game_info[game_info['season'] == season_2018_2019]
print(f"Number of games in 2018-2019 season: {len(games_2018_2019)}")

# Convert date to datetime
games_2018_2019['date'] = pd.to_datetime(games_2018_2019['date'])

# Sort by date
games_2018_2019 = games_2018_2019.sort_values('date')

# Create output directory for processed data
import os
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

# Define function to handle missing values in team factor data
def fill_missing_factor_data(df):
    # Fill missing values with mean for each team
    for col in ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']:
        if col in df.columns:
            # For away team stats
            if col.startswith('a_'):
                team_col = 'away_team'
            # For home team stats
            else:
                team_col = 'home_team'

            # Group by team and fill missing values with team mean
            team_means = df.groupby(team_col)[col].transform('mean')
            df[col] = df[col].fillna(team_means)

            # If still missing (new teams), fill with overall mean
            df[col] = df[col].fillna(df[col].mean())

    return df

# Process team factor data
print("Processing team factor data...")
team_factor_10_processed = fill_missing_factor_data(team_factor_10[team_factor_10['season'] == season_2018_2019])
team_factor_20_processed = fill_missing_factor_data(team_factor_20[team_factor_20['season'] == season_2018_2019])
team_factor_30_processed = fill_missing_factor_data(team_factor_30[team_factor_30['season'] == season_2018_2019])

# Save processed data
team_factor_10_processed.to_csv('processed_data/team_factor_10_processed.csv')
team_factor_20_processed.to_csv('processed_data/team_factor_20_processed.csv')
team_factor_30_processed.to_csv('processed_data/team_factor_30_processed.csv')

# Process team full data
print("Processing team full data...")
# Function to handle missing values in team full data
def fill_missing_full_data(df):
    # Get all stat columns (excluding game info columns)
    stat_cols = [col for col in df.columns if col.startswith('a_') or col.startswith('h_')]

    # Fill missing values with mean for each team
    for col in stat_cols:
        # For away team stats
        if col.startswith('a_'):
            team_col = 'away_team'
        # For home team stats
        else:
            team_col = 'home_team'

        # Group by team and fill missing values with team mean
        team_means = df.groupby(team_col)[col].transform('mean')
        df[col] = df[col].fillna(team_means)

        # If still missing (new teams), fill with overall mean
        df[col] = df[col].fillna(df[col].mean())

    return df

team_full_10_processed = fill_missing_full_data(team_full_10[team_full_10['season'] == season_2018_2019])
team_full_20_processed = fill_missing_full_data(team_full_20[team_full_20['season'] == season_2018_2019])
team_full_30_processed = fill_missing_full_data(team_full_30[team_full_30['season'] == season_2018_2019])

# Save processed data
team_full_10_processed.to_csv('processed_data/team_full_10_processed.csv')
team_full_20_processed.to_csv('processed_data/team_full_20_processed.csv')
team_full_30_processed.to_csv('processed_data/team_full_30_processed.csv')

# Create features for team performance patterns
print("Creating features for team performance patterns...")

# Function to calculate team's recent performance (win percentage in last N games)
def calculate_recent_performance(games_df, n_games=10):
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

# Calculate recent performance for different windows
games_with_recent_5 = calculate_recent_performance(games_2018_2019, 5)
games_with_recent_10 = calculate_recent_performance(games_2018_2019, 10)
games_with_recent_15 = calculate_recent_performance(games_2018_2019, 15)

# Save processed data
games_with_recent_5.to_csv('processed_data/games_with_recent_5.csv')
games_with_recent_10.to_csv('processed_data/games_with_recent_10.csv')
games_with_recent_15.to_csv('processed_data/games_with_recent_15.csv')

# Create features for back-to-back games
print("Creating features for back-to-back games...")

# Function to identify back-to-back games
def identify_back_to_back(games_df):
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

# Identify back-to-back games
games_with_b2b = identify_back_to_back(games_2018_2019)

# Save processed data
games_with_b2b.to_csv('processed_data/games_with_b2b.csv')

# Create features for team matchup history
print("Creating features for team matchup history...")

# Function to calculate head-to-head record
def calculate_head_to_head(games_df):
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

# Calculate head-to-head record
games_with_h2h = calculate_head_to_head(games_2018_2019)

# Save processed data
games_with_h2h.to_csv('processed_data/games_with_h2h.csv')

# Create a combined dataset with all features
print("Creating combined dataset...")

# Merge all processed datasets
combined_data = games_2018_2019.copy()

# Add recent performance features
combined_data = pd.merge(
    combined_data,
    games_with_recent_10[['game_id', 'away_last_n_win_pct', 'home_last_n_win_pct']],
    on='game_id',
    how='left'
)

# Add back-to-back features
combined_data = pd.merge(
    combined_data,
    games_with_b2b[['game_id', 'away_back_to_back', 'home_back_to_back']],
    on='game_id',
    how='left'
)

# Add head-to-head features
combined_data = pd.merge(
    combined_data,
    games_with_h2h[['game_id', 'away_vs_home_win_pct']],
    on='game_id',
    how='left'
)

# Add team factor features (using 10-game averages)
factor_cols = ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
combined_data = pd.merge(
    combined_data,
    team_factor_10_processed[['game_id'] + factor_cols],
    on='game_id',
    how='left'
)

# Calculate ELO difference
combined_data['elo_diff'] = combined_data['home_elo_i'] - combined_data['away_elo_i']

# Calculate point differential
combined_data['point_diff'] = combined_data['home_score'] - combined_data['away_score']

# Define upset based on ELO ratings
combined_data['favorite'] = np.where(combined_data['home_elo_i'] > combined_data['away_elo_i'], 1, 0)
combined_data['upset'] = np.where(combined_data['favorite'] != combined_data['result'], 1, 0)

# Fill missing values
combined_data = combined_data.fillna({
    'away_last_n_win_pct': 0.5,
    'home_last_n_win_pct': 0.5,
    'away_back_to_back': 0,
    'home_back_to_back': 0,
    'away_vs_home_win_pct': 0.5
})

# Save combined dataset
combined_data.to_csv('processed_data/combined_data.csv')

# Split data into training and testing sets
print("Splitting data into training and testing sets...")

# Sort by date
combined_data = combined_data.sort_values('date')

# Use first 80% of season for training, last 20% for testing
train_size = int(len(combined_data) * 0.8)
train_data = combined_data.iloc[:train_size]
test_data = combined_data.iloc[train_size:]

# Save train and test datasets
train_data.to_csv('processed_data/train_data.csv')
test_data.to_csv('processed_data/test_data.csv')

# Create feature sets for modeling
print("Creating feature sets for modeling...")

# Define features for prediction model
features = [
    'elo_diff',
    'away_last_n_win_pct', 'home_last_n_win_pct',
    'away_back_to_back', 'home_back_to_back',
    'away_vs_home_win_pct',
    'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
    'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp'
]

# Create X and y for training and testing
X_train = train_data[features]
y_train = train_data['result']
X_test = test_data[features]
y_test = test_data['result']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature sets
np.save('processed_data/X_train.npy', X_train_scaled)
np.save('processed_data/y_train.npy', y_train.values)
np.save('processed_data/X_test.npy', X_test_scaled)
np.save('processed_data/y_test.npy', y_test.values)

# Save feature names
with open('processed_data/feature_names.txt', 'w') as f:
    for feature in features:
        f.write(f"{feature}\n")

print("Data preparation completed. Processed data saved to processed_data/")
