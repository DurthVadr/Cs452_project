import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter

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
games_2018_2019 = game_info[game_info['season'] == season_2018_2019].copy()
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
    """Handle missing values in team factor data."""
    df = df.copy()  # Create explicit copy
    
    # Check first if team_id column exists, if not try to find it
    team_id_column = 'team_id'
    if 'team_id' not in df.columns:
        # Common alternative names for team ID columns
        possible_columns = ['team', 'franchise_id', 'team_name', 'home_team', 'away_team']
        for col in possible_columns:
            if col in df.columns:
                team_id_column = col
                print(f"Using '{col}' column as team identifier")
                break
        # If still not found, print available columns and raise error
        if team_id_column == 'team_id':
            print("Available columns:", df.columns.tolist())
            print("Warning: No team identifier column found. Using index for grouping.")
            # As a fallback, add a dummy team_id column based on index
            df['team_id'] = df.index
            team_id_column = 'team_id'
    
    # Only process numeric columns - explicitly exclude date and object columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        if team_id_column in col or 'season' in col or 'game_id' in col or 'date' in col:
            continue
        
        # Get team-specific means for each column
        team_means = df.groupby(team_id_column)[col].transform('mean')
        # Fill with team means first
        df.loc[:, col] = df[col].fillna(team_means)
        
        # Fill any remaining NaNs with overall column mean
        df.loc[:, col] = df[col].fillna(df[col].mean())
    
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
    """Handle missing values in team full data."""
    df = df.copy()  # Create explicit copy
    
    # Check first if team_id column exists, if not try to find it
    team_id_column = 'team_id'
    if 'team_id' not in df.columns:
        # Common alternative names for team ID columns
        possible_columns = ['team', 'franchise_id', 'team_name', 'home_team', 'away_team']
        for col in possible_columns:
            if col in df.columns:
                team_id_column = col
                print(f"Using '{col}' column as team identifier")
                break
        # If still not found, print available columns and raise error
        if team_id_column == 'team_id':
            print("Available columns:", df.columns.tolist())
            print("Warning: No team identifier column found. Using index for grouping.")
            # As a fallback, add a dummy team_id column based on index
            df['team_id'] = df.index
            team_id_column = 'team_id'
    
    # Only process numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        if team_id_column in col or 'season' in col or 'game_id' in col or 'date' in col:
            continue
        
        # Get team-specific means for each column
        team_means = df.groupby(team_id_column)[col].transform('mean')
        # Fill with team means first
        df.loc[:, col] = df[col].fillna(team_means)
        
        # Fill any remaining NaNs with overall column mean
        df.loc[:, col] = df[col].fillna(df[col].mean())
    
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

# Function to calculate team's streak before the current game (positive for winning streak, negative for losing streak)
def calculate_streak(games_df):
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
            # Update the appropriate column in the original dataframe with the streak BEFORE this game
            if game['away_team'] == team:
                df.at[idx, 'away_streak'] = prev_streaks[idx]
            else:
                df.at[idx, 'home_streak'] = prev_streaks[idx]

    return df

# Function to calculate weighted recent performance (more weight to recent games)
def calculate_weighted_performance(games_df, windows=[3, 5, 10], weights=[0.5, 0.3, 0.2]):
    # Create a copy to avoid modifying the original dataframe
    df = games_df.copy()

    # Initialize columns for weighted performance
    df['away_weighted_win_pct'] = np.nan
    df['home_weighted_win_pct'] = np.nan

    # Get unique teams
    teams = set(df['away_team'].unique()) | set(df['home_team'].unique())

    # For each team, calculate weighted win percentage before each game
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

# Calculate recent performance for different windows
games_with_recent_5 = calculate_recent_performance(games_2018_2019, 5)
games_with_recent_10 = calculate_recent_performance(games_2018_2019, 10)
games_with_recent_15 = calculate_recent_performance(games_2018_2019, 15)

# Calculate streak and weighted performance
games_with_streak = calculate_streak(games_2018_2019)
games_with_weighted = calculate_weighted_performance(games_2018_2019)

# Save processed data
games_with_recent_5.to_csv('processed_data/games_with_recent_5.csv')
games_with_recent_10.to_csv('processed_data/games_with_recent_10.csv')
games_with_recent_15.to_csv('processed_data/games_with_recent_15.csv')
games_with_streak.to_csv('processed_data/games_with_streak.csv')
games_with_weighted.to_csv('processed_data/games_with_weighted.csv')

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

# Add streak features
print("Adding streak features...")
combined_data = pd.merge(
    combined_data,
    games_with_streak[['game_id', 'away_streak', 'home_streak']],
    on='game_id',
    how='left'
)

# Add weighted performance features
print("Adding weighted performance features...")
combined_data = pd.merge(
    combined_data,
    games_with_weighted[['game_id', 'away_weighted_win_pct', 'home_weighted_win_pct']],
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

# Calculate point differential (for analysis only, not to be used as a feature)
combined_data['point_diff'] = combined_data['home_score'] - combined_data['away_score']

# Create differential features between home and away teams
print("Creating differential features between home and away teams...")
combined_data['eFGp_diff'] = combined_data['h_eFGp'] - combined_data['a_eFGp']
combined_data['FTr_diff'] = combined_data['h_FTr'] - combined_data['a_FTr']
combined_data['ORBp_diff'] = combined_data['h_ORBp'] - combined_data['a_ORBp']
combined_data['TOVp_diff'] = combined_data['h_TOVp'] - combined_data['a_TOVp']

# Create interaction features for Four Factors
print("Creating interaction features for Four Factors...")
combined_data['h_eFGp_x_TOVp'] = combined_data['h_eFGp'] * (1 - combined_data['h_TOVp'])
combined_data['a_eFGp_x_TOVp'] = combined_data['a_eFGp'] * (1 - combined_data['a_TOVp'])
combined_data['h_eFGp_x_ORBp'] = combined_data['h_eFGp'] * combined_data['h_ORBp']
combined_data['a_eFGp_x_ORBp'] = combined_data['a_eFGp'] * combined_data['a_ORBp']

# Define upset based on ELO ratings
combined_data['favorite'] = np.where(combined_data['home_elo_i'] > combined_data['away_elo_i'], 1, 0)
combined_data['upset'] = np.where(combined_data['favorite'] != combined_data['result'], 1, 0)

# Fill missing values
combined_data = combined_data.fillna({
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

# Original class distribution
print("\n--- Original Class Distribution ---")
print("Checking upset distribution in training data...")
train_upset_count = Counter(train_data['upset'])[1]
train_non_upset_count = Counter(train_data['upset'])[0]
print(f"Upset games: {train_upset_count} ({train_upset_count/(train_upset_count+train_non_upset_count)*100:.1f}%)")
print(f"Non-upset games: {train_non_upset_count} ({train_non_upset_count/(train_upset_count+train_non_upset_count)*100:.1f}%)")

# Save the original training data
train_data.to_csv('processed_data/train_data_original.csv')

# APPROACH 1: Random Oversampling
print("\n--- Applying Random Oversampling ---")
X_train_original = train_data.drop(columns=['result', 'upset'])
y_train_original = train_data[['result', 'upset']]

ros = RandomOverSampler(random_state=42)
X_resampled_ros, _ = ros.fit_resample(X_train_original, train_data['upset'])

# Map the resampled indices back to the original DataFrame to include all columns
train_data_ros = pd.DataFrame()
for i, idx in enumerate(ros.sample_indices_):
    if i < len(train_data):
        train_data_ros = pd.concat([train_data_ros, train_data.iloc[idx:idx+1]], axis=0)
    else:
        # This is a synthetic sample, duplicate an existing upset sample
        original_idx = ros.sample_indices_[i % len(train_data)]
        train_data_ros = pd.concat([train_data_ros, train_data.iloc[original_idx:original_idx+1]], axis=0)

# Check new class distribution
ros_upset_count = Counter(train_data_ros['upset'])[1]
ros_non_upset_count = Counter(train_data_ros['upset'])[0]
print(f"After Random Oversampling:")
print(f"Upset games: {ros_upset_count} ({ros_upset_count/(ros_upset_count+ros_non_upset_count)*100:.1f}%)")
print(f"Non-upset games: {ros_non_upset_count} ({ros_non_upset_count/(ros_upset_count+ros_non_upset_count)*100:.1f}%)")

# Save the Random Oversampled training data
train_data_ros.to_csv('processed_data/train_data_ros.csv')

# APPROACH 2: SMOTE
print("\n--- Applying SMOTE ---")
# Use only numeric features for SMOTE
X_train_numeric = train_data.select_dtypes(include=['number']).drop(columns=['result', 'upset'])
y_train = train_data['upset']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train_numeric, y_train)

# Create a new DataFrame from the resampled data
train_data_smote = pd.DataFrame(X_resampled_smote, columns=X_train_numeric.columns)

# Add the target columns
train_data_smote['upset'] = y_resampled_smote

# Handle the result column appropriately
# Default value for result - will be overridden based on upset
train_data_smote['result'] = 0 

# For original samples, use original result
for i in range(min(len(train_data), len(train_data_smote))):
    train_data_smote.loc[i, 'result'] = train_data.iloc[i]['result']
    
# For synthetic upset samples, result should be opposite of favorite
for i in range(len(train_data), len(train_data_smote)):
    # If upset=1, then result is opposite of favorite
    # If this team is favorite (1) then result should be 0 for an upset
    # If this team is not favorite (0) then result should be 1 for an upset
    favorite = 1 if train_data_smote.loc[i, 'home_elo_i'] > train_data_smote.loc[i, 'away_elo_i'] else 0
    train_data_smote.loc[i, 'result'] = 1 - favorite

# Add back the non-numeric columns (except result and upset that we've already handled)
for col in train_data.columns:
    if col not in train_data_smote.columns and col not in ['result', 'upset']:
        # Create a series of the right length
        if len(train_data_smote) > len(train_data):
            # We need to pad the original values
            if col == 'date':
                # Use the last date in the training set for synthetic samples
                last_date = train_data['date'].max()
                values = train_data['date'].tolist() + [last_date] * (len(train_data_smote) - len(train_data))
                train_data_smote[col] = values
            else:
                # For categorical columns, find values from samples with matching upset value
                values = []
                # First, add all original values
                values.extend(train_data[col].tolist())
                
                # Then, for each synthetic sample, find a suitable value from original data
                synthetic_count = len(train_data_smote) - len(train_data)
                upset_samples = train_data[train_data['upset'] == 1]
                
                if len(upset_samples) > 0:
                    # Get values from upset samples and repeat as needed
                    upset_values = upset_samples[col].tolist()
                    while len(values) < len(train_data_smote):
                        values.append(upset_values[len(values) % len(upset_values)])
                else:
                    # No upset samples, use any value
                    filler = train_data[col].iloc[0]
                    values.extend([filler] * synthetic_count)
                    
                train_data_smote[col] = values
        else:
            # If the SMOTE data is smaller or equal to original data, just take the right number of values
            train_data_smote[col] = train_data[col].head(len(train_data_smote)).values

# Check new class distribution
smote_upset_count = sum(train_data_smote['upset'] == 1)
smote_non_upset_count = sum(train_data_smote['upset'] == 0)
print(f"After SMOTE:")
print(f"Upset games: {smote_upset_count} ({smote_upset_count/(smote_upset_count+smote_non_upset_count)*100:.1f}%)")
print(f"Non-upset games: {smote_non_upset_count} ({smote_non_upset_count/(smote_upset_count+smote_non_upset_count)*100:.1f}%)")

# Save the SMOTE training data
train_data_smote.to_csv('processed_data/train_data_smote.csv')

# Create feature sets for modeling
print("Creating feature sets for modeling...")

# Define features for prediction model
features = [
    'elo_diff',
    'away_last_n_win_pct', 'home_last_n_win_pct',
    'away_back_to_back', 'home_back_to_back',
    'away_vs_home_win_pct',
    'a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp',
    'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp',
    # Differential features
    'eFGp_diff', 'FTr_diff', 'ORBp_diff', 'TOVp_diff',
    # Interaction features
    'h_eFGp_x_TOVp', 'a_eFGp_x_TOVp', 'h_eFGp_x_ORBp', 'a_eFGp_x_ORBp',
    # Momentum features
    'away_streak', 'home_streak',
    'away_weighted_win_pct', 'home_weighted_win_pct'
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
