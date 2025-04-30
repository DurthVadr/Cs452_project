import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Load the datasets
game_info = pd.read_csv('data/game_info.csv', index_col=0)
team_stats = pd.read_csv('data/team_stats.csv', index_col=0)
team_factor_10 = pd.read_csv('data/team_factor_10.csv', index_col=0)
team_factor_20 = pd.read_csv('data/team_factor_20.csv', index_col=0)
team_factor_30 = pd.read_csv('data/team_factor_30.csv', index_col=0)
team_full_10 = pd.read_csv('data/team_full_10.csv', index_col=0)
team_full_20 = pd.read_csv('data/team_full_20.csv', index_col=0)
team_full_30 = pd.read_csv('data/team_full_30.csv', index_col=0)

# Create output directory for plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Basic information about each dataset
datasets = {
    'game_info': game_info,
    'team_stats': team_stats,
    'team_factor_10': team_factor_10,
    'team_factor_20': team_factor_20,
    'team_factor_30': team_factor_30,
    'team_full_10': team_full_10,
    'team_full_20': team_full_20,
    'team_full_30': team_full_30
}

# Create a summary report
with open('data_summary.md', 'w') as f:
    f.write("# NBA Data Exploration Summary\n\n")
    f.write("## Dataset Overview\n\n")

    for name, df in datasets.items():
        f.write(f"### {name}\n")
        f.write(f"- Shape: {df.shape}\n")
        f.write(f"- Columns: {', '.join(df.columns)}\n")
        f.write(f"- Missing values: {df.isna().sum().sum()}\n\n")

# Analyze game_info dataset
print("Analyzing game_info dataset...")

# Check for 2018-2019 season data
seasons = game_info['season'].unique()
print(f"Available seasons: {seasons}")

# Filter for 2018-2019 season if available (typically coded as 1819)
season_2018_2019 = None
for season in seasons:
    if str(season).endswith('19'):
        season_2018_2019 = season
        break

if season_2018_2019:
    games_2018_2019 = game_info[game_info['season'] == season_2018_2019]
    print(f"Number of games in 2018-2019 season: {len(games_2018_2019)}")

    # Add to summary report
    with open('data_summary.md', 'a') as f:
        f.write(f"## 2018-2019 Season Analysis\n\n")
        f.write(f"- Number of games: {len(games_2018_2019)}\n")
        f.write(f"- Date range: {games_2018_2019['date'].min()} to {games_2018_2019['date'].max()}\n\n")
else:
    print("2018-2019 season data not found. Using most recent season available.")
    most_recent_season = max(seasons)
    games_most_recent = game_info[game_info['season'] == most_recent_season]

    # Add to summary report
    with open('data_summary.md', 'a') as f:
        f.write(f"## Most Recent Season Analysis ({most_recent_season})\n\n")
        f.write(f"- Number of games: {len(games_most_recent)}\n")
        f.write(f"- Date range: {games_most_recent['date'].min()} to {games_most_recent['date'].max()}\n\n")

# Analyze home court advantage
home_wins = len(game_info[game_info['result'] == 1])
away_wins = len(game_info[game_info['result'] == 0])
total_games = len(game_info)

home_win_pct = home_wins / total_games * 100
away_win_pct = away_wins / total_games * 100

# Plot home vs away win percentage
plt.figure(figsize=(10, 6))
plt.bar(['Home Wins', 'Away Wins'], [home_win_pct, away_win_pct], color=['blue', 'red'])
plt.title('Home vs Away Win Percentage')
plt.ylabel('Win Percentage (%)')
plt.ylim(0, 100)
for i, v in enumerate([home_win_pct, away_win_pct]):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.savefig('plots/home_away_win_pct.png')

# Add to summary report
with open('data_summary.md', 'a') as f:
    f.write(f"## Home Court Advantage\n\n")
    f.write(f"- Home team win percentage: {home_win_pct:.1f}%\n")
    f.write(f"- Away team win percentage: {away_win_pct:.1f}%\n\n")

# Analyze ELO ratings
print("Analyzing ELO ratings...")

# Distribution of initial ELO ratings
plt.figure(figsize=(12, 6))
plt.hist(game_info['away_elo_i'], bins=30, alpha=0.5, label='Away Teams')
plt.hist(game_info['home_elo_i'], bins=30, alpha=0.5, label='Home Teams')
plt.title('Distribution of Initial ELO Ratings')
plt.xlabel('ELO Rating')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('plots/elo_distribution.png')

# ELO prediction accuracy
elo_correct = game_info[game_info['elo_pred'] == game_info['result']]
elo_accuracy = len(elo_correct) / len(game_info) * 100

# Add to summary report
with open('data_summary.md', 'a') as f:
    f.write(f"## ELO Rating Analysis\n\n")
    f.write(f"- ELO prediction accuracy: {elo_accuracy:.1f}%\n")
    f.write(f"- Average home team initial ELO: {game_info['home_elo_i'].mean():.1f}\n")
    f.write(f"- Average away team initial ELO: {game_info['away_elo_i'].mean():.1f}\n\n")

# Analyze upsets
print("Analyzing upsets...")

# Define an upset as when the team with lower ELO rating wins
game_info['favorite'] = np.where(game_info['home_elo_i'] > game_info['away_elo_i'], 1, 0)
game_info['upset'] = np.where(game_info['favorite'] != game_info['result'], 1, 0)

upset_rate = game_info['upset'].mean() * 100

# Add to summary report
with open('data_summary.md', 'a') as f:
    f.write(f"## Upset Analysis\n\n")
    f.write(f"- Upset rate: {upset_rate:.1f}%\n")
    f.write(f"- Total upsets: {game_info['upset'].sum()} out of {len(game_info)} games\n\n")

# Plot upset rate by season
upset_by_season = game_info.groupby('season')['upset'].mean() * 100

plt.figure(figsize=(12, 6))
upset_by_season.plot(kind='bar')
plt.title('Upset Rate by Season')
plt.xlabel('Season')
plt.ylabel('Upset Rate (%)')
plt.axhline(y=upset_rate, color='r', linestyle='-', label=f'Overall Average: {upset_rate:.1f}%')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/upset_rate_by_season.png')

# Analyze Four Factors
print("Analyzing Four Factors...")

# Check for missing values in team_factor datasets
missing_factor_10 = team_factor_10.isna().sum().sum()
missing_factor_20 = team_factor_20.isna().sum().sum()
missing_factor_30 = team_factor_30.isna().sum().sum()

# Add to summary report
with open('data_summary.md', 'a') as f:
    f.write(f"## Four Factors Analysis\n\n")
    f.write(f"- Missing values in 10-game average: {missing_factor_10}\n")
    f.write(f"- Missing values in 20-game average: {missing_factor_20}\n")
    f.write(f"- Missing values in 30-game average: {missing_factor_30}\n\n")

# Analyze team statistics
print("Analyzing team statistics...")

# Calculate average team statistics
avg_stats = team_stats.groupby('team').mean()

# Top 5 teams by points scored
top_scoring_teams = avg_stats.sort_values('PTS', ascending=False).head(5)

# Add to summary report
with open('data_summary.md', 'a') as f:
    f.write(f"## Team Statistics Analysis\n\n")
    f.write(f"- Number of unique teams: {team_stats['team'].nunique()}\n")
    f.write(f"- Top 5 scoring teams: {', '.join(top_scoring_teams.index)}\n\n")

print("Data exploration completed. Results saved to data_summary.md")
