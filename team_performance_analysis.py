import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)
team_stats = pd.read_csv('data/team_stats.csv', index_col=0)
team_factor_10 = pd.read_csv('data/team_factor_10.csv', index_col=0)
team_factor_20 = pd.read_csv('data/team_factor_20.csv', index_col=0)
team_factor_30 = pd.read_csv('data/team_factor_30.csv', index_col=0)

# Create output directory for team performance analysis
import os
if not os.path.exists('team_analysis'):
    os.makedirs('team_analysis')

# Filter for 2018-2019 season
season_2018_2019 = 1819
season_data = combined_data[combined_data['season'] == season_2018_2019]
season_team_stats = team_stats[team_stats['game_id'].isin(season_data['game_id'])]

# Convert date to datetime
season_data['date'] = pd.to_datetime(season_data['date'])

# Analyze team performance patterns
print("Analyzing team performance patterns...")

# Get unique teams
teams = set(season_data['away_team'].unique()) | set(season_data['home_team'].unique())
print(f"Number of teams in 2018-2019 season: {len(teams)}")

# Calculate team win percentages
team_records = {}
for team in teams:
    # Games where team played
    team_games = season_data[(season_data['away_team'] == team) | (season_data['home_team'] == team)]

    # Calculate wins
    wins = len(team_games[
        ((team_games['away_team'] == team) & (team_games['result'] == 0)) |
        ((team_games['home_team'] == team) & (team_games['result'] == 1))
    ])

    # Calculate win percentage
    win_pct = wins / len(team_games) * 100

    # Store record
    team_records[team] = {
        'games': len(team_games),
        'wins': wins,
        'losses': len(team_games) - wins,
        'win_pct': win_pct
    }

# Convert to DataFrame
team_records_df = pd.DataFrame.from_dict(team_records, orient='index')
team_records_df = team_records_df.sort_values('win_pct', ascending=False)
team_records_df.to_csv('team_analysis/team_records.csv')

# Visualize team win percentages
plt.figure(figsize=(12, 8))
sns.barplot(x=team_records_df.index, y='win_pct', data=team_records_df)
plt.title('Team Win Percentages (2018-2019 Season)')
plt.xlabel('Team')
plt.ylabel('Win Percentage (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('team_analysis/team_win_percentages.png')

# Calculate average team statistics
team_avg_stats = {}
for team in teams:
    # Get team's stats
    team_stats_home = season_team_stats[season_team_stats['team'] == team]

    # Calculate averages
    if len(team_stats_home) > 0:
        avg_stats = team_stats_home.mean(numeric_only=True)
        team_avg_stats[team] = avg_stats.to_dict()

# Convert to DataFrame
team_avg_stats_df = pd.DataFrame.from_dict(team_avg_stats, orient='index')
team_avg_stats_df.to_csv('team_analysis/team_avg_stats.csv')

# Identify key performance indicators
print("Identifying key performance indicators...")

# Calculate correlation between team stats and win percentage
team_stats_with_record = pd.merge(
    team_records_df[['win_pct']],
    team_avg_stats_df,
    left_index=True,
    right_index=True
)

# Calculate correlations
correlations = team_stats_with_record.corr()['win_pct'].sort_values(ascending=False)
correlations = correlations.drop('win_pct')  # Remove self-correlation
correlations.to_csv('team_analysis/stat_win_correlations.csv')

# Visualize top 10 positive correlations
plt.figure(figsize=(12, 8))
top_positive = correlations.head(10)
sns.barplot(x=top_positive.index, y=top_positive.values)
plt.title('Top 10 Statistics Positively Correlated with Win Percentage')
plt.xlabel('Statistic')
plt.ylabel('Correlation with Win Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('team_analysis/top_positive_correlations.png')

# Visualize top 10 negative correlations
plt.figure(figsize=(12, 8))
top_negative = correlations.tail(10)
sns.barplot(x=top_negative.index, y=top_negative.values)
plt.title('Top 10 Statistics Negatively Correlated with Win Percentage')
plt.xlabel('Statistic')
plt.ylabel('Correlation with Win Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('team_analysis/top_negative_correlations.png')

# Analyze Four Factors impact on winning
print("Analyzing Four Factors impact on winning...")

# Get Four Factors data for 2018-2019 season
season_factor_10 = team_factor_10[team_factor_10['season'] == season_2018_2019]
season_factor_20 = team_factor_20[team_factor_20['season'] == season_2018_2019]
season_factor_30 = team_factor_30[team_factor_30['season'] == season_2018_2019]

# Calculate Four Factors differentials
season_factor_10['eFGp_diff'] = season_factor_10['h_eFGp'] - season_factor_10['a_eFGp']
season_factor_10['FTr_diff'] = season_factor_10['h_FTr'] - season_factor_10['a_FTr']
season_factor_10['ORBp_diff'] = season_factor_10['h_ORBp'] - season_factor_10['a_ORBp']
season_factor_10['TOVp_diff'] = season_factor_10['a_TOVp'] - season_factor_10['h_TOVp']  # Note: lower TOV% is better

# Calculate win percentage when team has advantage in each factor
factor_win_pcts = {}

# eFG%
efg_advantage = season_factor_10[season_factor_10['eFGp_diff'] > 0]
factor_win_pcts['eFGp'] = (efg_advantage['result'] == 1).mean() * 100

# FTr
ftr_advantage = season_factor_10[season_factor_10['FTr_diff'] > 0]
factor_win_pcts['FTr'] = (ftr_advantage['result'] == 1).mean() * 100

# ORB%
orb_advantage = season_factor_10[season_factor_10['ORBp_diff'] > 0]
factor_win_pcts['ORBp'] = (orb_advantage['result'] == 1).mean() * 100

# TOV%
tov_advantage = season_factor_10[season_factor_10['TOVp_diff'] > 0]
factor_win_pcts['TOVp'] = (tov_advantage['result'] == 1).mean() * 100

# Convert to DataFrame
factor_win_pcts_df = pd.DataFrame.from_dict(factor_win_pcts, orient='index', columns=['win_pct'])
factor_win_pcts_df.to_csv('team_analysis/four_factors_win_pct.csv')

# Visualize Four Factors impact
plt.figure(figsize=(10, 6))
sns.barplot(x=factor_win_pcts_df.index, y='win_pct', data=factor_win_pcts_df)
plt.title('Win Percentage When Team Has Advantage in Four Factors')
plt.xlabel('Four Factor')
plt.ylabel('Win Percentage (%)')
plt.axhline(y=50, color='r', linestyle='--', label='50% (No Advantage)')
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig('team_analysis/four_factors_win_pct.png')

# Analyze team performance over time
print("Analyzing team performance over time...")

# Function to calculate rolling win percentage
def calculate_rolling_win_pct(team, window=10):
    # Get games where team played
    team_games = season_data[(season_data['away_team'] == team) | (season_data['home_team'] == team)]
    team_games = team_games.sort_values('date')

    # Calculate win/loss for each game
    team_games['team_win'] = np.where(
        ((team_games['away_team'] == team) & (team_games['result'] == 0)) |
        ((team_games['home_team'] == team) & (team_games['result'] == 1)),
        1, 0
    )

    # Calculate rolling win percentage
    team_games['rolling_win_pct'] = team_games['team_win'].rolling(window=window, min_periods=1).mean() * 100

    return team_games[['date', 'rolling_win_pct']]

# Calculate rolling win percentage for top 5 teams
top_teams = team_records_df.head(5).index.tolist()
rolling_win_pcts = {}

for team in top_teams:
    rolling_win_pcts[team] = calculate_rolling_win_pct(team)

# Visualize rolling win percentage for top teams
plt.figure(figsize=(14, 8))
for team in top_teams:
    plt.plot(rolling_win_pcts[team]['date'], rolling_win_pcts[team]['rolling_win_pct'], label=team)
plt.title('Rolling 10-Game Win Percentage for Top 5 Teams')
plt.xlabel('Date')
plt.ylabel('Win Percentage (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('team_analysis/top_teams_rolling_win_pct.png')

# Analyze team clustering based on playing style
print("Analyzing team clustering based on playing style...")

# Select relevant statistics for clustering
clustering_stats = [
    'FGp', 'eFGp', '3PAr', 'FTr', 'ORBp', 'DRBp', 'ASTp', 'TOVp', 'STLp', 'BLKp', 'PTS'
]

# Prepare data for clustering
cluster_data = team_avg_stats_df[clustering_stats]

# Standardize data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Perform PCA for visualization
pca = PCA(n_components=2)
cluster_data_pca = pca.fit_transform(cluster_data_scaled)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(cluster_data_scaled)

# Add cluster labels to team data
team_clusters = pd.DataFrame({
    'team': cluster_data.index,
    'cluster': clusters,
    'pca_1': cluster_data_pca[:, 0],
    'pca_2': cluster_data_pca[:, 1]
})
team_clusters.to_csv('team_analysis/team_clusters.csv', index=False)

# Visualize team clusters
plt.figure(figsize=(12, 8))
for cluster in range(4):
    cluster_teams = team_clusters[team_clusters['cluster'] == cluster]
    plt.scatter(cluster_teams['pca_1'], cluster_teams['pca_2'], label=f'Cluster {cluster+1}')

    # Add team labels
    for i, row in cluster_teams.iterrows():
        plt.annotate(row['team'], (row['pca_1'], row['pca_2']), fontsize=9)

plt.title('Team Clustering Based on Playing Style')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('team_analysis/team_clusters.png')

# Analyze cluster characteristics
cluster_stats = {}
for cluster in range(4):
    cluster_teams = team_clusters[team_clusters['cluster'] == cluster]['team']
    cluster_stats[f'Cluster {cluster+1}'] = team_avg_stats_df.loc[cluster_teams, clustering_stats].mean()

# Convert to DataFrame
cluster_stats_df = pd.DataFrame.from_dict(cluster_stats, orient='index')
cluster_stats_df.to_csv('team_analysis/cluster_stats.csv')

# Analyze upset patterns
print("Analyzing upset patterns...")

# Define upsets based on ELO ratings
season_data['favorite'] = np.where(season_data['home_elo_i'] > season_data['away_elo_i'], 1, 0)
season_data['upset'] = np.where(season_data['favorite'] != season_data['result'], 1, 0)

# Calculate upset rate
upset_rate = season_data['upset'].mean() * 100

# Calculate upset rate by team (when team is favorite)
team_upset_rates = {}
for team in teams:
    # Games where team is favorite
    team_favorite_games = season_data[
        ((season_data['away_team'] == team) & (season_data['favorite'] == 0)) |
        ((season_data['home_team'] == team) & (season_data['favorite'] == 1))
    ]

    if len(team_favorite_games) > 0:
        # Calculate upset rate
        upset_rate = team_favorite_games['upset'].mean() * 100

        # Store upset rate
        team_upset_rates[team] = {
            'games_as_favorite': len(team_favorite_games),
            'upsets': team_favorite_games['upset'].sum(),
            'upset_rate': upset_rate
        }

# Convert to DataFrame
team_upset_rates_df = pd.DataFrame.from_dict(team_upset_rates, orient='index')
team_upset_rates_df = team_upset_rates_df.sort_values('upset_rate', ascending=False)
team_upset_rates_df.to_csv('team_analysis/team_upset_rates.csv')

# Visualize team upset rates
plt.figure(figsize=(14, 8))
sns.barplot(x=team_upset_rates_df.index, y='upset_rate', data=team_upset_rates_df)
plt.title('Upset Rate When Team is Favorite')
plt.xlabel('Team')
plt.ylabel('Upset Rate (%)')
plt.xticks(rotation=90)
plt.axhline(y=season_data['upset'].mean() * 100, color='r', linestyle='--',
            label=f'League Average: {season_data["upset"].mean() * 100:.1f}%')
plt.legend()
plt.tight_layout()
plt.savefig('team_analysis/team_upset_rates.png')

# Analyze factors contributing to upsets
print("Analyzing factors contributing to upsets...")

# Calculate statistics for games with upsets vs. no upsets
upset_games = season_data[season_data['upset'] == 1]
non_upset_games = season_data[season_data['upset'] == 0]

# Calculate ELO difference statistics
elo_diff_upset = abs(upset_games['home_elo_i'] - upset_games['away_elo_i'])
elo_diff_non_upset = abs(non_upset_games['home_elo_i'] - non_upset_games['away_elo_i'])

# Calculate average ELO difference
avg_elo_diff_upset = elo_diff_upset.mean()
avg_elo_diff_non_upset = elo_diff_non_upset.mean()

# Visualize ELO difference distribution
plt.figure(figsize=(12, 6))
sns.histplot(elo_diff_upset, kde=True, label='Upset Games', alpha=0.5)
sns.histplot(elo_diff_non_upset, kde=True, label='Non-Upset Games', alpha=0.5)
plt.title('ELO Difference Distribution: Upset vs. Non-Upset Games')
plt.xlabel('ELO Difference')
plt.ylabel('Frequency')
plt.axvline(x=avg_elo_diff_upset, color='blue', linestyle='--',
            label=f'Avg. Upset: {avg_elo_diff_upset:.1f}')
plt.axvline(x=avg_elo_diff_non_upset, color='orange', linestyle='--',
            label=f'Avg. Non-Upset: {avg_elo_diff_non_upset:.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('team_analysis/elo_diff_upset_distribution.png')

# Analyze back-to-back games and upsets
back_to_back_upset_rate = season_data[
    (season_data['away_back_to_back'] == 1) | (season_data['home_back_to_back'] == 1)
]['upset'].mean() * 100

non_back_to_back_upset_rate = season_data[
    (season_data['away_back_to_back'] == 0) & (season_data['home_back_to_back'] == 0)
]['upset'].mean() * 100

# Visualize back-to-back impact on upsets
plt.figure(figsize=(10, 6))
plt.bar(['Back-to-Back Games', 'Non-Back-to-Back Games'],
        [back_to_back_upset_rate, non_back_to_back_upset_rate])
plt.title('Upset Rate: Back-to-Back vs. Non-Back-to-Back Games')
plt.ylabel('Upset Rate (%)')
plt.ylim(0, 100)
for i, v in enumerate([back_to_back_upset_rate, non_back_to_back_upset_rate]):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('team_analysis/back_to_back_upset_rate.png')

# Create summary report
with open('team_analysis/team_performance_summary.md', 'w') as f:
    f.write("# Team Performance Analysis\n\n")

    f.write("## Team Records\n\n")
    f.write("### Top 5 Teams by Win Percentage\n\n")
    for i, (team, row) in enumerate(team_records_df.head(5).iterrows()):
        f.write(f"{i+1}. {team}: {row['win_pct']:.1f}% ({row['wins']}-{row['losses']})\n")

    f.write("\n### Bottom 5 Teams by Win Percentage\n\n")
    for i, (team, row) in enumerate(team_records_df.tail(5).iterrows()):
        f.write(f"{i+1}. {team}: {row['win_pct']:.1f}% ({row['wins']}-{row['losses']})\n")

    f.write("\n## Key Performance Indicators\n\n")
    f.write("### Top 5 Statistics Positively Correlated with Winning\n\n")
    for i, (stat, corr) in enumerate(correlations.head(5).items()):
        f.write(f"{i+1}. {stat}: {corr:.3f}\n")

    f.write("\n### Top 5 Statistics Negatively Correlated with Winning\n\n")
    for i, (stat, corr) in enumerate(correlations.tail(5).items()):
        f.write(f"{i+1}. {stat}: {corr:.3f}\n")

    f.write("\n## Four Factors Analysis\n\n")
    f.write("Win percentage when team has advantage in each factor:\n\n")
    for factor, win_pct in factor_win_pcts.items():
        f.write(f"- {factor}: {win_pct:.1f}%\n")

