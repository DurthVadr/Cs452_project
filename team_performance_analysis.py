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
    'FGp', 'eFGp', '3PAr', 'FTr', 'ORBp', 'DRBp', 'ASTp', 'TOVp', 'STLp', 'BLKp', 'PTS', 'ORtg', 'DRtg', '3Pp', 'TRBp'
]

# Prepare data for clustering
cluster_data = team_avg_stats_df[clustering_stats]

# Standardize data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)
cluster_data_scaled_df = pd.DataFrame(cluster_data_scaled, index=cluster_data.index, columns=clustering_stats)

# Create custom scores for each cluster type
# 1. Offensive Juggernauts: High pace, excellent shooting (especially 3PT), average/below-average defense
offensive_juggernaut_metrics = {
    'positive': ['PTS', 'ORtg', 'eFGp', '3Pp', '3PAr', 'ASTp'],  # Higher is better
    'negative': []  # Lower is better
}

# 2. Defensive Stalwarts: Strong defensive metrics, lower pace, efficient offense
defensive_stalwart_metrics = {
    'positive': ['STLp', 'BLKp'],  # Higher is better
    'negative': ['DRtg', 'TOVp']  # Lower is better (DRtg inverted below)
}

# 3. Interior Presence: Strong rebounding (especially offensive), inside scoring, less 3PT emphasis
interior_presence_metrics = {
    'positive': ['ORBp', 'DRBp', 'TRBp', 'BLKp', 'FTr'],  # Higher is better
    'negative': ['3PAr']  # Lower is better
}

# 4. Balanced/Mid-Tier: More balanced statistical profile
# This will be determined by teams that don't strongly fit into the other categories

# Calculate scores for each team for each cluster type
offensive_juggernaut_score = cluster_data_scaled_df[offensive_juggernaut_metrics['positive']].mean(axis=1)
defensive_stalwart_score = cluster_data_scaled_df[defensive_stalwart_metrics['positive']].mean(axis=1) - cluster_data_scaled_df[defensive_stalwart_metrics['negative']].mean(axis=1)
interior_presence_score = cluster_data_scaled_df[interior_presence_metrics['positive']].mean(axis=1) - cluster_data_scaled_df[interior_presence_metrics['negative']].mean(axis=1)

# Create a DataFrame with these scores
team_scores = pd.DataFrame({
    'offensive_juggernaut': offensive_juggernaut_score,
    'defensive_stalwart': defensive_stalwart_score,
    'interior_presence': interior_presence_score
}, index=cluster_data.index)

# Calculate a "balance" score (lower means more balanced)
team_scores['score_variance'] = team_scores.var(axis=1)
team_scores['balanced_score'] = -team_scores['score_variance']  # Invert so higher means more balanced

# Determine the dominant characteristic for each team
team_scores['dominant_type'] = team_scores[['offensive_juggernaut', 'defensive_stalwart', 'interior_presence', 'balanced_score']].idxmax(axis=1)

# Map the dominant type to cluster numbers
cluster_mapping = {
    'offensive_juggernaut': 0,  # Cluster 1
    'defensive_stalwart': 1,    # Cluster 2
    'interior_presence': 2,     # Cluster 3
    'balanced_score': 3         # Cluster 4
}

team_scores['cluster'] = team_scores['dominant_type'].map(cluster_mapping)

# Create a DataFrame for visualization
team_clusters = pd.DataFrame({
    'team': cluster_data.index,
    'cluster': team_scores['cluster'],
    'offensive_score': offensive_juggernaut_score,
    'defensive_score': defensive_stalwart_score,
    'interior_score': interior_presence_score,
    'balanced_score': team_scores['balanced_score'],
    'dominant_type': team_scores['dominant_type']
})

# Save to CSV
team_clusters.to_csv('team_analysis/team_clusters.csv', index=False)

# Create a 2D visualization using offensive score and interior score as axes
plt.figure(figsize=(14, 10))

# Define cluster names for the legend
cluster_names = {
    0: 'Offensive Juggernauts',
    1: 'Defensive Stalwarts',
    2: 'Interior Presence',
    3: 'Balanced Teams'
}

# Define colors and markers for each cluster
colors = ['#FF9500', '#4285F4', '#34A853', '#EA4335']
markers = ['o', 's', '^', 'D']

# Plot each cluster
for cluster in range(4):
    cluster_teams = team_clusters[team_clusters['cluster'] == cluster]
    plt.scatter(
        cluster_teams['offensive_score'],
        cluster_teams['interior_score'],
        label=cluster_names[cluster],
        color=colors[cluster],
        marker=markers[cluster],
        s=120,  # Larger point size
        alpha=0.8
    )

    # Add team labels
    for i, row in cluster_teams.iterrows():
        plt.annotate(
            row['team'],
            (row['offensive_score'], row['interior_score']),
            fontsize=10,
            fontweight='bold',
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

# Add reference lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add quadrant descriptions
plt.annotate('High Offense, Low Interior', xy=(1, -1), xytext=(1.5, -1.5),
             fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
plt.annotate('High Offense, High Interior', xy=(1, 1), xytext=(1.5, 1.5),
             fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
plt.annotate('Low Offense, Low Interior', xy=(-1, -1), xytext=(-1.5, -1.5),
             fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))
plt.annotate('Low Offense, High Interior', xy=(-1, 1), xytext=(-1.5, 1.5),
             fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.7))

plt.title('NBA Team Clustering (2018-2019 Season)', fontsize=16, fontweight='bold')
plt.xlabel('Offensive Prowess', fontsize=14)
plt.ylabel('Interior Presence', fontsize=14)
plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('team_analysis/team_clusters.png', dpi=300)

# Create a second visualization showing defensive score vs offensive score
plt.figure(figsize=(14, 10))

# Plot each cluster
for cluster in range(4):
    cluster_teams = team_clusters[team_clusters['cluster'] == cluster]
    plt.scatter(
        cluster_teams['offensive_score'],
        cluster_teams['defensive_score'],
        label=cluster_names[cluster],
        color=colors[cluster],
        marker=markers[cluster],
        s=120,
        alpha=0.8
    )

    # Add team labels
    for i, row in cluster_teams.iterrows():
        plt.annotate(
            row['team'],
            (row['offensive_score'], row['defensive_score']),
            fontsize=10,
            fontweight='bold',
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

# Add reference lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.title('NBA Teams: Offensive vs Defensive Profile (2018-2019)', fontsize=16, fontweight='bold')
plt.xlabel('Offensive Prowess', fontsize=14)
plt.ylabel('Defensive Prowess', fontsize=14)
plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('team_analysis/team_clusters_offense_defense.png', dpi=300)

# Analyze cluster characteristics
cluster_stats = {}
cluster_labels = {
    0: 'Offensive Juggernauts',
    1: 'Defensive Stalwarts',
    2: 'Interior Presence',
    3: 'Balanced Teams'
}

for cluster in range(4):
    cluster_teams = team_clusters[team_clusters['cluster'] == cluster]['team']
    cluster_stats[cluster_labels[cluster]] = team_avg_stats_df.loc[cluster_teams, clustering_stats].mean()

    # Add the custom scores to the cluster stats
    cluster_teams_df = team_clusters[team_clusters['cluster'] == cluster]
    cluster_stats[cluster_labels[cluster]]['offensive_score'] = cluster_teams_df['offensive_score'].mean()
    cluster_stats[cluster_labels[cluster]]['defensive_score'] = cluster_teams_df['defensive_score'].mean()
    cluster_stats[cluster_labels[cluster]]['interior_score'] = cluster_teams_df['interior_score'].mean()
    cluster_stats[cluster_labels[cluster]]['balanced_score'] = cluster_teams_df['balanced_score'].mean()

    # Add the teams in this cluster
    cluster_stats[cluster_labels[cluster]]['teams'] = ', '.join(cluster_teams.tolist())
    cluster_stats[cluster_labels[cluster]]['team_count'] = len(cluster_teams)

# Convert to DataFrame
cluster_stats_df = pd.DataFrame.from_dict(cluster_stats, orient='index')

# Create cluster descriptions based on their key characteristics
cluster_descriptions = {
    'Offensive Juggernauts': "Teams with high scoring, efficient shooting (especially from three), and often average or below-average defense.",
    'Defensive Stalwarts': "Teams with strong defensive metrics, lower pace, and efficient (though not necessarily high-volume) offense.",
    'Interior Presence': "Teams that excel in rebounding (particularly offensive rebounding) and score frequently near the basket.",
    'Balanced Teams': "Teams with a more balanced statistical profile, lacking extreme strengths or weaknesses."
}

# Add descriptions to the DataFrame
cluster_stats_df['description'] = pd.Series(cluster_descriptions)

# Save to CSV
cluster_stats_df.to_csv('team_analysis/cluster_stats.csv')

# Create a visual summary of the clusters showing their key metrics
plt.figure(figsize=(12, 8))

# Prepare data for bar chart
cluster_metrics = ['offensive_score', 'defensive_score', 'interior_score', 'balanced_score']
x = np.arange(len(cluster_labels))
width = 0.2
multiplier = 0

# Plot bars for each metric
for metric in cluster_metrics:
    offset = width * multiplier
    metric_values = [cluster_stats_df.loc[cluster_labels[i], metric] for i in range(4)]

    plt.bar(x + offset, metric_values, width, label=metric.replace('_', ' ').title())
    multiplier += 1

# Add labels and formatting
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Cluster Type', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Cluster Characteristics by Key Metrics', fontsize=14, fontweight='bold')
plt.xticks(x + width * 1.5, [cluster_labels[i] for i in range(4)], rotation=15)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('team_analysis/cluster_characteristics.png', dpi=300)

# Create a radar chart to visualize the key stats for each cluster
plt.figure(figsize=(14, 10))
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Function to create a radar chart
def radar_chart(ax, angles, values, color, label):
    # Close the plot (connect last point to first)
    values = np.append(values, values[0])
    angles = np.append(angles, angles[0])

    # Plot data
    ax.plot(angles, values, color=color, linewidth=2, label=label)
    ax.fill(angles, values, color=color, alpha=0.25)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    # Remove y-axis labels
    ax.set_yticklabels([])

    # Add gridlines
    ax.grid(True, alpha=0.3)

    return ax

# Select key stats for radar chart
categories = ['PTS', 'ORtg', 'DRtg', 'eFGp', '3PAr', 'ORBp', 'DRBp', 'ASTp', 'STLp', 'BLKp']
num_categories = len(categories)

# Set up angles for radar chart (equally spaced around a circle)
angles = np.linspace(0, 2*np.pi, num_categories, endpoint=False).tolist()

# Create subplots (2x2 grid of radar charts)
fig, axs = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
axs = axs.flatten()

# Normalize the data for radar chart
normalized_stats = {}
for category in categories:
    # For DRtg, lower is better, so invert the normalization
    if category == 'DRtg':
        min_val = cluster_stats_df[category].min()
        max_val = cluster_stats_df[category].max()
        normalized_stats[category] = [(max_val - cluster_stats_df.loc[cluster_labels[i], category]) / (max_val - min_val) for i in range(4)]
    else:
        min_val = cluster_stats_df[category].min()
        max_val = cluster_stats_df[category].max()
        normalized_stats[category] = [(cluster_stats_df.loc[cluster_labels[i], category] - min_val) / (max_val - min_val) for i in range(4)]

# Plot each cluster as a radar chart
for i in range(4):
    values = [normalized_stats[category][i] for category in categories]
    radar_chart(axs[i], angles, values, colors[i], cluster_labels[i])
    axs[i].set_title(f"{cluster_labels[i]}\n({int(cluster_stats_df.loc[cluster_labels[i], 'team_count'])} teams)",
                    fontsize=14, fontweight='bold', pad=20)

    # Add team names to each radar chart
    teams_text = f"Teams: {cluster_stats_df.loc[cluster_labels[i], 'teams']}"
    axs[i].text(0.5, -0.15, teams_text, transform=axs[i].transAxes,
               ha='center', va='center', fontsize=10, wrap=True)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.savefig('team_analysis/cluster_radar_charts.png', dpi=300, bbox_inches='tight')

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

    f.write("\n## Team Playing Style Clusters\n\n")
    f.write("Teams are clustered based on their playing style into four distinct categories:\n\n")
    f.write("1. **Offensive Juggernauts**: Teams with high scoring, efficient shooting (especially from three), and often average or below-average defense\n")
    f.write("2. **Defensive Stalwarts**: Teams with strong defensive metrics, lower pace, and efficient (though not necessarily high-volume) offense\n")
    f.write("3. **Interior Presence**: Teams that excel in rebounding (particularly offensive rebounding) and score frequently near the basket\n")
    f.write("4. **Balanced Teams**: Teams with a more balanced statistical profile, lacking extreme strengths or weaknesses\n\n")

    f.write("### Cluster Details\n\n")
    for cluster_name, row in cluster_stats_df.iterrows():
        f.write(f"**{cluster_name}** ({int(row['team_count'])} teams)\n\n")
        f.write(f"{row['description']}\n\n")
        f.write(f"**Teams**: {row['teams']}\n\n")
        f.write(f"**Key stats**:\n")
        f.write(f"- Points: {row['PTS']:.1f}\n")
        f.write(f"- Offensive Rating: {row['ORtg']:.1f}\n")
        f.write(f"- Defensive Rating: {row['DRtg']:.1f}\n")
        f.write(f"- Effective FG%: {row['eFGp']:.3f}\n")
        f.write(f"- 3-Point Attempt Rate: {row['3PAr']:.3f}\n")
        f.write(f"- Offensive Rebounding %: {row['ORBp']:.1f}\n")
        f.write(f"- Defensive Rebounding %: {row['DRBp']:.1f}\n")
        f.write(f"- Assist %: {row['ASTp']:.1f}\n")
        f.write(f"- Block %: {row['BLKp']:.1f}\n")
        f.write(f"- Steal %: {row['STLp']:.1f}\n\n")

