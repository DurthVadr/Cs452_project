"""
Streak Impact Analysis

This script analyzes the impact of team performance streaks on game outcomes.
It visualizes how recent performance can be a stronger predictor than season-long statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def main():
    print("Analyzing impact of team performance streaks on game outcomes...")

    # Create output directories if they don't exist
    os.makedirs('team_analysis', exist_ok=True)
    os.makedirs('../team_analysis', exist_ok=True)

    # Define possible paths for the data files
    possible_paths = [
        'processed_data/combined_data.csv',
        'processed_data/games_with_streak.csv',
        '../processed_data/combined_data.csv',
        '../processed_data/games_with_streak.csv',
        '../a_beast/data/processed/combined_data.csv',
        '../a_project/data/processed/combined_data.csv'
    ]

    # Try to load the data from any of the possible paths
    combined_data = None
    for path in possible_paths:
        try:
            combined_data = pd.read_csv(path, index_col=0)
            print(f"Loaded data from {path} with {len(combined_data)} rows")
            break
        except FileNotFoundError:
            continue

    # If we couldn't load the data from any path
    if combined_data is None:
        print("Error: Could not find processed data with streak information.")
        print("Please run data_preparation.py first to generate the required data.")
        return

    # Ensure we have the necessary columns
    required_columns = ['away_streak', 'home_streak', 'result']
    if not all(col in combined_data.columns for col in required_columns):
        print("Error: The loaded data does not contain the required streak columns.")
        print(f"Available columns: {combined_data.columns.tolist()}")
        return

    # Analyze streak impact on game outcomes
    analyze_streak_impact(combined_data)

    # Analyze recent performance vs. season-long performance
    if 'away_last_n_win_pct' in combined_data.columns and 'home_last_n_win_pct' in combined_data.columns:
        analyze_recent_vs_season_performance(combined_data)
    else:
        print("Warning: Recent performance columns not found. Skipping recent vs. season performance analysis.")

    print("Analysis completed. Results saved to team_analysis directory.")

def analyze_streak_impact(data):
    """
    Analyze how team streaks impact game outcomes.

    Args:
        data: DataFrame with game data including streak information
    """
    print("Analyzing impact of streaks on game outcomes...")

    # Create a copy of the data
    df = data.copy()

    # Group home team streaks and calculate win percentage
    home_streak_results = df.groupby('home_streak').agg(
        games=('result', 'count'),
        home_wins=('result', 'sum'),
    ).reset_index()

    home_streak_results['win_pct'] = home_streak_results['home_wins'] / home_streak_results['games'] * 100

    # Group away team streaks and calculate win percentage
    away_streak_results = df.groupby('away_streak').agg(
        games=('result', 'count'),
        away_wins=('result', lambda x: (1 - x).sum()),  # Away win is when result = 0
    ).reset_index()

    away_streak_results['win_pct'] = away_streak_results['away_wins'] / away_streak_results['games'] * 100

    # Filter to include only streaks with sufficient sample size (at least 10 games)
    home_streak_results = home_streak_results[home_streak_results['games'] >= 10]
    away_streak_results = away_streak_results[away_streak_results['games'] >= 10]

    # Create visualization of home streak impact
    plt.figure(figsize=(14, 8))

    # Plot home streak impact
    plt.subplot(1, 2, 1)
    sns.barplot(x='home_streak', y='win_pct', data=home_streak_results)
    plt.title('Home Team Win % by Streak Length')
    plt.xlabel('Streak Length (+ = Winning, - = Losing)')
    plt.ylabel('Win Percentage (%)')
    plt.axhline(y=50, color='r', linestyle='--', label='50% (Even odds)')
    plt.grid(True)
    plt.legend()

    # Plot away streak impact
    plt.subplot(1, 2, 2)
    sns.barplot(x='away_streak', y='win_pct', data=away_streak_results)
    plt.title('Away Team Win % by Streak Length')
    plt.xlabel('Streak Length (+ = Winning, - = Losing)')
    plt.ylabel('Win Percentage (%)')
    plt.axhline(y=50, color='r', linestyle='--', label='50% (Even odds)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('../team_analysis', exist_ok=True)

    # Save to both potential locations
    plt.savefig('team_analysis/streak_impact_on_outcomes.png')
    plt.savefig('../team_analysis/streak_impact_on_outcomes.png')
    print("Saved streak impact visualization to team_analysis/streak_impact_on_outcomes.png")

    # Create a combined visualization showing streak differential impact
    analyze_streak_differential(df)

def analyze_streak_differential(data):
    """
    Analyze how the difference in team streaks impacts game outcomes.

    Args:
        data: DataFrame with game data including streak information
    """
    print("Analyzing impact of streak differential on game outcomes...")

    # Create a copy of the data
    df = data.copy()

    # Calculate streak differential (home streak - away streak)
    df['streak_diff'] = df['home_streak'] - df['away_streak']

    # Group by streak differential and calculate home win percentage
    streak_diff_results = df.groupby('streak_diff').agg(
        games=('result', 'count'),
        home_wins=('result', 'sum'),
    ).reset_index()

    streak_diff_results['win_pct'] = streak_diff_results['home_wins'] / streak_diff_results['games'] * 100

    # Filter to include only streak differentials with sufficient sample size (at least 5 games)
    streak_diff_results = streak_diff_results[streak_diff_results['games'] >= 5]

    # Create visualization
    plt.figure(figsize=(14, 8))

    # Plot streak differential impact
    bars = sns.barplot(x='streak_diff', y='win_pct', data=streak_diff_results)

    # Add game count as text on each bar
    for i, p in enumerate(bars.patches):
        game_count = streak_diff_results.iloc[i]['games']
        bars.annotate(f'n={game_count}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'bottom',
                     xytext = (0, 5), textcoords = 'offset points')

    plt.title('Home Team Win % by Streak Differential (Home Streak - Away Streak)')
    plt.xlabel('Streak Differential')
    plt.ylabel('Home Team Win Percentage (%)')
    plt.axhline(y=50, color='r', linestyle='--', label='50% (Even odds)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('team_analysis/streak_differential_impact.png')
    plt.savefig('../team_analysis/streak_differential_impact.png')
    print("Saved streak differential visualization to team_analysis directories")

def analyze_recent_vs_season_performance(data):
    """
    Analyze how recent performance compares to season-long performance as a predictor.

    Args:
        data: DataFrame with game data including recent performance metrics
    """
    print("Analyzing recent performance vs. season-long performance...")

    # Create a copy of the data
    df = data.copy()

    # Calculate recent performance differential (home - away)
    df['recent_perf_diff'] = df['home_last_n_win_pct'] - df['away_last_n_win_pct']

    # Create bins for recent performance differential
    df['recent_perf_diff_bin'] = pd.cut(
        df['recent_perf_diff'],
        bins=[-1, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 1],
        labels=['-0.4 to -1', '-0.2 to -0.4', '-0.1 to -0.2', '0 to -0.1',
                '0 to 0.1', '0.1 to 0.2', '0.2 to 0.4', '0.4 to 1']
    )

    # Group by recent performance differential bin and calculate home win percentage
    recent_perf_results = df.groupby('recent_perf_diff_bin').agg(
        games=('result', 'count'),
        home_wins=('result', 'sum'),
    ).reset_index()

    recent_perf_results['win_pct'] = recent_perf_results['home_wins'] / recent_perf_results['games'] * 100

    # Create visualization
    plt.figure(figsize=(14, 8))

    # Plot recent performance differential impact
    bars = sns.barplot(x='recent_perf_diff_bin', y='win_pct', data=recent_perf_results)

    # Add game count as text on each bar
    for i, p in enumerate(bars.patches):
        game_count = recent_perf_results.iloc[i]['games']
        bars.annotate(f'n={game_count}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'bottom',
                     xytext = (0, 5), textcoords = 'offset points')

    plt.title('Home Team Win % by Recent Performance Differential')
    plt.xlabel('Recent Win % Differential (Home - Away)')
    plt.ylabel('Home Team Win Percentage (%)')
    plt.axhline(y=50, color='r', linestyle='--', label='50% (Even odds)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('team_analysis/recent_performance_impact.png')
    plt.savefig('../team_analysis/recent_performance_impact.png')
    print("Saved recent performance visualization to team_analysis directories")

if __name__ == "__main__":
    main()
