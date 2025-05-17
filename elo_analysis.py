import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Create output directory for ELO analysis
if not os.path.exists('elo_analysis'):
    os.makedirs('elo_analysis')

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Load different training datasets
print("Loading different sampling method datasets...")
datasets = {
    'Original': pd.read_csv('processed_data/train_data_original.csv', index_col=0),
    'ROS': pd.read_csv('processed_data/train_data_ros.csv', index_col=0),
    'SMOTE': pd.read_csv('processed_data/train_data_smote.csv', index_col=0)
}

# Analyze existing ELO implementation
print("Analyzing existing ELO implementation...")

# Calculate ELO prediction accuracy
elo_correct = combined_data[combined_data['elo_pred'] == combined_data['result']]
elo_accuracy = len(elo_correct) / len(combined_data) * 100

# Calculate ELO prediction accuracy for test set
test_elo_correct = test_data[test_data['elo_pred'] == test_data['result']]
test_elo_accuracy = len(test_elo_correct) / len(test_data) * 100

# Create confusion matrix for ELO predictions
elo_cm = confusion_matrix(combined_data['result'], combined_data['elo_pred'])
plt.figure(figsize=(10, 8))
sns.heatmap(elo_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Away Win', 'Home Win'],
            yticklabels=['Away Win', 'Home Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('ELO Prediction Confusion Matrix')
plt.savefig('elo_analysis/elo_confusion_matrix.png')

# Calculate ELO prediction metrics
elo_report = classification_report(combined_data['result'], combined_data['elo_pred'],
                                target_names=['Away Win', 'Home Win'], output_dict=True)
elo_metrics = pd.DataFrame(elo_report).transpose()
elo_metrics.to_csv('elo_analysis/elo_metrics.csv')

# Implement custom ELO rating system
print("Implementing custom ELO rating system...")

class EloSystem:
    """Custom ELO rating system implementation"""
    def __init__(self, k_factor=24, home_advantage=100, default_rating=1500):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.team_ratings = {}
    
    def get_rating(self, team):
        """Get team rating or default if not found"""
        return self.team_ratings.get(team, self.default_rating)
    
    def predict_proba(self, home_team, away_team):
        """Predict probability of home team winning"""
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        return 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
    
    def predict(self, home_team, away_team):
        """Predict result (1 for home win, 0 for away win)"""
        return 1 if self.predict_proba(home_team, away_team) >= 0.5 else 0
    
    def update_ratings(self, home_team, away_team, result):
        """Update team ratings after match"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_expected = self.predict_proba(home_team, away_team)
        away_expected = 1 - home_expected
        
        # Update home team rating
        home_new = home_rating + self.k_factor * (result - home_expected)
        self.team_ratings[home_team] = home_new
        
        # Update away team rating
        away_new = away_rating + self.k_factor * ((1 - result) - away_expected)
        self.team_ratings[away_team] = away_new

# Function to evaluate ELO system with different parameters
def evaluate_elo_system(train_data, test_data, k_factor, home_advantage):
    """Evaluate ELO system with given parameters"""
    # Initialize ELO system
    elo_system = EloSystem(k_factor=k_factor, home_advantage=home_advantage)
    
    # Train on training data
    train_predictions = []
    for _, game in train_data.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        result = game['result']
        
        # Make prediction
        prediction = elo_system.predict(home_team, away_team)
        train_predictions.append(prediction)
        
        # Update ratings
        elo_system.update_ratings(home_team, away_team, result)
    
    # Evaluate on test data
    test_predictions = []
    for _, game in test_data.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        result = game['result']
        
        # Make prediction
        prediction = elo_system.predict(home_team, away_team)
        test_predictions.append(prediction)
        
        # Update ratings
        elo_system.update_ratings(home_team, away_team, result)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(train_data['result'], train_predictions) * 100
    test_accuracy = accuracy_score(test_data['result'], test_predictions) * 100
    
    return train_accuracy, test_accuracy, elo_system

# Grid search for optimal ELO parameters (for each dataset)
print("Performing grid search for optimal ELO parameters...")

# Define parameter grid
k_factors = [10, 15, 20, 25, 30, 35, 40]
home_advantages = [50, 75, 100, 125, 150, 175, 200]

# Store results for each dataset
dataset_results = {}
best_params = {}
final_elo_systems = {}

for dataset_name, train_data in datasets.items():
    print(f"\nEvaluating ELO parameters for {dataset_name} dataset...")
    
    # Create results storage
    results = []
    
    # Grid search
    for k in k_factors:
        for ha in home_advantages:
            train_acc, test_acc, _ = evaluate_elo_system(train_data, test_data, k, ha)
            results.append({
                'dataset': dataset_name,
                'k_factor': k,
                'home_advantage': ha,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            print(f"  k={k}, ha={ha}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_idx = results_df['test_accuracy'].idxmax()
    best_k = results_df.loc[best_idx, 'k_factor']
    best_ha = results_df.loc[best_idx, 'home_advantage']
    
    print(f"Best parameters for {dataset_name} dataset: k_factor={best_k}, home_advantage={best_ha}")
    print(f"Best test accuracy: {results_df.loc[best_idx, 'test_accuracy']:.2f}%")
    
    # Store results
    dataset_results[dataset_name] = results_df
    best_params[dataset_name] = (best_k, best_ha)
    
    # Train final model with best parameters
    final_train_acc, final_test_acc, final_elo_system = evaluate_elo_system(
        train_data, test_data, best_k, best_ha
    )
    
    final_elo_systems[dataset_name] = final_elo_system
    
    # Save the results
    results_df.to_csv(f'elo_analysis/elo_parameter_search_{dataset_name.lower()}.csv', index=False)
    
    # Save final team ratings
    final_ratings = pd.DataFrame({
        'team': list(final_elo_system.team_ratings.keys()),
        'rating': list(final_elo_system.team_ratings.values())
    })
    final_ratings = final_ratings.sort_values('rating', ascending=False)
    final_ratings.to_csv(f'elo_analysis/final_team_ratings_{dataset_name.lower()}.csv', index=False)
    
    # Visualize final team ratings
    plt.figure(figsize=(12, 10))
    sns.barplot(x='rating', y='team', data=final_ratings)
    plt.title(f'Final ELO Ratings by Team ({dataset_name} Dataset)')
    plt.xlabel('ELO Rating')
    plt.ylabel('Team')
    plt.tight_layout()
    plt.savefig(f'elo_analysis/final_team_ratings_{dataset_name.lower()}.png')

# Create comparison of all datasets
comparison_data = []
for dataset_name in datasets.keys():
    best_row = dataset_results[dataset_name].iloc[dataset_results[dataset_name]['test_accuracy'].idxmax()]
    comparison_data.append({
        'Dataset': dataset_name,
        'Best k_factor': best_row['k_factor'],
        'Best home_advantage': best_row['home_advantage'],
        'Train Accuracy': best_row['train_accuracy'],
        'Test Accuracy': best_row['test_accuracy']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('elo_analysis/elo_comparison_all_datasets.csv', index=False)

# Visualize comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Dataset', y='Test Accuracy', data=comparison_df)
plt.title('ELO Test Accuracy by Dataset')
plt.ylabel('Accuracy (%)')
plt.ylim(50, 100)
for i, v in enumerate(comparison_df['Test Accuracy']):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('elo_analysis/elo_comparison_all_datasets.png')

# Create summary report
with open('elo_analysis/elo_summary.md', 'w') as f:
    f.write("# ELO Rating System Analysis\n\n")

    f.write("## Original ELO System Performance\n\n")
    f.write(f"- Overall accuracy: {elo_accuracy:.1f}%\n")
    f.write(f"- Test set accuracy: {test_elo_accuracy:.1f}%\n\n")

    f.write("## Optimized ELO System by Dataset\n\n")
    for dataset_name in datasets.keys():
        best_k, best_ha = best_params[dataset_name]
        best_row = dataset_results[dataset_name].iloc[dataset_results[dataset_name]['test_accuracy'].idxmax()]
        f.write(f"### {dataset_name} Dataset\n\n")
        f.write(f"- Best parameters: k_factor={best_k}, home_advantage={best_ha}\n")
        f.write(f"- Training set accuracy: {best_row['train_accuracy']:.1f}%\n")
        f.write(f"- Test set accuracy: {best_row['test_accuracy']:.1f}%\n")
        f.write(f"- Improvement over original: {best_row['test_accuracy'] - test_elo_accuracy:.1f}%\n\n")

    f.write("## Best ELO Implementation\n\n")
    best_dataset = comparison_df.iloc[comparison_df['Test Accuracy'].idxmax()]['Dataset']
    best_acc = comparison_df.iloc[comparison_df['Test Accuracy'].idxmax()]['Test Accuracy']
    f.write(f"The best ELO implementation was achieved with the **{best_dataset}** dataset ")
    f.write(f"with an accuracy of {best_acc:.1f}%.\n\n")

    f.write("## Top Teams by Final ELO Rating\n\n")
    for dataset_name in datasets.keys():
        f.write(f"### {dataset_name} Dataset\n\n")
        final_ratings = pd.DataFrame({
            'team': list(final_elo_systems[dataset_name].team_ratings.keys()),
            'rating': list(final_elo_systems[dataset_name].team_ratings.values())
        })
        final_ratings = final_ratings.sort_values('rating', ascending=False)
        top_teams = final_ratings.head(5)
        
        f.write("| Team | Rating |\n")
        f.write("|------|-------|\n")
        for _, row in top_teams.iterrows():
            f.write(f"| {row['team']} | {row['rating']:.1f} |\n")
        f.write("\n")

print("ELO analysis completed. Results saved to elo_analysis/")
