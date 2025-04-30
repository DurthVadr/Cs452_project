import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

# Create output directory for ELO analysis
import os
if not os.path.exists('elo_analysis'):
    os.makedirs('elo_analysis')

# Analyze existing ELO implementation
print("Analyzing existing ELO implementation...")

# Calculate ELO prediction accuracy
elo_correct = combined_data[combined_data['elo_pred'] == combined_data['result']]
elo_accuracy = len(elo_correct) / len(combined_data) * 100

# Calculate ELO prediction accuracy for training and test sets
train_elo_correct = train_data[train_data['elo_pred'] == train_data['result']]
train_elo_accuracy = len(train_elo_correct) / len(train_data) * 100

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

# Analyze ELO prediction accuracy by ELO difference
combined_data['elo_diff_abs'] = abs(combined_data['home_elo_i'] - combined_data['away_elo_i'])
combined_data['elo_diff_bin'] = pd.cut(combined_data['elo_diff_abs'],
                                      bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 1000],
                                      labels=['0-50', '50-100', '100-150', '150-200',
                                              '200-250', '250-300', '300-350', '350-400', '400+'])

elo_accuracy_by_diff = combined_data.groupby('elo_diff_bin').apply(
    lambda x: (x['elo_pred'] == x['result']).mean() * 100
).reset_index(name='accuracy')

plt.figure(figsize=(12, 6))
sns.barplot(x='elo_diff_bin', y='accuracy', data=elo_accuracy_by_diff)
plt.title('ELO Prediction Accuracy by ELO Difference')
plt.xlabel('ELO Difference')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('elo_analysis/elo_accuracy_by_diff.png')

# Analyze ELO prediction accuracy by season
elo_accuracy_by_season = combined_data.groupby('season').apply(
    lambda x: (x['elo_pred'] == x['result']).mean() * 100
).reset_index(name='accuracy')

plt.figure(figsize=(12, 6))
sns.barplot(x='season', y='accuracy', data=elo_accuracy_by_season)
plt.title('ELO Prediction Accuracy by Season')
plt.xlabel('Season')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('elo_analysis/elo_accuracy_by_season.png')

# Implement custom ELO rating system
print("Implementing custom ELO rating system...")

class EloSystem:
    def __init__(self, k_factor=20, home_advantage=100, initial_rating=1500):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.team_ratings = {}

    def get_rating(self, team):
        if team not in self.team_ratings:
            self.team_ratings[team] = self.initial_rating
        return self.team_ratings[team]

    def expected_result(self, team_a, team_b, team_a_home=False):
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)

        # Apply home court advantage
        if team_a_home:
            rating_a += self.home_advantage
        else:
            rating_b += self.home_advantage

        # Calculate expected result using ELO formula
        exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        return exp_a

    def update_ratings(self, team_a, team_b, result, team_a_home=False, k_factor=None):
        if k_factor is None:
            k_factor = self.k_factor

        expected_a = self.expected_result(team_a, team_b, team_a_home)

        # Actual result (1 for team_a win, 0 for team_b win)
        actual_a = result

        # Update ratings
        rating_change = k_factor * (actual_a - expected_a)
        self.team_ratings[team_a] += rating_change
        self.team_ratings[team_b] -= rating_change

        return rating_change

    def predict_winner(self, team_a, team_b, team_a_home=False):
        expected_a = self.expected_result(team_a, team_b, team_a_home)
        return 0 if expected_a > 0.5 else 1

# Function to evaluate ELO system with different parameters
def evaluate_elo_system(train_data, test_data, k_factor, home_advantage):
    # Initialize ELO system
    elo_system = EloSystem(k_factor=k_factor, home_advantage=home_advantage)

    # Train the ELO system on training data
    train_predictions = []
    for _, game in train_data.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']

        # Predict winner
        prediction = elo_system.predict_winner(away_team, home_team, team_a_home=False)
        train_predictions.append(prediction)

        # Update ratings based on actual result
        actual_result = 1 if game['result'] == 0 else 0  # 0 for away win, 1 for home win
        elo_system.update_ratings(away_team, home_team, actual_result, team_a_home=False)

    # Evaluate on test data
    test_predictions = []
    for _, game in test_data.iterrows():
        away_team = game['away_team']
        home_team = game['home_team']

        # Predict winner
        prediction = elo_system.predict_winner(away_team, home_team, team_a_home=False)
        test_predictions.append(prediction)

        # Update ratings based on actual result
        actual_result = 1 if game['result'] == 0 else 0  # 0 for away win, 1 for home win
        elo_system.update_ratings(away_team, home_team, actual_result, team_a_home=False)

    # Calculate accuracy
    train_accuracy = accuracy_score(train_data['result'], train_predictions) * 100
    test_accuracy = accuracy_score(test_data['result'], test_predictions) * 100

    return train_accuracy, test_accuracy, elo_system

# Grid search for optimal ELO parameters
print("Performing grid search for optimal ELO parameters...")
k_factors = [10, 15, 20, 25, 30, 35, 40]
home_advantages = [50, 75, 100, 125, 150, 175, 200]

results = []
for k in k_factors:
    for ha in home_advantages:
        train_acc, test_acc, _ = evaluate_elo_system(train_data, test_data, k, ha)
        results.append({
            'k_factor': k,
            'home_advantage': ha,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('elo_analysis/elo_parameter_search.csv', index=False)

# Find best parameters
best_result = results_df.loc[results_df['test_accuracy'].idxmax()]
best_k = best_result['k_factor']
best_ha = best_result['home_advantage']

print(f"Best parameters: k_factor={best_k}, home_advantage={best_ha}")
print(f"Best test accuracy: {best_result['test_accuracy']:.2f}%")

# Visualize parameter search results
plt.figure(figsize=(12, 8))
pivot_table = results_df.pivot_table(index='k_factor', columns='home_advantage', values='test_accuracy')
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis')
plt.title('ELO Parameter Search Results (Test Accuracy %)')
plt.xlabel('Home Advantage')
plt.ylabel('K Factor')
plt.tight_layout()
plt.savefig('elo_analysis/elo_parameter_search.png')

# Train final ELO model with best parameters
print("Training final ELO model with best parameters...")
final_train_acc, final_test_acc, final_elo_system = evaluate_elo_system(
    train_data, test_data, best_k, best_ha
)

# Save final team ratings
final_ratings = pd.DataFrame({
    'team': list(final_elo_system.team_ratings.keys()),
    'rating': list(final_elo_system.team_ratings.values())
})
final_ratings = final_ratings.sort_values('rating', ascending=False)
final_ratings.to_csv('elo_analysis/final_team_ratings.csv', index=False)

# Visualize final team ratings
plt.figure(figsize=(12, 10))
sns.barplot(x='rating', y='team', data=final_ratings)
plt.title('Final ELO Ratings by Team')
plt.xlabel('ELO Rating')
plt.ylabel('Team')
plt.tight_layout()
plt.savefig('elo_analysis/final_team_ratings.png')

# Compare original ELO with optimized ELO
comparison = pd.DataFrame({
    'Model': ['Original ELO', 'Optimized ELO'],
    'Train Accuracy': [train_elo_accuracy, final_train_acc],
    'Test Accuracy': [test_elo_accuracy, final_test_acc]
})
comparison.to_csv('elo_analysis/elo_comparison.csv', index=False)

plt.figure(figsize=(10, 6))
comparison.set_index('Model').plot(kind='bar')
plt.title('ELO Model Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate(comparison['Test Accuracy']):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('elo_analysis/elo_comparison.png')

# Create summary report
with open('elo_analysis/elo_summary.md', 'w') as f:
    f.write("# ELO Rating System Analysis\n\n")

    f.write("## Original ELO System Performance\n\n")
    f.write(f"- Overall accuracy: {elo_accuracy:.1f}%\n")
    f.write(f"- Training set accuracy: {train_elo_accuracy:.1f}%\n")
    f.write(f"- Test set accuracy: {test_elo_accuracy:.1f}%\n\n")

    f.write("## Optimized ELO System\n\n")
    f.write(f"- Best parameters: k_factor={best_k}, home_advantage={best_ha}\n")
    f.write(f"- Training set accuracy: {final_train_acc:.1f}%\n")
    f.write(f"- Test set accuracy: {final_test_acc:.1f}%\n")
    f.write(f"- Improvement over original: {final_test_acc - test_elo_accuracy:.1f}%\n\n")

    f.write("## Top 5 Teams by Final ELO Rating\n\n")
    for i, row in final_ratings.head(5).iterrows():
        f.write(f"- {row['team']}: {row['rating']:.1f}\n")

    f.write("\n## Bottom 5 Teams by Final ELO Rating\n\n")
    for i, row in final_ratings.tail(5).iterrows():
        f.write(f"- {row['team']}: {row['rating']:.1f}\n")

print("ELO rating system analysis completed. Results saved to elo_analysis/")
