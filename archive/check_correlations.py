import pandas as pd
import numpy as np

# Load test data
test = pd.read_csv('processed_data/test_data.csv')

# Load feature names
with open('processed_data/feature_names.txt', 'r') as f:
    features = [line.strip() for line in f.readlines() if line.strip()]

# Extract features and target
X_test = test[features]
y_test = test['result']

# Calculate correlations
correlations = []
for feature in features:
    correlations.append((feature, np.corrcoef(X_test[feature], y_test)[0, 1]))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Print top 10 correlations
print("Top 10 feature correlations with target:")
for feature, corr in correlations[:10]:
    print(f'{feature}: {corr:.4f}')

# Check for perfect correlations between features
print("\nChecking for perfect correlations between features...")
perfect_corrs = []
for i, feature1 in enumerate(features):
    for feature2 in features[i+1:]:
        corr = np.corrcoef(X_test[feature1], X_test[feature2])[0, 1]
        if abs(corr) > 0.99:
            perfect_corrs.append((feature1, feature2, corr))

if perfect_corrs:
    print("Found perfect correlations between features:")
    for f1, f2, corr in perfect_corrs:
        print(f'{f1} and {f2}: {corr:.4f}')
else:
    print("No perfect correlations found between features.")

# Check if any feature or combination can perfectly predict the target
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train a simple model on each feature individually
print("\nChecking if any single feature can perfectly predict the target...")
for feature in features:
    X = X_test[[feature]].values
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y_test)
    y_pred = rf.predict(X)
    acc = accuracy_score(y_test, y_pred)
    if acc > 0.95:
        print(f'{feature}: {acc:.4f}')

# Check if the point differential is in the test data
if 'point_diff' in test.columns:
    print("\nChecking if point_diff is leaking information...")
    corr = np.corrcoef(test['point_diff'], y_test)[0, 1]
    print(f'Correlation between point_diff and result: {corr:.4f}')
    
    # Check if result can be derived from point_diff
    derived_result = (test['point_diff'] > 0).astype(int)
    match = (derived_result == y_test).mean()
    print(f'Match between derived result and actual result: {match:.4f}')
