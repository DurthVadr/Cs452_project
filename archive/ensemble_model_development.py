import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

# Create output directory for ensemble model
if not os.path.exists('ensemble_model'):
    os.makedirs('ensemble_model')

# Load the processed data
print("Loading processed data...")
combined_data = pd.read_csv('processed_data/combined_data.csv', index_col=0)
train_data = pd.read_csv('processed_data/train_data.csv', index_col=0)
test_data = pd.read_csv('processed_data/test_data.csv', index_col=0)

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

# Define upsets based on ELO ratings
print("Identifying upsets based on ELO ratings...")
combined_data['favorite'] = np.where(combined_data['home_elo_i'] > combined_data['away_elo_i'], 1, 0)
combined_data['upset'] = np.where(combined_data['favorite'] != combined_data['result'], 1, 0)

train_data['favorite'] = np.where(train_data['home_elo_i'] > train_data['away_elo_i'], 1, 0)
train_data['upset'] = np.where(train_data['favorite'] != train_data['result'], 1, 0)

test_data['favorite'] = np.where(test_data['home_elo_i'] > test_data['away_elo_i'], 1, 0)
test_data['upset'] = np.where(test_data['favorite'] != test_data['result'], 1, 0)

# Calculate upset rate
upset_rate = combined_data['upset'].mean() * 100
print(f"Upset rate in the dataset: {upset_rate:.1f}%")

# Approach 1: Train separate models for upset and non-upset games
print("\nApproach 1: Training separate models for upset and non-upset games...")

# Split data for non-upset games
non_upset_train = train_data[train_data['upset'] == 0]
X_train_non_upset = non_upset_train[features]
y_train_non_upset = non_upset_train['result']

# Scale features for non-upset games
scaler_non_upset = StandardScaler()
X_train_non_upset_scaled = scaler_non_upset.fit_transform(X_train_non_upset)

# Split data for upset games
upset_train = train_data[train_data['upset'] == 1]
X_train_upset = upset_train[features]
y_train_upset = upset_train['result']

# Scale features for upset games
scaler_upset = StandardScaler()
X_train_upset_scaled = scaler_upset.fit_transform(X_train_upset)

# Train model for non-upset games
non_upset_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
non_upset_model.fit(X_train_non_upset_scaled, y_train_non_upset)

# Train model for upset games
upset_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
upset_model.fit(X_train_upset_scaled, y_train_upset)

# Save models
joblib.dump(non_upset_model, 'ensemble_model/non_upset_model.pkl')
joblib.dump(upset_model, 'ensemble_model/upset_model.pkl')
joblib.dump(scaler_non_upset, 'ensemble_model/scaler_non_upset.pkl')
joblib.dump(scaler_upset, 'ensemble_model/scaler_upset.pkl')

# Evaluate on test set
print("Evaluating separate models approach...")

# Predict using the appropriate model based on whether the game is expected to be an upset
test_predictions = []
for i, row in test_data.iterrows():
    # Get features for this game
    game_features = np.array(row[features]).reshape(1, -1)

    # Determine if this is likely to be an upset game
    home_team_favored = row['home_elo_i'] > row['away_elo_i']
    elo_diff = abs(row['home_elo_i'] - row['away_elo_i'])

    # Calculate upset probability based on ELO difference
    # The smaller the ELO difference, the higher the upset probability
    upset_prob = np.exp(-0.005 * elo_diff)

    # Use upset threshold based on the dataset's upset rate
    upset_threshold = 0.5  # Adjust this threshold based on validation

    if upset_prob > upset_threshold:
        # This game has high upset potential, use upset model
        scaled_features = scaler_upset.transform(game_features)
        prediction = upset_model.predict(scaled_features)[0]
    else:
        # This is likely a non-upset game, use non-upset model
        scaled_features = scaler_non_upset.transform(game_features)
        prediction = non_upset_model.predict(scaled_features)[0]

    test_predictions.append(prediction)

# Calculate accuracy
approach1_accuracy = accuracy_score(y_test, test_predictions)
print(f"Approach 1 Accuracy: {approach1_accuracy:.4f}")

# Approach 2: Develop an upset prediction model and combine with main model
print("\nApproach 2: Developing an upset prediction model and combining with main model...")

# Train a model to predict whether a game will be an upset
X_train_upset_pred = train_data[features]
y_train_upset_pred = train_data['upset']

# Scale features
scaler_upset_pred = StandardScaler()
X_train_upset_pred_scaled = scaler_upset_pred.fit_transform(X_train_upset_pred)

# Train upset prediction model
upset_prediction_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
upset_prediction_model.fit(X_train_upset_pred_scaled, y_train_upset_pred)

# Train main outcome prediction model
main_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
main_model.fit(X_train_scaled, y_train)

# Save models
joblib.dump(upset_prediction_model, 'ensemble_model/upset_prediction_model.pkl')
joblib.dump(main_model, 'ensemble_model/main_model.pkl')
joblib.dump(scaler_upset_pred, 'ensemble_model/scaler_upset_pred.pkl')
joblib.dump(scaler, 'ensemble_model/scaler_main.pkl')

# Evaluate on test set
print("Evaluating combined model approach...")

# First predict if each game will be an upset
X_test_scaled_upset = scaler_upset_pred.transform(X_test)
upset_predictions = upset_prediction_model.predict(X_test_scaled_upset)

# Then predict the outcome using the main model
main_predictions = main_model.predict(X_test_scaled)

# Combine predictions
combined_predictions = []
for i in range(len(y_test)):
    if upset_predictions[i] == 1:
        # This is predicted to be an upset
        # If favorite is home team (1), predict away team wins (0)
        # If favorite is away team (0), predict home team wins (1)
        combined_predictions.append(1 - test_data.iloc[i]['favorite'])
    else:
        # Not an upset, use main model prediction
        combined_predictions.append(main_predictions[i])

# Calculate accuracy
approach2_accuracy = accuracy_score(y_test, combined_predictions)
print(f"Approach 2 Accuracy: {approach2_accuracy:.4f}")

# Approach 3: Ensemble model with weighted voting
print("\nApproach 3: Creating an ensemble model with weighted voting...")

# Train multiple models with different strengths
model1 = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model3 = LogisticRegression(
    C=0.001,
    penalty='l2',
    solver='liblinear',
    random_state=42
)

# Create an ensemble model with weighted voting
ensemble_model = VotingClassifier(
    estimators=[
        ('gb', model1),
        ('rf', model2),
        ('lr', model3)
    ],
    voting='soft',  # Use probability estimates for voting
    weights=[2, 1, 1]  # Give more weight to the best model (Gradient Boosting)
)

# Train the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Save the ensemble model
joblib.dump(ensemble_model, 'ensemble_model/ensemble_model.pkl')

# Evaluate on test set
ensemble_predictions = ensemble_model.predict(X_test_scaled)
approach3_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Approach 3 Accuracy: {approach3_accuracy:.4f}")

# Approach 4: Meta-model approach (stacking)
print("\nApproach 4: Implementing a meta-model approach (stacking)...")

# Train base models
base_models = [
    GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=3, min_samples_split=5, random_state=42),
    RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42),
    LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=42)
]

# Split training data for meta-model
from sklearn.model_selection import KFold

# Generate predictions from base models using k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    # Train the model on the full training set and predict on test set
    model.fit(X_train_scaled, y_train)
    meta_features_test[:, i] = model.predict_proba(X_test_scaled)[:, 1]

    # Use cross-validation to generate out-of-fold predictions for training meta-model
    for train_idx, val_idx in kf.split(X_train_scaled):
        # Split data
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train model on training fold
        model.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        meta_features_train[val_idx, i] = model.predict_proba(X_val_fold)[:, 1]

# Add original features to meta-features
meta_features_train_with_orig = np.hstack((meta_features_train, X_train_scaled))
meta_features_test_with_orig = np.hstack((meta_features_test, X_test_scaled))

# Train meta-model
meta_model = LogisticRegression(C=0.1, random_state=42)
meta_model.fit(meta_features_train_with_orig, y_train)

# Save meta-model and base models
joblib.dump(meta_model, 'ensemble_model/meta_model.pkl')
for i, model in enumerate(base_models):
    joblib.dump(model, f'ensemble_model/base_model_{i}.pkl')

# Evaluate meta-model
meta_predictions = meta_model.predict(meta_features_test_with_orig)
approach4_accuracy = accuracy_score(y_test, meta_predictions)
print(f"Approach 4 Accuracy: {approach4_accuracy:.4f}")

# Approach 5: Adaptive ensemble with upset-specific features
print("\nApproach 5: Creating an adaptive ensemble with upset-specific features...")

# Create additional features specifically for upset detection
train_data['elo_diff_abs'] = abs(train_data['home_elo_i'] - train_data['away_elo_i'])
train_data['favorite_back_to_back'] = np.where(
    train_data['favorite'] == 1,
    train_data['home_back_to_back'],
    train_data['away_back_to_back']
)
train_data['underdog_back_to_back'] = np.where(
    train_data['favorite'] == 0,
    train_data['home_back_to_back'],
    train_data['away_back_to_back']
)

test_data['elo_diff_abs'] = abs(test_data['home_elo_i'] - test_data['away_elo_i'])
test_data['favorite_back_to_back'] = np.where(
    test_data['favorite'] == 1,
    test_data['home_back_to_back'],
    test_data['away_back_to_back']
)
test_data['underdog_back_to_back'] = np.where(
    test_data['favorite'] == 0,
    test_data['home_back_to_back'],
    test_data['away_back_to_back']
)

# Enhanced features for upset prediction
upset_features = features + ['elo_diff_abs', 'favorite_back_to_back', 'underdog_back_to_back']

# Prepare data for upset prediction model
X_train_upset_enhanced = train_data[upset_features]
y_train_upset_enhanced = train_data['upset']

X_test_upset_enhanced = test_data[upset_features]

# Scale features
scaler_upset_enhanced = StandardScaler()
X_train_upset_enhanced_scaled = scaler_upset_enhanced.fit_transform(X_train_upset_enhanced)
X_test_upset_enhanced_scaled = scaler_upset_enhanced.transform(X_test_upset_enhanced)

# Train enhanced upset prediction model
upset_model_enhanced = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=4,  # Slightly deeper trees for more complex patterns
    min_samples_split=5,
    random_state=42
)
upset_model_enhanced.fit(X_train_upset_enhanced_scaled, y_train_upset_enhanced)

# Train regular outcome prediction model
regular_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
regular_model.fit(X_train_scaled, y_train)

# Save models
joblib.dump(upset_model_enhanced, 'ensemble_model/upset_model_enhanced.pkl')
joblib.dump(regular_model, 'ensemble_model/regular_model.pkl')
joblib.dump(scaler_upset_enhanced, 'ensemble_model/scaler_upset_enhanced.pkl')

# Predict upsets with enhanced model
upset_proba = upset_model_enhanced.predict_proba(X_test_upset_enhanced_scaled)[:, 1]

# Predict outcomes with regular model
regular_proba = regular_model.predict_proba(X_test_scaled)[:, 1]

# Adaptive combination based on upset probability
adaptive_predictions = []
for i in range(len(y_test)):
    upset_probability = upset_proba[i]

    if upset_probability > 0.5:  # High chance of upset
        # If favorite is home team (1), predict away team wins (0)
        # If favorite is away team (0), predict home team wins (1)
        adaptive_predictions.append(1 - test_data.iloc[i]['favorite'])
    else:
        # Use regular model prediction
        adaptive_predictions.append(1 if regular_proba[i] > 0.5 else 0)

# Calculate accuracy
approach5_accuracy = accuracy_score(y_test, adaptive_predictions)
print(f"Approach 5 Accuracy: {approach5_accuracy:.4f}")

# Compare all approaches
print("\nComparison of all approaches:")
approaches = {
    "Approach 1: Separate models": approach1_accuracy,
    "Approach 2: Upset prediction + main model": approach2_accuracy,
    "Approach 3: Weighted voting ensemble": approach3_accuracy,
    "Approach 4: Meta-model (stacking)": approach4_accuracy,
    "Approach 5: Adaptive ensemble with upset features": approach5_accuracy
}

# Find the best approach
best_approach = max(approaches.items(), key=lambda x: x[1])
print(f"Best approach: {best_approach[0]} with accuracy {best_approach[1]:.4f}")

# Calculate improvement over original model
original_accuracy = 0.6707  # From previous model building
improvement = (best_approach[1] - original_accuracy) * 100
print(f"Improvement over original model: {improvement:.2f}%")

# Create visualization of approach comparison
plt.figure(figsize=(12, 8))
approaches_df = pd.DataFrame({
    'Accuracy': approaches.values()
}, index=approaches.keys())
approaches_df = approaches_df.sort_values('Accuracy', ascending=False)

sns.barplot(x='Accuracy', y=approaches_df.index, data=approaches_df)
plt.title('Comparison of Ensemble Approaches', fontsize=16)
plt.xlabel('Accuracy', fontsize=14)
plt.xlim(0.65, 0.75)  # Adjust as
