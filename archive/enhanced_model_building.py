"""
Enhanced model building script for the NBA prediction project.

This script builds and evaluates machine learning models for predicting NBA game outcomes
using advanced data augmentation techniques:
1. SMOTE for class balancing
2. Data generation for creating additional synthetic examples
3. Advanced ensemble techniques
4. Evaluation on real test data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
import joblib
import os
import time
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN

# Import utility modules
from utils.logging_config import setup_logger
from utils.common import ensure_directory_exists, load_processed_data, load_feature_names

# Set up logger
logger = setup_logger("enhanced_model", f"logs/enhanced_model_{datetime.now().strftime('%Y%m%d')}.log")
logger.info("Starting enhanced model building")

# Set plot style
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def load_data():
    """Load and prepare data for model building."""
    logger.info("Loading processed data")
    
    try:
        # Load processed data
        combined_data, train_data, test_data = load_processed_data()
        features = load_feature_names()
        
        # Create X and y for training and testing
        X_train = train_data[features]
        y_train = train_data['result']
        X_test = test_data[features]
        y_test = test_data['result']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Log class distribution before augmentation
        class_dist_before = pd.Series(y_train).value_counts(normalize=True)
        logger.info(f"Class distribution before augmentation: {class_dist_before.to_dict()}")
        
        # Identify upsets in the training data
        train_data['favorite'] = np.where(train_data['home_elo_i'] > train_data['away_elo_i'], 1, 0)
        train_data['upset'] = np.where(train_data['favorite'] != train_data['result'], 1, 0)
        
        # Calculate upset rate
        upset_rate = train_data['upset'].mean() * 100
        logger.info(f"Upset rate in the training dataset: {upset_rate:.1f}%")
        
        logger.info(f"Data loaded successfully. Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def generate_synthetic_data(X_train_scaled, y_train, train_data, features):
    """Generate synthetic data using multiple techniques."""
    logger.info("Generating synthetic data")
    
    # 1. Apply SMOTE for class balancing
    logger.info("Applying SMOTE for class balancing")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Log SMOTE results
    smote_dist = pd.Series(y_smote).value_counts(normalize=True)
    logger.info(f"Class distribution after SMOTE: {smote_dist.to_dict()}")
    logger.info(f"Data shape after SMOTE: {X_smote.shape}")
    
    # 2. Apply ADASYN for additional synthetic examples
    logger.info("Applying ADASYN for additional synthetic examples")
    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
    
    # Log ADASYN results
    adasyn_dist = pd.Series(y_adasyn).value_counts(normalize=True)
    logger.info(f"Class distribution after ADASYN: {adasyn_dist.to_dict()}")
    logger.info(f"Data shape after ADASYN: {X_adasyn.shape}")
    
    # 3. Generate synthetic upset examples based on ELO differences
    logger.info("Generating synthetic upset examples")
    
    # Identify upsets in the training data
    upset_indices = train_data[train_data['upset'] == 1].index
    upset_X = X_train_scaled[train_data.index.isin(upset_indices)]
    upset_y = y_train[train_data.index.isin(upset_indices)]
    
    # Create noise to add to upset examples
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, upset_X.shape)
    upset_X_noisy = upset_X + noise
    
    # Combine all synthetic data
    X_combined = np.vstack([X_smote, X_adasyn, upset_X_noisy])
    y_combined = np.concatenate([y_smote, y_adasyn, upset_y])
    
    # Log combined results
    combined_dist = pd.Series(y_combined).value_counts(normalize=True)
    logger.info(f"Class distribution after all augmentation: {combined_dist.to_dict()}")
    logger.info(f"Final augmented data shape: {X_combined.shape}")
    
    return X_combined, y_combined, combined_dist

def build_enhanced_models(X_train_enhanced, y_train_enhanced):
    """Build enhanced models using the augmented data."""
    logger.info("Building enhanced models")
    
    # Define base models
    base_models = {
        'logistic': LogisticRegression(C=0.01, penalty='l2', solver='liblinear', random_state=42),
        'rf': RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42)
    }
    
    # Train base models
    for name, model in base_models.items():
        logger.info(f"Training {name} model")
        model.fit(X_train_enhanced, y_train_enhanced)
    
    # Create voting ensemble
    logger.info("Creating voting ensemble")
    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        voting='soft',
        weights=[1, 2, 2]  # Give more weight to tree-based models
    )
    voting_ensemble.fit(X_train_enhanced, y_train_enhanced)
    
    # Create stacking ensemble
    logger.info("Creating stacking ensemble")
    stacking_ensemble = StackingClassifier(
        estimators=[(name, model) for name, model in base_models.items()],
        final_estimator=LogisticRegression(C=0.1, random_state=42),
        cv=5
    )
    stacking_ensemble.fit(X_train_enhanced, y_train_enhanced)
    
    # Add ensembles to models dictionary
    enhanced_models = {**base_models}
    enhanced_models['voting'] = voting_ensemble
    enhanced_models['stacking'] = stacking_ensemble
    
    return enhanced_models

def evaluate_enhanced_models(models, X_train, y_train, X_test, y_test, features):
    """Evaluate enhanced models on the original test data."""
    logger.info("Evaluating enhanced models on real test data")
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{name} test accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Away Win', 'Home Win'],
                    yticklabels=['Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'models/enhanced_{name}_confusion_matrix.png')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame({
        'accuracy': [results[name]['accuracy'] for name in results]
    }, index=results.keys())
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    return results, results_df

def main():
    """Main function to run the enhanced model building process."""
    start_time = time.time()
    
    # Create output directories
    ensure_directory_exists('models')
    ensure_directory_exists('models/enhanced')
    
    try:
        # Load data
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, features, train_data, test_data, scaler = load_data()
        
        # Generate synthetic data
        X_train_enhanced, y_train_enhanced, class_dist_enhanced = generate_synthetic_data(
            X_train_scaled, y_train, train_data, features
        )
        
        # Save enhanced data
        np.save('models/enhanced/X_train_enhanced.npy', X_train_enhanced)
        np.save('models/enhanced/y_train_enhanced.npy', y_train_enhanced)
        
        # Build enhanced models
        enhanced_models = build_enhanced_models(X_train_enhanced, y_train_enhanced)
        
        # Save models
        for name, model in enhanced_models.items():
            joblib.dump(model, f'models/enhanced/{name}_model.pkl')
        
        # Evaluate models on real test data
        results, results_df = evaluate_enhanced_models(
            enhanced_models, X_train_scaled, y_train, X_test_scaled, y_test, features
        )
        results_df.to_csv('models/enhanced/model_accuracies.csv')
        
        # Compare with ELO predictions
        elo_accuracy = (test_data['elo_pred'] == test_data['result']).mean()
        logger.info(f"ELO prediction accuracy: {elo_accuracy:.4f}")
        
        # Create summary report
        with open('models/enhanced_model_summary.md', 'w') as f:
            f.write("# Enhanced NBA Game Prediction Model Summary\n\n")
            
            f.write("## Data Enhancement Techniques\n\n")
            f.write("This model uses multiple data enhancement techniques:\n\n")
            f.write("1. **SMOTE** (Synthetic Minority Over-sampling Technique) for class balancing\n")
            f.write("2. **ADASYN** (Adaptive Synthetic Sampling) for additional synthetic examples\n")
            f.write("3. **Synthetic upset examples** with added noise to improve upset prediction\n\n")
            
            f.write("### Class Distribution\n\n")
            f.write("| Class | Original | Enhanced |\n")
            f.write("|-------|----------|----------|\n")
            original_dist = pd.Series(y_train).value_counts(normalize=True)
            for cls in sorted(original_dist.index):
                before = original_dist.get(cls, 0)
                after = class_dist_enhanced.get(cls, 0)
                f.write(f"| {cls} | {before:.4f} | {after:.4f} |\n")
            
            f.write("\n## Enhanced Models\n\n")
            f.write("Test accuracy for enhanced models (evaluated on real test data):\n\n")
            f.write("| Model | Test Accuracy |\n")
            f.write("|-------|---------------|\n")
            for name, accuracy in results_df.iterrows():
                f.write(f"| {name} | {accuracy['accuracy']:.4f} |\n")
            
            f.write("\n## Comparison with ELO\n\n")
            f.write("| Model | Accuracy |\n")
            f.write("|-------|----------|\n")
            f.write(f"| ELO | {elo_accuracy:.4f} |\n")
            f.write(f"| {results_df.index[0]} | {results_df.iloc[0, 0]:.4f} |\n")
            
            f.write("\n## Conclusion\n\n")
            best_model = results_df.index[0]
            best_accuracy = results_df.iloc[0, 0]
            
            if best_accuracy > elo_accuracy:
                improvement = best_accuracy - elo_accuracy
                f.write(f"The best performing model is **{best_model}** with a test accuracy of {best_accuracy:.4f}.\n")
                f.write(f"This represents an improvement of {improvement:.4f} ({improvement*100:.2f}%) over the baseline ELO model.\n")
            else:
                f.write(f"The ELO model still outperforms our enhanced models with an accuracy of {elo_accuracy:.4f}.\n")
                f.write(f"The best machine learning model is **{best_model}** with a test accuracy of {best_accuracy:.4f}.\n")
            
            f.write("\n## Impact of Data Enhancement\n\n")
            f.write("The data enhancement techniques have allowed our complex models to learn from a more balanced and diverse dataset.\n")
            f.write("This is particularly important for predicting upset games, which are less common in the original dataset.\n")
            f.write("By training on enhanced data but evaluating on real test data, we ensure that our models generalize well to real-world scenarios.\n")
        
        logger.info(f"Enhanced model building completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in enhanced model building: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
