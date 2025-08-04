"""
Simplified Disease Risk Classification Analysis
============================================

A focused analysis that provides clear, concise results for diabetes prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Load and describe dataset
print("=" * 70)
print("STEP 1: DATASET LOADING AND DESCRIPTION")
print("=" * 70)

data = pd.read_csv('dataset/diabetes.csv')
print(f"Dataset Shape: {data.shape}")
print(f"Features: {list(data.columns)}")
print(f"Target: Outcome (0 = No Diabetes, 1 = Diabetes)")
print()

# Missing values analysis
print("MISSING VALUES ANALYSIS:")
missing_df = pd.DataFrame({
    'Zero Count': (data == 0).sum(),
    'Zero Percentage': ((data == 0).sum() / len(data) * 100).round(2)
})
print(missing_df[missing_df['Zero Count'] > 0])
print()

# Class imbalance
print("CLASS DISTRIBUTION:")
class_dist = data['Outcome'].value_counts()
print(f"No Diabetes (0): {class_dist[0]} ({class_dist[0]/len(data)*100:.1f}%)")
print(f"Diabetes (1): {class_dist[1]} ({class_dist[1]/len(data)*100:.1f}%)")
print()

# EDA - Feature correlations
print("FEATURE CORRELATIONS WITH OUTCOME:")
correlations = data.corr()['Outcome'].sort_values(ascending=False)
print(correlations[1:])  # Exclude Outcome itself
print()

# Preprocessing
print("=" * 70)
print("STEP 2: DATA PREPROCESSING")
print("=" * 70)

# Handle missing values (zeros indicate missing data)
data_copy = data.copy()
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_impute:
    data_copy[col] = data_copy[col].replace(0, np.nan)

# Separate features and target
X = data_copy.drop('Outcome', axis=1)
y = data_copy['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print()

# Model training and evaluation
print("=" * 70)
print("STEP 3: MODEL TRAINING AND EVALUATION")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Hyperparameter tuning
    if name == 'Logistic Regression':
        params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
    else:
        params = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
    
    grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    # Make predictions
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}",
        'ROC-AUC': f"{roc_auc:.4f}",
        'roc_auc_value': roc_auc
    })
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

# Display results
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

# Determine best model
best_model_idx = max(range(len(results)), key=lambda i: results[i]['roc_auc_value'])
best_model = results[best_model_idx]['Model']

print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
print("\nWHY THIS MODEL PERFORMED BEST:")

if best_model == 'Logistic Regression':
    print("• Linear relationships captured well in medical data")
    print("• Less prone to overfitting with limited dataset")
    print("• Interpretable coefficients for medical insights")
elif best_model == 'Random Forest':
    print("• Captured non-linear patterns effectively")
    print("• Robust to outliers and missing data")
    print("• Feature importance provides clinical insights")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - All steps executed successfully!")
print("=" * 70)
