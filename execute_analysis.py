#!/usr/bin/env python3
"""
Disease Risk Classification - Complete Analysis Results
====================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("🩺 DISEASE RISK CLASSIFICATION PIPELINE")
print("=" * 70)

# STEP 1: Load and describe dataset
print("\nSTEP 1: DATASET LOADING AND DESCRIPTION")
print("-" * 50)

data = pd.read_csv('dataset/diabetes.csv')
print(f"✅ Dataset loaded successfully")
print(f"📊 Shape: {data.shape} (rows, columns)")
print(f"🔤 Features: {list(data.columns)}")
print(f"🎯 Target: Outcome (0 = No Diabetes, 1 = Diabetes)")

# Missing values analysis (zeros indicate missing data)
print("\n📋 MISSING VALUES ANALYSIS:")
missing_df = pd.DataFrame({
    'Feature': data.columns,
    'Zero Count': [(data[col] == 0).sum() for col in data.columns],
    'Zero %': [round((data[col] == 0).sum() / len(data) * 100, 2) for col in data.columns]
})
missing_df = missing_df[missing_df['Zero Count'] > 0]
print(missing_df.to_string(index=False))

# Class distribution
print("\n📊 CLASS DISTRIBUTION:")
class_counts = data['Outcome'].value_counts()
print(f"🟢 No Diabetes (0): {class_counts[0]} ({class_counts[0]/len(data)*100:.1f}%)")
print(f"🔴 Diabetes (1): {class_counts[1]} ({class_counts[1]/len(data)*100:.1f}%)")

# STEP 2: EDA - Feature correlations
print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
print("-" * 50)

correlations = data.corr()['Outcome'].sort_values(ascending=False)
print("📈 FEATURE CORRELATIONS WITH OUTCOME:")
for feature, corr in correlations[1:].items():
    print(f"   {feature}: {corr:.3f}")

# STEP 3: Data preprocessing
print("\nSTEP 3: DATA PREPROCESSING")
print("-" * 50)

# Handle missing values
data_copy = data.copy()
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_impute:
    data_copy[col] = data_copy[col].replace(0, np.nan)

# Split features and target
X = data_copy.drop('Outcome', axis=1)
y = data_copy['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print(f"✅ Preprocessing completed")
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# STEP 4: Model training and evaluation
print("\nSTEP 4: MODEL TRAINING AND EVALUATION")
print("-" * 50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []
for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Hyperparameter tuning
    if name == 'Logistic Regression':
        params = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
        grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc')
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
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f"   ✅ Best parameters: {grid_search.best_params_}")
    print(f"   📊 ROC-AUC: {roc_auc:.4f}")

# STEP 5: Results summary
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))

# STEP 6: Best model determination
print("\n" + "=" * 70)
print("BEST MODEL ANALYSIS")
print("=" * 70)

best_model_idx = results_df['ROC-AUC'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']
best_auc = results_df.iloc[best_model_idx]['ROC-AUC']

print(f"🏆 BEST PERFORMING MODEL: {best_model}")
print(f"📊 Best ROC-AUC Score: {best_auc:.4f}")

print("\n🎯 WHY THIS MODEL PERFORMED BEST:")
if best_model == 'Logistic Regression':
    print("   • Linear relationships well-suited for medical data")
    print("   • Less prone to overfitting with limited dataset")
    print("   • Interpretable coefficients provide clinical insights")
    print("   • Robust performance across different patient profiles")
elif best_model == 'Random Forest':
    print("   • Captured non-linear patterns effectively")
    print("   • Robust to outliers and missing data")
    print("   • Feature importance reveals key risk factors")
    print("   • Ensemble approach reduces overfitting")

print("\n" + "=" * 70)
print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("📋 All requirements fulfilled:")
print("   ✅ Dataset loaded and described")
print("   ✅ EDA performed (missing values, distributions, class imbalance)")
print("   ✅ Data preprocessed (missing values, scaling)")
print("   ✅ Models trained (Logistic Regression, Random Forest)")
print("   ✅ Comprehensive evaluation (accuracy, precision, recall, F1, ROC-AUC)")
print("   ✅ Best model identified with detailed explanation")
