#!/usr/bin/env python3
"""
Disease Risk Classification - Final Model Results
===============================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load and process data
data = pd.read_csv('dataset/diabetes.csv')

# Handle missing values (0 indicates missing)
data_processed = data.copy()
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    data_processed[col] = data_processed[col].replace(0, np.nan)

X = data_processed.drop('Outcome', axis=1)
y = data_processed['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("🩺 DISEASE RISK CLASSIFICATION - FINAL MODEL RESULTS")
print("=" * 70)
print()

# Train final models
print("📊 TRAINING FINAL MODELS...")
print("-" * 50)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_processed, y_train)
lr_pred = lr.predict(X_test_processed)
lr_proba = lr.predict_proba(X_test_processed)[:, 1]

# Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_processed, y_train)
rf_pred = rf.predict(X_test_processed)
rf_proba = rf.predict_proba(X_test_processed)[:, 1]

# Calculate exact metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print()
print("🏆 FINAL RESULTS TABLE")
print("=" * 70)
results = [
    ["Logistic Regression", f"{lr_accuracy:.4f}", f"{lr_precision:.4f}", 
     f"{lr_recall:.4f}", f"{lr_f1:.4f}", f"{lr_auc:.4f}"],
    ["Random Forest", f"{rf_accuracy:.4f}", f"{rf_precision:.4f}", 
     f"{rf_recall:.4f}", f"{rf_f1:.4f}", f"{rf_auc:.4f}"]
]

for model, acc, prec, rec, f1, auc in results:
    print(f"{model:<20} | Accuracy: {acc} | Precision: {prec} | Recall: {rec} | F1: {f1} | ROC-AUC: {auc}")

print()
print("🎯 WINNER ANALYSIS")
print("=" * 70)
if lr_auc > rf_auc:
    winner = "Logistic Regression"
    winner_auc = lr_auc
    print(f"🏆 BEST MODEL: {winner}")
    print(f"📈 ROC-AUC Score: {winner_auc:.4f}")
    print()
    print("🎯 Why Logistic Regression Won:")
    print("   • Linear relationships well-suited for medical data")
    print("   • Less prone to overfitting with limited dataset")
    print("   • Interpretable coefficients provide clinical insights")
    print("   • Robust performance across different patient profiles")
else:
    winner = "Random Forest"
    winner_auc = rf_auc
    print(f"🏆 BEST MODEL: {winner}")
    print(f"📈 ROC-AUC Score: {winner_auc:.4f}")
    print()
    print("🎯 Why Random Forest Won:")
    print("   • Captured non-linear patterns effectively")
    print("   • Robust to outliers and missing data")
    print("   • Feature importance reveals key risk factors")

print()
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
print("All models evaluated with comprehensive metrics")
print("Best model identified with detailed explanation")
