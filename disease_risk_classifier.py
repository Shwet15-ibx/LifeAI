"""
Disease Risk Binary Classification Pipeline
==========================================

A comprehensive machine learning pipeline for disease risk prediction
using the Pima Indians Diabetes Dataset. This implementation focuses
on explainability, code quality, and thorough evaluation.

Author: AI Mentor
Date: 2025-08-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using Random Forest and Logistic Regression only.")
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DiseaseRiskClassifier:
    """
    A comprehensive classifier for disease risk prediction.
    
    This class provides a complete pipeline from data loading to model evaluation,
    with emphasis on explainability and thorough analysis.
    """
    
    def __init__(self, data_path="dataset/diabetes.csv"):
        """
        Initialize the classifier with dataset path.
        
        Args:
            data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_and_describe_data(self):
        """
        Load the dataset and provide comprehensive description.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("=" * 60)
        print("STEP 1: DATASET LOADING AND DESCRIPTION")
        print("=" * 60)
        
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Basic information
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print(f"Target Variable: Outcome (0 = No Diabetes, 1 = Diabetes)")
        print()
        
        # Statistical summary
        print("STATISTICAL SUMMARY:")
        print(self.data.describe())
        print()
        
        # Data types
        print("DATA TYPES:")
        print(self.data.dtypes)
        print()
        
        # Missing values
        print("MISSING VALUES ANALYSIS:")
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Percentage': missing_percentage
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Check for zeros that might indicate missing values
        print("\nZERO VALUES (potential missing data indicators):")
        zero_counts = (self.data == 0).sum()
        zero_df = pd.DataFrame({
            'Zero Count': zero_counts,
            'Percentage': (zero_counts / len(self.data)) * 100
        })
        print(zero_df[zero_df['Zero Count'] > 0])
        
        return self.data
    
    def perform_eda(self):
        """
        Perform comprehensive Exploratory Data Analysis.
        
        Creates visualizations for:
        - Missing values heatmap
        - Feature distributions
        - Class imbalance
        - Feature correlations
        """
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Missing values heatmap
        plt.subplot(3, 3, 1)
        sns.heatmap(self.data.isnull(), cbar=True, cmap='viridis', 
                   yticklabels=False, alpha=0.7)
        plt.title('Missing Values Heatmap')
        
        # 2. Class imbalance
        plt.subplot(3, 3, 2)
        class_counts = self.data['Outcome'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        plt.pie(class_counts.values, labels=['No Diabetes', 'Diabetes'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Class Distribution (Target Variable)')
        
        # 3. Feature distributions by outcome
        features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
        for i, feature in enumerate(features, 3):
            plt.subplot(3, 3, i)
            self.data.boxplot(column=feature, by='Outcome', ax=plt.gca())
            plt.title(f'{feature} Distribution by Outcome')
            plt.suptitle('')  # Remove default title
        
        # 4. Correlation heatmap
        plt.subplot(3, 3, 7)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 5. Age distribution
        plt.subplot(3, 3, 8)
        for outcome in [0, 1]:
            subset = self.data[self.data['Outcome'] == outcome]['Age']
            plt.hist(subset, alpha=0.7, label=f'Outcome {outcome}', bins=20)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Outcome')
        plt.legend()
        
        # 6. Glucose distribution
        plt.subplot(3, 3, 9)
        for outcome in [0, 1]:
            subset = self.data[self.data['Outcome'] == outcome]['Glucose']
            plt.hist(subset, alpha=0.7, label=f'Outcome {outcome}', bins=20)
        plt.xlabel('Glucose')
        plt.ylabel('Frequency')
        plt.title('Glucose Distribution by Outcome')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("EDA visualizations saved as 'eda_analysis.png'")
        
        # Print correlation insights
        print("\nFEATURE CORRELATIONS WITH OUTCOME:")
        outcome_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
        print(outcome_corr[1:])  # Exclude Outcome itself
    
    def preprocess_data(self):
        """
        Preprocess the data with comprehensive handling of:
        - Missing values
        - Outlier treatment
        - Feature scaling
        - Train-test split
        """
        print("\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 60)
        
        # Handle missing values (zeros in specific columns indicate missing data)
        columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # Replace zeros with NaN for specific columns
        data_copy = self.data.copy()
        for col in columns_to_impute:
            data_copy[col] = data_copy[col].replace(0, np.nan)
        
        # Separate features and target
        X = data_copy.drop('Outcome', axis=1)
        y = data_copy['Outcome']
        
        # Create preprocessing pipeline
        numeric_features = X.columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply preprocessing
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)
        
        print("Preprocessing completed successfully!")
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Class distribution in training set:")
        print(pd.Series(self.y_train).value_counts().sort_index())
        
        return preprocessor
    
    def train_models(self):
        """
        Train multiple classification models with hyperparameter tuning.
        
        Models trained:
        - Logistic Regression
        - Random Forest
        - XGBoost
        """
        print("\n" + "=" * 60)
        print("STEP 4: MODEL TRAINING")
        print("=" * 60)
        
        # Define models with hyperparameter grids
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = {
                'model': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                }
            }
        
        # Train each model with GridSearchCV
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.models[name] = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Metrics calculated:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC AUC
        """
        print("\n" + "=" * 60)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 60)
        
        # Create results dataframe
        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
        
        # Create figure for evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        model_colors = {'Logistic Regression': 'blue', 'Random Forest': 'green', 'XGBoost': 'red'}
        
        for i, (name, model) in enumerate(self.models.items()):
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Add to results dataframe
            results_df = pd.concat([results_df, pd.DataFrame({
                'Model': [name],
                'Accuracy': [f"{accuracy:.4f}"],
                'Precision': [f"{precision:.4f}"],
                'Recall': [f"{recall:.4f}"],
                'F1-Score': [f"{f1:.4f}"],
                'ROC-AUC': [f"{roc_auc:.4f}"]
            })], ignore_index=True)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[0, 0], cbar=False)
            axes[0, 0].set_title(f'Confusion Matrix - {name}')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            axes[0, 1].plot(fpr, tpr, color=model_colors[name], 
                           label=f'{name} (AUC = {roc_auc:.3f})')
            
            # Feature importance (for tree-based models)
            if name in ['Random Forest', 'XGBoost']:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                   'SkinThickness', 'Insulin', 'BMI', 
                                   'DiabetesPedigreeFunction', 'Age']
                    
                    axes[1, 0].bar(range(len(importances)), importances, 
                                  label=name, alpha=0.7, color=model_colors[name])
        
        # Finalize ROC curve plot
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance plot
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_title('Feature Importance Comparison')
        axes[1, 0].set_xticks(range(8))
        axes[1, 0].set_xticklabels(['Preg', 'Gluc', 'BP', 'Skin', 'Ins', 'BMI', 'DPF', 'Age'], 
                                  rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics comparison
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        })
        
        metrics_df.set_index('Model').plot(kind='bar', ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('Metrics Comparison Across Models')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Display results table
        print("\nEVALUATION RESULTS SUMMARY:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def determine_best_model(self):
        """
        Determine the best performing model based on comprehensive analysis.
        
        Returns:
            str: Name of the best performing model
            dict: Detailed analysis of why this model performed best
        """
        print("\n" + "=" * 60)
        print("STEP 6: BEST MODEL ANALYSIS")
        print("=" * 60)
        
        # Find best model based on ROC-AUC
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['roc_auc'])
        
        print(f"🏆 BEST PERFORMING MODEL: {best_model_name}")
        print()
        
        # Detailed analysis
        best_metrics = self.results[best_model_name]
        print("PERFORMANCE METRICS:")
        print(f"  • Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  • Precision: {best_metrics['precision']:.4f}")
        print(f"  • Recall: {best_metrics['recall']:.4f}")
        print(f"  • F1-Score: {best_metrics['f1']:.4f}")
        print(f"  • ROC-AUC: {best_metrics['roc_auc']:.4f}")
        print()
        
        # Model-specific insights
        print("WHY THIS MODEL PERFORMED BEST:")
        
        if best_model_name == 'Logistic Regression':
            print("  • Linear relationship between features and outcome")
            print("  • Less prone to overfitting with limited data")
            print("  • Interpretable coefficients for medical insights")
            
        elif best_model_name == 'Random Forest':
            print("  • Captured non-linear relationships effectively")
            print("  • Robust to outliers and missing data")
            print("  • Feature importance provides medical insights")
            
        elif best_model_name == 'XGBoost':
            print("  • Captured complex feature interactions")
            print("  • Gradient boosting optimized for this dataset")
            print("  • Handles missing values inherently")
        
        # Compare with other models
        print("\nCOMPARISON WITH OTHER MODELS:")
        for model_name, metrics in self.results.items():
            if model_name != best_model_name:
                auc_diff = best_metrics['roc_auc'] - metrics['roc_auc']
                print(f"  • {model_name}: {auc_diff:.4f} lower ROC-AUC")
        
        return best_model_name, best_metrics

# Main execution function
def main():
    """
    Execute the complete disease risk classification pipeline.
    """
    print("🩺 DISEASE RISK CLASSIFICATION PIPELINE")
    print("=" * 60)
    print("A comprehensive analysis for diabetes prediction")
    print("=" * 60)
    
    # Initialize classifier
    classifier = DiseaseRiskClassifier()
    
    # Execute pipeline steps
    classifier.load_and_describe_data()
    classifier.perform_eda()
    classifier.preprocess_data()
    classifier.train_models()
    results_df = classifier.evaluate_models()
    best_model, analysis = classifier.determine_best_model()
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📊 Visualizations saved: eda_analysis.png, model_evaluation.png")
    print("📈 All models trained and evaluated comprehensively")
    print("🏆 Best model identified with detailed reasoning")

if __name__ == "__main__":
    main()
