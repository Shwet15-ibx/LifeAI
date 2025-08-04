# 🩺 LifeAI - Disease Risk Classification Project

**A Production-Ready Machine Learning Pipeline for Disease Risk Prediction**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

LifeAI is a comprehensive machine learning pipeline designed for **disease risk prediction** using the **Pima Indians Diabetes Dataset**. This project demonstrates end-to-end ML development with focus on **explainability, code quality, and production readiness** - perfect for technical interviews and portfolio showcases.

### 🎯 Key Features

- **Complete ML Pipeline**: From data loading to model deployment
- **Explainable AI**: Clear feature importance and model interpretability
- **Production Ready**: Clean, documented, and modular code structure
- **Comprehensive Evaluation**: Multiple metrics with detailed analysis
- **Zero Setup**: Ready to run with single command

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
Git
```

### Installation & Usage
```bash
# Clone the repository
git clone https://github.com/Shwet15-ibx/LifeAI.git
cd LifeAI

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python execute_analysis.py

# Or run the comprehensive pipeline
python disease_risk_classifier.py
```

## 📊 Project Structure

```
LifeAI/
├── dataset/
│   └── diabetes.csv          # Pima Indians Diabetes Dataset
├── disease_risk_classifier.py # Main comprehensive pipeline
├── execute_analysis.py       # Simplified analysis script
├── display_results.py        # Results visualization script
├── requirements.txt          # All dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

## 🧪 Dataset Description

**Pima Indians Diabetes Dataset**
- **Samples**: 768 patients
- **Features**: 8 medical predictors
- **Target**: Diabetes outcome (0/1)
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases

### Feature Details
| Feature | Description | Range |
|---------|-------------|--------|
| Pregnancies | Number of times pregnant | 0-17 |
| Glucose | Plasma glucose concentration | 0-199 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 0-122 |
| SkinThickness | Triceps skin fold thickness (mm) | 0-99 |
| Insulin | 2-Hour serum insulin (mu U/ml) | 0-846 |
| BMI | Body mass index | 0-67.1 |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0.078-2.42 |
| Age | Age (years) | 21-81 |

## 🔍 Analysis Results

### Final Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **77.92%** | **72.97%** | **62.62%** | **67.42%** | **0.8317** |
| Random Forest | 77.92% | 71.43% | 62.62% | 66.77% | 0.8187 |

### 🏆 Winner: Logistic Regression

**Why Logistic Regression Won:**
- **Superior ROC-AUC**: 0.8317 vs 0.8187
- **Medical interpretability**: Coefficients provide clinical insights
- **Robust performance**: Less prone to overfitting
- **Linear relationships**: Well-suited for medical data patterns

### Key Clinical Insights
- **Glucose** is the strongest predictor (correlation: 0.336)
- **BMI** and **Age** are significant secondary predictors
- **Class distribution**: 65.1% no diabetes, 34.9% diabetes
- **Missing data handled**: 227-374 zeros imputed as missing values

## 🛠️ Technical Implementation

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Code Architecture

#### 1. Data Pipeline
- **Missing value handling**: Median imputation for zero values
- **Feature scaling**: StandardScaler for normalization
- **Train-test split**: 80-20 split with stratification

#### 2. Model Training
- **Hyperparameter tuning**: GridSearchCV for optimal parameters
- **Cross-validation**: 5-fold CV for robust evaluation
- **Multiple algorithms**: Logistic Regression, Random Forest

#### 3. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity (true positive rate)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🎯 Usage Examples

### Basic Usage
```python
# Run the complete analysis
python execute_analysis.py

# Get detailed results
python display_results.py

# Access comprehensive pipeline
python disease_risk_classifier.py
```

### Custom Analysis
```python
# Import in your own scripts
from disease_risk_classifier import DiseaseRiskClassifier

# Initialize classifier
classifier = DiseaseRiskClassifier()

# Run complete pipeline
classifier.load_and_describe_data()
classifier.perform_eda()
classifier.preprocess_data()
classifier.train_models()
results = classifier.evaluate_models()
best_model = classifier.determine_best_model()
```

## 📈 Visualizations

The pipeline generates:
- **Missing values analysis** with heatmaps
- **Feature distributions** by outcome
- **Class imbalance** visualization
- **Model comparison** charts
- **ROC curves** for each model
- **Feature importance** plots

## 🔧 Interview Preparation

### Key Questions This Project Answers
1. **Data Preprocessing**: How do you handle missing values?
2. **Model Selection**: Why choose Logistic Regression over Random Forest?
3. **Evaluation**: How do you interpret ROC-AUC scores?
4. **Production**: How would you deploy this in production?
5. **Bias**: How do you handle class imbalance?

### Technical Discussion Points
- **Feature engineering** decisions
- **Model interpretability** vs performance trade-offs
- **Cross-validation** strategy
- **Hyperparameter tuning** approach
- **Production deployment** considerations

## 🤝 Contributing

This project is designed for educational and interview purposes. Feel free to:
- Fork and experiment
- Add new models or features
- Improve documentation
- Share your results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** team for the amazing ML library
- **Pima Indians Diabetes Dataset** contributors
- **Open source** community for continuous support

---

**Ready for immediate use in technical interviews!** 🚀

For questions or issues, please open an issue in the GitHub repository.
