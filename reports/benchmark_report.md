# Machine Learning Benchmark Report: Predicting Student Academic Performance

**Generated:** 2026-08-28 18:28:07  
**Total Cohort Size:** 1600 student records  

---

## Executive Summary
This report summarizes the empirical performance of multiple competitive Machine Learning algorithms applied to predicting student educational outcomes across three analytical dimensions:
1. **Continuous Final Grade Prediction (Regression)**
2. **Binary Pass / Fail Classification**
3. **Multi-tier Academic Performance Category (At Risk, Satisfactory, Good, Excellent)**

---

## Best Performing Machine Learning Models

| Task | Selected Best Algorithm | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :--- |
| **Final Grade Prediction (Regression)** | **Lasso Regression** | **$R^2$ = 0.9216** | **RMSE = 3.307** (MAE = 2.667) |
| **Pass/Fail Outcome (Binary Classification)** | **Logistic Regression** | **Accuracy = 95.63%** | **F1-Score = 95.53%** (ROC-AUC = 0.9844) |
| **Performance Tier (Multi-Class Classification)** | **Logistic Regression** | **Accuracy = 84.38%** | **Weighted F1 = 84.39%** |

---

## Complete Model Leaderboard

### 1. Regression Models (Final Grade 0-100)
| Model Architecture | $R^2$ Score | RMSE | MAE | MAPE (%) | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression | **0.9223** | 3.29 | 2.66 | 3.96% | 0.9250 |
| Linear Regression | **0.9222** | 3.29 | 2.66 | 3.96% | 0.9249 |
| Lasso Regression | **0.9216** | 3.31 | 2.67 | 3.97% | 0.9261 |
| Multi-Layer Perceptron (MLP) | **0.9118** | 3.51 | 2.82 | 4.25% | 0.9187 |
| XGBoost | **0.9089** | 3.56 | 2.86 | 4.26% | 0.9102 |
| LightGBM | **0.9077** | 3.59 | 2.87 | 4.34% | 0.9073 |
| Gradient Boosting | **0.9074** | 3.59 | 2.88 | 4.32% | 0.9117 |
| Random Forest | **0.9035** | 3.67 | 2.93 | 4.39% | 0.9053 |
| Support Vector Regressor (SVR) | **0.8842** | 4.02 | 3.10 | 4.88% | 0.8891 |
| Decision Tree | **0.8742** | 4.19 | 3.31 | 4.97% | 0.8700 |

### 2. Pass / Fail Binary Classification
| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| XGBoost | **95.94%** | 95.87% | 95.94% | **95.86%** | 0.9860 |
| Support Vector Classifier (SVC) | **95.63%** | 95.55% | 95.63% | **95.56%** | 0.9833 |
| Logistic Regression | **95.63%** | 95.55% | 95.63% | **95.53%** | 0.9844 |
| Random Forest | **95.63%** | 95.55% | 95.63% | **95.53%** | 0.9867 |
| Gradient Boosting | **95.63%** | 95.55% | 95.63% | **95.53%** | 0.9839 |
| LightGBM | **95.00%** | 94.91% | 95.00% | **94.93%** | 0.9854 |
| Decision Tree | **94.06%** | 93.92% | 94.06% | **93.95%** | 0.8975 |
| Multi-Layer Perceptron (MLP) | **93.75%** | 93.67% | 93.75% | **93.71%** | 0.9735 |

### 3. Performance Category Multi-Class Classification
| Model Architecture | Accuracy | Precision | Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | **84.38%** | 84.92% | 84.38% | **84.39%** |
| Random Forest | **84.38%** | 84.73% | 84.38% | **84.20%** |
| XGBoost | **84.06%** | 84.49% | 84.06% | **84.11%** |
| LightGBM | **83.75%** | 84.25% | 83.75% | **83.73%** |
| Support Vector Classifier (SVC) | **82.50%** | 82.61% | 82.50% | **82.46%** |
| Decision Tree | **81.87%** | 82.58% | 81.87% | **81.94%** |
| Gradient Boosting | **81.87%** | 82.28% | 81.87% | **81.89%** |
| Multi-Layer Perceptron (MLP) | **77.81%** | 78.39% | 77.81% | **77.93%** |

---

## Key Predictive Drivers (Feature Importance)
Top educational, demographic, and behavioral features ranked by predictive weight:

- **Previous Grades**: `35.25%` relative contribution
- **Study Time**: `11.80%` relative contribution
- **Failures**: `11.43%` relative contribution
- **Attendance Rate**: `9.04%` relative contribution
- **Parental Education High School**: `8.47%` relative contribution
- **Tutoring No**: `7.67%` relative contribution
- **Parental Education Master/Doctorate**: `4.96%` relative contribution
- **Academic Risk Index**: `4.43%` relative contribution

---

## Generated Diagnostic Artifacts
The following visual diagnostics have been rendered in `reports/figures/`:
1. `grade_distribution.png` - Final grade histogram & KDE with pass/fail line
2. `correlation_heatmap.png` - Pearson correlation matrix
3. `attendance_vs_grade.png` - Attendance rate vs. final grade regression trend
4. `study_time_vs_grade.png` - Weekly study hours impact across performance tiers
5. `parental_education_impact.png` - Grade distributions by parental education level
6. `model_regression_comparison.png` - Comparison of R2 and RMSE across all regressors
7. `model_classification_comparison.png` - Comparison of accuracy and F1 scores across classifiers
8. `regression_residuals_plot.png` - Predicted vs actual grades and residual error curve
9. `confusion_matrices_plot.png` - Confusion matrix heatmaps for classification tasks
10. `feature_importance_ranking.png` - Ranked feature contribution bar chart
