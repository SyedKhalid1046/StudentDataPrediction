import os
import nbformat as nbf

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title Markdown
    cells.append(nbf.v4.new_markdown_cell("""# Predicting Student Academic Performance Using Machine Learning Algorithms and Educational Data
**End-to-End Machine Learning Pipeline: From Raw Educational Data to Multi-Target Predictive Modeling & Prescriptive Interventions**

---

### Project Objectives
1. **Regression Task**: Predict continuous Final Academic Grade ($G_3$ / Score 0-100) using demographic, socioeconomic, behavioral, and past academic metrics.
2. **Binary Classification Task**: Predict Pass/Fail status with calibrated probabilities.
3. **Multi-Class Classification Task**: Predict standardized Performance Tier (*At Risk, Satisfactory, Good, Excellent*).
4. **Feature Importance & XAI**: Identify the strongest empirical drivers of student outcomes.
5. **Prescriptive Guidance**: Provide automated, actionable intervention recommendations for students and educators.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""# 1. Imports and Environment Setup
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in python path
_ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.eda import EDAPerformer
from src.models_regression import RegressionModelSuite
from src.models_classification import ClassificationModelSuite
from src.predictor import StudentPredictor

# Styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("Environment configured successfully!")
"""))

    # Step 1: Data Ingestion
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Data Ingestion & Profile Inspection
We load the student performance educational benchmark dataset containing demographics, attendance, study habits, parental education, and past performance.
"""))

    cells.append(nbf.v4.new_code_cell("""# Ingest data using DataLoader
data_path = os.path.join(_ROOT_DIR, "data", "student_performance_data.csv")
loader = DataLoader(data_path)
df = loader.df

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(8)
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics of numerical columns
df.describe().round(2)
"""))

    # Step 2: EDA
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Exploratory Data Analysis (EDA)
Let's analyze feature correlations, grade distributions, and the effect of attendance, study hours, and parental background on final performance.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Final Grade Distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(data=df, x="final_grade", hue="pass_fail_status", kde=True, palette={"Pass": "#10B981", "Fail": "#EF4444"}, bins=25, ax=ax)
ax.axvline(df["final_grade"].mean(), color="#3B82F6", linestyle="--", linewidth=2, label=f"Mean Grade ({df['final_grade'].mean():.1f})")
ax.axvline(60.0, color="#EF4444", linestyle=":", linewidth=2, label="Pass Threshold (60.0)")
ax.set_title("Student Final Grade Distribution with Pass/Fail Threshold", fontweight="bold")
ax.set_xlabel("Final Grade (Score 0-100)")
ax.legend()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Feature Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Pearson Correlation Heatmap of Educational Features", fontweight="bold")
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Attendance Rate vs Final Grade
plt.figure(figsize=(9, 4.5))
sns.regplot(data=df, x="attendance_rate", y="final_grade", scatter_kws={"alpha": 0.5, "color": "#6366F1"}, line_kws={"color": "#4338CA", "linewidth": 2.5})
plt.title("Impact of Class Attendance Rate on Final Grade", fontweight="bold")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("Final Grade (Score 0-100)")
plt.show()
"""))

    # Step 3: Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## Step 3: Domain-Specific Feature Engineering
We synthesize domain indices derived from educational research:
- **`academic_risk_index`**: Weighted penalty index of poor attendance, high absences, previous failures, and deficient study time.
- **`study_efficiency`**: Ratio of previous academic score to weekly study hours.
- **`socioeconomic_support_score`**: Combined index of parental education level, internet access, and family support.
- **`attendance_tier`**: Categorized attendance brackets.
"""))

    cells.append(nbf.v4.new_code_cell("""# Apply feature engineering
fe = FeatureEngineer()
df_engineered = fe.transform(df, is_training=True)

print("Engineered features sample:")
df_engineered[["student_id", "academic_risk_index", "study_efficiency", "socioeconomic_support_score", "attendance_tier"]].head()
"""))

    # Step 4: Preprocessing
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Preprocessing & Stratified Train/Test Splitting
We build a scikit-learn `ColumnTransformer` featuring `StandardScaler` for continuous numeric features and `OneHotEncoder` for categorical factors, with an 80/20 stratified train/test split.
"""))

    cells.append(nbf.v4.new_code_cell("""prep = Preprocessor()
splits = prep.fit_transform_dataset(df_engineered, test_size=0.2, random_state=42)

print(f"X_train Shape: {splits['X_train'].shape} | X_test Shape: {splits['X_test'].shape}")
print(f"Total Encoded Feature Columns: {len(splits['feature_names'])}")
"""))

    # Step 5: Regression Models
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Regression Models Benchmark (Continuous Grade Prediction)
We evaluate 10 regression algorithms with 5-Fold Cross-Validation:
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, Gradient Boosting
- XGBoost, LightGBM
- Support Vector Regressor (SVR), Multi-Layer Perceptron (MLP)
"""))

    cells.append(nbf.v4.new_code_cell("""# Train & Cross-Validate Regression Models
reg_suite = RegressionModelSuite(random_state=42)
reg_cv = reg_suite.train_and_cross_validate(splits["X_train"], splits["y_reg_train"], cv=5)
reg_results = reg_suite.evaluate_test_set(splits["X_test"], splits["y_reg_test"])

# Leaderboard DataFrame
reg_leaderboard = pd.DataFrame.from_dict(reg_results, orient="index").sort_values(by="R2_Score", ascending=False)
print(f"Best Regression Model: {reg_suite.best_model_name}")
reg_leaderboard
"""))

    # Step 6: Classification Models
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Classification Models Benchmark
### 6.1 Binary Classification: Pass / Fail Prediction
"""))

    cells.append(nbf.v4.new_code_cell("""# Pass/Fail Classification
pf_suite = ClassificationModelSuite(task_type="binary", random_state=42)
pf_cv = pf_suite.train_and_cross_validate(splits["X_train"], splits["y_pf_train"], cv=5)
pf_results = pf_suite.evaluate_test_set(splits["X_test"], splits["y_pf_test"])

pf_leaderboard = pd.DataFrame.from_dict(pf_results, orient="index").sort_values(by="F1_Score", ascending=False)
print(f"Best Pass/Fail Model: {pf_suite.best_model_name}")
pf_leaderboard[["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]]
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 6.2 Multi-Class Classification: Performance Category Prediction
Categories: *At Risk, Satisfactory, Good, Excellent*
"""))

    cells.append(nbf.v4.new_code_cell("""# Performance Category Classification
cat_suite = ClassificationModelSuite(task_type="multiclass", random_state=42)
cat_cv = cat_suite.train_and_cross_validate(splits["X_train"], splits["y_cat_train"], cv=5)
cat_results = cat_suite.evaluate_test_set(splits["X_test"], splits["y_cat_test"])

cat_leaderboard = pd.DataFrame.from_dict(cat_results, orient="index").sort_values(by="F1_Score", ascending=False)
print(f"Best Category Model: {cat_suite.best_model_name}")
cat_leaderboard[["Accuracy", "Precision", "Recall", "F1_Score"]]
"""))

    # Step 7: Feature Importance
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Feature Importance & Explainable AI (XAI)
What variables have the greatest empirical influence on student final outcomes?
"""))

    cells.append(nbf.v4.new_code_cell("""# Extract Feature Importances from Best Model
feat_importances = reg_suite.get_feature_importances(splits["feature_names"])
top_10 = list(feat_importances.items())[:10]

plt.figure(figsize=(9, 5))
names = [k.replace('_', ' ').title() for k, v in top_10]
vals = [v for k, v in top_10]

plt.barh(names[::-1], vals[::-1], color="#3B82F6", edgecolor="#1E3A8A")
plt.xlabel("Relative Importance Score (%)", fontweight="bold")
plt.title("Top 10 Drivers of Student Academic Performance", fontweight="bold")
for i, v in enumerate(vals[::-1]):
    plt.text(v + 0.3, i, f"{v:.1f}%", va="center", fontweight="bold")
plt.show()
"""))

    # Step 8: Inference
    cells.append(nbf.v4.new_markdown_cell("""## Step 8: Real-Time Inference & Prescriptive AI Interventions
We test the `StudentPredictor` on an at-risk profile to observe predictions and automated recommendations.
"""))

    cells.append(nbf.v4.new_code_cell("""# Instantiate Predictor
predictor = StudentPredictor(models_dir=os.path.join(_ROOT_DIR, "models"))

# At-Risk Student Test Case
at_risk_student = {
    "gender": "Male",
    "age": 18,
    "parental_education": "High School",
    "study_time": 4.0,
    "attendance_rate": 66.0,
    "previous_grades": 50.0,
    "extracurricular_activities": "No",
    "internet_access": "Yes",
    "tutoring": "No",
    "family_support": "No",
    "health": "Fair",
    "absences": 14,
    "failures": 1
}

res = predictor.predict_single(at_risk_student)

print("=" * 60)
print(f"PREDICTED FINAL GRADE: {res['predicted_grade']} / 100 (20-Scale: {res['predicted_grade_20_scale']})")
print(f"PASS/FAIL OUTCOME:     {res['pass_fail_status']} (Pass Probability: {res['pass_probability']}%)")
print(f"PERFORMANCE TIER:      {res['performance_category']}")
print(f"ACADEMIC RISK INDEX:   {res['academic_risk_index']} / 100 [{res['risk_tier']}]")
print("=" * 60)
print(f"\nPRESCRIPTIVE AI RECOMMENDATIONS ({len(res['recommendations'])}):")
for r in res['recommendations']:
    print(f" • [{r['category']} - {r['priority']}] {r['action']}")
    print(f"   -> Impact: {r['expected_impact']}")
"""))

    nb.cells = cells
    
    os.makedirs(os.path.join(_ROOT_DIR, "notebooks"), exist_ok=True)
    nb_path = os.path.join(_ROOT_DIR, "notebooks", "student_performance_ml_pipeline.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook successfully written to: {nb_path}")

if __name__ == "__main__":
    create_notebook()
