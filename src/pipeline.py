import os
import sys
import json
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib

# Ensure root directory is on python path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.eda import EDAPerformer
from src.models_regression import RegressionModelSuite
from src.models_classification import ClassificationModelSuite
from src.evaluation import EvaluationVisualizer

class MLPipeline:
    """Master class coordinating the student performance ML training and evaluation lifecycle."""
    
    def __init__(self, data_path: str = "data/student_performance_data.csv", models_dir: str = "models", reports_dir: str = "reports"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(reports_dir, "figures")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def run_pipeline(self) -> Dict[str, Any]:
        """Execute complete end-to-end pipeline."""
        start_time = time.time()
        print("=" * 80)
        print("STARTING END-TO-END STUDENT ACADEMIC PERFORMANCE ML PIPELINE")
        print("=" * 80)
        
        # 1. Data Ingestion & Profile
        print("\n[Stage 1/6] Ingesting & Validating Dataset...")
        loader = DataLoader(self.data_path)
        raw_df = loader.df
        profile = loader.get_summary_profile()
        print(f" Loaded {len(raw_df)} records across {len(raw_df.columns)} initial columns.")
        
        # 2. Exploratory Data Analysis & Plots
        print("\n[Stage 2/6] Performing Exploratory Data Analysis (EDA)...")
        eda = EDAPerformer(raw_df, output_dir=self.figures_dir)
        eda_figs = eda.generate_all_visualizations()
        eda_summary = eda.get_eda_summary_dict()
        print(f" Generated {len(eda_figs)} publication-ready analytical charts in '{self.figures_dir}'.")
        
        # 3. Feature Engineering
        print("\n[Stage 3/6] Engineering Domain-Specific Academic Indicators...")
        fe = FeatureEngineer()
        df_feat = fe.transform(raw_df, is_training=True)
        print(f" Added engineered features: Academic Risk Index, Study Efficiency, Socioeconomic Support Index, Attendance Tiers.")
        
        # 4. Preprocessing & Dataset Splitting
        print("\n[Stage 4/6] Preprocessing, Encoding & Splitting Data (80/20 Stratified)...")
        preprocessor = Preprocessor()
        splits = preprocessor.fit_transform_dataset(df_feat, test_size=0.2, random_state=42)
        print(f" Training samples: {splits['X_train'].shape[0]} | Testing samples: {splits['X_test'].shape[0]}")
        print(f" Transformed Feature Dimensions: {splits['X_train'].shape[1]}")
        
        # 5. Model Training & Cross-Validation
        print("\n[Stage 5/6] Training & Tuning Multi-Model Suites with 5-Fold Cross-Validation...")
        
        # 5a. Regression (Final Grade)
        print(" -> Training 10 Regression Models for Continuous Grade Prediction...")
        reg_suite = RegressionModelSuite(random_state=42)
        reg_cv = reg_suite.train_and_cross_validate(splits["X_train"], splits["y_reg_train"], cv=5)
        reg_test_results = reg_suite.evaluate_test_set(splits["X_test"], splits["y_reg_test"])
        print(f"    Best Regression Model: '{reg_suite.best_model_name}' (R2: {reg_test_results[reg_suite.best_model_name]['R2_Score']:.4f}, RMSE: {reg_test_results[reg_suite.best_model_name]['RMSE']:.2f})")
        
        # 5b. Binary Classification (Pass/Fail)
        print(" -> Training 8 Classification Models for Pass/Fail Prediction...")
        pf_suite = ClassificationModelSuite(task_type="binary", random_state=42)
        pf_cv = pf_suite.train_and_cross_validate(splits["X_train"], splits["y_pf_train"], cv=5)
        pf_test_results = pf_suite.evaluate_test_set(splits["X_test"], splits["y_pf_test"])
        print(f"    Best Pass/Fail Model: '{pf_suite.best_model_name}' (Accuracy: {pf_test_results[pf_suite.best_model_name]['Accuracy']:.4f}, F1: {pf_test_results[pf_suite.best_model_name]['F1_Score']:.4f})")
        
        # 5c. Multi-Class Classification (Performance Categories)
        print(" -> Training 8 Classification Models for Performance Category Prediction...")
        cat_suite = ClassificationModelSuite(task_type="multiclass", random_state=42)
        cat_cv = cat_suite.train_and_cross_validate(splits["X_train"], splits["y_cat_train"], cv=5)
        cat_test_results = cat_suite.evaluate_test_set(splits["X_test"], splits["y_cat_test"])
        print(f"    Best Category Model: '{cat_suite.best_model_name}' (Accuracy: {cat_test_results[cat_suite.best_model_name]['Accuracy']:.4f}, F1: {cat_test_results[cat_suite.best_model_name]['F1_Score']:.4f})")
        
        # 6. Evaluation Visualizations & Diagnostics
        print("\n[Stage 6/6] Generating Comprehensive Model Diagnostics & Leaderboard Visualizations...")
        eval_viz = EvaluationVisualizer(output_dir=self.figures_dir)
        
        fig_reg_comp = eval_viz.plot_regression_comparison(reg_test_results)
        fig_pf_comp = eval_viz.plot_classification_comparison(pf_test_results, title="Pass/Fail Classification")
        fig_cat_comp = eval_viz.plot_classification_comparison(cat_test_results, title="Performance Category Classification")
        
        feat_importances = reg_suite.get_feature_importances(splits["feature_names"])
        fig_feat_imp = eval_viz.plot_feature_importance(feat_importances, top_n=12)
        
        y_reg_pred_best = reg_suite.best_model.predict(splits["X_test"])
        fig_residuals = eval_viz.plot_actual_vs_predicted(splits["y_reg_test"], y_reg_pred_best, reg_suite.best_model_name)
        
        cm_pf = pf_test_results[pf_suite.best_model_name]["Confusion_Matrix"]
        cm_cat = cat_test_results[cat_suite.best_model_name]["Confusion_Matrix"]
        fig_cm = eval_viz.plot_confusion_matrices(
            cm_pf=cm_pf,
            cm_cat=cm_cat,
            pf_labels=list(preprocessor.label_encoder_pass_fail.classes_),
            cat_labels=list(preprocessor.label_encoder_category.classes_)
        )
        
        # 7. Serialize Artifacts
        print("\nSaving Optimized Model Artifacts to 'models/'...")
        preprocessor.save(os.path.join(self.models_dir, "preprocessor.joblib"))
        joblib.dump(reg_suite.best_model, os.path.join(self.models_dir, "best_regression_model.joblib"))
        joblib.dump(pf_suite.best_model, os.path.join(self.models_dir, "best_pass_fail_model.joblib"))
        joblib.dump(cat_suite.best_model, os.path.join(self.models_dir, "best_category_model.joblib"))
        
        metadata = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_records": len(raw_df),
            "features_used": splits["feature_names"],
            "best_regression_model": reg_suite.best_model_name,
            "best_regression_metrics": reg_test_results[reg_suite.best_model_name],
            "best_pass_fail_model": pf_suite.best_model_name,
            "best_pass_fail_metrics": pf_test_results[pf_suite.best_model_name],
            "best_category_model": cat_suite.best_model_name,
            "best_category_metrics": cat_test_results[cat_suite.best_model_name],
            "regression_leaderboard": reg_test_results,
            "pass_fail_leaderboard": pf_test_results,
            "category_leaderboard": cat_test_results,
            "feature_importances": feat_importances
        }
        joblib.dump(metadata, os.path.join(self.models_dir, "model_metadata.joblib"))
        
        # Save JSON metrics report
        metrics_json_path = os.path.join(self.reports_dir, "model_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Write Markdown Benchmark Report
        self._write_markdown_report(metadata, eda_summary)
        
        elapsed = time.time() - start_time
        print(f"\n Pipeline Execution Completed in {elapsed:.2f} seconds.")
        print(f" Model artifacts saved to '{self.models_dir}'")
        print(f" Metrics & reports saved to '{self.reports_dir}'")
        print("=" * 80)
        
        return metadata

    def _write_markdown_report(self, meta: Dict[str, Any], eda_summary: Dict[str, Any]) -> None:
        """Generate comprehensive Markdown report summarizing pipeline results."""
        report_path = os.path.join(self.reports_dir, "benchmark_report.md")
        
        md = f"""# Machine Learning Benchmark Report: Predicting Student Academic Performance

**Generated:** {meta['trained_at']}  
**Total Cohort Size:** {meta['dataset_records']} student records  

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
| **Final Grade Prediction (Regression)** | **{meta['best_regression_model']}** | **$R^2$ = {meta['best_regression_metrics']['R2_Score']}** | **RMSE = {meta['best_regression_metrics']['RMSE']}** (MAE = {meta['best_regression_metrics']['MAE']}) |
| **Pass/Fail Outcome (Binary Classification)** | **{meta['best_pass_fail_model']}** | **Accuracy = {meta['best_pass_fail_metrics']['Accuracy']*100:.2f}%** | **F1-Score = {meta['best_pass_fail_metrics']['F1_Score']*100:.2f}%** (ROC-AUC = {meta['best_pass_fail_metrics']['ROC_AUC']}) |
| **Performance Tier (Multi-Class Classification)** | **{meta['best_category_model']}** | **Accuracy = {meta['best_category_metrics']['Accuracy']*100:.2f}%** | **Weighted F1 = {meta['best_category_metrics']['F1_Score']*100:.2f}%** |

---

## Complete Model Leaderboard

### 1. Regression Models (Final Grade 0-100)
| Model Architecture | $R^2$ Score | RMSE | MAE | MAPE (%) | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for m, met in sorted(meta['regression_leaderboard'].items(), key=lambda x: x[1]['R2_Score'], reverse=True):
            md += f"| {m} | **{met['R2_Score']:.4f}** | {met['RMSE']:.2f} | {met['MAE']:.2f} | {met['MAPE_Percent']:.2f}% | {met['CV_R2_Mean']:.4f} |\n"
            
        md += """
### 2. Pass / Fail Binary Classification
| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
        for m, met in sorted(meta['pass_fail_leaderboard'].items(), key=lambda x: x[1]['F1_Score'], reverse=True):
            roc_str = f"{met['ROC_AUC']:.4f}" if isinstance(met['ROC_AUC'], float) else met['ROC_AUC']
            md += f"| {m} | **{met['Accuracy']*100:.2f}%** | {met['Precision']*100:.2f}% | {met['Recall']*100:.2f}% | **{met['F1_Score']*100:.2f}%** | {roc_str} |\n"
            
        md += """
### 3. Performance Category Multi-Class Classification
| Model Architecture | Accuracy | Precision | Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
"""
        for m, met in sorted(meta['category_leaderboard'].items(), key=lambda x: x[1]['F1_Score'], reverse=True):
            md += f"| {m} | **{met['Accuracy']*100:.2f}%** | {met['Precision']*100:.2f}% | {met['Recall']*100:.2f}% | **{met['F1_Score']*100:.2f}%** |\n"
            
        md += """
---

## Key Predictive Drivers (Feature Importance)
Top educational, demographic, and behavioral features ranked by predictive weight:

"""
        for feat, score in list(meta['feature_importances'].items())[:8]:
            md += f"- **{feat.replace('_', ' ').title()}**: `{score:.2f}%` relative contribution\n"
            
        md += """
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
"""
        with open(report_path, "w") as f:
            f.write(md)

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run_pipeline()
