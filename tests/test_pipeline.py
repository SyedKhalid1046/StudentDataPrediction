"""
Comprehensive Unit and Integration Test Suite
Validates Data Loading, Feature Engineering, Preprocessing, ML Model Suites, Predictor Inference, and Flask API endpoints.
"""

import os
import sys
import unittest
import json
import numpy as np
import pandas as pd

# Bootstrap root path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.models_regression import RegressionModelSuite
from src.models_classification import ClassificationModelSuite
from src.predictor import StudentPredictor
from app import app

class TestStudentMLPipeline(unittest.TestCase):
    """Test suite covering the machine learning pipeline and web endpoints."""
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(_ROOT_DIR, "data", "student_performance_data.csv")
        cls.loader = DataLoader(cls.data_path)
        cls.fe = FeatureEngineer()
        cls.prep = Preprocessor()
        cls.predictor = StudentPredictor(models_dir=os.path.join(_ROOT_DIR, "models"))
        cls.client = app.test_client()
        
    def test_01_data_loader(self):
        """Test dataset loading, schema verification, and summary profile."""
        self.assertIsNotNone(self.loader.df)
        self.assertGreaterEqual(len(self.loader.df), 500)
        
        profile = self.loader.get_summary_profile()
        self.assertIn("total_records", profile)
        self.assertIn("final_grade", profile["numeric_columns"])
        self.assertIn("gender", profile["categorical_columns"])
        
    def test_02_feature_engineering(self):
        """Test domain feature creation (risk index, study efficiency, socioeconomic index)."""
        df_feat = self.fe.transform(self.loader.df.head(50))
        
        self.assertIn("academic_risk_index", df_feat.columns)
        self.assertIn("study_efficiency", df_feat.columns)
        self.assertIn("socioeconomic_support_score", df_feat.columns)
        self.assertIn("attendance_tier", df_feat.columns)
        
        # Risk index should be bounded between 0 and 100
        self.assertTrue((df_feat["academic_risk_index"] >= 0).all())
        self.assertTrue((df_feat["academic_risk_index"] <= 100).all())
        
    def test_03_preprocessing_splits(self):
        """Test ColumnTransformer encoding, scaling, and stratified train/test split."""
        df_feat = self.fe.transform(self.loader.df)
        splits = self.prep.fit_transform_dataset(df_feat, test_size=0.2, random_state=42)
        
        self.assertEqual(splits["X_train"].shape[0] + splits["X_test"].shape[0], len(df_feat))
        self.assertEqual(len(splits["y_reg_train"]), splits["X_train"].shape[0])
        self.assertGreater(splits["X_train"].shape[1], 15)
        
    def test_04_regression_suite(self):
        """Test training and evaluation of regression models."""
        df_feat = self.fe.transform(self.loader.df.head(200))
        splits = self.prep.fit_transform_dataset(df_feat, test_size=0.25, random_state=42)
        
        reg_suite = RegressionModelSuite()
        reg_suite.train_and_cross_validate(splits["X_train"], splits["y_reg_train"], cv=3)
        results = reg_suite.evaluate_test_set(splits["X_test"], splits["y_reg_test"])
        
        self.assertIsNotNone(reg_suite.best_model_name)
        self.assertIn(reg_suite.best_model_name, results)
        self.assertGreater(results[reg_suite.best_model_name]["R2_Score"], 0.70)
        
    def test_05_classification_suite(self):
        """Test binary and multi-class classification models."""
        df_feat = self.fe.transform(self.loader.df.head(200))
        splits = self.prep.fit_transform_dataset(df_feat, test_size=0.25, random_state=42)
        
        # Binary
        pf_suite = ClassificationModelSuite(task_type="binary")
        pf_suite.train_and_cross_validate(splits["X_train"], splits["y_pf_train"], cv=3)
        pf_res = pf_suite.evaluate_test_set(splits["X_test"], splits["y_pf_test"])
        self.assertGreater(pf_res[pf_suite.best_model_name]["Accuracy"], 0.80)
        
    def test_06_single_prediction_inference(self):
        """Test StudentPredictor single prediction with prescriptive guidance."""
        self.assertTrue(self.predictor.is_ready())
        
        sample = {
            "gender": "Female",
            "age": 17,
            "parental_education": "Bachelor",
            "study_time": 12.0,
            "attendance_rate": 92.0,
            "previous_grades": 85.0,
            "extracurricular_activities": "Yes",
            "internet_access": "Yes",
            "tutoring": "No",
            "family_support": "Yes",
            "health": "Good",
            "absences": 2,
            "failures": 0
        }
        
        res = self.predictor.predict_single(sample)
        
        self.assertIn("predicted_grade", res)
        self.assertIn("pass_fail_status", res)
        self.assertIn("performance_category", res)
        self.assertIn("pass_probability", res)
        self.assertIn("recommendations", res)
        self.assertGreaterEqual(res["predicted_grade"], 0.0)
        self.assertLessEqual(res["predicted_grade"], 100.0)
        self.assertGreaterEqual(len(res["recommendations"]), 1)
        
    def test_07_batch_prediction_inference(self):
        """Test StudentPredictor batch prediction."""
        df_sample = self.loader.df.head(10).drop(columns=["final_grade", "pass_fail_status", "performance_category"], errors="ignore")
        res_df = self.predictor.predict_batch(df_sample)
        
        self.assertEqual(len(res_df), 10)
        self.assertIn("predicted_final_grade", res_df.columns)
        self.assertIn("predicted_pass_fail", res_df.columns)
        self.assertIn("pass_probability_pct", res_df.columns)
        
    def test_08_flask_api_endpoints(self):
        """Test Flask REST API routes."""
        # GET /
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        
        # GET /api/summary
        res_sum = self.client.get("/api/summary")
        self.assertEqual(res_sum.status_code, 200)
        data_sum = json.loads(res_sum.data)
        self.assertEqual(data_sum["status"], "success")
        
        # GET /api/models
        res_models = self.client.get("/api/models")
        self.assertEqual(res_models.status_code, 200)
        data_models = json.loads(res_models.data)
        self.assertEqual(data_models["status"], "success")
        
        # POST /api/predict
        payload = {
            "gender": "Male",
            "age": 18,
            "parental_education": "High School",
            "study_time": 6.0,
            "attendance_rate": 78.0,
            "previous_grades": 62.0,
            "extracurricular_activities": "No",
            "internet_access": "Yes",
            "tutoring": "No",
            "family_support": "No",
            "health": "Fair",
            "absences": 8,
            "failures": 1
        }
        res_pred = self.client.post("/api/predict", json=payload)
        self.assertEqual(res_pred.status_code, 200)
        data_pred = json.loads(res_pred.data)
        self.assertEqual(data_pred["status"], "success")
        self.assertIn("predicted_grade", data_pred["prediction"])

    def test_09_batch_prediction_edge_cases(self):
        """Test batch prediction with missing columns, alias headers, and NaN values."""
        # Messy headers
        df_messy = pd.DataFrame([{
            "Student ID": "STU_99",
            "Gender": "Male",
            "Study Time (hrs)": 10.0,
            "Attendance Rate": 82.0,
            "Previous Grade": 78.0,
            "Parent Education": "Bachelor"
        }])
        res_messy = self.predictor.predict_batch(df_messy)
        self.assertEqual(len(res_messy), 1)
        self.assertIn("predicted_final_grade", res_messy.columns)
        self.assertIn("risk_tier", res_messy.columns)

        # Missing values (NaN)
        df_nans = pd.DataFrame([{
            "student_id": "STU_100",
            "attendance_rate": np.nan,
            "study_time": None,
            "previous_grades": 70.0
        }])
        res_nans = self.predictor.predict_batch(df_nans)
        self.assertEqual(len(res_nans), 1)
        self.assertFalse(np.isnan(res_nans["predicted_final_grade"].iloc[0]))

    def test_10_batch_api_upload(self):
        """Test /api/batch-predict endpoint with uploaded CSV data."""
        sample_csv = "student_id,attendance_rate,study_time,previous_grades\nSTU_1,90,12,85\nSTU_2,65,4,50\n"
        import io
        data = {
            "file": (io.BytesIO(sample_csv.encode("utf-8")), "cohort.csv")
        }
        res = self.client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = json.loads(res.data)
        self.assertEqual(json_data["status"], "success")
        self.assertEqual(json_data["summary"]["total_students"], 2)
        self.assertEqual(len(json_data["records"]), 2)

if __name__ == "__main__":
    unittest.main()
