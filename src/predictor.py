import os
import sys
from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure root directory is on python path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor
from src.dataset_normalizer import SchemaAutoMapper, FlexibleDatasetReader

class StudentPredictor:
    """Class to load ML artifacts and generate academic predictions with prescriptive guidance."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.feature_engineer = FeatureEngineer()
        self.schema_mapper = SchemaAutoMapper()
        self.preprocessor: Optional[Preprocessor] = None
        self.reg_model: Optional[Any] = None
        self.pf_model: Optional[Any] = None
        self.cat_model: Optional[Any] = None
        self.model_metadata: Dict[str, Any] = {}
        
        self.load_artifacts()
        
    def load_artifacts(self) -> None:
        """Load serialized models, preprocessors, and evaluation metadata."""
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        reg_path = os.path.join(self.models_dir, "best_regression_model.joblib")
        pf_path = os.path.join(self.models_dir, "best_pass_fail_model.joblib")
        cat_path = os.path.join(self.models_dir, "best_category_model.joblib")
        meta_path = os.path.join(self.models_dir, "model_metadata.joblib")
        
        if os.path.exists(prep_path):
            self.preprocessor = Preprocessor.load(prep_path)
        if os.path.exists(reg_path):
            self.reg_model = joblib.load(reg_path)
        if os.path.exists(pf_path):
            self.pf_model = joblib.load(pf_path)
        if os.path.exists(cat_path):
            self.cat_model = joblib.load(cat_path)
        if os.path.exists(meta_path):
            self.model_metadata = joblib.load(meta_path)
            
    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return bool(self.preprocessor and self.reg_model and self.pf_model and self.cat_model)
        
    def predict_single(self, student: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate academic predictions and intervention guidance for a single student.
        
        Parameters:
            student (dict): Student features dictionary.
            
        Returns:
            dict: Predictions, probabilities, risk indicators, and recommendations.
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded. Please run pipeline training first.")
            
        # Defaults for missing optional features
        std_dict = {
            "gender": str(student.get("gender", "Female")),
            "age": int(student.get("age", 17)),
            "parental_education": str(student.get("parental_education", "Some College")),
            "study_time": float(student.get("study_time", 8.0)),
            "attendance_rate": float(student.get("attendance_rate", 85.0)),
            "previous_grades": float(student.get("previous_grades", 70.0)),
            "extracurricular_activities": str(student.get("extracurricular_activities", "No")),
            "internet_access": str(student.get("internet_access", "Yes")),
            "tutoring": str(student.get("tutoring", "No")),
            "family_support": str(student.get("family_support", "Yes")),
            "health": str(student.get("health", "Good")),
            "absences": int(student.get("absences", max(0, int((100 - float(student.get("attendance_rate", 85.0))) * 0.35)))),
            "failures": int(student.get("failures", 0))
        }
        
        df_single = pd.DataFrame([std_dict])
        df_feat = self.feature_engineer.transform(df_single, is_training=False)
        X_mat = self.preprocessor.transform_features(df_feat)
        
        # 1. Continuous Grade Prediction
        pred_grade = float(np.clip(self.reg_model.predict(X_mat)[0], 0.0, 100.0))
        pred_grade = round(pred_grade, 1)
        
        # 2. Pass / Fail Classification & Probability
        pf_encoded = self.pf_model.predict(X_mat)[0]
        pass_fail_label = self.preprocessor.label_encoder_pass_fail.inverse_transform([pf_encoded])[0]
        
        pass_prob = 0.5
        if hasattr(self.pf_model, "predict_proba"):
            probs = self.pf_model.predict_proba(X_mat)[0]
            # Identify index of 'Pass' class
            pass_idx = list(self.preprocessor.label_encoder_pass_fail.classes_).index("Pass") if "Pass" in self.preprocessor.label_encoder_pass_fail.classes_ else 1
            pass_prob = round(float(probs[pass_idx]) * 100, 1)
            
        # 3. Performance Category Classification & Distribution
        cat_encoded = self.cat_model.predict(X_mat)[0]
        category_label = self.preprocessor.label_encoder_category.inverse_transform([cat_encoded])[0]
        
        cat_probs = {}
        if hasattr(self.cat_model, "predict_proba"):
            probs_cat = self.cat_model.predict_proba(X_mat)[0]
            for cat_name, prob in zip(self.preprocessor.label_encoder_category.classes_, probs_cat):
                cat_probs[cat_name] = round(float(prob) * 100, 1)
                
        # 4. Risk Level Assessment
        risk_score = float(df_feat["academic_risk_index"].iloc[0])
        if risk_score >= 40.0 or pass_prob < 50.0 or pred_grade < 55.0:
            risk_tier = "High Risk"
            risk_color = "#EF4444"
        elif risk_score >= 20.0 or pass_prob < 75.0 or pred_grade < 70.0:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B"
        else:
            risk_tier = "Low Risk"
            risk_color = "#10B981"
            
        # 5. Prescriptive AI Educational Interventions
        recommendations = self._generate_recommendations(std_dict, pred_grade, pass_prob, risk_score)
        
        return {
            "predicted_grade": pred_grade,
            "predicted_grade_20_scale": round((pred_grade / 100.0) * 20.0, 1),
            "pass_fail_status": pass_fail_label,
            "pass_probability": pass_prob,
            "performance_category": category_label,
            "category_probabilities": cat_probs,
            "academic_risk_index": risk_score,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "study_efficiency": float(df_feat["study_efficiency"].iloc[0]),
            "socioeconomic_support_score": float(df_feat["socioeconomic_support_score"].iloc[0]),
            "recommendations": recommendations
        }
        
    def _generate_recommendations(self, student: Dict[str, Any], pred_grade: float, pass_prob: float, risk_score: float) -> List[Dict[str, str]]:
        """Generate targeted, actionable interventions tailored to the student's profile."""
        recs = []
        
        # Attendance intervention
        att = student["attendance_rate"]
        if att < 80.0:
            gain = round((85.0 - att) * 0.16, 1)
            recs.append({
                "category": "Attendance",
                "priority": "High",
                "action": f"Improve class attendance from {att:.0f}% to 85%+",
                "expected_impact": f"Projected grade boost of +{gain} to +{gain+2.5} points."
            })
        elif att < 90.0:
            recs.append({
                "category": "Attendance",
                "priority": "Medium",
                "action": "Maintain consistent classroom participation and minimize unexcused absences.",
                "expected_impact": "Stabilizes academic retention and exam readiness."
            })
            
        # Study Hours intervention
        study = student["study_time"]
        if study < 7.0:
            suggested = round(study + 4.0, 1)
            gain = round(4.0 * 0.52, 1)
            recs.append({
                "category": "Study Habits",
                "priority": "High",
                "action": f"Increase structured weekly study time from {study:.1f} hrs to {suggested:.1f} hrs.",
                "expected_impact": f"Estimated grade improvement of +{gain} points."
            })
            
        # Tutoring intervention
        if student["tutoring"].lower() != "yes" and (pred_grade < 70.0 or student["failures"] > 0):
            recs.append({
                "category": "Academic Support",
                "priority": "High",
                "action": "Enroll in specialized peer or 1-on-1 tutoring sessions in core subjects.",
                "expected_impact": "Expected grade lift of +3.0 to +5.0 points and failure mitigation."
            })
            
        # Prior Failures intervention
        if student["failures"] > 0:
            recs.append({
                "category": "Remediation",
                "priority": "Critical",
                "action": f"Student has {student['failures']} prior subject failure(s). Implement an academic recovery contract.",
                "expected_impact": "Prevents compounding credit deficit and reduces dropout risk."
            })
            
        # Extracurricular & Wellbeing
        if student["extracurricular_activities"].lower() != "yes":
            recs.append({
                "category": "Engagement",
                "priority": "Low",
                "action": "Participate in structured campus clubs, athletics, or academic societies.",
                "expected_impact": "Enhances school connectedness and long-term academic motivation."
            })
            
        if not recs:
            recs.append({
                "category": "Advanced Enrichment",
                "priority": "Maintenance",
                "action": "Student demonstrates exemplary performance. Recommend advanced placement/honors coursework.",
                "expected_impact": "Maximizes collegiate competitive positioning."
            })
            
        return recs

    def _normalize_batch_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and ensure all standard features exist with default values."""
        df_norm, _ = self.schema_mapper.process(df)
        return df_norm

    def predict_batch(
        self,
        df: Union[pd.DataFrame, str, bytes, Any],
        filename: str = "dataset.csv",
        return_metadata: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Execute batch predictions across an arbitrary student dataset DataFrame or file input.
        
        Parameters:
            df: pd.DataFrame or file input (path, bytes, stream).
            filename: Original file name (used for format detection if file input is provided).
            return_metadata: If True, returns (results_dataframe, schema_mapping_metadata).
            
        Returns:
            pd.DataFrame or (pd.DataFrame, dict): Enriched predictions DataFrame with risk triage and metadata.
        """
        if not self.is_ready():
            raise RuntimeError("Models not loaded. Please run pipeline training first.")
            
        # 1. Ingest input if raw file or stream provided
        if not isinstance(df, pd.DataFrame):
            df_raw = FlexibleDatasetReader.read(df, filename=filename)
        else:
            df_raw = df.copy()
            
        # 2. Flexible schema normalization and fuzzy mapping
        df_norm, schema_meta = self.schema_mapper.process(df_raw)
        
        # 3. Feature engineering & transformation
        df_feat = self.feature_engineer.transform(df_norm, is_training=False)
        X_mat = self.preprocessor.transform_features(df_feat)
        
        # 4. Multi-model inference
        pred_grades = np.clip(self.reg_model.predict(X_mat), 0.0, 100.0).round(1)
        pf_encoded = self.pf_model.predict(X_mat)
        pf_labels = self.preprocessor.label_encoder_pass_fail.inverse_transform(pf_encoded)
        
        cat_encoded = self.cat_model.predict(X_mat)
        cat_labels = self.preprocessor.label_encoder_category.inverse_transform(cat_encoded)
        
        pass_probs = []
        if hasattr(self.pf_model, "predict_proba"):
            probs = self.pf_model.predict_proba(X_mat)
            pass_idx = list(self.preprocessor.label_encoder_pass_fail.classes_).index("Pass") if "Pass" in self.preprocessor.label_encoder_pass_fail.classes_ else 1
            pass_probs = (probs[:, pass_idx] * 100).round(1)
        else:
            pass_probs = np.where(pf_labels == "Pass", 85.0, 35.0)
            
        # 5. Determine risk tier and color
        risk_scores = df_feat["academic_risk_index"].values
        risk_tiers = []
        risk_colors = []
        
        for r_score, p_prob, p_grade in zip(risk_scores, pass_probs, pred_grades):
            if r_score >= 40.0 or p_prob < 50.0 or p_grade < 55.0:
                risk_tiers.append("High Risk")
                risk_colors.append("#EF4444")
            elif r_score >= 20.0 or p_prob < 75.0 or p_grade < 70.0:
                risk_tiers.append("Moderate Risk")
                risk_colors.append("#F59E0B")
            else:
                risk_tiers.append("Low Risk")
                risk_colors.append("#10B981")
                
        # 6. Construct enriched results DataFrame (preserving extra columns from uploaded dataset)
        res_df = df_norm.copy()
        
        # Preserve extra unrecognized columns for user context if not conflicting
        for col in schema_meta.get("extra_columns", []):
            if col in df_raw.columns and col not in res_df.columns:
                res_df[col] = df_raw[col].values
                
        res_df["predicted_final_grade"] = pred_grades
        res_df["predicted_pass_fail"] = pf_labels
        res_df["pass_probability_pct"] = pass_probs
        res_df["predicted_performance_category"] = cat_labels
        res_df["academic_risk_index"] = risk_scores
        res_df["risk_tier"] = risk_tiers
        res_df["risk_color"] = risk_colors
        
        # Attach schema metadata as dataframe attribute for convenience
        res_df.attrs["schema_metadata"] = schema_meta
        
        if return_metadata:
            return res_df, schema_meta
        return res_df

if __name__ == "__main__":
    predictor = StudentPredictor()
    if predictor.is_ready():
        sample_student = {
            "gender": "Female",
            "age": 17,
            "parental_education": "Bachelor",
            "study_time": 10.5,
            "attendance_rate": 92.0,
            "previous_grades": 84.0,
            "extracurricular_activities": "Yes",
            "internet_access": "Yes",
            "tutoring": "No",
            "family_support": "Yes",
            "health": "Good",
            "absences": 2,
            "failures": 0
        }
        res = predictor.predict_single(sample_student)
        print("Sample Prediction Result:")
        print(f"Predicted Grade: {res['predicted_grade']} ({res['performance_category']})")
        print(f"Pass Probability: {res['pass_probability']}% | Risk Tier: {res['risk_tier']}")
        print(f"Recommendations ({len(res['recommendations'])}):")
        for r in res['recommendations']:
            print(f" - [{r['category']} - {r['priority']}] {r['action']}")
    else:
        print("Models not trained yet. Run pipeline.py to train and save models.")
