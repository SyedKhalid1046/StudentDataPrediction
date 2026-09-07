"""
Feature Engineering Module for Student Academic Performance Prediction
Creates domain-specific indicators, risk scores, interaction terms, and normalized metrics.
"""

from typing import Optional
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Class to extract and engineer domain-specific educational features."""
    
    def __init__(self):
        pass
        
    def transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Parameters:
            df (pd.DataFrame): Raw or cleaned input DataFrame.
            is_training (bool): Whether target variables should be preserved.
            
        Returns:
            pd.DataFrame: Augmented DataFrame with engineered features.
        """
        df_feat = df.copy()
        
        n_rows = len(df_feat)
        
        # 1. Academic Risk Index (Composite penalty score 0 to 100)
        # Higher score = higher risk of academic underperformance
        if "attendance_rate" in df_feat.columns and not df_feat["attendance_rate"].empty:
            att_rate = pd.to_numeric(df_feat["attendance_rate"], errors="coerce").fillna(85.0)
        else:
            att_rate = pd.Series([85.0] * n_rows, index=df_feat.index, dtype=float)
            
        if "absences" in df_feat.columns and not df_feat["absences"].empty:
            absences = pd.to_numeric(df_feat["absences"], errors="coerce").fillna(4.0)
        else:
            absences = pd.Series([4.0] * n_rows, index=df_feat.index, dtype=float)
            
        if "failures" in df_feat.columns and not df_feat["failures"].empty:
            failures = pd.to_numeric(df_feat["failures"], errors="coerce").fillna(0.0)
        else:
            failures = pd.Series([0.0] * n_rows, index=df_feat.index, dtype=float)
            
        if "study_time" in df_feat.columns and not df_feat["study_time"].empty:
            study_hours = pd.to_numeric(df_feat["study_time"], errors="coerce").fillna(8.0)
        else:
            study_hours = pd.Series([8.0] * n_rows, index=df_feat.index, dtype=float)
            
        if "previous_grades" in df_feat.columns and not df_feat["previous_grades"].empty:
            prev_grades = pd.to_numeric(df_feat["previous_grades"], errors="coerce").fillna(70.0)
        else:
            prev_grades = pd.Series([70.0] * n_rows, index=df_feat.index, dtype=float)
        
        # Risk components
        att_penalty = np.maximum(0, (85.0 - att_rate) * 1.5)
        absence_penalty = np.minimum(30.0, absences * 2.0)
        failure_penalty = failures * 20.0
        study_penalty = np.maximum(0, (7.0 - study_hours) * 3.0)
        grade_penalty = np.maximum(0, (65.0 - prev_grades) * 1.2)
        
        academic_risk_index = att_penalty + absence_penalty + failure_penalty + study_penalty + grade_penalty
        df_feat["academic_risk_index"] = np.clip(np.round(academic_risk_index, 2), 0.0, 100.0)
        
        # 2. Study Efficiency Ratio (Previous grade produced per study hour)
        df_feat["study_efficiency"] = np.round(prev_grades / (study_hours + 1.0), 3)
        
        # 3. Socioeconomic & Institutional Support Score (0 to 10 scale)
        edu_score_map = {
            "none": 0.0,
            "high school": 2.5,
            "some college": 5.0,
            "bachelor": 7.5,
            "master/doctorate": 10.0
        }
        
        if "parental_education" in df_feat.columns and not df_feat["parental_education"].empty:
            edu_series = df_feat["parental_education"].astype(str).str.strip().str.lower()
            edu_score = edu_series.map(edu_score_map).fillna(5.0)
        else:
            edu_score = pd.Series([5.0] * n_rows, index=df_feat.index, dtype=float)
            
        def _get_binary_series(col_name: str, default_val: str = "yes") -> pd.Series:
            if col_name in df_feat.columns and not df_feat[col_name].empty:
                return df_feat[col_name].astype(str).str.strip().str.lower()
            return pd.Series([default_val] * n_rows, index=df_feat.index)
            
        net_series = _get_binary_series("internet_access", "yes")
        fam_series = _get_binary_series("family_support", "yes")
        tut_series = _get_binary_series("tutoring", "no")
        act_series = _get_binary_series("extracurricular_activities", "no")
        
        net_score = np.where(net_series == "yes", 2.0, 0.0)
        fam_score = np.where(fam_series == "yes", 2.5, 0.0)
        tut_score = np.where(tut_series == "yes", 2.5, 0.0)
        act_score = np.where(act_series == "yes", 1.5, 0.0)
        
        socioeconomic_support_score = (edu_score * 0.35) + (net_score + fam_score + tut_score + act_score) * 0.75
        df_feat["socioeconomic_support_score"] = np.clip(np.round(socioeconomic_support_score, 2), 0.0, 10.0)
        
        # 4. Attendance Risk Tier (Categorical)
        df_feat["attendance_tier"] = pd.cut(
            att_rate,
            bins=[-np.inf, 70.0, 85.0, 95.0, np.inf],
            labels=["Critical (<70%)", "Moderate (70-85%)", "Good (85-95%)", "Excellent (>95%)"]
        ).astype(str)
        
        # 5. Prior Grade Tier
        df_feat["previous_grade_tier"] = pd.cut(
            prev_grades,
            bins=[-np.inf, 55.0, 70.0, 85.0, np.inf],
            labels=["Low (<55)", "Average (55-70)", "High (70-85)", "Superior (>85)"]
        ).astype(str)
        
        # 6. Interaction Term: Study Time x Attendance Rate
        df_feat["study_attendance_interaction"] = np.round((study_hours * att_rate) / 100.0, 3)
        
        return df_feat

if __name__ == "__main__":
    from src.data_loader import DataLoader
    loader = DataLoader("data/student_performance_data.csv")
    fe = FeatureEngineer()
    df_engineered = fe.transform(loader.df)
    print("Feature Engineering Completed:")
    print(df_engineered[["student_id", "academic_risk_index", "study_efficiency", "socioeconomic_support_score", "attendance_tier"]].head())
