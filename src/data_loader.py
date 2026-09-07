"""
Data Loader and Validation Module
Handles loading datasets from CSV or Excel, schema verification, and summary profiling.
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

REQUIRED_FEATURES = [
    "gender",
    "age",
    "parental_education",
    "study_time",
    "attendance_rate",
    "previous_grades",
    "extracurricular_activities",
    "internet_access"
]

TARGET_COLUMNS = ["final_grade", "pass_fail_status", "performance_category"]

class DataLoader:
    """Class to load, validate, and summarize educational datasets."""
    
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        if file_path:
            self.load_data(file_path)
            
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .csv, .xlsx, or .xls")
            
        self.file_path = file_path
        self._clean_and_normalize()
        return self.df
        
    def _clean_and_normalize(self) -> None:
        """Perform initial normalization on column names and missing values."""
        if self.df is None:
            return
            
        # Strip column whitespace and convert to lowercase standard
        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
        
        # Ensure student_id exists or generate it
        if "student_id" not in self.df.columns:
            self.df.insert(0, "student_id", [f"STU_{1000 + i}" for i in range(len(self.df))])
            
        # Handle numeric types
        numeric_cols = ["age", "study_time", "attendance_rate", "previous_grades", "absences", "failures", "final_grade"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                
        # Forward/median fill for missing values if any
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
        for col in self.df.select_dtypes(include=["object", "category", "string", "str"]).columns:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                
        # If target classification columns are missing but final_grade is present, generate them
        if "final_grade" in self.df.columns:
            if "pass_fail_status" not in self.df.columns:
                self.df["pass_fail_status"] = np.where(self.df["final_grade"] >= 60.0, "Pass", "Fail")
            if "performance_category" not in self.df.columns:
                conds = [
                    (self.df["final_grade"] < 55.0),
                    (self.df["final_grade"] >= 55.0) & (self.df["final_grade"] < 70.0),
                    (self.df["final_grade"] >= 70.0) & (self.df["final_grade"] < 85.0),
                    (self.df["final_grade"] >= 85.0)
                ]
                cats = ["At Risk", "Satisfactory", "Good", "Excellent"]
                self.df["performance_category"] = np.select(conds, cats, default="Satisfactory")
                
    def get_summary_profile(self) -> Dict[str, Any]:
        """Generate high-level metadata and statistical summaries."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
            
        numeric_summary = self.df.describe().to_dict()
        missing_values = self.df.isnull().sum().to_dict()
        
        categorical_columns = self.df.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
        cat_distributions = {col: self.df[col].value_counts().to_dict() for col in categorical_columns}
        
        return {
            "total_records": len(self.df),
            "total_features": len(self.df.columns),
            "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": categorical_columns,
            "missing_values": missing_values,
            "numeric_summary": numeric_summary,
            "categorical_distributions": cat_distributions
        }

if __name__ == "__main__":
    loader = DataLoader("data/student_performance_data.csv")
    profile = loader.get_summary_profile()
    print("Dataset Loaded Successfully:")
    print(f"Total records: {profile['total_records']}, Total features: {profile['total_features']}")
