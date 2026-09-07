"""
Data Preprocessing and Pipeline Transformation Module
Encapsulates encoders, scalers, imputers, and dataset splitters.
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

NUMERIC_FEATURES = [
    "age",
    "study_time",
    "attendance_rate",
    "previous_grades",
    "absences",
    "failures",
    "academic_risk_index",
    "study_efficiency",
    "socioeconomic_support_score",
    "study_attendance_interaction"
]

CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "extracurricular_activities",
    "internet_access",
    "tutoring",
    "family_support",
    "health",
    "attendance_tier",
    "previous_grade_tier"
]

NUMERIC_DEFAULTS = {
    "age": 17.0,
    "study_time": 8.0,
    "attendance_rate": 85.0,
    "previous_grades": 70.0,
    "absences": 4.0,
    "failures": 0.0,
    "academic_risk_index": 20.0,
    "study_efficiency": 7.5,
    "socioeconomic_support_score": 5.0,
    "study_attendance_interaction": 6.8
}

CATEGORICAL_DEFAULTS = {
    "gender": "Female",
    "parental_education": "Some College",
    "extracurricular_activities": "No",
    "internet_access": "Yes",
    "tutoring": "No",
    "family_support": "Yes",
    "health": "Good",
    "attendance_tier": "Good (85-95%)",
    "previous_grade_tier": "High (70-85)"
}

class Preprocessor:
    """Class to manage data transformation pipelines for training and inference."""
    
    def __init__(self):
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder_pass_fail = LabelEncoder()
        self.label_encoder_category = LabelEncoder()
        self.feature_names_out: List[str] = []
        self.is_fitted: bool = False
        
    def _build_column_transformer(self) -> ColumnTransformer:
        """Construct the scikit-learn ColumnTransformer."""
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, NUMERIC_FEATURES),
                ("cat", cat_pipeline, CATEGORICAL_FEATURES)
            ],
            remainder="drop"
        )
        return transformer
        
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit preprocessor pipelines and label encoders on dataframe."""
        self.column_transformer = self._build_column_transformer()
        self.column_transformer.fit(df)
        
        # Extract feature names
        num_cols = NUMERIC_FEATURES
        cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names_out = num_cols + cat_cols
        
        if "pass_fail_status" in df.columns:
            self.label_encoder_pass_fail.fit(df["pass_fail_status"])
        else:
            self.label_encoder_pass_fail.fit(["Fail", "Pass"])
            
        if "performance_category" in df.columns:
            self.label_encoder_category.fit(df["performance_category"])
        else:
            self.label_encoder_category.fit(["At Risk", "Satisfactory", "Good", "Excellent"])
            
        self.is_fitted = True
        return self
        
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input features to scaled numerical matrix."""
        if not self.is_fitted or self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        df_input = df.copy()
        for col, default_val in NUMERIC_DEFAULTS.items():
            if col not in df_input.columns:
                df_input[col] = default_val
            else:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(default_val)
                
        for col, default_val in CATEGORICAL_DEFAULTS.items():
            if col not in df_input.columns:
                df_input[col] = default_val
            else:
                df_input[col] = df_input[col].fillna(default_val).astype(str)
                
        return self.column_transformer.transform(df_input)
        
    def fit_transform_dataset(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fit preprocessor and split dataset into Train/Test partitions for all target types.
        
        Returns:
            Dict containing X_train, X_test, y_reg_train, y_reg_test, y_pf_train, y_pf_test,
            y_cat_train, y_cat_test, and raw test dataframe.
        """
        self.fit(df)
        
        # Targets
        y_reg = df["final_grade"].values
        y_pf = self.label_encoder_pass_fail.transform(df["pass_fail_status"])
        y_cat = self.label_encoder_category.transform(df["performance_category"])
        
        # Stratify on performance category to maintain balanced representation in test split
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=df["performance_category"]
        )
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        X_train = self.transform_features(df_train)
        X_test = self.transform_features(df_test)
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_reg_train": y_reg[train_idx],
            "y_reg_test": y_reg[test_idx],
            "y_pf_train": y_pf[train_idx],
            "y_pf_test": y_pf[test_idx],
            "y_cat_train": y_cat[train_idx],
            "y_cat_test": y_cat[test_idx],
            "df_train": df_train,
            "df_test": df_test,
            "feature_names": self.feature_names_out
        }
        
    def save(self, filepath: str) -> None:
        """Serialize preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "Preprocessor":
        """Load serialized preprocessor from disk."""
        return joblib.load(filepath)

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    
    loader = DataLoader("data/student_performance_data.csv")
    fe = FeatureEngineer()
    df_feat = fe.transform(loader.df)
    
    prep = Preprocessor()
    data_splits = prep.fit_transform_dataset(df_feat)
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {data_splits['X_train'].shape}, X_test shape: {data_splits['X_test'].shape}")
    print(f"Transformed feature count: {len(data_splits['feature_names'])}")
