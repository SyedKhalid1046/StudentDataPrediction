"""
Classification Models Suite for Pass/Fail & Performance Tier Prediction
Trains, tunes, cross-validates, and evaluates binary and multi-class classification models.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib

class ClassificationModelSuite:
    """Suite of classification algorithms for Pass/Fail and Performance Category predictions."""
    
    def __init__(self, task_type: str = "binary", random_state: int = 42):
        """
        Parameters:
            task_type (str): 'binary' (Pass/Fail) or 'multiclass' (Performance Categories).
            random_state (int): Seed for reproducibility.
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models: Dict[str, Any] = self._init_models()
        self.fitted_models: Dict[str, Any] = {}
        self.cv_scores: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        
    def _init_models(self) -> Dict[str, Any]:
        """Instantiate candidate classification algorithms."""
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=8, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=6, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=self.random_state),
            "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.07, max_depth=4, subsample=0.85, random_state=self.random_state, n_jobs=-1, eval_metric="logloss"),
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.07, max_depth=4, num_leaves=15, random_state=self.random_state, n_jobs=-1, verbose=-1),
            "Support Vector Classifier (SVC)": SVC(C=2.0, kernel="rbf", probability=True, random_state=self.random_state),
            "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, alpha=0.01, random_state=self.random_state)
        }
        
    def train_and_cross_validate(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Fit all classification models and run stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scoring = "accuracy" if self.task_type == "binary" else "f1_weighted"
        
        for name, model in self.models.items():
            acc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
            f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1)
            
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            
            self.cv_scores[name] = {
                "cv_accuracy_mean": float(np.mean(acc_scores)),
                "cv_accuracy_std": float(np.std(acc_scores)),
                "cv_f1_mean": float(np.mean(f1_scores)),
                "cv_f1_std": float(np.std(f1_scores))
            }
            
        # Select best model based on CV F1-Score
        self.best_model_name = max(self.cv_scores, key=lambda k: self.cv_scores[k]["cv_f1_mean"])
        self.best_model = self.fitted_models[self.best_model_name]
        
        return self.cv_scores
        
    def evaluate_test_set(self, X_test: np.ndarray, y_test: np.ndarray, class_names: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate all classification models on unseen test partition."""
        test_results = {}
        
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            roc_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                    if self.task_type == "binary":
                        roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                    else:
                        roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"))
                except Exception:
                    roc_auc = None
                    
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            test_results[name] = {
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4),
                "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else "N/A",
                "Confusion_Matrix": cm,
                "CV_F1_Mean": round(self.cv_scores.get(name, {}).get("cv_f1_mean", 0.0), 4)
            }
            
        return test_results

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import Preprocessor
    
    loader = DataLoader("data/student_performance_data.csv")
    fe = FeatureEngineer()
    df_feat = fe.transform(loader.df)
    
    prep = Preprocessor()
    splits = prep.fit_transform_dataset(df_feat)
    
    # Binary Classification (Pass/Fail)
    suite_pf = ClassificationModelSuite(task_type="binary")
    suite_pf.train_and_cross_validate(splits["X_train"], splits["y_pf_train"])
    results_pf = suite_pf.evaluate_test_set(splits["X_test"], splits["y_pf_test"])
    
    print("\nPass/Fail Classification Benchmark:")
    for m, met in results_pf.items():
        print(f"[{m}] Accuracy: {met['Accuracy']:.4f} | F1: {met['F1_Score']:.4f} | ROC-AUC: {met['ROC_AUC']}")
    print(f"\nBest Pass/Fail Model: {suite_pf.best_model_name}")
    
    # Multi-Class Classification (Performance Category)
    suite_cat = ClassificationModelSuite(task_type="multiclass")
    suite_cat.train_and_cross_validate(splits["X_train"], splits["y_cat_train"])
    results_cat = suite_cat.evaluate_test_set(splits["X_test"], splits["y_cat_test"])
    
    print("\nPerformance Category Classification Benchmark:")
    for m, met in results_cat.items():
        print(f"[{m}] Accuracy: {met['Accuracy']:.4f} | F1: {met['F1_Score']:.4f}")
    print(f"\nBest Category Model: {suite_cat.best_model_name}")
