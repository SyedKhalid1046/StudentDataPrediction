"""
Regression Models Suite for Final Grade Prediction
Trains, tunes, cross-validates, and evaluates continuous grade predictors.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
import joblib

class RegressionModelSuite:
    """Suite of regression algorithms for student grade prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = self._init_models()
        self.fitted_models: Dict[str, Any] = {}
        self.cv_scores: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        
    def _init_models(self) -> Dict[str, Any]:
        """Instantiate competitive candidate regression models with tuned hyperparameters."""
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Lasso Regression": Lasso(alpha=0.05, random_state=self.random_state),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_split=8, random_state=self.random_state),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=6, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=self.random_state),
            "XGBoost": xgb.XGBRegressor(n_estimators=150, learning_rate=0.07, max_depth=4, subsample=0.85, random_state=self.random_state, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.07, max_depth=4, num_leaves=15, random_state=self.random_state, n_jobs=-1, verbose=-1),
            "Support Vector Regressor (SVR)": SVR(C=3.0, epsilon=0.15, kernel="rbf"),
            "Multi-Layer Perceptron (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, alpha=0.01, random_state=self.random_state)
        }
        
    def train_and_cross_validate(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Fit all regression models and perform k-fold cross-validation."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            # Cross-validation for RMSE and R2
            neg_mse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
            r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
            
            rmse_scores = np.sqrt(-neg_mse_scores)
            
            # Fit model on entire training partition
            model.fit(X_train, y_train)
            self.fitted_models[name] = model
            
            self.cv_scores[name] = {
                "cv_rmse_mean": float(np.mean(rmse_scores)),
                "cv_rmse_std": float(np.std(rmse_scores)),
                "cv_r2_mean": float(np.mean(r2_scores)),
                "cv_r2_std": float(np.std(r2_scores))
            }
            
        # Select best model based on CV R2
        self.best_model_name = max(self.cv_scores, key=lambda k: self.cv_scores[k]["cv_r2_mean"])
        self.best_model = self.fitted_models[self.best_model_name]
        
        return self.cv_scores
        
    def evaluate_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute evaluation metrics across all fitted regression models on test set."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
        
        test_results = {}
        for name, model in self.fitted_models.items():
            y_pred = model.predict(X_test)
            
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1.0)))) * 100
            
            test_results[name] = {
                "RMSE": round(float(rmse), 3),
                "MAE": round(float(mae), 3),
                "R2_Score": round(float(r2), 4),
                "MAPE_Percent": round(float(mape), 2),
                "CV_R2_Mean": round(self.cv_scores.get(name, {}).get("cv_r2_mean", 0.0), 4)
            }
            
        return test_results
        
    def get_feature_importances(self, feature_names: list) -> Dict[str, float]:
        """Extract feature importances from best model if supported."""
        if self.best_model is None:
            return {}
            
        importances = None
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_)
            
        if importances is not None and len(importances) == len(feature_names):
            norm_imp = (importances / np.sum(importances)) * 100
            feat_imp_dict = {feat: round(float(imp), 2) for feat, imp in zip(feature_names, norm_imp)}
            # Sort descending
            return dict(sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True))
            
        return {}

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.preprocessing import Preprocessor
    
    loader = DataLoader("data/student_performance_data.csv")
    fe = FeatureEngineer()
    df_feat = fe.transform(loader.df)
    
    prep = Preprocessor()
    splits = prep.fit_transform_dataset(df_feat)
    
    suite = RegressionModelSuite()
    suite.train_and_cross_validate(splits["X_train"], splits["y_reg_train"])
    results = suite.evaluate_test_set(splits["X_test"], splits["y_reg_test"])
    
    print("\nRegression Models Benchmark:")
    for model_name, metrics in results.items():
        print(f"[{model_name}] R2: {metrics['R2_Score']:.4f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}")
    print(f"\nBest Regression Model: {suite.best_model_name}")
