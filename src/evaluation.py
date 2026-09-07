"""
Model Evaluation and Metrics Visualizer Module
Computes comparative performance tables, metrics summaries, and visual diagnostics.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationVisualizer:
    """Class to generate model evaluation charts and performance leaderboards."""
    
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_regression_comparison(self, reg_results: Dict[str, Dict[str, float]]) -> str:
        """Plot comparative R2, RMSE, and MAE bar charts across regression models."""
        filepath = os.path.join(self.output_dir, "model_regression_comparison.png")
        
        df_res = pd.DataFrame.from_dict(reg_results, orient="index").reset_index()
        df_res.rename(columns={"index": "Model"}, inplace=True)
        df_res.sort_values(by="R2_Score", ascending=True, inplace=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # Plot R2 Score
        sns.barplot(
            data=df_res,
            y="Model",
            x="R2_Score",
            palette="mako",
            ax=axes[0]
        )
        axes[0].set_title("Coefficient of Determination ($R^2$ Score)", fontsize=13, fontweight="bold", pad=10)
        axes[0].set_xlabel("$R^2$ Score (Higher is Better)", fontsize=10, fontweight="bold")
        axes[0].set_xlim(0, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_width():.3f}",
                             (p.get_width() - 0.08 if p.get_width() > 0.1 else 0.02, p.get_y() + p.get_height() / 2),
                             ha="center", va="center", color="white" if p.get_width() > 0.15 else "black",
                             fontweight="bold", fontsize=9)
                             
        # Plot RMSE
        sns.barplot(
            data=df_res,
            y="Model",
            x="RMSE",
            palette="rocket_r",
            ax=axes[1]
        )
        axes[1].set_title("Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold", pad=10)
        axes[1].set_xlabel("RMSE (Lower is Better)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("")
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_width():.2f}",
                             (p.get_width() / 2, p.get_y() + p.get_height() / 2),
                             ha="center", va="center", color="white", fontweight="bold", fontsize=9)
                             
        plt.suptitle("Comparative Performance Benchmark Across Regression Algorithms", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_classification_comparison(self, cls_results: Dict[str, Dict[str, Any]], title: str = "Pass/Fail Classification") -> str:
        """Plot comparative Accuracy and F1-Score bar charts across classification models."""
        filename = "model_classification_comparison.png" if "Pass" in title else "model_category_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        
        df_res = pd.DataFrame.from_dict(cls_results, orient="index").reset_index()
        df_res.rename(columns={"index": "Model"}, inplace=True)
        df_res.sort_values(by="F1_Score", ascending=True, inplace=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # Accuracy
        sns.barplot(
            data=df_res,
            y="Model",
            x="Accuracy",
            palette="viridis",
            ax=axes[0]
        )
        axes[0].set_title(f"Classification Accuracy - {title}", fontsize=13, fontweight="bold", pad=10)
        axes[0].set_xlabel("Accuracy (0 - 1.0)", fontsize=10, fontweight="bold")
        axes[0].set_xlim(0, 1.0)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_width():.3f}",
                             (p.get_width() - 0.08 if p.get_width() > 0.1 else 0.02, p.get_y() + p.get_height() / 2),
                             ha="center", va="center", color="white", fontweight="bold", fontsize=9)
                             
        # F1-Score
        sns.barplot(
            data=df_res,
            y="Model",
            x="F1_Score",
            palette="flare",
            ax=axes[1]
        )
        axes[1].set_title(f"Weighted F1-Score - {title}", fontsize=13, fontweight="bold", pad=10)
        axes[1].set_xlabel("F1-Score (0 - 1.0)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("")
        axes[1].set_xlim(0, 1.0)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_width():.3f}",
                             (p.get_width() - 0.08 if p.get_width() > 0.1 else 0.02, p.get_y() + p.get_height() / 2),
                             ha="center", va="center", color="white", fontweight="bold", fontsize=9)
                             
        plt.suptitle(f"Algorithm Benchmark: {title}", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_feature_importance(self, feature_importances: Dict[str, float], top_n: int = 12) -> str:
        """Plot horizontal bar chart of top predictor feature importances."""
        filepath = os.path.join(self.output_dir, "feature_importance_ranking.png")
        if not feature_importances:
            return ""
            
        top_features = list(feature_importances.items())[:top_n]
        names = [f[0].replace("_", " ").title() for f in top_features]
        values = [f[1] for f in top_features]
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        y_pos = np.arange(len(names))
        
        ax.barh(y_pos, values[::-1], color="#3B82F6", edgecolor="#1E3A8A", height=0.65, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative Importance Score (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"Top {top_n} Educational & Behavioral Features Driving Student Performance", fontsize=13, fontweight="bold", pad=12)
        
        for i, v in enumerate(values[::-1]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9, color="#1E293B")
            
        ax.set_xlim(0, max(values) * 1.15)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        """Plot Actual vs Predicted Final Grades and Residual distribution."""
        filepath = os.path.join(self.output_dir, "regression_residuals_plot.png")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        
        # 1. Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.55, color="#6366F1", edgecolor="none", s=35)
        min_v = min(min(y_true), min(y_pred))
        max_v = max(max(y_true), max(y_pred))
        axes[0].plot([min_v, max_v], [min_v, max_v], color="#DC2626", linestyle="--", linewidth=2, label="Perfect Fit Line ($y=x$)")
        axes[0].set_title(f"Actual vs Predicted Grade ({model_name})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Actual Student Grade", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Predicted Student Grade", fontsize=10, fontweight="bold")
        axes[0].legend(frameon=True, facecolor="white")
        
        # 2. Residual Distribution
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, color="#0EA5E9", ax=axes[1], bins=20)
        axes[1].axvline(0, color="#DC2626", linestyle="--", linewidth=2)
        axes[1].set_title("Prediction Error (Residuals) Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Residual (Actual - Predicted)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("Frequency", fontsize=10, fontweight="bold")
        
        plt.suptitle(f"Model Diagnostic Analysis: {model_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_confusion_matrices(self, cm_pf: list, cm_cat: list, pf_labels: list, cat_labels: list) -> str:
        """Plot confusion matrix heatmaps for Pass/Fail and Category models."""
        filepath = os.path.join(self.output_dir, "confusion_matrices_plot.png")
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Pass/Fail CM
        sns.heatmap(cm_pf, annot=True, fmt="d", cmap="Blues", xticklabels=pf_labels, yticklabels=pf_labels, ax=axes[0], cbar=False)
        axes[0].set_title("Pass/Fail Confusion Matrix", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("True Label", fontsize=10, fontweight="bold")
        
        # Category CM
        sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Greens", xticklabels=cat_labels, yticklabels=cat_labels, ax=axes[1], cbar=False)
        axes[1].set_title("Performance Category Confusion Matrix", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Predicted Category", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("True Category", fontsize=10, fontweight="bold")
        
        plt.suptitle("Classification Confusion Matrix Evaluations", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
