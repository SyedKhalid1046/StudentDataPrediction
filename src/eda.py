"""
Exploratory Data Analysis (EDA) and Visualization Module
Generates statistical analyses, correlation structures, distributions, and publication-ready plots.
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import seaborn as sns

class EDAPerformer:
    """Class to perform EDA and generate analytical visual figures."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "reports/figures"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set modern plotting theme
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.sans-serif": "Arial",
            "axes.edgecolor": "#E2E8F0",
            "axes.linewidth": 1.2,
            "grid.color": "#EDF2F7",
            "grid.linestyle": "--",
            "figure.autolayout": True
        })
        
    def generate_all_visualizations(self) -> List[str]:
        """Generate and save all core EDA visualizations to disk."""
        generated_files = []
        
        generated_files.append(self.plot_grade_distribution())
        generated_files.append(self.plot_correlation_heatmap())
        generated_files.append(self.plot_attendance_vs_grade())
        generated_files.append(self.plot_study_time_vs_grade())
        generated_files.append(self.plot_parental_education_impact())
        generated_files.append(self.plot_performance_categories())
        generated_files.append(self.plot_academic_risk_distribution())
        
        return generated_files
        
    def plot_grade_distribution(self) -> str:
        """Plot Final Grade distribution with Pass/Fail threshold."""
        filepath = os.path.join(self.output_dir, "grade_distribution.png")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        sns.histplot(
            data=self.df,
            x="final_grade",
            hue="pass_fail_status",
            kde=True,
            palette={"Pass": "#10B981", "Fail": "#EF4444"},
            bins=25,
            ax=ax,
            alpha=0.6
        )
        
        mean_val = self.df["final_grade"].mean()
        ax.axvline(mean_val, color="#3B82F6", linestyle="--", linewidth=2, label=f"Mean Grade ({mean_val:.1f})")
        ax.axvline(60.0, color="#EF4444", linestyle=":", linewidth=2, label="Pass/Fail Threshold (60.0)")
        
        ax.set_title("Student Final Grade Distribution with Pass/Fail Cutoff", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Final Grade (Score 0-100)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Student Count", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_correlation_heatmap(self) -> str:
        """Plot heatmap of Pearson correlation coefficients across numerical features."""
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=1.0,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
            ax=ax
        )
        
        ax.set_title("Correlation Heatmap of Student Academic & Behavioral Factors", fontsize=14, fontweight="bold", pad=14)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_attendance_vs_grade(self) -> str:
        """Plot Attendance Rate vs Final Grade with regression trendline."""
        filepath = os.path.join(self.output_dir, "attendance_vs_grade.png")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        sns.regplot(
            data=self.df,
            x="attendance_rate",
            y="final_grade",
            scatter_kws={"alpha": 0.5, "color": "#6366F1", "s": 35},
            line_kws={"color": "#4338CA", "linewidth": 2.5},
            ax=ax
        )
        
        ax.set_title("Impact of Class Attendance Rate on Final Grade", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Attendance Rate (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Grade (Score 0-100)", fontsize=11, fontweight="bold")
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_study_time_vs_grade(self) -> str:
        """Plot Study Hours vs Final Grade across Extracurricular status."""
        filepath = os.path.join(self.output_dir, "study_time_vs_grade.png")
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        sns.scatterplot(
            data=self.df,
            x="study_time",
            y="final_grade",
            hue="performance_category",
            palette={"At Risk": "#EF4444", "Satisfactory": "#F59E0B", "Good": "#3B82F6", "Excellent": "#10B981"},
            alpha=0.75,
            s=45,
            ax=ax
        )
        
        sns.regplot(
            data=self.df,
            x="study_time",
            y="final_grade",
            scatter=False,
            color="#1E293B",
            line_kws={"linestyle": "--", "linewidth": 2},
            ax=ax
        )
        
        ax.set_title("Weekly Study Hours vs. Final Grade by Performance Tier", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Weekly Study Hours", fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Grade (Score 0-100)", fontsize=11, fontweight="bold")
        ax.legend(title="Performance Tier", frameon=True, facecolor="white")
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_parental_education_impact(self) -> str:
        """Boxplot of Final Grade across Parental Education levels."""
        filepath = os.path.join(self.output_dir, "parental_education_impact.png")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        order = ["None", "High School", "Some College", "Bachelor", "Master/Doctorate"]
        valid_order = [lvl for lvl in order if lvl in self.df["parental_education"].values]
        
        sns.boxplot(
            data=self.df,
            x="parental_education",
            y="final_grade",
            order=valid_order,
            palette="Blues_r",
            ax=ax,
            boxprops=dict(alpha=0.8)
        )
        
        sns.stripplot(
            data=self.df,
            x="parental_education",
            y="final_grade",
            order=valid_order,
            color="black",
            alpha=0.15,
            jitter=0.2,
            size=3,
            ax=ax
        )
        
        ax.set_title("Distribution of Final Academic Grades by Parental Education Level", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Parental Education Level", fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Grade (Score 0-100)", fontsize=11, fontweight="bold")
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_performance_categories(self) -> str:
        """Bar plot showing count and percentage of students per performance tier."""
        filepath = os.path.join(self.output_dir, "performance_category_breakdown.png")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        
        cat_counts = self.df["performance_category"].value_counts().reindex(["At Risk", "Satisfactory", "Good", "Excellent"]).fillna(0)
        colors = ["#EF4444", "#F59E0B", "#3B82F6", "#10B981"]
        
        bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="#334155", linewidth=1.2, alpha=0.85)
        
        total = len(self.df)
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax.annotate(f"{int(height)}\n({pct:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
        ax.set_title("Student Cohort Breakdown by Performance Category", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Performance Category", fontsize=11, fontweight="bold")
        ax.set_ylabel("Student Count", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(cat_counts.values) * 1.18)
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath
        
    def plot_academic_risk_distribution(self) -> str:
        """Scatter of Academic Risk Index vs Final Grade."""
        filepath = os.path.join(self.output_dir, "academic_risk_vs_performance.png")
        if "academic_risk_index" not in self.df.columns:
            return ""
            
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        
        sns.scatterplot(
            data=self.df,
            x="academic_risk_index",
            y="final_grade",
            hue="pass_fail_status",
            palette={"Pass": "#10B981", "Fail": "#EF4444"},
            alpha=0.7,
            s=40,
            ax=ax
        )
        
        ax.axvline(35.0, color="#EF4444", linestyle="--", linewidth=1.8, label="High Risk Threshold (35+)")
        ax.set_title("Academic Risk Index vs. Observed Final Grade", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Engineered Academic Risk Index (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Final Grade (Score 0-100)", fontsize=11, fontweight="bold")
        ax.legend(frameon=True, facecolor="white")
        
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def get_eda_summary_dict(self) -> Dict[str, Any]:
        """Return structured summary metrics for JSON API serialization."""
        numeric_cols = ["age", "study_time", "attendance_rate", "previous_grades", "absences", "failures", "final_grade"]
        valid_cols = [c for c in numeric_cols if c in self.df.columns]
        
        stats = {}
        for col in valid_cols:
            stats[col] = {
                "mean": round(float(self.df[col].mean()), 2),
                "std": round(float(self.df[col].std()), 2),
                "min": round(float(self.df[col].min()), 2),
                "max": round(float(self.df[col].max()), 2),
                "median": round(float(self.df[col].median()), 2)
            }
            
        category_counts = self.df["performance_category"].value_counts().to_dict() if "performance_category" in self.df.columns else {}
        pass_fail_counts = self.df["pass_fail_status"].value_counts().to_dict() if "pass_fail_status" in self.df.columns else {}
        
        corr_matrix = self.df[valid_cols].corr().round(3).to_dict()
        
        return {
            "total_students": len(self.df),
            "statistics": stats,
            "category_distribution": category_counts,
            "pass_fail_distribution": pass_fail_counts,
            "correlation_matrix": corr_matrix
        }

if __name__ == "__main__":
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    
    loader = DataLoader("data/student_performance_data.csv")
    fe = FeatureEngineer()
    df_feat = fe.transform(loader.df)
    
    eda = EDAPerformer(df_feat)
    figures = eda.generate_all_visualizations()
    print("EDA Visualizations Generated Successfully:")
    for f in figures:
        if f:
            print(f" - {f}")
