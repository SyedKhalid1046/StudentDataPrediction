"""
Educational Dataset Generator for Student Academic Performance Prediction
Creates realistic, statistically correlated benchmark student datasets adhering to educational research standards.
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic educational dataset with realistic multi-variate correlations.
    
    Parameters:
        n_samples (int): Number of student records to generate.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Complete student performance dataset.
    """
    np.random.seed(random_state)
    
    # 1. Student ID & Demographics
    student_ids = [f"STU_{1000 + i}" for i in range(n_samples)]
    genders = np.random.choice(["Female", "Male"], size=n_samples, p=[0.52, 0.48])
    ages = np.random.choice([15, 16, 17, 18, 19, 20, 21], size=n_samples, p=[0.10, 0.25, 0.35, 0.18, 0.08, 0.03, 0.01])
    
    # 2. Socioeconomic & Parental Background
    parent_edu_levels = ["None", "High School", "Some College", "Bachelor", "Master/Doctorate"]
    parent_edu_probs = [0.08, 0.32, 0.25, 0.23, 0.12]
    parental_education = np.random.choice(parent_edu_levels, size=n_samples, p=parent_edu_probs)
    
    family_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.70, 0.30])
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15])
    tutoring = np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62])
    extracurricular = np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    health_status = np.random.choice(["Poor", "Fair", "Good", "Excellent"], size=n_samples, p=[0.08, 0.22, 0.45, 0.25])
    
    # Numerical factors with realistic distributions
    study_hours_base = np.random.gamma(shape=3.5, scale=2.5, size=n_samples) # Mean ~8.75 hrs/week
    study_hours = np.clip(study_hours_base, 1.0, 30.0)
    
    attendance_rate_base = np.random.beta(a=9, b=2, size=n_samples) * 100 # Skewed towards 80-95%
    attendance_rate = np.clip(attendance_rate_base, 40.0, 100.0)
    
    absences = np.round((100 - attendance_rate) * 0.35 + np.random.poisson(lam=1.5, size=n_samples)).astype(int)
    absences = np.clip(absences, 0, 35)
    
    failures_prob = np.where(attendance_rate < 70, 0.45, np.where(attendance_rate < 85, 0.15, 0.04))
    failures = np.random.binomial(n=3, p=failures_prob, size=n_samples)
    
    # 3. Base Academic Aptitude & Previous Grades
    edu_weight_map = {"None": -5.0, "High School": -1.5, "Some College": 1.5, "Bachelor": 4.5, "Master/Doctorate": 7.5}
    edu_weights = np.array([edu_weight_map[e] for e in parental_education])
    
    net_weight = np.where(internet_access == "Yes", 3.0, -3.0)
    fam_weight = np.where(family_support == "Yes", 3.0, -2.0)
    tut_weight = np.where(tutoring == "Yes", 4.0, 0.0)
    act_weight = np.where(extracurricular == "Yes", 2.0, -1.0)
    
    # Latent academic capability
    latent_ability = (
        0.42 * (attendance_rate - 78) +
        1.35 * (study_hours - 8.5) +
        edu_weights +
        net_weight +
        fam_weight +
        tut_weight +
        act_weight -
        9.0 * failures -
        0.65 * absences +
        np.random.normal(0, 5.0, size=n_samples)
    )
    
    previous_grades = 72.0 + 0.62 * latent_ability + np.random.normal(0, 3.5, size=n_samples)
    previous_grades = np.clip(np.round(previous_grades, 1), 30.0, 99.5)
    
    # 4. Final Grade (G3 / Final Exam Score 0-100)
    final_grade_raw = (
        0.65 * previous_grades +
        0.16 * (attendance_rate - 70) +
        0.52 * (study_hours - 8) -
        3.8 * failures -
        0.3 * absences +
        0.35 * edu_weights +
        np.where(tutoring == "Yes", 2.2, 0.0) +
        23.5 + 
        np.random.normal(0, 3.2, size=n_samples)
    )
    
    final_grade = np.clip(np.round(final_grade_raw, 1), 20.0, 99.5)
    
    # 5. Derived Classification Targets
    # Binary: Pass (>= 60) vs Fail (< 60)
    pass_fail = np.where(final_grade >= 60.0, "Pass", "Fail")
    
    # Multi-class Performance Category:
    # At Risk (< 55), Satisfactory (55-69), Good (70-84), Excellent (85-100)
    conditions = [
        (final_grade < 55.0),
        (final_grade >= 55.0) & (final_grade < 70.0),
        (final_grade >= 70.0) & (final_grade < 85.0),
        (final_grade >= 85.0)
    ]
    categories = ["At Risk", "Satisfactory", "Good", "Excellent"]
    performance_category = np.select(conditions, categories, default="Satisfactory")
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": genders,
        "age": ages,
        "parental_education": parental_education,
        "study_time": np.round(study_hours, 1),
        "attendance_rate": np.round(attendance_rate, 1),
        "previous_grades": previous_grades,
        "extracurricular_activities": extracurricular,
        "internet_access": internet_access,
        "tutoring": tutoring,
        "family_support": family_support,
        "health": health_status,
        "absences": absences,
        "failures": failures,
        "final_grade": final_grade,
        "pass_fail_status": pass_fail,
        "performance_category": performance_category
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_student_dataset(n_samples=1600, random_state=42)
    
    csv_path = os.path.join("data", "student_performance_data.csv")
    excel_path = os.path.join("data", "student_performance_data.xlsx")
    
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    
    print(f"Generated dataset with {len(df)} records.")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Excel to: {excel_path}")
    print("\nDataset Summary Preview:")
    print(df.describe())
    print("\nPerformance Category Distribution:")
    print(df["performance_category"].value_counts())
    print("\nPass/Fail Distribution:")
    print(df["pass_fail_status"].value_counts(normalize=True))
