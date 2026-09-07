"""
Flexible Dataset Reader and Schema Auto-Mapper Module
Enables seamless ingestion of arbitrary student datasets with auto-encoding detection,
auto-delimiter sniffing, synonym/fuzzy column mapping, scale normalization, and robust imputation.
"""

import io
import re
import csv
import difflib
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd


# Canonical model feature list and default fallback values
CANONICAL_NUMERIC_DEFAULTS: Dict[str, float] = {
    "age": 17.0,
    "study_time": 8.0,
    "attendance_rate": 85.0,
    "previous_grades": 70.0,
    "absences": 4.0,
    "failures": 0.0
}

CANONICAL_CATEGORICAL_DEFAULTS: Dict[str, str] = {
    "gender": "Female",
    "parental_education": "Some College",
    "extracurricular_activities": "No",
    "internet_access": "Yes",
    "tutoring": "No",
    "family_support": "Yes",
    "health": "Good"
}

# Comprehensive synonyms and aliases dictionary for fuzzy/synonym matching
SYNONYM_MAP: Dict[str, List[str]] = {
    "student_id": [
        "student_id", "studentid", "student id", "id", "roll_no", "rollno", "roll number",
        "roll_num", "reg_no", "regno", "registration_no", "registration_number", "student_no",
        "student_num", "student_code", "student_name", "name", "student", "candidate_id",
        "learner_id", "admission_no", "enrollment_no", "matric_no", "usn", "uid"
    ],
    "gender": [
        "gender", "sex", "male_female", "m_f", "gen", "student_gender", "gender_identity"
    ],
    "age": [
        "age", "student_age", "years_old", "age_years", "dob_years", "student_age_years"
    ],
    "parental_education": [
        "parental_education", "parent_education", "parenteducation", "parental_edu",
        "guardian_education", "medu", "fedu", "mother_education", "father_education",
        "parents_education", "parent_edu", "guardians_education", "parent_qualification",
        "mother_edu", "father_edu", "family_education", "parent_ed", "parental_qualification"
    ],
    "study_time": [
        "study_time", "studytime", "study_hours", "studyhours", "weekly_study_hours",
        "study_time_weekly", "hours_studied", "study_duration", "weekly_study_time",
        "study_hrs", "self_study_hours", "revision_hours", "study_time_hrs", "study_hours_week",
        "study_hours_weekly", "study_time_per_week", "time_studied", "daily_study_hours"
    ],
    "attendance_rate": [
        "attendance_rate", "attendancerate", "attendance", "attendance_pct",
        "attendance_percentage", "attendance_percent", "att_rate", "att_pct",
        "attendance_ratio", "presence", "presence_rate", "class_attendance",
        "attendance_score", "attendance_pct_score", "attended_percent", "attendance_fraction"
    ],
    "previous_grades": [
        "previous_grades", "previousgrades", "previous_grade", "prev_score", "last_gpa",
        "past_marks", "g1", "g2", "prev_grade", "prior_grades", "prior_grade", "past_grade",
        "past_grades", "previous_score", "prev_marks", "past_score", "last_marks", "gpa",
        "cgpa", "exam_score", "prior_score", "internal_marks", "midterm_score", "term1_score",
        "term1_grade", "g1_grade", "g2_grade", "pre_exam_score", "past_score_pct", "past_gpa",
        "prior_marks", "previous_marks", "previous_academic_score", "test_score", "prior_gpa"
    ],
    "extracurricular_activities": [
        "extracurricular_activities", "extracurricular", "extracurriculars", "activities",
        "extra_curricular", "sports_activities", "co_curricular", "clubs",
        "activities_participation", "extra_activities", "after_school_activities"
    ],
    "internet_access": [
        "internet_access", "internet", "internetaccess", "has_internet", "net_access",
        "web_access", "online_access", "connectivity", "home_internet", "internet_at_home"
    ],
    "tutoring": [
        "tutoring", "tutor", "tuition", "extra_classes", "extra_tutoring", "coaching",
        "paid_classes", "supplementary_classes", "private_tutor", "tuition_classes"
    ],
    "family_support": [
        "family_support", "familysupport", "famsup", "parental_support", "family_help",
        "home_support", "parent_support", "guardian_support"
    ],
    "health": [
        "health", "health_status", "wellness", "medical_condition", "health_condition",
        "physical_health", "general_health", "student_health"
    ],
    "absences": [
        "absences", "absence", "days_absent", "absent_days", "missed_classes",
        "unexcused_absences", "leaves", "absent_count", "total_absences", "number_of_absences"
    ],
    "failures": [
        "failures", "failure", "past_failures", "failed_subjects", "number_of_failures",
        "backlogs", "arrears", "fail_count", "remedials", "previous_failures", "course_failures"
    ],
    "final_grade": [
        "final_grade", "finalgrade", "g3", "final_score", "actual_grade", "exam_final",
        "target", "ground_truth", "final_marks", "final_result_score"
    ],
    "pass_fail_status": [
        "pass_fail_status", "pass_fail", "passfail", "result", "outcome", "passed", "status"
    ],
    "performance_category": [
        "performance_category", "performance_tier", "grade_tier", "category", "achievement_level"
    ]
}


class FlexibleDatasetReader:
    """Robust parser for CSV, TSV, TXT, XLSX, and XLS student datasets."""

    ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    COMMON_DELIMITERS = [",", ";", "\t", "|"]

    @classmethod
    def read(cls, file_input: Union[str, bytes, io.BytesIO, Any], filename: str = "dataset.csv") -> pd.DataFrame:
        """
        Read arbitrary student dataset file with auto-detection of format, encoding, and delimiter.
        
        Parameters:
            file_input: File path, bytes, BytesIO stream, or Werkzeug FileStorage.
            filename: Original file name.
            
        Returns:
            pd.DataFrame: Parsed DataFrame with stripped column headers.
        """
        filename_lower = filename.lower()

        # Handle Werkzeug FileStorage or file-like object
        if hasattr(file_input, "read"):
            raw_bytes = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        elif isinstance(file_input, str) and not raw_bytes_needed(file_input):
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        else:
            raise ValueError("Unsupported file input type provided.")

        if not raw_bytes or len(raw_bytes.strip()) == 0:
            raise ValueError("The uploaded file is completely empty (0 bytes).")

        # 1. Handle Excel formats (.xlsx, .xls)
        if filename_lower.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes))
                return cls._clean_dataframe(df)
            except Exception as e:
                raise ValueError(f"Failed to parse Excel workbook: {str(e)}")

        # 2. Handle Text / CSV / TSV formats
        return cls._read_csv_with_auto_detection(raw_bytes)

    @classmethod
    def _read_csv_with_auto_detection(cls, raw_bytes: bytes) -> pd.DataFrame:
        """Attempt multi-encoding and multi-delimiter parsing for delimited text files."""
        decoded_text: Optional[str] = None
        used_encoding = "utf-8"

        # Try decoding with supported encodings
        for enc in cls.ENCODINGS_TO_TRY:
            try:
                decoded_text = raw_bytes.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            # Fallback with replacement
            decoded_text = raw_bytes.decode("latin-1", errors="replace")
            used_encoding = "latin-1 (forced)"

        # Strip null bytes if present
        decoded_text = decoded_text.replace("\x00", "")

        # Sniff delimiter from sample text
        sample = "\n".join([line for line in decoded_text.splitlines()[:25] if line.strip()])
        detected_delimiter = None

        if sample:
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=";,|\t, ")
                detected_delimiter = dialect.delimiter
            except Exception:
                detected_delimiter = None

        # Delimiters priority list to test
        delims_to_test = []
        if detected_delimiter:
            delims_to_test.append(detected_delimiter)
        for d in cls.COMMON_DELIMITERS:
            if d not in delims_to_test:
                delims_to_test.append(d)

        # Attempt parsing with pandas
        last_error = None
        for delim in delims_to_test:
            try:
                df = pd.read_csv(
                    io.StringIO(decoded_text),
                    sep=delim,
                    engine="python",
                    on_bad_lines="skip"
                )
                # Ensure it didn't collapse everything into a single column when multi-column exists
                if df.shape[1] > 1 or len(delims_to_test) == 1:
                    return cls._clean_dataframe(df)
                elif df.shape[1] == 1 and delim == delims_to_test[-1]:
                    return cls._clean_dataframe(df)
            except Exception as ex:
                last_error = ex
                continue

        # Final fallback with auto sep=None
        try:
            df = pd.read_csv(io.StringIO(decoded_text), sep=None, engine="python", on_bad_lines="skip")
            return cls._clean_dataframe(df)
        except Exception as ex:
            raise ValueError(f"Could not parse CSV file. Error: {last_error or ex}")

    @classmethod
    def _clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unneeded blank rows/columns and clean column headers."""
        if df.empty:
            raise ValueError("The uploaded spreadsheet contains no data rows.")

        # Drop columns that are completely Unnamed or empty
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

        # Clean column names (strip whitespace and remove quotes)
        clean_cols = []
        for col in df.columns:
            c_str = str(col).strip()
            c_str = re.sub(r'^["\']+|["\']+$', '', c_str)
            clean_cols.append(c_str)
        df.columns = clean_cols

        if df.empty:
            raise ValueError("The spreadsheet contains headers but zero valid data records.")

        return df


def raw_bytes_needed(path_str: str) -> bool:
    """Helper to verify if path is a file on disk."""
    import os
    return not os.path.exists(path_str)


class SchemaAutoMapper:
    """Intelligent Schema Mapper with fuzzy synonym resolution, value normalizers, and fallback imputation."""

    def __init__(self, synonym_map: Optional[Dict[str, List[str]]] = None):
        self.synonym_map = synonym_map or SYNONYM_MAP

    def _clean_col_name(self, col: str) -> str:
        """Normalize string by removing punctuation, extra spaces, and lowercasing."""
        s = str(col).strip().lower()
        s = re.sub(r'[\(\)\[\]\{\}\%\:\;\,\.\-\/\\]', ' ', s)
        s = re.sub(r'\s+', '_', s).strip('_')
        return s

    def _get_root_name(self, clean_name: str) -> str:
        """Strip common prefix prefixes to isolate the semantic root word."""
        root = clean_name
        for prefix in ["student_", "learner_", "candidate_", "school_", "class_", "academic_"]:
            if root.startswith(prefix) and len(root) > len(prefix):
                root = root[len(prefix):]
        return root

    def map_columns(self, df: pd.DataFrame) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Map uploaded DataFrame columns to canonical model features using multi-stage matching:
        1. Exact synonym / alias match.
        2. Exact root-token match.
        3. Significant keyword match.
        4. High-confidence fuzzy similarity on semantic root words.
        
        Returns:
            Tuple of:
            - mapped_columns: Dict[canonical_name, uploaded_column_name]
            - imputed_features: List[canonical_name] that were missing
            - extra_columns: List[uploaded_column_name] that were not used
        """
        mapped: Dict[str, str] = {}
        used_uploaded_cols = set()

        raw_columns = list(df.columns)
        cleaned_to_raw = {self._clean_col_name(c): c for c in raw_columns}

        # Stage 1: Exact Synonym / Canonical Match
        for canonical, synonyms in self.synonym_map.items():
            if canonical in mapped:
                continue

            for syn in synonyms:
                clean_syn = self._clean_col_name(syn)
                for clean_col, raw_col in cleaned_to_raw.items():
                    if raw_col not in used_uploaded_cols:
                        if clean_col == clean_syn or clean_col == self._clean_col_name(canonical):
                            mapped[canonical] = raw_col
                            used_uploaded_cols.add(raw_col)
                            break
                if canonical in mapped:
                    break

        # Stage 2: Root-name exact match (e.g. "student_age" vs "age", "student_gender" vs "gender")
        for canonical, synonyms in self.synonym_map.items():
            if canonical in mapped:
                continue

            for syn in synonyms:
                root_syn = self._get_root_name(self._clean_col_name(syn))
                for clean_col, raw_col in cleaned_to_raw.items():
                    if raw_col not in used_uploaded_cols:
                        root_col = self._get_root_name(clean_col)
                        if root_col == root_syn or root_col == self._get_root_name(self._clean_col_name(canonical)):
                            # Prevent generic names from matching wrong canonicals
                            if root_col in ["name", "student_name"] and canonical != "student_id":
                                continue
                            mapped[canonical] = raw_col
                            used_uploaded_cols.add(raw_col)
                            break
                if canonical in mapped:
                    break

        # Stage 3: Semantic Keyword Match (token overlap on core domain words)
        domain_keywords = {
            "attendance_rate": ["attendance", "presence", "attended"],
            "study_time": ["study_time", "study_hours", "studytime", "studyhours", "hours_studied"],
            "previous_grades": ["previous_grade", "prev_grade", "past_grade", "prior_grade", "past_marks", "gpa", "cgpa", "prev_score", "past_score", "g1", "g2"],
            "parental_education": ["parent_education", "parental_edu", "guardian_education", "mother_education", "father_education", "medu", "fedu", "parent_edu"],
            "extracurricular_activities": ["extracurricular", "extra_curricular", "co_curricular", "clubs", "activities"],
            "internet_access": ["internet", "internet_access", "net_access", "wifi", "online_access"],
            "tutoring": ["tutoring", "tutor", "tuition", "coaching", "extra_classes"],
            "family_support": ["family_support", "parental_support", "family_help", "famsup"],
            "health": ["health", "wellness", "medical_condition"],
            "absences": ["absences", "days_absent", "absent_days", "missed_classes"],
            "failures": ["failures", "past_failures", "failed_subjects", "backlogs", "arrears"],
            "gender": ["gender", "sex", "male_female"],
            "age": ["student_age", "age_years", "dob_years"]
        }

        for canonical, kws in domain_keywords.items():
            if canonical in mapped:
                continue

            for clean_col, raw_col in cleaned_to_raw.items():
                if raw_col in used_uploaded_cols:
                    continue

                for kw in kws:
                    clean_kw = self._clean_col_name(kw)
                    if clean_kw in clean_col:
                        # Extra check: ensure "name" doesn't match "age"
                        if "name" in clean_col and canonical != "student_id":
                            continue
                        mapped[canonical] = raw_col
                        used_uploaded_cols.add(raw_col)
                        break
                if canonical in mapped:
                    break

        # Stage 4: Fuzzy Match on Root Words with High Threshold (>= 0.88)
        for canonical, synonyms in self.synonym_map.items():
            if canonical in mapped:
                continue

            best_match = None
            best_score = 0.0

            for clean_col, raw_col in cleaned_to_raw.items():
                if raw_col in used_uploaded_cols:
                    continue

                root_col = self._get_root_name(clean_col)
                if root_col in ["name", "student_name"] and canonical != "student_id":
                    continue

                for syn in synonyms:
                    root_syn = self._get_root_name(self._clean_col_name(syn))
                    score = difflib.SequenceMatcher(None, root_col, root_syn).ratio()
                    if score > best_score and score >= 0.88:
                        best_score = score
                        best_match = raw_col

            if best_match:
                mapped[canonical] = best_match
                used_uploaded_cols.add(best_match)

        # Determine imputed vs extra
        all_required_features = list(CANONICAL_NUMERIC_DEFAULTS.keys()) + list(CANONICAL_CATEGORICAL_DEFAULTS.keys())
        imputed = [f for f in all_required_features if f not in mapped]
        extra = [c for c in raw_columns if c not in used_uploaded_cols]

        return mapped, imputed, extra

    def normalize_values(self, df_mapped: pd.DataFrame, mapped_columns: Dict[str, str]) -> pd.DataFrame:
        """
        Clean, sanitize, scale, and standardize values in the mapped DataFrame.
        """
        df_norm = pd.DataFrame(index=df_mapped.index)

        # 1. Student ID
        if "student_id" in mapped_columns:
            raw_id = df_mapped[mapped_columns["student_id"]].astype(str).str.strip()
            df_norm["student_id"] = raw_id.replace({"": np.nan, "nan": np.nan, "None": np.nan})
            df_norm["student_id"] = df_norm["student_id"].fillna(
                pd.Series([f"STU_{1001 + i}" for i in range(len(df_norm))], index=df_norm.index)
            )
        else:
            df_norm["student_id"] = [f"STU_{1001 + i}" for i in range(len(df_norm))]

        # 2. Gender
        if "gender" in mapped_columns:
            g_series = df_mapped[mapped_columns["gender"]].astype(str).str.strip().str.lower()
            df_norm["gender"] = g_series.apply(self._normalize_gender)
        else:
            df_norm["gender"] = CANONICAL_CATEGORICAL_DEFAULTS["gender"]

        # 3. Age
        if "age" in mapped_columns:
            age_num = pd.to_numeric(df_mapped[mapped_columns["age"]], errors="coerce")
            df_norm["age"] = age_num.fillna(CANONICAL_NUMERIC_DEFAULTS["age"]).clip(10, 30).round().astype(int)
        else:
            df_norm["age"] = int(CANONICAL_NUMERIC_DEFAULTS["age"])

        # 4. Attendance Rate (Auto-scale 0.0-1.0 fraction to 0-100%)
        if "attendance_rate" in mapped_columns:
            att_raw = df_mapped[mapped_columns["attendance_rate"]]
            att_clean = att_raw.astype(str).str.replace("%", "").str.strip()
            att_num = pd.to_numeric(att_clean, errors="coerce")

            # Check if attendance is represented as fraction (e.g. 0.85 instead of 85%)
            valid_att = att_num.dropna()
            if not valid_att.empty and valid_att.max() <= 1.05 and valid_att.mean() <= 1.0:
                att_num = att_num * 100.0

            df_norm["attendance_rate"] = att_num.fillna(CANONICAL_NUMERIC_DEFAULTS["attendance_rate"]).clip(0.0, 100.0).round(1)
        else:
            df_norm["attendance_rate"] = CANONICAL_NUMERIC_DEFAULTS["attendance_rate"]

        # 5. Study Time
        if "study_time" in mapped_columns:
            st_raw = df_mapped[mapped_columns["study_time"]]
            st_clean = st_raw.astype(str).str.replace(r"(?i)hrs?|hours?", "", regex=True).str.strip()
            st_num = pd.to_numeric(st_clean, errors="coerce")
            df_norm["study_time"] = st_num.fillna(CANONICAL_NUMERIC_DEFAULTS["study_time"]).clip(0.0, 40.0).round(1)
        else:
            df_norm["study_time"] = CANONICAL_NUMERIC_DEFAULTS["study_time"]

        # 6. Previous Grades (Auto-detect GPA scales 4.0, 10.0, 20.0 and normalize to 0-100)
        if "previous_grades" in mapped_columns:
            pg_raw = df_mapped[mapped_columns["previous_grades"]]
            pg_clean = pg_raw.astype(str).str.replace("%", "").str.strip()
            pg_num = pd.to_numeric(pg_clean, errors="coerce")

            valid_pg = pg_num.dropna()
            if not valid_pg.empty:
                max_val = valid_pg.max()
                if max_val <= 4.05 and max_val > 0:
                    pg_num = (pg_num / 4.0) * 100.0  # 4.0 GPA scale
                elif max_val <= 10.05 and max_val > 4.05 and valid_pg.mean() <= 10.0:
                    pg_num = (pg_num / 10.0) * 100.0  # 10.0 GPA scale
                elif max_val <= 20.05 and max_val > 10.05 and valid_pg.mean() <= 20.0:
                    pg_num = (pg_num / 20.0) * 100.0  # 20-point scale (Portuguese G1/G2)

            df_norm["previous_grades"] = pg_num.fillna(CANONICAL_NUMERIC_DEFAULTS["previous_grades"]).clip(0.0, 100.0).round(1)
        else:
            df_norm["previous_grades"] = CANONICAL_NUMERIC_DEFAULTS["previous_grades"]

        # 7. Parental Education
        if "parental_education" in mapped_columns:
            pe_series = df_mapped[mapped_columns["parental_education"]].astype(str)
            df_norm["parental_education"] = pe_series.apply(self._normalize_parental_education)
        else:
            df_norm["parental_education"] = CANONICAL_CATEGORICAL_DEFAULTS["parental_education"]

        # 8. Binary Educational Context Features
        binary_features = [
            ("extracurricular_activities", "extracurricular_activities"),
            ("internet_access", "internet_access"),
            ("tutoring", "tutoring"),
            ("family_support", "family_support")
        ]
        for canonical, orig_key in binary_features:
            if canonical in mapped_columns:
                b_series = df_mapped[mapped_columns[canonical]].astype(str)
                df_norm[canonical] = b_series.apply(
                    lambda v: self._normalize_binary(v, default=CANONICAL_CATEGORICAL_DEFAULTS[canonical])
                )
            else:
                df_norm[canonical] = CANONICAL_CATEGORICAL_DEFAULTS[canonical]

        # 9. Health
        if "health" in mapped_columns:
            h_series = df_mapped[mapped_columns["health"]].astype(str)
            df_norm["health"] = h_series.apply(self._normalize_health)
        else:
            df_norm["health"] = CANONICAL_CATEGORICAL_DEFAULTS["health"]

        # 10. Failures
        if "failures" in mapped_columns:
            f_num = pd.to_numeric(df_mapped[mapped_columns["failures"]], errors="coerce")
            df_norm["failures"] = f_num.fillna(CANONICAL_NUMERIC_DEFAULTS["failures"]).clip(0, 10).round().astype(int)
        else:
            df_norm["failures"] = int(CANONICAL_NUMERIC_DEFAULTS["failures"])

        # 11. Absences (Estimate from attendance rate if missing)
        if "absences" in mapped_columns:
            abs_num = pd.to_numeric(df_mapped[mapped_columns["absences"]], errors="coerce")
            # Fill missing absences with estimate from attendance
            estimated = np.maximum(0, ((100.0 - df_norm["attendance_rate"]) * 0.35).round().astype(int))
            df_norm["absences"] = abs_num.fillna(estimated).clip(0, 50).round().astype(int)
        else:
            df_norm["absences"] = np.maximum(0, ((100.0 - df_norm["attendance_rate"]) * 0.35).round().astype(int))

        return df_norm

    @staticmethod
    def _normalize_gender(val: str) -> str:
        s = str(val).strip().lower()
        if s in ["m", "male", "boy", "man", "1", "1.0"]:
            return "Male"
        elif s in ["f", "female", "girl", "woman", "0", "0.0"]:
            return "Female"
        return "Female"

    @staticmethod
    def _normalize_binary(val: str, default: str = "Yes") -> str:
        s = str(val).strip().lower()
        if s in ["yes", "y", "true", "t", "1", "1.0", "positive", "si"]:
            return "Yes"
        elif s in ["no", "n", "false", "f", "0", "0.0", "negative"]:
            return "No"
        return default

    @staticmethod
    def _normalize_parental_education(val: str) -> str:
        s = str(val).strip().lower()
        if any(w in s for w in ["master", "phd", "doctor", "postgrad", "pg", "m.sc", "m.tech", "mba", "5", "5.0"]):
            return "Master/Doctorate"
        elif any(w in s for w in ["bachelor", "degree", "undergrad", "ug", "b.sc", "b.tech", "b.a", "b.com", "graduate", "4", "4.0"]):
            return "Bachelor"
        elif any(w in s for w in ["college", "diploma", "associate", "vocational", "some", "3", "3.0"]):
            return "Some College"
        elif any(w in s for w in ["high school", "secondary", "12th", "10th", "matric", "hsc", "ssc", "2", "2.0", "1", "1.0"]):
            return "High School"
        elif any(w in s for w in ["none", "no education", "uneducated", "illiterate", "0", "0.0"]):
            return "None"
        return "Some College"

    @staticmethod
    def _normalize_health(val: str) -> str:
        s = str(val).strip().lower()
        if any(w in s for w in ["excellent", "very good", "5", "5.0", "optimal"]):
            return "Excellent"
        elif any(w in s for w in ["good", "4", "4.0", "fine"]):
            return "Good"
        elif any(w in s for w in ["fair", "average", "moderate", "3", "3.0", "ok"]):
            return "Fair"
        elif any(w in s for w in ["poor", "bad", "sick", "low", "1", "2", "1.0", "2.0"]):
            return "Poor"
        return "Good"

    def process(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        End-to-end dataset transformation with schema mapping metadata generation.
        
        Returns:
            Tuple of:
            - df_norm: Normalized pd.DataFrame ready for FeatureEngineering and Preprocessor
            - schema_metadata: Dict with mapped_columns, imputed_features, extra_columns, etc.
        """
        mapped_cols, imputed_feats, extra_cols = self.map_columns(df_raw)
        df_norm = self.normalize_values(df_raw, mapped_cols)

        # Calculate confidence score based on core features detected
        core_features = ["attendance_rate", "study_time", "previous_grades", "failures", "absences", "gender", "age"]
        core_detected = sum(1 for f in core_features if f in mapped_cols)
        confidence_pct = round((core_detected / len(core_features)) * 100, 1)

        metadata = {
            "mapped_columns": mapped_cols,
            "imputed_features": imputed_feats,
            "extra_columns": extra_cols,
            "original_columns": list(df_raw.columns),
            "total_features_detected": len(mapped_cols),
            "total_features_imputed": len(imputed_feats),
            "detection_confidence_pct": confidence_pct,
            "total_rows": len(df_norm)
        }

        return df_norm, metadata
