"""
Comprehensive Test Suite for Flexible Schema Ingestion and Resilient Batch Inference
Validates CSV/TSV/Excel parsing, auto-encoding, delimiter sniffing, fuzzy mapping,
scale normalization, missing column imputation, extra column preservation, and Flask API.
"""

import os
import sys
import io
import json
import unittest
import numpy as np
import pandas as pd

# Bootstrap root path
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.dataset_normalizer import FlexibleDatasetReader, SchemaAutoMapper
from src.predictor import StudentPredictor
from app import app


class TestFlexibleBatchInference(unittest.TestCase):
    """Test cases covering all flexible schema and multi-format batch inference requirements."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = StudentPredictor(models_dir=os.path.join(_ROOT_DIR, "models"))
        cls.mapper = SchemaAutoMapper()
        cls.client = app.test_client()

    def test_01_delimiter_auto_detection(self):
        """Test auto-detecting comma, semicolon, tab, and pipe delimiters."""
        data_rows = [
            ["student_id", "attendance_rate", "study_time", "previous_grades"],
            ["STU_101", "92.5", "12.0", "88.0"],
            ["STU_102", "74.0", "5.5", "61.0"]
        ]

        # Comma
        csv_comma = "\n".join([",".join(row) for row in data_rows])
        df_comma = FlexibleDatasetReader.read(csv_comma.encode("utf-8"), filename="test_comma.csv")
        self.assertEqual(df_comma.shape, (2, 4))
        self.assertIn("attendance_rate", df_comma.columns)

        # Semicolon
        csv_semi = "\n".join([";".join(row) for row in data_rows])
        df_semi = FlexibleDatasetReader.read(csv_semi.encode("utf-8"), filename="test_semi.csv")
        self.assertEqual(df_semi.shape, (2, 4))
        self.assertIn("attendance_rate", df_semi.columns)

        # Tab
        csv_tab = "\n".join(["\t".join(row) for row in data_rows])
        df_tab = FlexibleDatasetReader.read(csv_tab.encode("utf-8"), filename="test_tab.tsv")
        self.assertEqual(df_tab.shape, (2, 4))
        self.assertIn("attendance_rate", df_tab.columns)

        # Pipe
        csv_pipe = "\n".join(["|".join(row) for row in data_rows])
        df_pipe = FlexibleDatasetReader.read(csv_pipe.encode("utf-8"), filename="test_pipe.txt")
        self.assertEqual(df_pipe.shape, (2, 4))
        self.assertIn("attendance_rate", df_pipe.columns)

    def test_02_encoding_handling(self):
        """Test auto-decoding UTF-8, UTF-8-sig (with BOM), Latin-1, and Windows-1252."""
        text_content = "student_id,attendance_rate,study_time,previous_grades,remarks\nSTU_1,90,10,80,Élève très motivé\nSTU_2,80,6,65,Schülerin\n"

        # UTF-8 with BOM
        bom_bytes = b"\xef\xbb\xbf" + text_content.encode("utf-8")
        df_bom = FlexibleDatasetReader.read(bom_bytes, filename="bom.csv")
        self.assertEqual(len(df_bom), 2)
        self.assertEqual(df_bom.columns[0], "student_id")

        # Latin-1
        latin1_bytes = text_content.encode("latin-1")
        df_latin1 = FlexibleDatasetReader.read(latin1_bytes, filename="latin1.csv")
        self.assertEqual(len(df_latin1), 2)

        # Windows-1252 / CP1252
        cp1252_bytes = text_content.encode("cp1252")
        df_cp1252 = FlexibleDatasetReader.read(cp1252_bytes, filename="cp1252.csv")
        self.assertEqual(len(df_cp1252), 2)

    def test_03_excel_workbook_support(self):
        """Test parsing Excel .xlsx file."""
        df_input = pd.DataFrame([
            {"Roll_Number": "EXCEL_01", "Attendance_Pct": 95.0, "Study_Hours": 14.0, "GPA": 3.8},
            {"Roll_Number": "EXCEL_02", "Attendance_Pct": 65.0, "Study_Hours": 4.0, "GPA": 2.1}
        ])
        excel_buffer = io.BytesIO()
        df_input.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        df_read = FlexibleDatasetReader.read(excel_buffer.getvalue(), filename="students.xlsx")
        self.assertEqual(len(df_read), 2)
        self.assertIn("Roll_Number", df_read.columns)

    def test_04_fuzzy_synonym_mapping(self):
        """Test mapping non-standard column names using synonym dictionary and fuzzy matching."""
        df_raw = pd.DataFrame([
            {
                "Roll_No": "STU_881",
                "Sex": "M",
                "Student_Age": "18",
                "Parents_Education": "High School",
                "Weekly_Study_Hours": "9.5",
                "Attendance_Pct": "88%",
                "Past_Marks": "76.5",
                "Co_Curricular": "Yes",
                "Net_Access": "1",
                "Extra_Tuition": "No",
                "Parental_Support": "Yes",
                "Physical_Health": "Good",
                "Days_Absent": "4",
                "Backlogs": "0",
                "Student_Name": "Alexander Smith",
                "City": "Boston"
            }
        ])

        mapped, imputed, extra = self.mapper.map_columns(df_raw)

        self.assertEqual(mapped["student_id"], "Roll_No")
        self.assertEqual(mapped["gender"], "Sex")
        self.assertEqual(mapped["age"], "Student_Age")
        self.assertEqual(mapped["parental_education"], "Parents_Education")
        self.assertEqual(mapped["study_time"], "Weekly_Study_Hours")
        self.assertEqual(mapped["attendance_rate"], "Attendance_Pct")
        self.assertEqual(mapped["previous_grades"], "Past_Marks")
        self.assertEqual(mapped["extracurricular_activities"], "Co_Curricular")
        self.assertEqual(mapped["internet_access"], "Net_Access")
        self.assertEqual(mapped["tutoring"], "Extra_Tuition")
        self.assertEqual(mapped["family_support"], "Parental_Support")
        self.assertEqual(mapped["health"], "Physical_Health")
        self.assertEqual(mapped["absences"], "Days_Absent")
        self.assertEqual(mapped["failures"], "Backlogs")

        self.assertIn("Student_Name", extra)
        self.assertIn("City", extra)
        self.assertEqual(len(imputed), 0)

    def test_05_value_scale_normalization(self):
        """Test auto-scaling 4.0 GPA, 20-point scale, and fractional attendance."""
        # 1. 4.0 GPA scale (3.8 out of 4 -> 95%) & fractional attendance (0.94 -> 94%)
        df_gpa4 = pd.DataFrame([{
            "id": "STU_1",
            "attendance": 0.94,
            "study_time": 10.0,
            "gpa": 3.6
        }])
        mapped, _, _ = self.mapper.map_columns(df_gpa4)
        df_norm = self.mapper.normalize_values(df_gpa4, mapped)
        self.assertAlmostEqual(df_norm["attendance_rate"].iloc[0], 94.0, delta=0.5)
        self.assertAlmostEqual(df_norm["previous_grades"].iloc[0], 90.0, delta=0.5)

        # 2. 20-point Portuguese scale (G1 = 16 out of 20 -> 80%)
        df_port = pd.DataFrame([{
            "id": "STU_2",
            "attendance_rate": 88.0,
            "study_time": 10.0,
            "g1": 15.0
        }])
        mapped2, _, _ = self.mapper.map_columns(df_port)
        df_norm2 = self.mapper.normalize_values(df_port, mapped2)
        self.assertAlmostEqual(df_norm2["previous_grades"].iloc[0], 75.0, delta=0.5)

    def test_06_sparse_dataset_with_imputation(self):
        """Test batch prediction on minimal 2-column dataset with graceful defaults."""
        df_sparse = pd.DataFrame([
            {"student_id": "MINI_01", "attendance": 95.0, "study_hours": 14.0},
            {"student_id": "MINI_02", "attendance": 55.0, "study_hours": 2.0}
        ])

        res_df, meta = self.predictor.predict_batch(df_sparse, return_metadata=True)

        self.assertEqual(len(res_df), 2)
        self.assertIn("predicted_final_grade", res_df.columns)
        self.assertIn("predicted_pass_fail", res_df.columns)
        self.assertIn("risk_tier", res_df.columns)

        # Verify imputation metadata
        self.assertGreaterEqual(meta["total_features_imputed"], 5)
        self.assertIn("previous_grades", meta["imputed_features"])
        self.assertIn("gender", meta["imputed_features"])

        # High attendance + high study hours student should pass
        self.assertEqual(res_df["predicted_pass_fail"].iloc[0], "Pass")
        # Low attendance + low study hours student should have high risk
        self.assertIn(res_df["risk_tier"].iloc[1], ["High Risk", "Moderate Risk"])

    def test_07_extra_columns_preservation(self):
        """Test that extra non-feature columns are preserved in the prediction output."""
        df_extra = pd.DataFrame([
            {
                "student_id": "STU_EXTRA",
                "student_name": "Sarah Connor",
                "email": "sarah@example.com",
                "attendance_rate": 90.0,
                "study_time": 10.0,
                "previous_grades": 82.0,
                "advisor_notes": "Needs honors recommendation"
            }
        ])

        res_df, meta = self.predictor.predict_batch(df_extra, return_metadata=True)

        self.assertIn("student_name", res_df.columns)
        self.assertIn("email", res_df.columns)
        self.assertIn("advisor_notes", res_df.columns)
        self.assertEqual(res_df["student_name"].iloc[0], "Sarah Connor")
        self.assertEqual(res_df["advisor_notes"].iloc[0], "Needs honors recommendation")

    def test_08_batch_api_endpoint_various_formats(self):
        """Test /api/batch-predict endpoint with CSV, TSV, and Semicolon files."""
        # 1. Semicolon CSV with aliases
        csv_content = "roll_no;attendance_pct;weekly_study_hours;prev_score;gender\nS_10;92;11;85;Female\nS_20;62;3;48;Male\n"
        data = {
            "file": (io.BytesIO(csv_content.encode("utf-8")), "students_semicolon.csv")
        }
        res = self.client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)

        json_data = json.loads(res.data)
        self.assertEqual(json_data["status"], "success")
        self.assertIn("schema_mapping", json_data)
        self.assertEqual(json_data["summary"]["total_students"], 2)
        self.assertEqual(len(json_data["records"]), 2)
        self.assertEqual(json_data["records"][0]["student_id"], "S_10")

    def test_09_api_error_handling_and_validation(self):
        """Test API validation for empty files, invalid extensions, and missing file payload."""
        # 1. No file in request
        res1 = self.client.post("/api/batch-predict", data={}, content_type="multipart/form-data")
        self.assertEqual(res1.status_code, 400)
        self.assertIn("No file uploaded", json.loads(res1.data)["message"])

        # 2. Empty file
        empty_data = {
            "file": (io.BytesIO(b""), "empty.csv")
        }
        res2 = self.client.post("/api/batch-predict", data=empty_data, content_type="multipart/form-data")
        self.assertEqual(res2.status_code, 400)
        self.assertIn("empty", json.loads(res2.data)["message"].lower())

        # 3. Invalid extension
        invalid_ext_data = {
            "file": (io.BytesIO(b"some content"), "report.pdf")
        }
        res3 = self.client.post("/api/batch-predict", data=invalid_ext_data, content_type="multipart/form-data")
        self.assertEqual(res3.status_code, 400)
        self.assertIn("Unsupported file format", json.loads(res3.data)["message"])

    def test_10_cors_headers(self):
        """Test CORS headers presence on REST API endpoints."""
        res = self.client.get("/api/summary")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("OPTIONS", res.headers.get("Access-Control-Allow-Methods", ""))


if __name__ == "__main__":
    unittest.main()
