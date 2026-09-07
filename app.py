"""
Flask Web Application and REST API Server
Provides interactive dashboard, EDA analytics, real-time inference, and batch predictions.
"""

import os
import sys
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# Ensure project root is on sys.path
_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.data_loader import DataLoader
from src.eda import EDAPerformer
from src.predictor import StudentPredictor
from src.dataset_normalizer import FlexibleDatasetReader

app = Flask(__name__)
app.config["SECRET_KEY"] = "student-performance-ml-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

@app.after_request
def add_cors_headers(response):
    """Enable CORS for modern web and API integration."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle files exceeding 16 MB upload limit."""
    return jsonify({
        "status": "error",
        "message": "File exceeds the maximum upload limit of 16 MB. Please upload a smaller dataset."
    }), 413

# Initialize Predictor
predictor = StudentPredictor(models_dir=os.path.join(_ROOT_DIR, "models"))

@app.route("/")
def index():
    """Render main dashboard."""
    return render_template("index.html")

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Return dataset overview, column metadata, and EDA statistics."""
    try:
        data_path = os.path.join(_ROOT_DIR, "data", "student_performance_data.csv")
        loader = DataLoader(data_path)
        eda = EDAPerformer(loader.df)
        eda_summary = eda.get_eda_summary_dict()
        profile = loader.get_summary_profile()
        
        # Sample records for preview table
        preview_records = loader.df.head(15).to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "eda": eda_summary,
            "profile": {
                "total_records": profile["total_records"],
                "total_features": profile["total_features"],
                "numeric_columns": profile["numeric_columns"],
                "categorical_columns": profile["categorical_columns"]
            },
            "preview_records": preview_records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def get_models_benchmark():
    """Return model evaluation leaderboard and feature importance data."""
    try:
        meta_path = os.path.join(_ROOT_DIR, "reports", "model_metrics.json")
        if not os.path.exists(meta_path):
            return jsonify({"status": "error", "message": "Model metrics not found. Run pipeline first."}), 404
            
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        return jsonify({
            "status": "success",
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict_single():
    """Generate real-time prediction and recommendations for a single student."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400
            
        result = predictor.predict_single(data)
        return jsonify({
            "status": "success",
            "prediction": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """Upload CSV/Excel file, run batch inference with flexible schema, and return predictions."""
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded. Please select a CSV or Excel file."}), 400
            
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"status": "error", "message": "No file selected. Please choose a valid file."}), 400
            
        filename = file.filename.strip()
        filename_lower = filename.lower()
        allowed_exts = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
        if not any(filename_lower.endswith(ext) for ext in allowed_exts):
            return jsonify({
                "status": "error",
                "message": f"Unsupported file format '{filename}'. Allowed formats: .csv, .xlsx, .xls, .tsv"
            }), 400

        # Parse file using FlexibleDatasetReader
        try:
            raw_df = FlexibleDatasetReader.read(file, filename=filename)
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as ex:
            return jsonify({"status": "error", "message": f"Failed to parse uploaded spreadsheet: {str(ex)}"}), 400

        if raw_df.empty or len(raw_df) == 0:
            return jsonify({"status": "error", "message": "The uploaded spreadsheet contains no data rows."}), 400
            
        # Run batch prediction with schema mapping diagnostics
        results_df, schema_meta = predictor.predict_batch(raw_df, filename=filename, return_metadata=True)
        
        # Summary metrics
        total = len(results_df)
        pass_count = int((results_df["predicted_pass_fail"] == "Pass").sum())
        fail_count = total - pass_count
        avg_grade = float(results_df["predicted_final_grade"].mean())
        cat_counts = results_df["predicted_performance_category"].value_counts().to_dict()
        risk_counts = results_df["risk_tier"].value_counts().to_dict() if "risk_tier" in results_df.columns else {}
        
        return jsonify({
            "status": "success",
            "schema_mapping": schema_meta,
            "summary": {
                "total_students": total,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_rate_pct": round((pass_count / max(total, 1)) * 100, 1),
                "average_predicted_grade": round(avg_grade, 1),
                "category_breakdown": cat_counts,
                "risk_breakdown": risk_counts
            },
            "records": results_df.to_dict(orient="records")
        })
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference error: {str(e)}"}), 500

@app.route("/api/download-sample-csv", methods=["GET"])
def download_sample_csv():
    """Download template CSV for batch predictions."""
    sample_df = pd.DataFrame([
        {
            "student_id": "STU_2001",
            "gender": "Female",
            "age": 17,
            "parental_education": "Bachelor",
            "study_time": 12.0,
            "attendance_rate": 94.0,
            "previous_grades": 85.0,
            "extracurricular_activities": "Yes",
            "internet_access": "Yes",
            "tutoring": "No",
            "family_support": "Yes",
            "health": "Good",
            "absences": 2,
            "failures": 0
        },
        {
            "student_id": "STU_2002",
            "gender": "Male",
            "age": 18,
            "parental_education": "High School",
            "study_time": 4.5,
            "attendance_rate": 68.0,
            "previous_grades": 52.0,
            "extracurricular_activities": "No",
            "internet_access": "Yes",
            "tutoring": "No",
            "family_support": "No",
            "health": "Fair",
            "absences": 12,
            "failures": 2
        },
        {
            "student_id": "STU_2003",
            "gender": "Female",
            "age": 16,
            "parental_education": "Master/Doctorate",
            "study_time": 18.0,
            "attendance_rate": 98.0,
            "previous_grades": 92.0,
            "extracurricular_activities": "Yes",
            "internet_access": "Yes",
            "tutoring": "Yes",
            "family_support": "Yes",
            "health": "Excellent",
            "absences": 1,
            "failures": 0
        }
    ])
    output = io.StringIO()
    sample_df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="student_batch_prediction_template.csv"
    )

@app.route("/reports/figures/<path:filename>")
def serve_figures(filename):
    """Serve generated diagnostic and EDA figures."""
    figures_dir = os.path.join(_ROOT_DIR, "reports", "figures")
    return send_from_directory(figures_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
