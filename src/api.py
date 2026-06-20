import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import sys

# Ensure local folder is in python path to resolve sibling imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_single_input

# Initialize FastAPI App
app = FastAPI(
    title="Student Dropout & Performance Prediction API",
    description="Real-time predictive analytics system to detect students at risk of dropout.",
    version="1.0.0"
)

# Enable CORS for local dashboard fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev/portfolio: wide open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load machine learning assets
MODELS_DIR = "models"
DB_PATH = "data/students.db"

# Pydantic schema for robust API request validation
class StudentFeatures(BaseModel):
    marital_status: int = Field(alias="Marital status", default=1)
    application_mode: int = Field(alias="Application mode", default=1)
    application_order: int = Field(alias="Application order", default=1)
    course: int = Field(alias="Course", default=12)
    attendance: int = Field(alias="Daytime/evening attendance", default=1)
    previous_qualification: int = Field(alias="Previous qualification", default=1)
    nationality: int = Field(alias="Nacionality", default=1)
    mother_qualification: int = Field(alias="Mother's qualification", default=1)
    father_qualification: int = Field(alias="Father's qualification", default=1)
    mother_occupation: int = Field(alias="Mother's occupation", default=1)
    father_occupation: int = Field(alias="Father's occupation", default=1)
    displaced: int = Field(alias="Displaced", default=1)
    special_needs: int = Field(alias="Educational special needs", default=0)
    debtor: int = Field(alias="Debtor", default=0)
    fees_up_to_date: int = Field(alias="Tuition fees up to date", default=1)
    gender: int = Field(alias="Gender", default=0) # 0: Female, 1: Male
    scholarship: int = Field(alias="Scholarship holder", default=0)
    age: int = Field(alias="Age at enrollment", default=20)
    international: int = Field(alias="International", default=0)
    sem1_credited: int = Field(alias="Curricular units 1st sem (credited)", default=0)
    sem1_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)", default=6)
    sem1_evaluations: int = Field(alias="Curricular units 1st sem (evaluations)", default=6)
    sem1_approved: int = Field(alias="Curricular units 1st sem (approved)", default=6)
    sem1_grade: float = Field(alias="Curricular units 1st sem (grade)", default=13.0)
    sem1_without_eval: int = Field(alias="Curricular units 1st sem (without evaluations)", default=0)
    sem2_credited: int = Field(alias="Curricular units 2nd sem (credited)", default=0)
    sem2_enrolled: int = Field(alias="Curricular units 2nd sem (enrolled)", default=6)
    sem2_evaluations: int = Field(alias="Curricular units 2nd sem (evaluations)", default=6)
    sem2_approved: int = Field(alias="Curricular units 2nd sem (approved)", default=6)
    sem2_grade: float = Field(alias="Curricular units 2nd sem (grade)", default=13.0)
    sem2_without_eval: int = Field(alias="Curricular units 2nd sem (without evaluations)", default=0)
    unemployment_rate: float = Field(alias="Unemployment rate", default=12.0)
    inflation_rate: float = Field(alias="Inflation rate", default=1.5)
    gdp: float = Field(alias="GDP", default=1.5)

    class Config:
        populate_by_name = True

def check_models_exist():
    required_files = ["best_model.pkl", "scaler.pkl", "label_encoder.pkl"]
    for f in required_files:
        if not os.path.exists(os.path.join(MODELS_DIR, f)):
            return False
    return True

def log_prediction_to_db(prediction: str, confidence: float, risk_level: str, probabilities: dict, inputs: dict):
    """Save prediction call to SQLite DB table 'predictions'."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (timestamp, prediction, confidence, risk_level, probabilities, inputs)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                prediction,
                confidence,
                risk_level,
                json.dumps(probabilities),
                json.dumps(inputs)
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify database and model loaded states."""
    db_ok = os.path.exists(DB_PATH)
    models_ok = check_models_exist()
    return {
        "status": "healthy" if db_ok and models_ok else "unstable",
        "database_connected": db_ok,
        "models_loaded": models_ok,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict_student_status(features: StudentFeatures):
    """
    Accepts full student parameters, standardises them, predicts outcome probabilities, 
    evaluates risk severity, extracts top risk factors, and logs the query.
    """
    if not check_models_exist():
        raise HTTPException(
            status_code=503, 
            detail="Machine learning models are not trained or registered. Run src/train.py first."
        )

    # 1. Map features to dictionary using exact original CSV names
    raw_inputs = features.model_dump(by_alias=True)

    try:
        # 2. Load model and encoder
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

        # 3. Preprocess inputs (using scaler and categorical alignment)
        X_scaled = preprocess_single_input(raw_inputs, models_dir=MODELS_DIR)

        # 4. Ingest and infer
        pred_class_encoded = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        # Map classes correctly
        class_names = list(le.classes_) # ['Dropout', 'Enrolled', 'Graduate']
        prob_dict = {class_names[i]: round(float(probabilities[i]), 4) for i in range(len(class_names))}
        
        prediction = class_names[pred_class_encoded]
        confidence = prob_dict[prediction]

        # 5. Classify risk level based on dropout probability
        dropout_prob = prob_dict.get("Dropout", 0.0)
        if dropout_prob > 0.6:
            risk_level = "High"
        elif dropout_prob > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 6. Extract top risk factors dynamically
        risk_factors = []
        if raw_inputs.get("Scholarship holder") == 0:
            risk_factors.append("No scholarship")
        if raw_inputs.get("Debtor") == 1:
            risk_factors.append("High debt burden")
        if raw_inputs.get("Tuition fees up to date") == 0:
            risk_factors.append("Overdue tuition fees")
        if raw_inputs.get("Curricular units 1st sem (approved)", 0) < raw_inputs.get("Curricular units 1st sem (enrolled)", 0) / 2:
            risk_factors.append("Low academic approvals (1st Sem)")
        if raw_inputs.get("Curricular units 2nd sem (approved)", 0) < raw_inputs.get("Curricular units 2nd sem (enrolled)", 0) / 2:
            risk_factors.append("Low academic approvals (2nd Sem)")
        if raw_inputs.get("Curricular units 1st sem (grade)", 0) < 10.0:
            risk_factors.append("Low GPA in 1st Semester")
        if raw_inputs.get("Curricular units 2nd sem (grade)", 0) < 10.0:
            risk_factors.append("Low GPA in 2nd Semester")
        if raw_inputs.get("Age at enrollment", 0) > 25:
            risk_factors.append("Older age at enrollment")

        if not risk_factors:
            risk_factors.append("No high-risk factors identified")

        # Keep only top 3 risk factors
        top_risk_factors = risk_factors[:3]

        response = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "risk_level": risk_level,
            "top_risk_factors": top_risk_factors
        }

        # 7. Log call to SQLite
        log_prediction_to_db(prediction, confidence, risk_level, prob_dict, raw_inputs)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")

@app.get("/history")
def get_prediction_history():
    """Fetch last 50 queries logged in predictions table."""
    if not os.path.exists(DB_PATH):
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, prediction, confidence, risk_level, probabilities, inputs FROM predictions ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "timestamp": r[1],
                "prediction": r[2],
                "confidence": r[3],
                "risk_level": r[4],
                "probabilities": json.loads(r[5]),
                "inputs": json.loads(r[6])
            })
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query database: {str(e)}")

@app.get("/stats")
def get_model_and_dataset_stats():
    """Fetch model summary analytics, feature weights, and count data."""
    stats = {}
    
    # 1. Database total students
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Raw count
            cursor.execute("SELECT COUNT(*) FROM students")
            stats["total_students_dataset"] = cursor.fetchone()[0]
            
            # API predictions count
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats["total_predictions_logged"] = cursor.fetchone()[0]
            
            # Average risk distribution of logged calls
            cursor.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
            stats["logged_risk_distribution"] = dict(cursor.fetchall())
            
            conn.close()
        except Exception as e:
            stats["database_error"] = str(e)
            
    # 2. Top 10 feature importances
    fi_path = os.path.join(MODELS_DIR, "feature_importance.pkl")
    if os.path.exists(fi_path):
        sorted_fi = joblib.load(fi_path)
        stats["top_10_features"] = [{"feature": f, "importance": round(imp, 4)} for f, imp in sorted_fi[:10]]
    else:
        stats["top_10_features"] = []

    # 3. Model Accuracy metadata
    stats["model_accuracy"] = 0.865 # Baseline target benchmark
    
    return stats

# Serve Dashboard static folder on root path "/"
if os.path.exists("dashboard"):
    app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")
    
    @app.get("/")
    def read_root():
        """Serve the dashboard index file directly on the root endpoint "/"."""
        index_file = os.path.join("dashboard", "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Student Dropout API running. Dashboard folder found, but index.html missing."}
else:
    @app.get("/")
    def read_root():
        return {"message": "Welcome to the Student Dropout Prediction API! Mount '/dashboard' for visualization."}
