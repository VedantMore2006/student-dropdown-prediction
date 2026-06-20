#!/usr/bin/env python3
"""
Student Dropout Prediction System — Comprehensive Project Validation Suite

This script statically validates the entire project without launching
servers, training models, or running production workflows.
"""

import os
import sys
import json
import logging
import importlib
import traceback
from typing import Dict, List, Tuple, Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Test Result Container ────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str, purpose: str):
        self.name = name
        self.purpose = purpose
        self.passed = False
        self.errors: List[str] = []
        self.details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"

# ── Utility helpers ──────────────────────────────────────────────────────────

def _check_path(*parts: str) -> bool:
    return (PROJECT_ROOT / os.path.join(*parts)).exists()

def _read_file(*parts: str) -> str:
    path = PROJECT_ROOT / os.path.join(*parts)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")

def _safe_import(module_name: str) -> Tuple[Any, str]:
    try:
        mod = importlib.import_module(module_name)
        return mod, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Project Structure Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_project_structure() -> TestResult:
    tr = TestResult(
        "Project Structure Validation",
        "Verify that all expected directories and files exist per README and project plan."
    )
    missing: List[str] = []

    expected_dirs = ["src", "data", "models", "dashboard", "notebooks"]
    for d in expected_dirs:
        if not (PROJECT_ROOT / d).is_dir():
            missing.append(f"directory/{d}")

    expected_files = [
        "README.md",
        "requirements.txt",
        "StudentDropout_ProjectPlan.md",
        "dataset.csv",
        "src/load_data.py",
        "src/preprocessing.py",
        "src/train.py",
        "src/api.py",
        "dashboard/index.html",
        "notebooks/eda.ipynb",
        "notebooks/statistics.ipynb",
    ]
    for f in expected_files:
        if not _check_path(f):
            missing.append(f"file/{f}")

    # Data files expected
    data_files = ["data/students.csv", "data/students.db"]
    for f in data_files:
        if not _check_path(f):
            missing.append(f"file/{f}")

    # Model artifacts expected
    model_files = [
        "models/best_model.pkl",
        "models/scaler.pkl",
        "models/label_encoder.pkl",
        "models/feature_importance.pkl",
        "models/final_columns.pkl",
        "models/original_columns.pkl",
    ]
    for f in model_files:
        if not _check_path(f):
            missing.append(f"file/{f}")

    # The project plan references assets/screenshots/ — check if it exists
    if not (PROJECT_ROOT / "assets").is_dir():
        missing.append("directory/assets/ (referenced in project plan)")

    if missing:
        tr.passed = False
        tr.errors.append(f"Missing {len(missing)} expected path(s): {', '.join(missing)}")
    else:
        tr.passed = True
        tr.details = "All expected directories and files present."

    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Dependency Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_dependencies() -> TestResult:
    tr = TestResult(
        "Dependency Validation",
        "Verify that requirements.txt declares all needed packages and they are installable / importable."
    )
    if not _check_path("requirements.txt"):
        tr.errors.append("requirements.txt is missing")
        return tr

    content = _read_file("requirements.txt")
    declared = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    critical_packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "imbalanced-learn": "imblearn",
        "xgboost": "xgboost",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "joblib": "joblib",
        "seaborn": "seaborn",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "ipykernel": "ipykernel",
    }

    # Check all critical packages are declared (accounting for version pins)
    undeclared = []
    for pkg_name in critical_packages:
        found = any(declared_line.startswith(pkg_name) or declared_line.startswith(pkg_name + "==")
                    or declared_line.startswith(pkg_name + ">=") for declared_line in declared)
        if not found:
            undeclared.append(pkg_name)
    if undeclared:
        tr.errors.append(f"Packages missing from requirements.txt: {', '.join(undeclared)}")

    # Check all declared packages can be imported (if installed)
    import_errors = []
    for line in declared:
        # Normalize package name to import name
        pkg = line.strip()
        if ">" in pkg or "<" in pkg or "=" in pkg or ";" in pkg or "#" in pkg:
            pkg = pkg.split(">")[0].split("<")[0].split("=")[0].split(";")[0].split("#")[0].strip()
        if not pkg or pkg.startswith("-"):
            continue
        # Map known package names to import names
        import_map = {
            "scikit-learn": "sklearn",
            "imbalanced-learn": "imblearn",
        }
        import_name = import_map.get(pkg, pkg)
        try:
            importlib.import_module(import_name)
        except ImportError:
            import_errors.append(pkg)
        except Exception:
            pass  # Some import successfully but may have issues

    if import_errors:
        tr.errors.append(f"Packages declared but not importable: {', '.join(import_errors)}")

    # Check for version pinning: requirements.txt has no version constraints
    has_version_pins = any(
        "==" in line or ">=" in line or "<=" in line or "~=" in line
        for line in declared
    )
    if not has_version_pins:
        tr.errors.append("No version pins in requirements.txt — risk of breaking changes")

    if not tr.errors:
        tr.passed = True
        tr.details = "All critical packages declared and importable."

    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Configuration Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_configuration() -> TestResult:
    tr = TestResult(
        "Configuration Validation",
        "Verify environment config, database schema, and dataset integrity."
    )

    # 3a. Database schema verification
    import sqlite3
    db_path = PROJECT_ROOT / "data" / "students.db"
    if not db_path.exists():
        tr.errors.append("Database file data/students.db not found")
        return tr

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {t[0] for t in cursor.fetchall()}

        expected_tables = {"students", "predictions"}
        missing_tables = expected_tables - tables
        if missing_tables:
            tr.errors.append(f"Missing DB tables: {missing_tables}")

        # Verify students table columns
        if "students" in tables:
            cursor.execute("PRAGMA table_info(students)")
            student_cols = {r[1] for r in cursor.fetchall()}
            expected_cols = {
                "Marital status", "Application mode", "Application order",
                "Course", "Daytime/evening attendance", "Previous qualification",
                "Nacionality", "Mother's qualification", "Father's qualification",
                "Mother's occupation", "Father's occupation", "Displaced",
                "Educational special needs", "Debtor", "Tuition fees up to date",
                "Gender", "Scholarship holder", "Age at enrollment", "International",
                "Target",
            }
            missing_cols = expected_cols - student_cols
            if missing_cols:
                tr.errors.append(f"Missing columns in 'students' table: {missing_cols}")

        if "predictions" in tables:
            cursor.execute("PRAGMA table_info(predictions)")
            pred_cols = {r[1] for r in cursor.fetchall()}
            for col in ["id", "timestamp", "prediction", "confidence", "risk_level", "probabilities", "inputs"]:
                if col not in pred_cols:
                    tr.errors.append(f"Missing column '{col}' in 'predictions' table")

        conn.close()
    except Exception as e:
        tr.errors.append(f"Database validation error: {e}")

    # 3b. Dataset CSV integrity
    csv_path = PROJECT_ROOT / "dataset.csv"
    if csv_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(str(csv_path))
            if df.shape[1] != 35:
                tr.errors.append(f"Expected 35 columns in dataset.csv, got {df.shape[1]}")
            if df.shape[0] != 4424:
                tr.errors.append(f"Expected 4424 rows in dataset.csv, got {df.shape[0]}")
            expected_targets = {"Dropout", "Enrolled", "Graduate"}
            actual_targets = set(df["Target"].unique())
            missing_targets = expected_targets - actual_targets
            if missing_targets:
                tr.errors.append(f"Missing target classes in dataset: {missing_targets}")
        except Exception as e:
            tr.errors.append(f"CSV validation error: {e}")
    else:
        tr.errors.append("dataset.csv not found at project root")

    tr.passed = len(tr.errors) == 0
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Import Integrity Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_import_integrity() -> TestResult:
    tr = TestResult(
        "Import Integrity Validation",
        "Verify that all internal imports resolve without errors."
    )

    # Check that src modules are importable
    # Since src/ has no __init__.py, we import modules directly via sys.path
    modules_to_check = [
        ("load_data", ["setup_directories", "ingest_data"]),
        ("preprocessing", [
            "load_data_from_db", "fit_and_save_preprocessors",
            "get_balanced_train_test", "preprocess_single_input",
        ]),
        ("train", ["train_and_evaluate_models"]),
    ]

    for mod_name, expected_funcs in modules_to_check:
        mod, err = _safe_import(mod_name)
        if err:
            tr.errors.append(f"Import '{mod_name}' failed: {err}")
        else:
            for func in expected_funcs:
                if not hasattr(mod, func):
                    tr.errors.append(f"Missing function {mod_name}.{func}")

    # Check for sys.path manipulation risk in api.py (sibling import)
    api_content = _read_file("src/api.py")
    if "sys.path.append" in api_content:
        if "from preprocessing import preprocess_single_input" not in api_content:
            tr.errors.append("api.py uses sys.path hack but import path may still be broken")

    # Check for circular import risk between modules
    deps = {
        "train.py": ["preprocessing"],
        "api.py": ["preprocessing"],
        "preprocessing.py": [],
        "load_data.py": [],
    }
    for mod, imported in deps.items():
        content = _read_file(f"src/{mod}")
        for dep in imported:
            if f"from {dep}" not in content and f"import {dep}" not in content:
                tr.errors.append(f"{mod} does not import its expected dependency '{dep}'")

    tr.passed = len(tr.errors) == 0
    if tr.passed:
        tr.details = "All imports resolve and required functions are present."
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Model Artifact Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_model_artifacts() -> TestResult:
    tr = TestResult(
        "Model Artifact Validation",
        "Verify that serialized model artifacts exist, are loadable, and have expected structure."
    )

    models_dir = PROJECT_ROOT / "models"

    # Check existence of all required model files
    required = ["best_model.pkl", "scaler.pkl", "label_encoder.pkl", "feature_importance.pkl"]
    for f in required:
        if not (models_dir / f).exists():
            tr.errors.append(f"Missing model artifact: {f}")

    if tr.errors:
        return tr

    # Try to load artifacts
    import joblib
    try:
        model = joblib.load(str(models_dir / "best_model.pkl"))
        # Check it's a classifier
        if not hasattr(model, "predict"):
            tr.errors.append("best_model.pkl does not have predict() method")
        if not hasattr(model, "predict_proba"):
            tr.errors.append("best_model.pkl does not have predict_proba() method")
        # Verify it's one of the expected classifier types
        expected_types = {"LogisticRegression", "RandomForestClassifier", "XGBClassifier", "SVC"}
        actual_type = type(model).__name__
        if actual_type not in expected_types:
            tr.errors.append(f"Unexpected model type: {actual_type}")
        else:
            tr.details += f"Model type: {actual_type}. "
    except Exception as e:
        tr.errors.append(f"Cannot load best_model.pkl: {e}")

    try:
        scaler = joblib.load(str(models_dir / "scaler.pkl"))
        from sklearn.preprocessing import StandardScaler
        if not isinstance(scaler, StandardScaler):
            tr.errors.append(f"scaler.pkl is not a StandardScaler (got {type(scaler).__name__})")
        else:
            # After one-hot encoding, we expect 76 features
            n_features = len(scaler.mean_) if hasattr(scaler, "mean_") else 0
            tr.details += f"Scaler features: {n_features}. "
            if n_features == 0:
                tr.errors.append("Scaler appears unfitted (0 feature means)")
    except Exception as e:
        tr.errors.append(f"Cannot load scaler.pkl: {e}")

    try:
        le = joblib.load(str(models_dir / "label_encoder.pkl"))
        from sklearn.preprocessing import LabelEncoder
        if not isinstance(le, LabelEncoder):
            tr.errors.append(f"label_encoder.pkl is not a LabelEncoder (got {type(le).__name__})")
        else:
            expected_classes = ["Dropout", "Enrolled", "Graduate"]
            actual_classes = list(le.classes_)
            if actual_classes != expected_classes:
                tr.errors.append(f"Label encoder classes mismatch: expected {expected_classes}, got {actual_classes}")
            else:
                tr.details += "Label encoder correctly maps Dropout/Enrolled/Graduate. "
    except Exception as e:
        tr.errors.append(f"Cannot load label_encoder.pkl: {e}")

    try:
        fi = joblib.load(str(models_dir / "feature_importance.pkl"))
        if not isinstance(fi, list) or len(fi) == 0:
            tr.errors.append("feature_importance.pkl is empty or not a list")
        else:
            tr.details += f"Feature importance has {len(fi)} entries. "
    except Exception as e:
        tr.errors.append(f"Cannot load feature_importance.pkl: {e}")

    tr.passed = len(tr.errors) == 0
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: API / Interface Validation (Static)
# ══════════════════════════════════════════════════════════════════════════════

def test_api_interface() -> TestResult:
    tr = TestResult(
        "API / Interface Validation",
        "Verify FastAPI endpoints are correctly defined without launching the server."
    )

    api_path = PROJECT_ROOT / "src" / "api.py"
    if not api_path.exists():
        tr.errors.append("src/api.py not found")
        return tr

    content = _read_file("src/api.py")

    # Check expected imports for FastAPI
    fastapi_imports = ["FastAPI", "HTTPException", "CORSMiddleware", "BaseModel", "Field"]
    for imp in fastapi_imports:
        if imp not in content:
            tr.errors.append(f"Missing required FastAPI import: {imp}")

    # Check expected endpoints
    endpoint_decorators = [
        ("@app.get(\"/health\")", "/health"),
        ("@app.post(\"/predict\")", "/predict"),
        ("@app.get(\"/history\")", "/history"),
        ("@app.get(\"/stats\")", "/stats"),
        ("@app.get(\"/\")", "/ (root)"),
    ]
    for decorator, route in endpoint_decorators:
        if decorator not in content:
            tr.errors.append(f"Missing endpoint: {route}")

    # Check that CORS is configured
    if "allow_origins" not in content:
        tr.errors.append("CORS middleware not configured — dashboard cross-origin fetch will fail")

    # Check that the Pydantic schema for /predict exists
    if "class StudentFeatures" not in content:
        tr.errors.append("Missing StudentFeatures Pydantic model for /predict endpoint")
    else:
        # Check that all 34 feature fields are defined
        field_count = content.count("Field(alias=")
        if field_count < 30:
            tr.errors.append(f"StudentFeatures likely incomplete: only ~{field_count} fields found (expected 34)")

    # Check static file serving for dashboard
    if "StaticFiles" not in content:
        tr.errors.append("StaticFiles not configured — dashboard may not be served")

    tr.passed = len(tr.errors) == 0
    if tr.passed:
        tr.details = "All 5 expected endpoints (/health, /predict, /history, /stats, /) defined with CORS."
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: Documentation Consistency Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_documentation_consistency() -> TestResult:
    tr = TestResult(
        "Documentation Consistency Validation",
        "Verify README claims match actual implementation."
    )

    readme = _read_file("README.md")

    # Check claimed features exist
    checks = [
        ("SQLite", "data/students.db" in str(list(PROJECT_ROOT.glob("data/*"))) or True),
    ]

    # README claims 8 charts in EDA notebook — verify
    nb_path = PROJECT_ROOT / "notebooks" / "eda.ipynb"
    if nb_path.exists():
        with open(str(nb_path)) as f:
            eda = json.load(f)
        plot_cells = sum(1 for c in eda.get("cells", [])
                         if any("plt.figure" in "".join(c.get("source", [])) or
                                "sns." in "".join(c.get("source", [])) for _ in [1]))
        # More accurate: count markdown cells with "Plot" headers
        plot_headers = sum(1 for c in eda.get("cells", [])
                          if c.get("cell_type") == "markdown" and "Plot" in "".join(c.get("source", "")))
        if plot_headers < 8:
            tr.errors.append(f"README claims 8 EDA plots, but notebook has ~{plot_headers} plot sections")

    # README claims 4 hypothesis tests in statistics notebook
    stats_path = PROJECT_ROOT / "notebooks" / "statistics.ipynb"
    if stats_path.exists():
        with open(str(stats_path)) as f:
            stats_nb = json.load(f)
        test_headers = sum(1 for c in stats_nb.get("cells", [])
                          if c.get("cell_type") == "markdown" and "Test" in "".join(c.get("source", "")))
        if test_headers < 4:
            tr.errors.append(f"README claims 4 statistical tests, but notebook has ~{test_headers} test sections")

    # README references 'Admission grade' column in dataset — verify
    # Note: the README now has a doc note explaining this discrepancy
    csv_path = PROJECT_ROOT / "dataset.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(str(csv_path), nrows=1)
        if "Admission grade" not in df.columns:
            # Check if README has the explanatory note
            readme = _read_file("README.md")
            if "Admission grade" in readme and "does not rely on a standalone admission grade" in readme:
                tr.details += "README documents the Admission grade column discrepancy. "
            else:
                tr.errors.append("README references 'Admission grade' but column does not exist in dataset.csv")

    # README references badges with versions
    if "Python 3.14" in readme:
        tr.details += "Python 3.14 badge present. "
    # FastAPI badge version
    if "FastAPI" in readme and "v0.100" in readme:
        tr.details += "FastAPI badge present. "

    # Verify uvicorn command in README works as documented
    if "uvicorn src.api:app --reload" not in readme:
        tr.errors.append("README lauch command missing or different from expected")

    # Verify project plan checklist items — many unchecked
    plan = _read_file("StudentDropout_ProjectPlan.md")
    unchecked_items = plan.count("- [ ]")
    if unchecked_items > 0:
        tr.details += f"Project plan has {unchecked_items} unchecked items. "

    tr.passed = len(tr.errors) == 0
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Static Logic Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_static_logic() -> TestResult:
    tr = TestResult(
        "Static Logic Validation",
        "Verify critical business logic is implemented (not stubs/placeholders)."
    )

    # Check preprocessing: SMOTE application, encoding, scaling
    prep = _read_file("src/preprocessing.py")
    logic_checks = {
        "SMOTE usage": "SMOTE" in prep,
        "LabelEncoder fit": "LabelEncoder" in prep,
        "StandardScaler fit": "StandardScaler" in prep,
        "One-hot encoding": "get_dummies" in prep,
        "Train/test split": "train_test_split" in prep,
        "Joblib save": "joblib.dump" in prep,
    }
    for check, present in logic_checks.items():
        if not present:
            tr.errors.append(f"Missing preprocessing logic: {check}")

    # Check training: multi-model, CV, evaluation
    train = _read_file("src/train.py")
    train_checks = {
        "Logistic Regression": "LogisticRegression" in train,
        "Random Forest": "RandomForestClassifier" in train,
        "XGBoost": "XGBClassifier" in train,
        "SVM": "SVC" in train,
        "Cross-validation": "cross_val_score" in train,
        "F1 score calculation": "f1_score" in train,
        "ROC-AUC calculation": "roc_auc_score" in train,
        "Classification report": "classification_report" in train,
        "Best model saving": "joblib.dump" in train and "best_model" in train,
    }
    for check, present in train_checks.items():
        if not present:
            tr.errors.append(f"Missing training logic: {check}")

    # Check API: risk factors, prediction logging, stats
    api = _read_file("src/api.py")
    api_checks = {
        "Risk factor extraction": "top_risk_factors" in api,
        "Risk level classification": "risk_level" in api,
        "Prediction logging to DB": "log_prediction_to_db" in api,
        "Model accuracy metadata": "model_accuracy" in api,
        "Health check endpoint": "/health" in api,
        "Prediction history": "/history" in api,
        "Model stats": "/stats" in api,
    }
    for check, present in api_checks.items():
        if not present:
            tr.errors.append(f"Missing API logic: {check}")

    # Check load_data for DB pipeline
    ld = _read_file("src/load_data.py")
    ld_checks = {
        "CSV copy to data/": "shutil.copy" in ld,
        "SQLite ingestion": "to_sql" in ld,
        "Predictions table creation": "CREATE TABLE" in ld,
        "Verification query": "SELECT" in ld and "GROUP BY" in ld,
    }
    for check, present in ld_checks.items():
        if not present:
            tr.errors.append(f"Missing data loading logic: {check}")

    tr.passed = len(tr.errors) == 0
    if tr.passed:
        tr.details = "All critical business logic paths are implemented."
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Security & Environment Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_security() -> TestResult:
    tr = TestResult(
        "Security & Environment Validation",
        "Detect hardcoded secrets, unsafe defaults, and missing environment protections."
    )

    findings = []

    # Check all .py files for hardcoded secrets / credentials / tokens
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if "venv" in str(py_file) or "__pycache__" in str(py_file):
            continue
        # Skip the test file itself
        if py_file.name == "validate_project.py":
            continue
        content = py_file.read_text()
        # Check for common secret patterns
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            lower = stripped.lower()
            # Filter known false positives
            if "allow_credentials" in lower:
                continue
            if any(k in lower for k in ["password", "secret", "token", "api_key", "apikey", "credential"]):
                if "=" in stripped and not stripped.endswith("input()"):
                    findings.append(f"{py_file.name}:{i}: Potential secret: {stripped[:80]}")

    # Check CORS is wide open — acceptable for dev/portfolio project
    api_content = _read_file("src/api.py")
    if 'allow_origins=["*"]' in api_content:
        if "# Dev/portfolio: wide open CORS" not in api_content:
            findings.append("src/api.py: CORS allows all origins (*) — consider adding explicit origins for production")

    # Check if .env or .gitignore exists
    if not _check_path(".gitignore"):
        findings.append("No .gitignore — risk of committing venv/, __pycache__/, .env")
    else:
        gitignore = _read_file(".gitignore")
        if "venv" not in gitignore:
            findings.append(".gitignore does not exclude venv/")
        if "__pycache__" not in gitignore:
            findings.append(".gitignore does not exclude __pycache__/")

    # Check for unsafe defaults in API (student features)
    if "default=0" in api_content:
        pass  # Some defaults of 0 are expected

    # Check venv is actually gitignored
    if _check_path(".gitignore"):
        gitignore = _read_file(".gitignore")
        # Check if .gitignore exists in the git repo
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "venv/"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True
        )
        if result.returncode == 0:
            findings.append("venv/ is tracked by git — should be in .gitignore")

    for f in findings:
        tr.errors.append(f)

    tr.passed = len(tr.errors) == 0
    if not tr.errors:
        tr.details = "No hardcoded secrets detected."
    else:
        tr.details = f"{len(tr.errors)} security finding(s)."
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: End-to-End Readiness Validation
# ══════════════════════════════════════════════════════════════════════════════

def test_e2e_readiness() -> TestResult:
    tr = TestResult(
        "End-to-End Readiness Validation",
        "Verify all components are present and correctly connected for successful execution without actually running."
    )

    critical_issues = []

    # 1. Data pipeline readiness
    if not _check_path("dataset.csv"):
        critical_issues.append("dataset.csv missing — load_data.py has no input")
    if not _check_path("data", "students.db"):
        critical_issues.append("students.db missing — run load_data.py first")

    # 2. Preprocessing readiness — check module can be used by both train.py and api.py
    prep_mod, err = _safe_import("preprocessing")
    if err:
        critical_issues.append(f"preprocessing module not importable: {err}")

    # 3. Training readiness — check train.py can import from preprocessing
    train_mod, err = _safe_import("train")
    if err:
        critical_issues.append(f"train module not importable: {err}")

    # 4. API readiness — check api.py can import preprocessing and has model artifacts
    if not _check_path("models", "best_model.pkl"):
        critical_issues.append("best_model.pkl missing — run train.py first")
    if not _check_path("models", "scaler.pkl"):
        critical_issues.append("scaler.pkl missing")
    if not _check_path("models", "label_encoder.pkl"):
        critical_issues.append("label_encoder.pkl missing")

    # 5. Dashboard readiness — check it references correct API endpoints
    dashboard = _read_file("dashboard/index.html")
    api_endpoints_in_dashboard = ["/predict", "/health", "/history", "/stats"]
    for ep in api_endpoints_in_dashboard:
        if ep not in dashboard:
            critical_issues.append(f"Dashboard missing reference to API endpoint {ep}")

    # 6. Check that the number of features in preprocessing matches scaler
    if _check_path("models", "final_columns.pkl") and _check_path("models", "scaler.pkl"):
        import joblib
        try:
            final_cols = joblib.load(str(PROJECT_ROOT / "models" / "final_columns.pkl"))
            scaler = joblib.load(str(PROJECT_ROOT / "models" / "scaler.pkl"))
            n_scaler_features = len(scaler.mean_)
            if len(final_cols) != n_scaler_features:
                critical_issues.append(
                    f"Mismatch: final_columns.pkl has {len(final_cols)} features "
                    f"but scaler expects {n_scaler_features}"
                )
        except Exception as e:
            critical_issues.append(f"Cannot compare model artifacts: {e}")

    # 7. Check venv activation instruction consistency
    readme = _read_file("README.md")
    if "source venv/bin/activate" not in readme:
        critical_issues.append("README venv activation command missing or incorrect")

    # 8. Check for __init__.py in src/ (prevents reliable imports)
    if not _check_path("src", "__init__.py"):
        critical_issues.append("src/__init__.py missing — may cause import issues in some contexts")

    for issue in critical_issues:
        tr.errors.append(issue)

    tr.passed = len(tr.errors) == 0
    if tr.passed:
        tr.details = "All components appear connected and ready for execution."
    return tr


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Run all tests and generate report
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  STUDENT DROPOUT PREDICTION SYSTEM — PROJECT VALIDATION SUITE")
    print("=" * 70)
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    print("=" * 70)
    print()

    tests = [
        ("1. Project Structure Validation", test_project_structure),
        ("2. Dependency Validation", test_dependencies),
        ("3. Configuration Validation", test_configuration),
        ("4. Import Integrity Validation", test_import_integrity),
        ("5. Model Artifact Validation", test_model_artifacts),
        ("6. API / Interface Validation", test_api_interface),
        ("7. Documentation Consistency", test_documentation_consistency),
        ("8. Static Logic Validation", test_static_logic),
        ("9. Security & Environment", test_security),
        ("10. E2E Readiness Validation", test_e2e_readiness),
    ]

    results: List[TestResult] = []
    for title, test_fn in tests:
        log.info(f"Running: {title}")
        try:
            result = test_fn()
        except Exception as e:
            result = TestResult(title, "Error during test execution")
            result.errors.append(f"Unexpected error: {traceback.format_exc()}")
        results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {title}")
        if not result.passed:
            for err in result.errors[:3]:
                print(f"      └─ {err}")
            if len(result.errors) > 3:
                print(f"      └─ ... and {len(result.errors) - 3} more error(s)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_errors = sum(len(r.errors) for r in results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}]  {r.name}")
        if r.details:
            print(f"         {r.details}")

    print()
    print(f"  Passed: {passed}/{len(results)}  |  Failed: {failed}/{len(results)}  |  Total Issues: {total_errors}")

    # ── Readiness Score ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL READINESS SCORE")
    print("=" * 70)

    # Base score: each test contributes up to 10 points
    # Each error deducts proportionally
    score = 100
    for r in results:
        if not r.passed:
            # Deduct per error, max -10 per test
            deduction = min(len(r.errors) * 5, 10)
            score -= deduction

    score = max(0, min(100, score))

    print(f"  Score: {score}/100")
    print()

    # Critical issues summary
    all_errors = []
    for r in results:
        for e in r.errors:
            all_errors.append(f"  [{r.name}] {e}")

    if all_errors:
        print("  Critical Issues:")
        for e in all_errors:
            print(f"    • {e}")
        print()

    # Recommendations
    print("  Recommendations:")
    recommendations = [
        "1. Pin dependency versions in requirements.txt (e.g., pandas==2.2.0)",
        "2. Add src/__init__.py for proper package imports",
        "3. Create .gitignore excluding venv/, __pycache__/, *.pyc",
        "4. Add 'Admission grade' column to dataset or update README (discrepancy found)",
        "5. Add assets/screenshots/ directory as referenced in project plan",
        "6. Consider adding Dockerfile and docker-compose.yml for deployment",
        "7. Add input validation for edge cases in /predict endpoint",
        "8. Consider adding unit tests for preprocessing (preprocess_single_input edge cases)",
    ]
    for rec in recommendations:
        print(f"    {rec}")

    print()
    print("=" * 70)

    return 0 if score >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
