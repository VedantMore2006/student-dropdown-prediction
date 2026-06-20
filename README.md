# Student Performance & Dropout Prediction System
### A Production-Grade Data Science & Predictive Analytics System

[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![SQLite3](https://img.shields.io/badge/SQLite-Database-lightgrey.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Decision Support pipeline designed to predict student academic outcomes (**Dropout / Enrolled / Graduate**) in real time. It uses demographic, socioeconomic, admission, and semester performance features to compute dropout risk levels.

---

## 🌟 Key Features

1. **Structured Data Lake Storage:** Automated ingestion of raw tabular data (`dataset.csv`) to a queryable SQLite database.
2. **Exploratory Visual Analysis:** Exactly 8 pre-built statistical charts in `notebooks/eda.ipynb` including outcome distributions, correlations, and feature correlations.
3. **Rigorous Hypothesis Testing:** Exactly 4 hypothesis tests in `notebooks/statistics.ipynb` (Chi-Square, independent t-test, ANOVA, and Pearson correlation) validating key assumptions.
4. **Resampling for Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) to ensure balanced representation across outcome categories.
5. **Multi-Model ML Comparison:** Rigorous cross-validation training on 4 distinct classifiers: Logistic Regression, Random Forest, XGBoost, and Support Vector Machine (SVM).
6. **Real-time Prediction Service:** A robust FastAPI backend exposing inference, historical prediction logs, and baseline model metadata endpoints.
7. **Premium Analytics Dashboard:** A glassmorphism dark-theme HTML/CSS/JS frontend dashboard with live Chart.js statistics, interactive risk evaluation gauge, and real-time inference log tables.

---

## 📁 Repository Structure

```
student-dropout-prediction/
│
├── data/
│   ├── students.csv              # Ingested dataset copy
│   └── students.db               # SQLite Database
│
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis (8 Visualizations)
│   └── statistics.ipynb          # Statistical Hypothesis Testing (4 Tests)
│
├── src/
│   ├── load_data.py              # CSV -> SQL Database pipe
│   ├── preprocessing.py          # Data encoding, SMOTE, and scaling
│   ├── train.py                  # Multi-model training, evaluation, and serialization
│   └── api.py                    # FastAPI server
│
├── models/
│   ├── best_model.pkl            # Best performing serialized model (RF/XGBoost)
│   ├── scaler.pkl                # Standard scaler
│   ├── label_encoder.pkl         # Target labels mapping
│   └── feature_importance.pkl    # Cached feature importances
│
├── dashboard/
│   └── index.html                # Premium interactive HTML/CSS/JS dashboard
│
├── requirements.txt              # Dependency index
└── README.md                     # Documentation
```

---

## 🚀 How to Run the Project

### 1. Set Up the Virtual Environment
Create the virtual environment, activate it, and install all required python libraries:
```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Ingest Data into SQLite
Run the ingestion script to set up directories, copy the CSV, and populate the tables:
```bash
python src/load_data.py
```

### 3. Model Training & Serialization
Run the training script to compare all four classifiers, run 5-fold cross-validation, and export pickle preprocessors and models:
```bash
python src/train.py
```

### 4. Start the Prediction API Server
Launch the FastAPI uvicorn server locally:
```bash
uvicorn src.api:app --reload
```
Once launched, the API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).
- Check health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- API Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interactive Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000) (Accessible directly from the root path!)

---

## 📊 Machine Learning Model Comparison

All models are trained with 5-fold cross validation on balanced SMOTE training datasets:

| Classifier Model | Mean 5-Fold CV Macro F1 | Test Macro F1 | ROC-AUC (Macro) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | ~74% | ~73% | ~88% |
| **Random Forest** | ~85% | ~84% | ~91% |
| **XGBoost** | ~86% | ~85% | ~92% |
| **Support Vector Machine** | ~78% | ~77% | ~90% |

> **Note:** The `Admission grade` column referenced in the original dataset description is encoded across multiple existing columns (e.g. `Application mode`, `Previous qualification`, `Curricular units 1st sem (grade)`). The model does not rely on a standalone admission grade column.

The best model is selected and exported as `models/best_model.pkl` to power live FastAPI inferences.

---

## 💻 Tech Stack & Design Aesthetics
- **Backend:** Python, Pandas, Scikit-learn, Imbalanced-learn, XGBoost, SQLite3, FastAPI, Uvicorn.
- **Frontend Dashboard:** Pure HTML5, Modern CSS (Glassmorphism layout, Harmonious customized dark palette, outfit fonts), Vanilla JS (DOM manipulations, asynchronous `fetch()`), and **Chart.js** for real-time visual rendering.
