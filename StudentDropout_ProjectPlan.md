# Student Performance & Dropout Prediction System
### A Data Science & AI Analytics Portfolio Project

> **Resume Target:** Data Science & AI Analytics Internship  
> **Author:** Vedant Vasant More  
> **Estimated Build Time:** 4–5 days  
> **Stack:** Python · Pandas · Scikit-learn · Seaborn · SQLite · FastAPI · HTML/CSS/JS

---

## Table of Contents

1. [Project Aim](#1-project-aim)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Project Architecture](#4-project-architecture)
5. [Phase-by-Phase Breakdown](#5-phase-by-phase-breakdown)
   - Phase 1: Data Collection & SQL Storage
   - Phase 2: Exploratory Data Analysis (EDA)
   - Phase 3: Statistical Analysis
   - Phase 4: Feature Engineering & Preprocessing
   - Phase 5: Model Training & Evaluation
   - Phase 6: API Deployment
   - Phase 7: Dashboard
6. [Deliverables Checklist](#6-deliverables-checklist)
7. [Resume Outcomes](#7-resume-outcomes)
8. [ATS Keyword Coverage](#8-ats-keyword-coverage)
9. [Folder Structure](#9-folder-structure)
10. [Resume Bullet Points](#10-resume-bullet-points-ready-to-copy)

---

## 1. Project Aim

To build a complete, end-to-end data science system that predicts whether a higher education student is at risk of dropping out — using demographic, socioeconomic, and academic performance data — and deploys the prediction as a real-time API with an interactive dashboard.

This project is designed to demonstrate **every core skill expected in a Data Science & AI Analytics internship role**: data handling, statistical reasoning, machine learning, visualization, and deployment.

---

## 2. Problem Statement

Student dropout is a critical challenge in higher education globally, particularly in developing regions. Early identification of at-risk students allows institutions to intervene with scholarships, counselling, or academic support — before it is too late.

**The system must answer:**
- Which students are most likely to drop out, and why?
- What are the strongest predictors of dropout vs. graduation?
- Can we quantify dropout probability for a new incoming student in real time?

**Classification target (3 classes):**
- `Dropout` — student left before completing the course
- `Enrolled` — student is currently active
- `Graduate` — student successfully completed the course

---

## 3. Dataset

| Property | Details |
|---|---|
| **Source** | UCI Machine Learning Repository / Kaggle |
| **Kaggle URL** | https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention |
| **Records** | 4,424 students |
| **Features** | 36 attributes |
| **Target Column** | `Target` (Dropout / Enrolled / Graduate) |
| **License** | Open / Public |

**Key feature groups in the dataset:**

| Group | Example Columns |
|---|---|
| Demographic | Age at enrollment, Gender, Nationality, Marital status |
| Socioeconomic | Father's occupation, Mother's qualification, Scholarship holder, Debtor |
| Academic (Admission) | Admission grade, Previous qualification grade, Application mode |
| Academic (Semester) | Curricular units approved (1st & 2nd sem), Grade averages |
| Macroeconomic | GDP, Inflation rate, Unemployment rate |

---

## 4. Project Architecture

```
Raw CSV Data
     │
     ▼
[Phase 1] SQLite Storage  ──── data/students.db
     │
     ▼
[Phase 2] EDA Notebook  ──────── notebooks/eda.ipynb
     │
     ▼
[Phase 3] Statistical Analysis ── notebooks/statistics.ipynb
     │
     ▼
[Phase 4] Feature Engineering ─── src/preprocessing.py
     │
     ▼
[Phase 5] Model Training ─────── src/train.py → models/best_model.pkl
     │
     ▼
[Phase 6] FastAPI Endpoint ────── src/api.py
     │
     ▼
[Phase 7] Dashboard ──────────── dashboard/index.html
```

---

## 5. Phase-by-Phase Breakdown

---

### Phase 1 — Data Collection & SQL Storage
**File:** `src/load_data.py`  
**Time:** ~2 hours

**What to do:**
- Download the CSV from Kaggle
- Load it into a **SQLite database** using `pandas` + `sqlite3`
- Create two tables: `students` (raw data) and `predictions` (log of API calls)
- Write SQL queries using `pd.read_sql()` to verify ingestion

**Key code patterns to use:**
```python
import pandas as pd
import sqlite3

df = pd.read_csv('data/students.csv')
conn = sqlite3.connect('data/students.db')
df.to_sql('students', conn, if_exists='replace', index=False)

# Query via pandas
df_query = pd.read_sql("SELECT Target, COUNT(*) FROM students GROUP BY Target", conn)
```

**What this proves to a recruiter:**
- You can ingest raw data into a structured store
- You know how to use SQL alongside Python
- You understand data pipelines start with storage, not notebooks

---

### Phase 2 — Exploratory Data Analysis (EDA)
**File:** `notebooks/eda.ipynb`  
**Time:** ~1 day

**What to do — produce exactly these 8 visualizations:**

| # | Visualization | Library | What it shows |
|---|---|---|---|
| 1 | Class distribution bar chart | Seaborn | Dropout vs. Enrolled vs. Graduate counts |
| 2 | Missing values heatmap | Seaborn | Data quality snapshot |
| 3 | Age distribution by outcome | Seaborn (boxplot) | Are older students more likely to drop? |
| 4 | Scholarship holder vs. dropout | Seaborn (countplot) | Socioeconomic impact |
| 5 | Semester grade correlation | Seaborn (scatter) | Academic performance trend |
| 6 | Correlation matrix heatmap | Seaborn | Feature relationships at a glance |
| 7 | GDP vs. dropout rate | Matplotlib (line) | Macroeconomic influence |
| 8 | Feature importance (post-model) | Matplotlib (barh) | Top 10 predictors |

**Key operations to demonstrate:**
```python
# Data cleaning
df.isnull().sum()
df.duplicated().sum()
df.describe()

# Data type handling
df['Target'] = df['Target'].astype('category')

# Outlier detection
Q1 = df['Admission grade'].quantile(0.25)
Q3 = df['Admission grade'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Admission grade'] < Q1 - 1.5*IQR) | (df['Admission grade'] > Q3 + 1.5*IQR)]
```

**What this proves to a recruiter:**
- You do not just run models blindly — you understand your data first
- You can produce publication-quality visualizations
- You think about data quality, not just accuracy scores

---

### Phase 3 — Statistical Analysis
**File:** `notebooks/statistics.ipynb`  
**Time:** ~3–4 hours

**What to do — run exactly these 4 statistical tests:**

| Test | Question it answers | Library |
|---|---|---|
| Chi-Square Test | Does scholarship status significantly affect dropout? | `scipy.stats` |
| T-Test (independent) | Is admission grade significantly different between dropouts and graduates? | `scipy.stats` |
| ANOVA | Do semester grades differ significantly across all 3 outcome groups? | `scipy.stats` |
| Pearson Correlation | What is the linear relationship between GDP and dropout rate? | `numpy` / `scipy` |

**Example pattern:**
```python
from scipy import stats

# T-test: dropout vs graduate admission grades
dropout_grades = df[df['Target'] == 'Dropout']['Admission grade']
graduate_grades = df[df['Target'] == 'Graduate']['Admission grade']

t_stat, p_value = stats.ttest_ind(dropout_grades, graduate_grades)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
# Interpret: if p < 0.05, difference is statistically significant
```

**What this proves to a recruiter:**
- You understand the difference between correlation and causation
- You know when to use which statistical test
- You can draw data-backed conclusions, not just visual guesses

---

### Phase 4 — Feature Engineering & Preprocessing
**File:** `src/preprocessing.py`  
**Time:** ~3 hours

**What to do:**
- Label encode the target column (`Dropout=0`, `Enrolled=1`, `Graduate=2`)
- One-hot encode categorical features (Marital status, Application mode, Nationality)
- Scale numerical features using `StandardScaler`
- Handle class imbalance using `SMOTE` from `imbalanced-learn`
- Split into train/test sets with stratification

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])

X = df.drop(['Target', 'Target_encoded'], axis=1)
y = df['Target_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

---

### Phase 5 — Model Training & Evaluation
**File:** `src/train.py`  
**Time:** ~1 day

**Train these 4 models and compare them:**

| Model | Why include it |
|---|---|
| Logistic Regression | Baseline, interpretable, expected in DS roles |
| Random Forest | Strong performer, gives feature importance |
| XGBoost | Shows awareness of gradient boosting |
| Support Vector Machine (SVM) | Demonstrates breadth of ML knowledge |

**Evaluation metrics to compute for each model:**

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
```

| Metric | Why it matters for this problem |
|---|---|
| F1 Score (macro) | Handles class imbalance — more honest than accuracy |
| Confusion Matrix | Shows exactly which classes get confused |
| ROC-AUC | Overall discrimination power |
| Cross-validation (5-fold) | Proves the model generalises, not just overfit |

**Save the best model:**
```python
import joblib
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
```

**Target numbers to aim for:**
- Random Forest F1: ~82–86%
- XGBoost F1: ~84–88%
- Do NOT inflate or fabricate these — honest numbers with good explanation beat fake high scores

---

### Phase 6 — FastAPI Prediction Endpoint
**File:** `src/api.py`  
**Time:** ~3 hours (you already know FastAPI)

**Endpoints to build:**

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Check if API is running |
| `POST` | `/predict` | Accept student features, return dropout probability |
| `GET` | `/history` | Return last 50 predictions from SQLite |
| `GET` | `/stats` | Return model accuracy + feature importance |

**Prediction response format:**
```json
{
  "prediction": "Dropout",
  "confidence": 0.78,
  "probabilities": {
    "Dropout": 0.78,
    "Enrolled": 0.14,
    "Graduate": 0.08
  },
  "risk_level": "High",
  "top_risk_factors": ["Low semester grades", "No scholarship", "High debt"]
}
```

Log every prediction to the `predictions` table in SQLite for the `/history` endpoint.

---

### Phase 7 — Interactive Dashboard
**File:** `dashboard/index.html`  
**Time:** ~4–5 hours (single HTML file)

**Dashboard sections to build:**

| Section | What it shows |
|---|---|
| Summary cards | Total students, dropout rate %, model accuracy |
| Class distribution chart | Chart.js doughnut chart |
| Top 10 feature importance | Horizontal bar chart |
| Prediction form | Input student details → hit Predict → see result |
| Risk gauge | Visual indicator: Low / Medium / High risk |
| Prediction history table | Last 10 predictions from API |

Connect the prediction form to the FastAPI `/predict` endpoint using `fetch()`.

---

## 6. Deliverables Checklist

### Code Files
- [ ] `src/load_data.py` — data ingestion to SQLite
- [ ] `notebooks/eda.ipynb` — EDA with 8 visualizations
- [ ] `notebooks/statistics.ipynb` — 4 statistical tests with interpretations
- [ ] `src/preprocessing.py` — cleaning, encoding, scaling, SMOTE
- [ ] `src/train.py` — 4 models, cross-validation, evaluation, model save
- [ ] `src/api.py` — FastAPI with 4 endpoints
- [ ] `dashboard/index.html` — interactive frontend

### Model Artifacts
- [ ] `models/best_model.pkl`
- [ ] `models/scaler.pkl`
- [ ] `models/label_encoder.pkl`

### Documentation
- [ ] `README.md` with screenshots, accuracy table, and how to run
- [ ] `requirements.txt`
- [ ] At least 2 screenshots in `assets/screenshots/`

### Data
- [ ] `data/students.csv` (original)
- [ ] `data/students.db` (SQLite database)

---

## 7. Resume Outcomes

These are the concrete, quantifiable achievements you will be able to write on your resume after completing this project:

### What you will have built:
1. **End-to-end DS pipeline** — from raw CSV ingestion to live prediction API
2. **Statistical analysis** — 4 hypothesis tests with real p-values and interpretations
3. **Multi-model comparison** — 4 classifiers evaluated on 3 metrics
4. **Production deployment** — FastAPI serving predictions in real time
5. **SQL-integrated workflow** — SQLite for both training data and prediction logging
6. **Interactive dashboard** — browser-based frontend with charts and live prediction

### What a recruiter sees:
- You think end-to-end, not just model accuracy
- You understand statistics, not just sklearn
- You can communicate results (dashboard + README)
- You connect real-world context (education, social impact) to technical work

---

## 8. ATS Keyword Coverage

| Missing keyword from your resume | How this project covers it |
|---|---|
| **Pandas / EDA** | `eda.ipynb` — entire notebook built on Pandas |
| **Statistical Analysis** | `statistics.ipynb` — Chi-Square, T-test, ANOVA, Pearson |
| **Scikit-learn** | `train.py` — 4 models, cross-validation, metrics |
| **Data Visualization** | 8 Seaborn + Matplotlib plots in EDA notebook |
| **SQL** | SQLite via `pd.read_sql()` and `sqlite3` |
| **Big Data / tabular data processing** | 4,424 records with 36 features — enough to say "large tabular dataset" |
| **Feature Engineering** | SMOTE, encoding, scaling in `preprocessing.py` |
| **Classification / Predictive Modeling** | Core deliverable of the project |
| **API / Deployment** | FastAPI production endpoint |

---

## 9. Folder Structure

```
student-dropout-prediction/
│
├── data/
│   ├── students.csv              # Original dataset
│   └── students.db               # SQLite database
│
├── notebooks/
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   └── statistics.ipynb          # Statistical hypothesis tests
│
├── src/
│   ├── load_data.py              # CSV → SQLite ingestion
│   ├── preprocessing.py          # Encoding, scaling, SMOTE
│   ├── train.py                  # Model training & evaluation
│   └── api.py                    # FastAPI prediction server
│
├── models/
│   ├── best_model.pkl            # Saved best classifier
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── label_encoder.pkl         # Fitted LabelEncoder
│
├── dashboard/
│   └── index.html                # Browser dashboard
│
├── assets/
│   └── screenshots/              # README screenshots
│
├── README.md
└── requirements.txt
```

---

## 10. Resume Bullet Points (Ready to Copy)

Once built, use these bullets on your resume under a Projects section. Fill in your actual numbers:

```
Student Dropout Prediction System                                    [Month] 2026
─────────────────────────────────────────────────────────────────────────────────
• Built an end-to-end student dropout prediction pipeline on 4,424 records across
  36 features using Pandas, Scikit-learn, and SQLite for data ingestion and storage.

• Conducted exploratory data analysis with 8 Seaborn/Matplotlib visualizations and
  4 statistical hypothesis tests (Chi-Square, T-test, ANOVA, Pearson Correlation)
  to identify key dropout predictors.

• Trained and compared 4 classifiers (Logistic Regression, Random Forest, XGBoost,
  SVM); achieved XX% macro F1 score with 5-fold cross-validation on Random Forest.

• Deployed the best model as a FastAPI REST API serving real-time dropout probability
  scores with per-class confidence and risk-level classification.

• Built a browser-based analytics dashboard displaying model insights, feature
  importance, and a live student risk assessment form connected to the prediction API.
```

---

## Final Note on Narrative

When asked about this project in an interview, your answer should be:

> "I built this because I work with STEM education NGOs in rural Maharashtra — Buldhana district — and I wanted to apply data science to a problem I have direct context for. Early dropout prediction has real impact: if an institution knows a student is at high risk in semester 1, they can intervene with scholarships or counselling before it is too late. I built the full pipeline from raw data to a live API, and the dashboard was designed so a college administrator — not a data scientist — could actually use it."

That answer connects your technical work to real human impact. No other candidate applying for that role will have that story.

---

*Document prepared for: Vedant Vasant More · Data Science & AI Analytics Internship Application*
