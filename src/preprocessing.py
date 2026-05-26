import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def load_data_from_db(db_path="data/students.db"):
    """Load the students table from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

def fit_and_save_preprocessors(df, models_dir="models"):
    """
    Fits and saves the LabelEncoder, StandardScaler, and categorical column lists.
    This guarantees consistency between training and live FastAPI inferences.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 1. Fit Target Label Encoder
    le = LabelEncoder()
    # Fit target: Dropout=0, Enrolled=1, Graduate=2
    # Standard classes order: ['Dropout', 'Enrolled', 'Graduate']
    # Let's ensure strict mapping:
    df['Target_encoded'] = le.fit_transform(df['Target'])
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    print("LabelEncoder fitted and saved.")
    print(f"Target classes mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop(columns=['Target', 'Target_encoded'])
    y = df['Target_encoded']

    # 2. Handle Categorical One-Hot Encoding
    # We will one-hot encode: 'Marital status', 'Application mode', 'Nacionality'
    categorical_cols = ['Marital status', 'Application mode', 'Nacionality']
    
    # We need to make sure continuous columns are scaled, and one-hot encoding columns are saved
    # First, let's keep track of original feature columns to reconstruct later in API
    original_cols = list(X.columns)
    joblib.dump(original_cols, os.path.join(models_dir, "original_columns.pkl"))

    # Convert specified categoricals to dummies
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Save the exact final column list (post dummy variable expansion) for alignment during prediction
    final_cols = list(X_encoded.columns)
    joblib.dump(final_cols, os.path.join(models_dir, "final_columns.pkl"))

    # 3. Fit StandardScaler on all features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=final_cols)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("StandardScaler fitted and saved.")

    return X_encoded, X_scaled, y

def get_balanced_train_test(X_scaled, y, test_size=0.2, random_state=42):
    """
    Splits data into stratified train/test sets and applies SMOTE
    on the training set to address class imbalances.
    """
    # Stratified split to preserve class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Original train size: {X_train.shape}, test size: {X_test.shape}")
    print(f"Original training target distribution:\n{y_train.value_counts()}")

    # Apply SMOTE only on training data to prevent data leakage
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Balanced training size after SMOTE: {X_train_bal.shape}")
    print(f"Balanced training target distribution:\n{y_train_bal.value_counts()}")

    return X_train_bal, X_test, y_train_bal, y_test

def preprocess_single_input(input_dict, models_dir="models"):
    """
    Preprocess a single incoming raw feature dictionary for live API inference.
    Aligned 100% with the fit transformations.
    """
    # Load serialised assets
    original_cols = joblib.load(os.path.join(models_dir, "original_columns.pkl"))
    final_cols = joblib.load(os.path.join(models_dir, "final_columns.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    # Convert to DataFrame
    df_single = pd.DataFrame([input_dict])

    # Ensure all original columns are present
    for col in original_cols:
        if col not in df_single.columns:
            df_single[col] = 0

    # Match original column order
    df_single = df_single[original_cols]

    # Convert dummy columns
    categorical_cols = ['Marital status', 'Application mode', 'Nacionality']
    df_encoded = pd.get_dummies(df_single, columns=categorical_cols)

    # Reconstruct all dummy columns matching the training layout
    for col in final_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure final columns are in the exact same order
    df_encoded = df_encoded[final_cols]

    # Standard scale
    scaled_values = scaler.transform(df_encoded)
    
    return scaled_values

if __name__ == "__main__":
    df = load_data_from_db()
    X_encoded, X_scaled, y = fit_and_save_preprocessors(df)
    X_train_bal, X_test, y_train_bal, y_test = get_balanced_train_test(X_scaled, y)
    print("Preprocessing verification completed successfully!")
