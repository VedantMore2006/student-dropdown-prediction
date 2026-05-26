import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from preprocessing import load_data_from_db, fit_and_save_preprocessors, get_balanced_train_test

def train_and_evaluate_models():
    print("Starting Phase 5: Model Training & Evaluation...")
    
    # 1. Load and preprocess data
    df = load_data_from_db()
    X_encoded, X_scaled, y = fit_and_save_preprocessors(df)
    X_train, X_test, y_train, y_test = get_balanced_train_test(X_scaled, y)
    
    # 2. Define the 4 classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    results = {}
    best_f1 = -1
    best_model_name = None
    best_model = None
    
    # 3. Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Fit model on balanced training data
        model.fit(X_train, y_train)
        
        # Predict on scaled test data
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        # 5-fold cross-validation on balanced training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        mean_cv_f1 = np.mean(cv_scores)
        
        # Multi-class ROC-AUC score (ovr - One vs Rest)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
        print(f"[{name}] Test Macro F1: {macro_f1:.4f} | 5-Fold CV Macro F1: {mean_cv_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # Store results
        results[name] = {
            "model": model,
            "test_macro_f1": macro_f1,
            "cv_macro_f1": mean_cv_f1,
            "roc_auc": roc_auc,
            "classification_report": classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Check if this is the best model based on macro F1
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            best_model = model
            
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    comparison_df = pd.DataFrame({
        name: {
            "Test Macro F1": f"{res['test_macro_f1']:.4f}",
            "CV Macro F1": f"{res['cv_macro_f1']:.4f}",
            "ROC-AUC": f"{res['roc_auc']:.4f}"
        } for name, res in results.items()
    }).T
    print(comparison_df)
    print("="*50)
    
    print(f"\nBest Model Selected: {best_model_name} with F1-score: {best_f1:.4f}")
    print("\nBest Model Classification Report:")
    print(results[best_model_name]['classification_report'])
    
    # 4. Save the best model
    models_dir = "models"
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Saved best model to {best_model_path}")
    
    # Save the name of the best model and features list
    feature_importances = None
    if best_model_name == "Random Forest":
        feature_importances = best_model.feature_importances_
    elif best_model_name == "XGBoost":
        feature_importances = best_model.feature_importances_
    elif best_model_name == "Logistic Regression":
        # Use absolute coefficients for feature importance
        feature_importances = np.mean(np.abs(best_model.coef_), axis=0)
        
    if feature_importances is not None:
        final_cols = joblib.load(os.path.join(models_dir, "final_columns.pkl"))
        # Map features to importances
        fi_dict = dict(zip(final_cols, [float(x) for x in feature_importances]))
        # Sort feature importances
        sorted_fi = sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)
        joblib.dump(sorted_fi, os.path.join(models_dir, "feature_importance.pkl"))
        print("Feature importances computed and saved.")
        
    print("\nPhase 5 complete!")

if __name__ == "__main__":
    train_and_evaluate_models()
