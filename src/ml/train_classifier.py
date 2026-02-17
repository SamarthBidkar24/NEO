
"""
Train Classifier

Trains a Random Forest Classifier to predict if an asteroid is detected.
"""

import sys
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.preprocessing import load_xy, FEATURE_COLUMNS

def train_classifier():
    print("Loading data for classification task...")
    try:
        X, y = load_xy(task="classification")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python scripts/generate_data.py ...' first.")
        sys.exit(1)

    print(f"Data loaded. Shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Train/Test Split (Stratified)
    print("Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Model
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = 0.0 # Handle case with only one class in test set
    
    print(f"\nModel Performance on Test Set:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")

    # Save Model and Features
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "classifier.joblib")
    features_path = os.path.join(models_dir, "classifier_features.json")

    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)

    print(f"Saving feature list to {features_path}...")
    with open(features_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=4)

    print("Done.")

if __name__ == "__main__":
    train_classifier()
