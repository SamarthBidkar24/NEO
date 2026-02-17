
"""
Train Regressor

Trains a Random Forest Regressor to predict asteroid apparent magnitude.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.preprocessing import load_xy, FEATURE_COLUMNS

def train_regressor():
    print("Loading data for regression task...")
    try:
        X, y = load_xy(task="regression")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python scripts/generate_data.py ...' first.")
        sys.exit(1)

    print(f"Data loaded. Shape: {X.shape}")

    # Train/Test Split
    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance on Test Set:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save Model and Features
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "regressor.joblib")
    features_path = os.path.join(models_dir, "regressor_features.json")

    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)

    print(f"Saving feature list to {features_path}...")
    with open(features_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=4)

    print("Done.")

if __name__ == "__main__":
    train_regressor()
