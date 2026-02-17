
"""
Model Training

Pipeline to train models, evaluate performance metrics, and save artifacts.
"""

import pandas as pd
import pathlib
import pickle
import logging
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from src.ml.preprocessing import NeoPreprocessor
from src.ml.models import NeoClassifier, NeoRegressor

logger = logging.getLogger(__name__)

def train_pipeline(data_path: str, models_dir: str):
    """
    Train both classifier and regressor.
    """
    logging.info("Starting Training Pipeline...")
    
    # 1. Load & Preprocess
    preprocessor = NeoPreprocessor()
    df = preprocessor.load_data(data_path)
    X, y_cls, y_reg = preprocessor.preprocess(df, is_training=True)
    
    # Save Scaler
    models_path = pathlib.Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    preprocessor.save_scaler(str(models_path / "scaler.pkl"))
    
    # Split done internally by user script or here? 
    # Let's do a simple split here for valid
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42
    )
    
    # 2. Train Classifier
    logger.info("Training Classifier...")
    clf = NeoClassifier()
    clf.fit(X_train, y_cls_train)
    
    # Eval Classifier
    y_pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_cls_test, y_pred_cls)
    f1 = f1_score(y_cls_test, y_pred_cls)
    logger.info(f"Classifier Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Save Classifier
    with open(models_path / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    # 3. Train Regressor
    logger.info("Training Regressor...")
    reg = NeoRegressor()
    reg.fit(X_train, y_reg_train)
    
    # Eval Regressor
    y_pred_reg = reg.predict(X_test)
    rmse = mean_squared_error(y_reg_test, y_pred_reg, squared=False)
    r2 = r2_score(y_reg_test, y_pred_reg)
    logger.info(f"Regressor Metrics -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Save Regressor
    with open(models_path / "regressor.pkl", "wb") as f:
        pickle.dump(reg, f)
        
    logger.info("Training Complete.")
