
"""
Data Preprocessing

Functions for loading and preparing NEO data for machine learning tasks.
"""

import pandas as pd
import pathlib

# constant list of feature columns
FEATURE_COLUMNS = [
    "a", 
    "e", 
    "i", 
    "H", 
    "distance_au", 
    "phase_angle_deg", 
    "ra_deg", 
    "dec_deg", 
    "limiting_mag", 
    "exposure_s", 
    "fov_deg",
    "eff_limiting_mag",
    "in_frame"
]

def load_xy(task: str, csv_path: str = "data/simulated_observations.csv"):
    """
    Load data and split into features and target.

    Parameters
    ----------
    task : str
        "regression" or "classification".
    csv_path : str, optional
        Path to the CSV file. Defaults to "data/simulated_observations.csv".

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
        regression -> apparent_magnitude
        classification -> is_detected

    Raises
    ------
    ValueError
        If task is not 'regression' or 'classification'.
        If csv_path does not exist.
    """
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)

    # Validate task
    if task == "regression":
        target_col = "apparent_magnitude"
    elif task == "classification":
        target_col = "is_detected"
    else:
        raise ValueError(f"Unknown task '{task}'. Must be 'regression' or 'classification'.")

    # Combine features and target to drop NaNs together
    cols_to_use = FEATURE_COLUMNS + [target_col]
    
    # Check if columns exist
    missing = [c for c in cols_to_use if c not in df.columns]
    if missing:
        raise KeyError(f"The following columns are missing from the CSV: {missing}")

    # Drop rows with NaNs in relevant columns
    df_clean = df.dropna(subset=cols_to_use).copy()

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[target_col]

    return X, y
