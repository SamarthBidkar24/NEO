
"""
Train Models Script

CLI entry point to train ML models.
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ml.train import train_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train NEO ML Models")
    parser.add_argument("--data-path", type=str, default="data/processed/neo_observations.parquet", help="Path to training data")
    parser.add_argument("--models-dir", type=str, default="models/", help="Directory to save models")
    
    args = parser.parse_args()
    
    try:
        train_pipeline(args.data_path, args.models_dir)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
