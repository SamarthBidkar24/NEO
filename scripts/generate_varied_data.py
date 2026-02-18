
import sys
import os
import pandas as pd
import logging
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.generator import generate_observations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Scenarios to ensure coverage of all dashboard buckets
    # Exposure: Short (<60), Long (>=60)
    # FOV: Narrow (<2.0), Wide (>=2.0)
    
    scenarios = [
        {"name": "Short_Narrow", "exposure_s": 30.0, "fov_deg": 1.0},
        {"name": "Short_Wide",   "exposure_s": 30.0, "fov_deg": 5.0},
        {"name": "Long_Narrow",  "exposure_s": 300.0, "fov_deg": 1.0},
        {"name": "Long_Wide",    "exposure_s": 300.0, "fov_deg": 5.0},
    ]
    
    # Common settings
    NEOS_PER_SCENARIO = 200
    START_DATE = "2026-01-01"
    END_DATE = "2026-01-08" # 1 week
    STEP_HOURS = 24
    
    # Baseline for Limiting Magnitude Calculation
    # Assume 30s exposure yields approx Mag 22.0 limit
    BASELINE_EXP = 30.0
    BASELINE_MAG = 22.0
    
    dfs = []
    
    print(f"Generating varied dataset with {len(scenarios)} scenarios...")
    
    for sc in scenarios:
        # Calculate Limiting Mag dynamically
        # Limiting Mag increases by approx 1.25 * log10(t/t0) (simple scaling for SNR)
        # Or more accurately 2.5 * log10(sqrt(t/t0)) = 1.25 log10
        import numpy as np
        current_limit = BASELINE_MAG + 1.25 * np.log10(sc["exposure_s"] / BASELINE_EXP)
        
        print(f"Running scenario: {sc['name']} (Exp={sc['exposure_s']}s -> LimMag ~{current_limit:.2f}, FOV={sc['fov_deg']}deg)")
        try:
            df = generate_observations(
                num_neos=NEOS_PER_SCENARIO,
                start_date=START_DATE,
                end_date=END_DATE,
                step_hours=STEP_HOURS,
                limiting_mag=current_limit,
                exposure_s=sc["exposure_s"],
                fov_deg=sc["fov_deg"]
            )
            df["scenario"] = sc["name"]
            dfs.append(df)
        except Exception as e:
            print(f"Failed scenario {sc['name']}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        
        output_path = "data/simulated_observations.csv"
        full_df.to_csv(output_path, index=False)
        print(f"Successfully saved combined dataset with {len(full_df)} observations to {output_path}")
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()
