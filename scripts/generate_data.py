
"""
Generate Data CLI

Runs the NEO detection simulation and saves the results to a CSV file.
"""

import sys
import os
import argparse
import pathlib
import logging
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.generator import generate_observations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Generate Varied Simulated NEO Observations")
    parser.add_argument("--neos", type=int, default=500, help="Number of NEOs to sample")
    parser.add_argument("--start", type=str, default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-07-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--step-hours", type=int, default=24, help="Time step in hours")
    parser.add_argument("--base-mag", type=float, default=22.0, help="Base Limiting Magnitude (at 30s)")
    
    args = parser.parse_args()
    
    # Define varied scenarios to cover the parameter space for ML training
    # We want the model to learn the effect of Exposure and FOV.
    # So we simulate the SAME objects under DIFFERENT conditions?
    # Or different objects?
    # Ideally, for "What-If" analysis, we want to know: "Object A, if obs with Exp=30 -> ?" vs "Object A, if obs with Exp=300 -> ?"
    # So we should simulate varied conditions for the SAME set of objects (or overlapping sets).
    
    scenarios = [
        (30.0, 1.0),
        (30.0, 5.0),
        (60.0, 2.0),
        (120.0, 2.0),
        (300.0, 1.0),
        (300.0, 5.0),
        (300.0, 10.0), # Wide survey
        (10.0, 2.0),   # Very short
    ]
    
    dfs = []
    logging.info(f"Generating data for {args.neos} NEOs across {len(scenarios)} scenarios...")
    
    import traceback
    
    for exp, fov in scenarios:
        logging.info(f"Scenario: Exposure={exp}s, FOV={fov}deg")
        try:
            df = generate_observations(
                num_neos=args.neos,
                start_date=args.start,
                end_date=args.end,
                step_hours=args.step_hours,
                limiting_mag=args.base_mag, # Passed as base
                exposure_s=exp,
                fov_deg=fov
            )
            df["scenario_id"] = f"exp_{exp}_fov_{fov}"
            dfs.append(df)
        except Exception:
            traceback.print_exc()
            continue

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        
        output_path = pathlib.Path("data/simulated_observations.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_df.to_csv(output_path, index=False)
        logging.info(f"Successfully generated {len(full_df)} observations. Saved to {output_path}")
    else:
        logging.error("No data generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
