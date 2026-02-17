
"""
Generate Data CLI

Runs the NEO detection simulation and saves the results to a CSV file.
"""

import sys
import os
import argparse
import pathlib
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.generator import generate_observations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Generate Simulated NEO Observations")
    parser.add_argument("--neos", type=int, required=True, help="Number of NEOs to sample")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--step-hours", type=int, required=True, help="Time step in hours")
    parser.add_argument("--limiting-mag", type=float, required=True, help="Limiting Magnitude")
    parser.add_argument("--exposure-s", type=float, required=True, help="Exposure time in seconds")
    parser.add_argument("--fov-deg", type=float, required=True, help="Field of View in degrees (Opposition Cone)")
    
    args = parser.parse_args()
    
    # Run Simulation
    try:
        df = generate_observations(
            num_neos=args.neos,
            start_date=args.start,
            end_date=args.end,
            step_hours=args.step_hours,
            limiting_mag=args.limiting_mag,
            exposure_s=args.exposure_s,
            fov_deg=args.fov_deg
        )
        
        # Save to CSV
        output_path = pathlib.Path("data/simulated_observations.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully generated {len(df)} observations. Saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error generating data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
