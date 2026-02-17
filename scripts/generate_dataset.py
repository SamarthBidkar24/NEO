
"""
Data Generation Script

CLI entry point to execute the simulation and generate the initial dataset.
"""

import sys
import os
import argparse
import pathlib
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.loader import NeoDataLoader
from src.simulation.generator import NeoSimulation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Generate NEO Observation Dataset")
    parser.add_argument("--db-path", type=str, default="data/raw/neodys.db", help="Path to NEODyS database")
    parser.add_argument("--kernels-dir", type=str, default="data/raw/kernels", help="Path to SPICE kernels directory")
    parser.add_argument("--output-file", type=str, default="data/processed/neo_observations.parquet", help="Output file path")
    parser.add_argument("--start-time", type=str, default="2022-01-01T00:00:00", help="Simulation start time (UTC)")
    parser.add_argument("--duration", type=float, default=24.0, help="Simulation duration in hours")
    parser.add_argument("--step-size", type=float, default=1.0, help="Time step in hours")
    parser.add_argument("--mag-limit", type=float, default=26.0, help="Detection magnitude limit")
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = pathlib.Path(__file__).parent.parent
    db_path = project_root / args.db_path
    kernels_dir = project_root / args.kernels_dir
    output_path = project_root / args.output_file
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        loader = NeoDataLoader(str(db_path), str(kernels_dir))
        sim = NeoSimulation(loader)
        
        df = sim.run_simulation(
            start_time_utc=args.start_time,
            duration_hours=args.duration,
            step_hours=args.step_size,
            mag_limit=args.mag_limit
        )
        
        if df.empty:
            logging.warning("Simulation returned empty results!")
        else:
            df.to_parquet(output_path)
            logging.info(f"Dataset saved to {output_path}. Shape: {df.shape}")
            
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
