
"""
Create Mock NEODyS Database

Generates a dummy SQLite database with random NEO orbital elements
because the official NEODyS server is blocking automated downloads.
"""

import sqlite3
import numpy as np
import pathlib
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = pathlib.Path("data/raw/neodys.db")

def create_mock_db(num_neos=1000):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if DB_PATH.exists():
        logging.info(f"Database {DB_PATH} already exists. Skipping mock creation.")
        return

    logging.info(f"Creating mock database at {DB_PATH} with {num_neos} objects...")
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Create Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS main(
            Name TEXT PRIMARY KEY,
            Epoch_MJD FLOAT,
            SemMajAxis_AU FLOAT,
            Ecc_ FLOAT,
            Incl_deg FLOAT,
            LongAscNode_deg FLOAT,
            ArgP_deg FLOAT,
            MeanAnom_deg FLOAT,
            AbsMag_ FLOAT,
            SlopeParamG_ FLOAT,
            Aphel_AU FLOAT,
            Perihel_AU FLOAT
        )
    """)
    con.commit()
    
    # Generate Random Data
    np.random.seed(42)
    
    names = [f"MockNEO_{i:04d}" for i in range(num_neos)]
    mjds = np.full(num_neos, 59600.0) # Approx 2022
    a = np.random.uniform(0.5, 3.5, num_neos) # Semi-major axis (AU)
    e = np.random.uniform(0.0, 0.9, num_neos) # Eccentricity
    i = np.random.uniform(0.0, 45.0, num_neos) # Inclination (deg)
    om = np.random.uniform(0.0, 360.0, num_neos) # Long Asc Node
    w = np.random.uniform(0.0, 360.0, num_neos) # Arg Perihelion
    ma = np.random.uniform(0.0, 360.0, num_neos) # Mean Anomaly
    h = np.random.uniform(15.0, 28.0, num_neos) # Absolute Magnitude
    g = np.full(num_neos, 0.15) # Slope parameter
    
    aphel = a * (1.0 + e)
    perihel = a * (1.0 - e)
    
    data = []
    for idx in range(num_neos):
        data.append((
            names[idx], float(mjds[idx]), float(a[idx]), float(e[idx]), float(i[idx]),
            float(om[idx]), float(w[idx]), float(ma[idx]), float(h[idx]), float(g[idx]),
            float(aphel[idx]), float(perihel[idx])
        ))
        
    cur.executemany("""
        INSERT INTO main VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    
    con.commit()
    con.close()
    logging.info("Mock database created successfully.")

if __name__ == "__main__":
    create_mock_db()
