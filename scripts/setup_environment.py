
"""
Setup Environment

Downloads necessary data (NEODyS database and SPICE kernels) to run the project.
"""

import os
import pathlib
import sqlite3
import urllib.request
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_RAW = pathlib.Path("data/raw")
KERNELS_DIR = DATA_RAW / "kernels"
DB_PATH = DATA_RAW / "neodys.db"

# URLs
NEODYS_URL = "https://newton.spacedys.com/~neodys2/neodys.cat"
KERNELS_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
KERNELS_URLS = {
    "spk/de432s.bsp": f"{KERNELS_BASE}/spk/planets/de432s.bsp",
    "lsk/naif0012.tls": f"{KERNELS_BASE}/lsk/naif0012.tls",
    "pck/gm_de431.tpc": f"{KERNELS_BASE}/pck/gm_de431.tpc"
}

def download_file(url, dest):
    dest = pathlib.Path(dest) # Ensure Path object
    if dest.exists():
        logging.info(f"File already exists: {dest}")
        return
    
    logging.info(f"Downloading {url} to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    )
    
    try:
        with urllib.request.urlopen(req, context=ctx) as u, open(dest, 'wb') as f:
            f.write(u.read())
        logging.info("Download complete.")
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        raise

def setup_neodys_db():
    if DB_PATH.exists():
        logging.info(f"Database already exists: {DB_PATH}")
        return

    cat_path = DATA_RAW / "neodys.cat"
    download_file(NEODYS_URL, cat_path)
    
    logging.info("Building SQLite database...")
    neo_dict = []
    
    with open(cat_path, "r", encoding="utf-8", errors="replace") as f:
        # Skip header (first 6 lines)
        lines = f.readlines()[6:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 10:
                neo_dict.append({
                    "Name": parts[0].replace("'", ""),
                    "Epoch_MJD": float(parts[1]),
                    "SemMajAxis_AU": float(parts[2]),
                    "Ecc_": float(parts[3]),
                    "Incl_deg": float(parts[4]),
                    "LongAscNode_deg": float(parts[5]),
                    "ArgP_deg": float(parts[6]),
                    "MeanAnom_deg": float(parts[7]),
                    "AbsMag_": float(parts[8]),
                    "SlopeParamG_": float(parts[9])
                })
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
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
    
    # Batch insert
    logging.info(f"Inserting {len(neo_dict)} NEOs...")
    cur.executemany("""
        INSERT OR IGNORE INTO main (
            Name, Epoch_MJD, SemMajAxis_AU, Ecc_, Incl_deg, 
            LongAscNode_deg, ArgP_deg, MeanAnom_deg, AbsMag_, SlopeParamG_
        ) VALUES (
            :Name, :Epoch_MJD, :SemMajAxis_AU, :Ecc_, :Incl_deg, 
            :LongAscNode_deg, :ArgP_deg, :MeanAnom_deg, :AbsMag_, :SlopeParamG_
        )
    """, neo_dict)
    
    # Update Perihel/Aphel
    logging.info("Updating derived parameters...")
    # SQL based update or Python? Python loop is safer/easier if SQL support varies.
    # Note: sqlite supports math? Not mostly.
    # Let's fetch, calc, update.
    
    cur.execute("SELECT Name, SemMajAxis_AU, Ecc_ FROM main")
    rows = cur.fetchall()
    updates = []
    for row in rows:
        name, a, e = row
        aphel = a * (1.0 + e)
        perihel = a * (1.0 - e)
        updates.append((aphel, perihel, name))
        
    cur.executemany("UPDATE main SET Aphel_AU=?, Perihel_AU=? WHERE Name=?", updates)
    
    con.commit()
    con.close()
    
    # Cleanup
    # cat_path.unlink() # Keep it or delete? Keep for cache.
    logging.info("Database setup complete.")

def setup_kernels():
    for rel_path, url in KERNELS_URLS.items():
        dest = KERNELS_DIR / rel_path
        download_file(url, dest)

def main():
    try:
        setup_kernels()
        setup_neodys_db()
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
