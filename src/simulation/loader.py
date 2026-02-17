
"""
Data Loader

Handles loading of the NEODyS database and SPICE kernels required for simulation.
"""

import pathlib
import sqlite3
import pandas as pd
import spiceypy
import math
import logging

logger = logging.getLogger(__name__)

class NeoDataLoader:
    def __init__(self, db_path: str, kernels_dir: str):
        """
        Initialize the data loader.

        Parameters
        ----------
        db_path : str
            Path to the neodys.db SQLite database.
        kernels_dir : str
            Path to the directory containing SPICE kernels.
        """
        self.db_path = pathlib.Path(db_path)
        self.kernels_dir = pathlib.Path(kernels_dir)
        
    def load_spice_kernels(self):
        """
        Load standard SPICE kernels.
        """
        # Load SPICE kernels - adjust paths as needed relative to kernels_dir
        # Assuming standard naming from the notebooks: de432s.bsp, naif0012.tls, gm_de431.tpc
        # We might need to find them or use provided paths.
        
        # Common kernels usually found in such projects
        kernels_to_load = [
            "spk/de432s.bsp",
            "lsk/naif0012.tls",
            "pck/gm_de431.tpc"
        ]
        
        for k in kernels_to_load:
            k_path = self.kernels_dir / k
            if k_path.exists():
                spiceypy.furnsh(str(k_path))
                logger.info(f"Loaded kernel: {k_path}")
            else:
                logger.warning(f"Kernel not found: {k_path}")
                # Try loading generic if specific not found, or raising error
                
    def get_neo_data(self) -> pd.DataFrame:
        """
        Query NEODyS database and return dataframe with orbital elements.
        Performs initial cleaning and unit conversions (deg -> rad, AU -> km where needed).
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        con = sqlite3.connect(str(self.db_path))
        try:
            # Read all NEOs
            neo_df = pd.read_sql("SELECT * FROM main", con)
        finally:
            con.close()

        # Clean data (remove known problematic NEOs as seen in notebooks)
        neos_2_del = ["2010LF64", "2010LB67", "2010OJ129", "2010HA71"]
        neo_df = neo_df.loc[~neo_df["Name"].isin(neos_2_del)].copy()
        neo_df.reset_index(drop=True, inplace=True)

        # Vectorized Conversions
        # AU to km for Perihelion
        # Note: We need to check if 'Perihel_AU' exists or we calculate it from SemMajAxis and Ecc
        if "Perihel_AU" in neo_df.columns:
             pass 
        else:
             # Perihelion = a * (1 - e)
             neo_df["Perihel_AU"] = neo_df["SemMajAxis_AU"] * (1.0 - neo_df["Ecc_"])

        # Create Perihel_km
        # spiceypy.convrt(x, "AU", "km") is scalar, use broadcasting or apply
        # One AU in km is roughly 1.496e8. We can use spice function or constant.
        # Ideally use constant from physics module, but for dataframe usually applying map is fine
        # or just multiplying by constant for speed.
        one_au_km = spiceypy.convrt(1.0, "AU", "km")
        neo_df["Perihel_km"] = neo_df["Perihel_AU"] * one_au_km

        # Degrees to Radians
        deg_cols = ["Incl_deg", "LongAscNode_deg", "ArgP_deg", "MeanAnom_deg"]
        rad_cols = ["Incl_rad", "LongAscNode_rad", "ArgP_rad", "MeanAnom_rad"]
        
        for deg_c, rad_c in zip(deg_cols, rad_cols):
            neo_df[rad_c] = np.radians(neo_df[deg_c])

        # Epoch MJD to ET
        # MJD to JD: JD = MJD + 2400000.5
        neo_df["Epoch_JD"] = neo_df["Epoch_MJD"] + 2400000.5
        
        # Vectorizing utc2et is tricky because it takes string. 
        # But we can Convert JD to ET relative to J2000.
        # ET = (JD - 2451545.0) * 86400.0 roughly, but SPICE is more precise (TDB vs UTC).
        # For simulation speed, we might want to use a vectorized approximation or loop if fast enough.
        # Notebook used apply with utc2et.
        neo_df["Epoch_et"] = neo_df["Epoch_JD"].apply(lambda x: spiceypy.utc2et(str(x) + " JD"))

        return neo_df

    def clear_kernels(self):
        """Unload all SPICE kernels."""
        spiceypy.kclear()

import numpy as np
