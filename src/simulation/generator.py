
"""
Simulation Generator

Core simulation loop that generates synthetic observations of NEOs over time.
Checks visibility criteria and creates the dataset for ML training.
"""

import pandas as pd
import numpy as np
import spiceypy
import logging
from tqdm import tqdm
from src.physics import photometry, orbit, constants
from src.simulation.loader import NeoDataLoader
import pathlib

logger = logging.getLogger(__name__)

# Default paths - assumed to be relative to project root or absolute
# In a real app these might come from config
DEFAULT_DB_PATH = "data/raw/neodys.db"
DEFAULT_KERNELS_DIR = "data/raw/kernels"

def generate_observations(
    num_neos: int, 
    start_date: str, 
    end_date: str, 
    step_hours: int, 
    limiting_mag: float, 
    exposure_s: float, 
    fov_deg: float
) -> pd.DataFrame:
    """
    Returns a DataFrame where each row is one NEO x one observation time.
    
    Columns include:
    neo_id, a, e, i, H,
    distance_au, phase_angle_deg, ra_deg, dec_deg,
    limiting_mag, exposure_s, fov_deg,
    apparent_magnitude, is_detected (0/1)
    """
    
    # 1. Setup Loader
    # Find project root (3 levels up from this file: src/simulation/generator.py -> src/simulation -> src -> root)
    root_dir = pathlib.Path(__file__).parent.parent.parent
    db_path = root_dir / DEFAULT_DB_PATH
    kernels_dir = root_dir / DEFAULT_KERNELS_DIR
    
    loader = NeoDataLoader(str(db_path), str(kernels_dir))
    loader.load_spice_kernels()
    neo_df = loader.get_neo_data()
    
    # 2. Sample NEOs
    if len(neo_df) > num_neos:
        neo_df = neo_df.sample(n=num_neos, random_state=42)
    else:
        logger.warning(f"Requested {num_neos} NEOs, but only {len(neo_df)} available.")
    
    logger.info(f"Simulating {len(neo_df)} NEOs form {start_date} to {end_date}...")

    # 3. Time Steps
    start_et = spiceypy.utc2et(start_date)
    end_et = spiceypy.utc2et(end_date)
    step_seconds = step_hours * 3600.0
    
    if end_et <= start_et:
        raise ValueError("End date must be after start date.")
        
    times_et = np.arange(start_et, end_et + step_seconds, step_seconds) # Include end date if step aligns? usually strictly less if arange. +epslion ensures coverage.
    
    observations = []
    
    # Pre-calculate Earth positions for all times to save SPICE calls
    earth_positions = {} # et -> vec
    for et in times_et:
        earth_positions[et] = orbit.get_earth_position(et)

    # 4. Simulation Loop
    records = neo_df.to_dict("records")
    
    for row in tqdm(records):
        for et in times_et:
            # Orbital Elements
            # Using our physics wrapper
            # orbit.orbital_elements_to_position expects elements.
            # But get_neo_data returns columns: Perihel_km, Ecc_, Incl_rad, LongAscNode_rad, ArgP_rad, MeanAnom_rad, Epoch_et
            
            try:
                # NEO Position (Sun-centric)
                sun2neo = orbit.orbital_elements_to_position(
                    perihel_km=row["Perihel_km"],
                    eccentricity=row["Ecc_"],
                    inclination_rad=row["Incl_rad"],
                    long_asc_node_rad=row["LongAscNode_rad"],
                    arg_perihelion_rad=row["ArgP_rad"],
                    mean_anomaly_rad=row["MeanAnom_rad"],
                    epoch_et=row["Epoch_et"],
                    current_et=et,
                    gm=constants.GM_SUN
                )
            except Exception:
                continue # Skip if propagation fails

            # Earth Position
            sun2earth = earth_positions[et]
            
            # Relative Vectors
            neo2earth = sun2earth - sun2neo
            neo2sun = -sun2neo
            earth2neo = -neo2earth
            
            # Distances / Conversions
            dist_earth_km = np.linalg.norm(neo2earth)
            dist_earth_au = dist_earth_km / constants.ONE_AU
            
            # Vectors in AU
            neo2earth_au = neo2earth / constants.ONE_AU
            neo2sun_au = neo2sun / constants.ONE_AU
            earth2neo_au = earth2neo / constants.ONE_AU
            
            # Apparent Magnitude
            app_mag = photometry.hg_app_mag(
                abs_mag=row["AbsMag_"],
                vec_obj2obs=neo2earth_au,
                vec_obj2ill=neo2sun_au,
                slope_g=row["SlopeParamG_"]
            )
            
            # Phase Angle (approx) - reused implicit calculation inside hg_app_mag but requested in output
            # Recalculate explicitly for output
            # Phase angle is angle at NEO between Earth and Sun
            # Vectors from NEO: neo2earth and neo2sun
            # Dot product
            v1 = neo2earth
            v2 = neo2sun
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            phase_rad = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
            phase_deg = np.degrees(phase_rad)
            
            # RA / Dec
            _, ra_rad, dec_rad = spiceypy.recrad(earth2neo) # Earth to NEO vector
            ra_deg = np.degrees(ra_rad)
            dec_deg = np.degrees(dec_rad)
            if ra_deg < 0: ra_deg += 360.0
            
            # Detection Logic
            # 1. Magnitude check
            # 2. FOV check (Opposition angle used in notebook)
            # Opposition vector = Sun -> Earth
            earth2sun = -sun2earth
            # Angle between Earth->NEO and Sun->Earth (Opposition direction)
            opp_angle_rad = spiceypy.vsep(earth2neo, sun2earth) 
            # Note: Notebook used sun2earth as opposition direction?
            # Opposition is Away from sun. From Earth, looking away from Sun.
            # Vector: Earth->Sun is -Sun->Earth.
            # Opposition direction is Sun->Earth (S->E). 
            # So angle between (E->N) and (S->E).
            opp_angle_deg = np.degrees(opp_angle_rad)
            
            # Notebook logic: (app_mag <= mag_detec) & (ang_dist_neo2opp_deg <= opp_range)
            # Here fov_deg is passed. Assuming fov_deg acts as the opposition range constraint 
            # (i.e. we are observing in the opposition cone).
            
            is_visible_mag = app_mag <= limiting_mag
            is_in_fov = opp_angle_deg <= (fov_deg / 2.0) # FOV usually diameter, opposition range usually radius from center?
            # Notebook said "opp_range = 15.0". AND "(ang_dist_neo2opp_deg <= opp_range)".
            # If fov_deg is 2.0 (from prompt), it's very narrow.
            # Assuming fov_deg represents the cone size we searched.
            # If we assume we surveyed the *entire* fov_deg region centered at opposition.
            
            # Strict interpretation: The user scans a region of size `fov_deg`.
            # If we assume the survey point IS opposition, then `opp_angle_deg <= fov_deg/2`.
            # Let's verify prompt: "fov_deg 2.0".
            # Let's use `opp_angle_deg <= fov_deg`. (Notebook style: range is the radius/threshold).
            
            is_detected = 1 if (is_visible_mag and opp_angle_deg <= fov_deg) else 0
            
            # Collect Data
            observations.append({
                "neo_id": row["Name"],
                "a": row["SemMajAxis_AU"],
                "e": row["Ecc_"],
                "i": row["Incl_deg"],
                "H": row["AbsMag_"],
                "distance_au": dist_earth_au,
                "phase_angle_deg": phase_deg,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "limiting_mag": limiting_mag,
                "exposure_s": exposure_s,
                "fov_deg": fov_deg,
                "apparent_magnitude": app_mag,
                "is_detected": is_detected,
                # Metadata
                "timestamp_utc": spiceypy.et2utc(et, "ISOC", 0)
            })

    df = pd.DataFrame(observations)
    return df
