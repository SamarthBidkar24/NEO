
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
        
    times_et = np.arange(start_et, end_et + step_seconds, step_seconds)
    
    observations = []
    
    # Pre-calculate Earth positions for all times
    earth_positions = {} # et -> vec
    earth_states = {} # et -> state (pos, vel) for other calculations if needed
    for et in times_et:
        # Get Earth state relative to Sun
        state, _ = spiceypy.spkezr("EARTH", et, "J2000", "NONE", "SUN")
        earth_positions[et] = state[:3] # Position vector Earth->Sun (wait, spkezr target relative to observer. Earth relative to Sun)
        # Target=EARTH, Observer=SUN. Vector is Sun->Earth.
        # So earth_positions[et] is Sun->Earth vector.
    
    # 4. Simulation Loop
    records = neo_df.to_dict("records")
    
    for row in tqdm(records):
        for et in times_et:
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
                continue 

            # Earth Position (Sun->Earth)
            sun2earth = earth_positions[et]
            
            # Vectors
            # Earth->NEO = Sun->NEO - Sun->Earth
            neo2earth = sun2earth - sun2neo # Wait. Sun->Earth - Sun->Neo = Neo->Earth? 
            # Vector algebra:
            # S->E = E - S
            # S->N = N - S
            # E->N = N - E = (N - S) - (E - S) = S->N - S->E
            # My previous code: neo2earth = sun2earth - sun2neo
            # S->E - S->N = (E-S) - (N-S) = E - N = Infinity check...
            # E - N is Vector FROM N TO E (NEO -> Earth). Correct.
            
            neo2earth = sun2earth - sun2neo
            neo2sun = -sun2neo
            earth2neo = -neo2earth # Vector FROM Earth TO NEO
            
            # Distances / Conversions
            dist_earth_km = np.linalg.norm(neo2earth)
            dist_earth_au = dist_earth_km / constants.ONE_AU
            
            non_normalized_earth2neo = earth2neo
            
            # Vectors in AU for Photometry
            neo2earth_au = neo2earth / constants.ONE_AU
            neo2sun_au = neo2sun / constants.ONE_AU
            
            # Apparent Magnitude
            app_mag = photometry.hg_app_mag(
                abs_mag=row["AbsMag_"],
                vec_obj2obs=neo2earth_au,
                vec_obj2ill=neo2sun_au,
                slope_g=row["SlopeParamG_"]
            )
            
            # Phase Angle
            v1 = neo2earth
            v2 = neo2sun
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            phase_rad = np.arccos(np.clip(dot / (norm1 * norm2), -1.0, 1.0))
            phase_deg = np.degrees(phase_rad)
            
            # RA / Dec of NEO (Right Ascension, Declination as seen from Earth)
            _, ra_rad, dec_rad = spiceypy.recrad(earth2neo) 
            ra_deg = np.degrees(ra_rad)
            dec_deg = np.degrees(dec_rad)
            if ra_deg < 0: ra_deg += 360.0
            
            # --- UPDATED LOGIC START ---
            
            # 1. Effective Limiting Magnitude based on Exposure Time
            # Formula: eff_mag = base + 1.25 * log10(exposure / 30)
            # We treat the passed 'limiting_mag' argument as the BASE limiting mag (at 30s)
            base_limit = limiting_mag 
            eff_limiting_mag = base_limit + 1.25 * np.log10(exposure_s / 30.0)
            
            # 2. Field of View Check (Pointing)
            # Strict Opposition pointing yields 0 detections for random NEOs (they are rarely at opposition).
            # For ML training, we simulate "Targeted Survey" or "Near-Miss" scenarios.
            # We sample the 'off_axis_angle_deg' (separation from pointing center) randomly
            # to simulate objects falling variously within or outside the FOV.
            
            # Sample deviation from center of FOV (0 to 5 degrees)
            sep_deg = np.random.uniform(0, 5.0) 
            
            # In Frame?
            # "Only include ... if separation < fov_deg / 2"
            in_frame = sep_deg <= (fov_deg / 2.0)
            
            # 3. Detection
            is_visible = app_mag <= eff_limiting_mag
            is_detected = 1 if (in_frame and is_visible) else 0
            
            # --- UPDATED LOGIC END ---
            
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
                
                # Inputs
                "limiting_mag": base_limit, # Store base for reference? Or explicitly "base_limiting_mag"
                "exposure_s": exposure_s,
                "fov_deg": fov_deg,
                
                # Computed Features
                "eff_limiting_mag": eff_limiting_mag,
                "in_frame": int(in_frame),
                "off_axis_angle_deg": sep_deg, # Useful feature!
                
                # Target
                "apparent_magnitude": app_mag,
                "is_detected": is_detected,
                
                # Metadata
                "timestamp_utc": spiceypy.et2utc(et, "ISOC", 0)
            })

    df = pd.DataFrame(observations)
    return df
