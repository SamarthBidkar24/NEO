
"""
Orbital Mechanics Module

Wraps spiceypy functions to convert orbital elements to state vectors and handle coordinate transformations.
"""

import spiceypy
import numpy as np
import typing as t
from src.physics.constants import GM_SUN

def get_earth_position(et: float) -> np.ndarray:
    """
    Get the position vector of Earth relative to the Sun (J2000 ecliptic).
    
    Parameters
    ----------
    et : float
        Ephemeris time (seconds past J2000).
        
    Returns
    -------
    position : np.ndarray
        3D position vector of Earth in km.
    """
    # targ=399 (Earth), obs=10 (Sun), ref="ECLIPJ2000"
    sun2earth_position_vec, _ = spiceypy.spkgps(targ=399, et=et, ref="ECLIPJ2000", obs=10)
    return sun2earth_position_vec

def orbital_elements_to_position(
    perihel_km: float,
    eccentricity: float,
    inclination_rad: float,
    long_asc_node_rad: float,
    arg_perihelion_rad: float,
    mean_anomaly_rad: float,
    epoch_et: float,
    current_et: float,
    gm: float = GM_SUN
) -> np.ndarray:
    """
    Compute position vector from Keplerian orbital elements using SPICE.
    
    Parameters
    ----------
    perihel_km : float
        Perihelion distance in km.
    eccentricity : float
        Eccentricity.
    inclination_rad : float
        Inclination in radians.
    long_asc_node_rad : float
        Longitude of ascending node in radians.
    arg_perihelion_rad : float
        Argument of perihelion in radians.
    mean_anomaly_rad : float
        Mean anomaly in radians.
    epoch_et : float
        Epoch of elements (ET).
    current_et : float
        Time at which to compute the state (ET).
    gm : float
        Gravitational parameter of the primary (default: Sun).
        
    Returns
    -------
    position : np.ndarray
        3D position vector relative to the primary in km.
    """
    # Create elements array expected by spiceypy.conics
    # [RP, ECC, INC, LNODE, ARGP, M0, T0, MU]
    # Note: SPICE expects mean anomaly at epoch T0.
    
    elts = [
        perihel_km,
        eccentricity,
        inclination_rad,
        long_asc_node_rad,
        arg_perihelion_rad,
        mean_anomaly_rad,
        epoch_et,
        gm
    ]
    
    state_vector = spiceypy.conics(elts, current_et)
    return state_vector[:3] # Return only position (x, y, z)

def utc_to_et(utc_str: str) -> float:
    """
    Convert UTC string to Ephemeris Time (ET).
    """
    return spiceypy.utc2et(utc_str)

def et_to_utc(et: float, format_str: str = "C") -> str:
    """
    Convert Ephemeris Time to UTC string.
    """
    return spiceypy.et2utc(et, format_str, 0)
