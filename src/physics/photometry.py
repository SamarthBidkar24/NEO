
"""
Photometry and Magnitude Calculations

Implements phase functions and H-G magnitude system for calculating apparent magnitude.
Reference: https://britastro.org/asteroids/dymock4.pdf
"""

import math
import typing as t
import numpy as np

def phase_func(index: int, phase_angle: float) -> float:
    """
    Phase function that is needed for the H-G visual / apparent magnitude function.
    The function has two versions, depending on the index ('1' or '2').

    Parameters
    ----------
    index : int
        Phase function index / version. 1 or 2.
    phase_angle : float
        Phase angle of the asteroid in radians.

    Returns
    -------
    phi : float
        Phase function result.
    """
    # Dictionaries that contain the A and B constants, depending on the index version
    a_factor = {1: 3.33, 2: 1.87}
    b_factor = {1: 0.63, 2: 1.22}

    # Phase function
    phi = math.exp(
        -1.0 * a_factor[index] * ((math.tan(0.5 * phase_angle) ** b_factor[index]))
    )

    # Return the phase function result
    return phi


def reduc_mag(abs_mag: float, phase_angle: float, slope_g: float = 0.15) -> float:
    """
    Compute the reduced magnitude of an object.

    Parameters
    ----------
    abs_mag : float
        Absolute magnitude of the object.
    phase_angle : float
        Phase angle of the object w.r.t. the illumination source and observer.
    slope_g : float, optional
        Slope parameter G. Default is 0.15.

    Returns
    -------
    reduced_magnitude : float
        Reduced magnitude of the object.
    """
    # Compute the reduced magnitude based on the equations given in the references
    reduced_magnitude = abs_mag - 2.5 * math.log10(
        (1.0 - slope_g) * phase_func(index=1, phase_angle=phase_angle)
        + slope_g * phase_func(index=2, phase_angle=phase_angle)
    )

    return reduced_magnitude


def hg_app_mag(
    abs_mag: float,
    vec_obj2obs: t.Union[t.List[float], t.Tuple[float, float, float], np.ndarray],
    vec_obj2ill: t.Union[t.List[float], t.Tuple[float, float, float], np.ndarray],
    slope_g: float = 0.15,
) -> float:
    """
    Compute the visual / apparent magnitude of an asteroid.
    Based on the H-G system.

    Parameters
    ----------
    abs_mag : float
        Absolute magnitude.
    vec_obj2obs : array-like
        3D vector from asteroid to observer (AU).
    vec_obj2ill : array-like
        3D vector from asteroid to illumination source (AU).
    slope_g : float, optional
        Slope parameter G. Default is 0.15.

    Returns
    -------
    app_mag : float
        Apparent magnitude.
    """
    # Ensure inputs are list-like or array-like
    if not isinstance(vec_obj2obs, (list, tuple, np.ndarray)):
         vec_obj2obs = list(vec_obj2obs)
    if not isinstance(vec_obj2ill, (list, tuple, np.ndarray)):
         vec_obj2ill = list(vec_obj2ill)

    # Compute the length of the two input vectors
    vec_obj2obs_norm = np.linalg.norm(vec_obj2obs)
    vec_obj2ill_norm = np.linalg.norm(vec_obj2ill)

    # Compute the phase angle of the asteroid
    # Dot product
    dotp_res = np.dot(vec_obj2obs, vec_obj2ill)
    
    # Clip value for acos stability
    cosine_angle = dotp_res / (vec_obj2obs_norm * vec_obj2ill_norm)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    obj_phase_angle = math.acos(cosine_angle)

    # Compute the reduced magnitude of the asteroid
    red_mag = reduc_mag(abs_mag, obj_phase_angle, slope_g)

    # Merge all information and compute the apparent magnitude
    app_mag = red_mag + 5.0 * math.log10(vec_obj2obs_norm * vec_obj2ill_norm)

    return app_mag
