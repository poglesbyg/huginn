"""Array geometry and direction helpers.

Conventions: x = east, y = north, z = up. Azimuth is measured counter-clockwise
from +x in degrees, elevation is degrees above the horizontal plane.
"""

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at ~20 C


def uca_with_apex(n_ring: int = 8, radius: float = 0.25, apex_height: float = 0.30) -> np.ndarray:
    """Uniform circular array plus one raised mic so elevation is observable.

    A planar ring alone cannot tell "above" from "below" the plane; the apex mic
    breaks that symmetry. Returns (M, 3) mic positions in metres.
    """
    ang = 2 * np.pi * np.arange(n_ring) / n_ring
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n_ring)], axis=1)
    return np.vstack([ring, [0.0, 0.0, apex_height]])


def unit_vector(az_deg, el_deg) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def az_el(vec) -> tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(vec, dtype=float)
    az = np.degrees(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    el = np.degrees(np.arctan2(vec[..., 2], np.hypot(vec[..., 0], vec[..., 1])))
    return az, el


def angular_separation_deg(az1, el1, az2, el2) -> np.ndarray:
    u, v = unit_vector(az1, el1), unit_vector(az2, el2)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


def wrap_deg(a):
    """Wrap an angle difference into [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0
