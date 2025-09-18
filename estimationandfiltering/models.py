

"""
estimationandfiltering/models.py

Dynamics models for the Filtering Module.
Implements 9-state Nearly-Constant-Acceleration (NCA) model with white jerk noise.
Units: km, km/s, km/s^2.
"""

from __future__ import annotations
import numpy as np

import os
try:
    from config.loader import load_config
    _CFG = load_config()
    _JERK_PSD_CFG = float(_CFG.get("filter", {}).get("jerk_psd_km2_s5", 1e-6))
except Exception:
    _CFG = None
    _JERK_PSD_CFG = float(os.environ.get("JERK_PSD", 1e-6))

# Default process noise spectral density (jerk PSD) in km^2/s^5
# Source of truth: config/pipeline.yaml -> filter.jerk_psd_km2_s5 (fallback to env JERK_PSD, else 1e-6)
JERK_PSD = _JERK_PSD_CFG

def jerk_psd_default() -> float:
    """Return the default jerk PSD (km^2/s^5) resolved from config/env."""
    return JERK_PSD


def F_Q(dt: float, qj: float = JERK_PSD) -> tuple[np.ndarray, np.ndarray]:
    """
    Build state transition F and process noise Q for 9-state NCA model.

    State vector: x = [r(3), v(3), a(3)]
    r in km, v in km/s, a in km/s^2.

    Args:
        dt: timestep in seconds
        qj: jerk PSD, km^2/s^5

    Returns:
        F (9x9), Q (9x9)
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    F = np.block([
        [I3, dt * I3, 0.5 * dt ** 2 * I3],
        [Z3, I3, dt * I3],
        [Z3, Z3, I3]
    ])

    # Continuous-time jerk spectral density integrated over dt
    q11 = (dt ** 5) / 20.0
    q12 = (dt ** 4) / 8.0
    q13 = (dt ** 3) / 6.0
    q22 = (dt ** 3) / 3.0
    q23 = (dt ** 2) / 2.0
    q33 = dt

    Q = qj * np.block([
        [q11 * I3, q12 * I3, q13 * I3],
        [q12 * I3, q22 * I3, q23 * I3],
        [q13 * I3, q23 * I3, q33 * I3],
    ])

    return F, Q


if __name__ == "__main__":
    # quick self-test
    dt = 1.0
    F, Q = F_Q(dt)
    print("[MODELS] jerk_psd_default:", JERK_PSD)
    print("[MODELS] F shape:", F.shape, "cond(F):", np.linalg.cond(F))
    print("[MODELS] Q diag (first 3):", np.diag(Q)[:3])