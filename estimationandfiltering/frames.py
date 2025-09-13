"""
frames.py — minimal ICRF → ECEF utilities (km, km/s, km/s^2)

Assumptions:
- Input state is expressed in ICRF (J2000 inertial) at UTC timestamps.
- We approximate UT1≈UTC for Earth rotation angle.
- ECEF is modeled as a simple Earth-fixed frame rotating about z with constant ω.

Formulas:
    r_e = R * r_i
    v_e = R * ( v_i - ω × r_i )
    a_e = R * ( a_i - 2 ω × v_i - ω × ( ω × r_i ) )
where ω = [0, 0, ω_EARTH] and R = Rz(θ) with θ = θ_GAST ≈ θ_GMST(UTC).

For high-precision needs (IAU2006/2000A, nutation/polar motion), this should be replaced
by a full IERS chain. For our WSS-facing export, this is sufficient as a first cut
and consistent across all modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OMEGA_EARTH = 7.2921150e-5  # rad/s

# GMST from UTC (approx), Vallado-like simple polynomial around J2000
# Reference: D.A. Vallado, Fundamentals of Astrodynamics and Applications, 4th ed.
# This is a compact approximation adequate for our purposes here.
def _gmst_angle_rad(ts_utc: pd.Series | np.ndarray) -> np.ndarray:
    # Convert to pandas datetime, ensuring we get a Series or DatetimeIndex
    if isinstance(ts_utc, (pd.Series, pd.DatetimeIndex)):
        dt = pd.to_datetime(ts_utc, utc=True)
    else:
        # Convert array-like to Series first
        dt = pd.Series(pd.to_datetime(ts_utc, utc=True))

    # Handle timezone conversion properly
    if dt.dt.tz is not None:
        # If timezone-aware, convert to UTC and remove timezone
        dt_naive = dt.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # If already timezone-naive, use as is
        dt_naive = dt

    # Get numpy datetime64[ns] values
    t = dt_naive.values

    # Julian Date UT1 (≈UTC)
    JD_UNIX_EPOCH = 2440587.5
    seconds = (t.astype('datetime64[ns]').astype('int64')) / 1e9
    JD = JD_UNIX_EPOCH + seconds / 86400.0
    T = (JD - 2451545.0) / 36525.0  # centuries since J2000

    # GMST in seconds (IAU-82 like)
    gmst_sec = (67310.54841
                + (876600.0 * 3600 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3)
    gmst_rad = np.deg2rad((gmst_sec % 86400.0) / 240.0)  # 240 = 86400/360
    return gmst_rad


def _Rz(theta: np.ndarray) -> np.ndarray:
    c = np.cos(theta); s = np.sin(theta)
    R = np.empty((theta.size, 3, 3))
    R[:,0,0] =  c; R[:,0,1] = -s; R[:,0,2] = 0.0
    R[:,1,0] =  s; R[:,1,1] =  c; R[:,1,2] = 0.0
    R[:,2,0] = 0.0; R[:,2,1] = 0.0; R[:,2,2] = 1.0
    return R


def icrf_to_ecef(times_utc,
                 r_km: np.ndarray,
                 v_kmps: np.ndarray,
                 a_kmps2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ICRF states to ECEF with constant Earth rotation rate.
    Inputs:
      times_utc : array-like of timestamps (str/pandas/np) parsable by pandas
      r_km      : (N,3) position in km
      v_kmps    : (N,3) velocity in km/s
      a_kmps2   : (N,3) acceleration in km/s^2
    Returns:
      (r_ecef, v_ecef, a_ecef) all (N,3)
    """
    r = np.asarray(r_km, float); v = np.asarray(v_kmps, float); a = np.asarray(a_kmps2, float)
    assert r.shape == v.shape == a.shape and r.shape[1] == 3

    # Pass times_utc directly to _gmst_angle_rad
    theta = _gmst_angle_rad(times_utc)
    R = _Rz(theta)

    # ω vector and cross products
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    wxr = np.cross(np.broadcast_to(w, r.shape), r)
    wxv = np.cross(np.broadcast_to(w, v.shape), v)
    wxwxr = np.cross(np.broadcast_to(w, r.shape), wxr)

    v_i_eff = v - wxr
    a_i_eff = a - 2.0 * wxv - wxwxr

    # Batch rotation: y_i = (R @ x_i)
    def _batch_rot(Rbat, X):
        # (N,3,3) @ (N,3) -> (N,3)
        return np.einsum('nij,nj->ni', Rbat, X, optimize=True)

    r_e = _batch_rot(R, r)
    v_e = _batch_rot(R, v_i_eff)
    a_e = _batch_rot(R, a_i_eff)
    return r_e, v_e, a_e