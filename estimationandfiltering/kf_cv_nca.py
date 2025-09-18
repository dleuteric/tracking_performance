"""
filtering/kf_cv_nca.py

Kalman Filter core for 9-state Nearly-Constant-Acceleration (NCA) model in ICRF.
Uses position-only measurements from GPM (adapter).
Units: km, km/s, km/s^2.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

# Config + logger
import os
try:
    from config.loader import load_config
    _CFG = load_config()
except Exception:
    _CFG = {"filter": {}, "logging": {}}

LOG_LEVEL = str(_CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)

# Filter knobs from YAML (with safe fallbacks)
_CHI2_GATE = float(_CFG.get("filter", {}).get("chi2_gate_3dof", 7.815))
_infl = _CFG.get("filter", {}).get("inflate", {})
_BAD_GEOM_FACTOR   = float(_infl.get("bad_geom_factor", 3.0))
_CONDA_THRESH      = float(_infl.get("condA_thresh", 1.0e8))
_CONDA_FACTOR      = float(_infl.get("condA_factor", 2.0))
_BETAMIN_THRESH_DEG= float(_infl.get("betamin_thresh_deg", 10.0))
_BETAMIN_FACTOR    = float(_infl.get("betamin_factor", 2.0))

# Initial covariance from YAML P0_diag, else default
import numpy as _np
_P0_DIAG = _CFG.get("filter", {}).get("P0_diag", [1e-3,1e-3,1e-3, 1.0,1.0,1.0, 10.0,10.0,10.0])
_P0_DIAG = _np.array(_P0_DIAG, dtype=float)

# Support both package and direct-script execution
try:
    from estimationandfiltering.models import F_Q, JERK_PSD
    from estimationandfiltering.adapter import Measurement
except Exception:
    import sys, pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from estimationandfiltering.models import F_Q, JERK_PSD
    from estimationandfiltering.adapter import Measurement

# Measurement model: position-only
H = np.hstack([np.eye(3), np.zeros((3, 6))])  # (3x9)


def predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """KF predict step."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update(x_pred: np.ndarray, P_pred: np.ndarray,
           z: np.ndarray, R: np.ndarray, meta: dict | None = None) -> Tuple[np.ndarray, np.ndarray, float, bool, float]:
    """KF update step. Returns (x_upd, P_upd, NIS, did_update, alpha_R)."""
    # R-inflation based on geometry/meta
    alpha = 1.0
    if meta:
        if bool(meta.get("bad_geom", False)):
            alpha *= _BAD_GEOM_FACTOR
        try:
            condA = float(meta.get("condA", _np.nan))
            if _np.isfinite(condA) and condA > _CONDA_THRESH:
                alpha *= _CONDA_FACTOR
        except Exception:
            pass
        try:
            bmin = float(meta.get("beta_min_deg", _np.nan))
            if _np.isfinite(bmin) and bmin < _BETAMIN_THRESH_DEG:
                alpha *= _BETAMIN_FACTOR
        except Exception:
            pass
    R_eff = alpha * R

    v = z - H @ x_pred
    S = H @ P_pred @ H.T + R_eff
    try:
        S_inv = _np.linalg.inv(S)
    except _np.linalg.LinAlgError:
        # ill-conditioned S, skip update
        return x_pred, P_pred, _np.nan, False, alpha

    nis = float(v.T @ S_inv @ v)  # Normalized Innovation Squared

    # Chi-square gate (3 dof)
    if not _np.isfinite(nis) or nis > _CHI2_GATE:
        return x_pred, P_pred, nis, False, alpha

    K = P_pred @ H.T @ S_inv
    x_upd = x_pred + K @ v
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    return x_upd, P_upd, nis, True, alpha


def run_forward(meas_list: List[Measurement],
                qj: float = JERK_PSD,
                P0_scale: float = 1.0) -> List[dict]:
    """
    Run a forward KF pass over a list of Measurement objects.
    Args:
        meas_list: list of Measurement (adapter output, in ICRF, km units).
        qj: jerk PSD (km^2/s^5).
        P0_scale: scaling factor for initial covariance.

    Returns:
        history: list of dicts with keys {t, x, P, nis, did_update}
    """
    if not meas_list:
        return []

    n = 9
    history: List[dict] = []

    # Initial state: from first measurement
    z0 = meas_list[0].z_km
    x = np.zeros(n)
    x[0:3] = z0  # position
    # v,a remain zero
    P = np.diag(_P0_DIAG) * P0_scale

    last_t = meas_list[0].t

    for m in meas_list:
        dt = (m.t - last_t).total_seconds()
        if dt < 0:
            dt = 0.0
        last_t = m.t

        # Predict
        F, Q = F_Q(dt, qj)
        x_pred, P_pred = predict(x, P, F, Q)

        # Update
        x, P, nis, did_update, aR = update(x_pred, P_pred, m.z_km, m.R_km2, m.meta)

        history.append({
            "t": m.t,
            "x": x.copy(),
            "P": P.copy(),
            "nis": nis,
            "did_update": did_update,
            "R_inflation": aR,
            "meta": m.meta,
        })

    _log("INFO", f"[KF ] Forward run complete: {len(history)} epochs")
    return history


if __name__ == "__main__":
    # self-test stub (requires adapter + a triangulation CSV)
    print("[KF ] Module imported. Use run_filter.py to drive a run.")