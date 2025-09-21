# estimationandfiltering/ew_rls.py
# Exponentially-Weighted Recursive Least Squares (EW-RLS) polynomial fit
# Reference: Huang et al., 2018, Sec. 4 (eqs. 16–21). Estimates position and velocity by
# fitting a 1D polynomial per axis with forgetting factor theta in (0,1].
#
# I/O contract:
#   * Input CSV can have flexible schemas. Time column candidates: t_s,time_s,t,time,epoch_s,epoch
#     Position column candidates per axis (km or m): x_km,x,x_icrf_km,x_ecef_km,x_m,x_icrf_m,x_ecef_m (same for y/z)
#   * Units are standardized to: t [s], position [km], velocity [km/s]
#   * Output CSV: t_s, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, frame, ewrls_order, ewrls_theta
#
# Typical use:
#   python estimationandfiltering/ew_rls.py --in_csv triangulation/<RUN_ID>/xhat_geo_*.csv \
#       --out_csv tracks/<RUN_ID>/ewrls_tracks.csv --theta 0.93 --order 4
#
# Notes:
#   - A 4th-order polynomial (order=4) aligns with the paper and often balances bias/variance well.
#   - For numerical stability, time is zero-based and scaled by (t_rel / t_scale) internally.
#   - Set theta closer to 1.0 for slower forgetting; smaller for more responsiveness to maneuvers.

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Helpers: schema mapping and numerical utilities
# ---------------------------------------------------------------------------

def _first(df: pd.DataFrame, candidates) -> str | None:
  for c in candidates:
    if c in df.columns:
      return c
  # fuzzy fallback: substring match
  for c in df.columns:
    name = c.lower()
    for cand in candidates:
      if cand in name:
        return c
  return None

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
  """
  Standardize an arbitrary triangulation-like CSV to ['t_s','x_km','y_km','z_km'].
  Detect meters vs km and convert to km.
  """
  t_col = _first(df, ["t_s","time_s","t","time","epoch_s","epoch"])
  if t_col is None:
    raise ValueError("No time column found (tried: t_s,time_s,t,time,epoch_s,epoch)")

  x_col = _first(df, ["x_km","x","x_icrf_km","x_ecef_km","x_m","x_icrf_m","x_ecef_m"])
  y_col = _first(df, ["y_km","y","y_icrf_km","y_ecef_km","y_m","y_icrf_m","y_ecef_m"])
  z_col = _first(df, ["z_km","z","z_icrf_km","z_ecef_km","z_m","z_icrf_m","z_ecef_m"])
  if x_col is None or y_col is None or z_col is None:
    raise ValueError(f"No position columns found. Available: {list(df.columns)}")

  tmp = df[[t_col, x_col, y_col, z_col]].copy()
  tmp.columns = ["t_s","x_raw","y_raw","z_raw"]

  def looks_meters(name: str) -> bool:
    n = name.lower()
    return ("_m" in n and "_km" not in n) or n.endswith("meters")

  meterish = looks_meters(x_col) or looks_meters(y_col) or looks_meters(z_col)
  scale = 1000.0 if meterish else 1.0

  # time to float seconds: accept numeric or ISO-8601 strings
  t_series = tmp["t_s"]
  try:
    t_numeric = t_series.astype(float)
  except Exception:
    # parse datetimes; prefer UTC if present
    dt = pd.to_datetime(t_series, utc=True, errors="coerce")
    if dt.isna().any():
      # fallback without forcing UTC
      dt = pd.to_datetime(t_series, utc=False, errors="coerce")
    if dt.isna().any():
      raise ValueError(f"Cannot parse time column to numeric or datetime: sample={t_series.iloc[0]!r}")
    # seconds since UNIX epoch (float)
    t_numeric = dt.astype("int64") / 1e9

  return pd.DataFrame({
    "t_s": t_numeric.astype(float),
    "x_km": tmp["x_raw"].astype(float) / scale,
    "y_km": tmp["y_raw"].astype(float) / scale,
    "z_km": tmp["z_raw"].astype(float) / scale
  })

def _poly_regressor(t_scaled: float, d: int) -> np.ndarray:
  """u(t) = [1, t, t^2, ..., t^(d-1)]"""
  return np.array([t_scaled**k for k in range(d)], dtype=float)

# ---------------------------------------------------------------------------
# Core EW-RLS per 1D signal
# ---------------------------------------------------------------------------

def _ew_rls_1d(t_rel: np.ndarray,
               z: np.ndarray,
               order: int = 4,
               theta: float = 0.93,
               P0: float = 1e6,
               t_scale: float | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  EW-RLS on a single axis.
  Parameters
  ----------
  t_rel : (N,) time in seconds, zero-based
  z     : (N,) measurements [km]
  order : polynomial order (>=1). d = order+1 coefficients
  theta : forgetting factor in (0,1], closer to 1 -> slower forgetting
  P0    : initial covariance scale
  t_scale : optional time scaling factor; if None, inferred from (max(t_rel)-min(t_rel))
  Returns
  -------
  a      : (d,) final polynomial coefficients in scaled time
  a_hist : (N,d) coefficient history
  z_hat  : (N,) fitted position [km]
  v_hat  : (N,) fitted velocity [km/s]
  """
  assert order >= 1, "order must be >= 1"
  N = int(len(t_rel))
  d = order + 1

  # Time scaling for numerical conditioning
  if t_scale is None:
    span = float(max(1.0, (t_rel.max() - t_rel.min())))
    t_scale = span if span > 0 else 1.0

  t_s = t_rel / t_scale

  P = np.eye(d) * float(P0)
  a = np.zeros(d, dtype=float)
  a_hist = np.zeros((N, d), dtype=float)
  z_hat = np.zeros(N, dtype=float)
  v_hat = np.zeros(N, dtype=float)

  for k in range(N):
    u = _poly_regressor(t_s[k], d)                 # (d,)
    denom = float(theta + u @ (P @ u))             # scalar
    K = (P @ u) / denom                            # (d,)
    innov = float(z[k] - u @ a)
    a = a + K * innov
    P = (P - np.outer(K, u) @ P) / theta           # Joseph form not required for LS

    a_hist[k] = a
    z_hat[k] = u @ a
    # derivative in original seconds: v = (1/t_scale) * d/dtau (poly)
    # d/dtau sum_{p=0}^{d-1} a_p * tau^p = sum_{p=1}^{d-1} p * a_p * tau^(p-1)
    tau = t_s[k]
    dv_dtau = 0.0
    for p in range(1, d):
      dv_dtau += p * a[p] * (tau ** (p - 1))
    v_hat[k] = dv_dtau / t_scale

  return a, a_hist, z_hat, v_hat

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ewrls_on_csv(xhat_geo_csv: str,
                     out_csv: str = "ewrls_tracks.csv",
                     frame: str = "ECEF",
                     order: int = 4,
                     theta: float = 0.93,
                     P0: float = 1e6) -> pd.DataFrame:
  """
  Run EW-RLS on a triangulated pseudo-position CSV (flexible schema).
  """
  df_in = pd.read_csv(xhat_geo_csv)
  df_std = _map_columns(df_in)

  t = df_std["t_s"].to_numpy(float)
  # zero-base time for conditioning regardless of absolute epoch
  t_rel = t - float(t[0])
  x = df_std["x_km"].to_numpy(float)
  y = df_std["y_km"].to_numpy(float)
  z = df_std["z_km"].to_numpy(float)

  _, _, xhat, vx = _ew_rls_1d(t_rel, x, order=order, theta=theta, P0=P0)
  _, _, yhat, vy = _ew_rls_1d(t_rel, y, order=order, theta=theta, P0=P0)
  _, _, zhat, vz = _ew_rls_1d(t_rel, z, order=order, theta=theta, P0=P0)

  # also provide ISO8601 time strings for plotting compatibility
  dt_iso = pd.to_datetime(t, unit="s", utc=True)
  time_iso = dt_iso.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
  out = pd.DataFrame({
    "t_s": t,
    "time": time_iso,
    "x_km": xhat, "y_km": yhat, "z_km": zhat,
    "vx_kms": vx, "vy_kms": vy, "vz_kms": vz,
    "frame": frame,
    "ewrls_order": order,
    "ewrls_theta": theta
  })
  Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
  out.to_csv(out_csv, index=False)
  return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser(description="EW-RLS polynomial fit (Huang et al. 2018, Sec. 4)")
  ap.add_argument("--in_csv", required=True,
                  help="triangulated CSV (time + x/y/z in km or m; flexible column names)")
  ap.add_argument("--out_csv", default="tracks/ewrls_tracks.csv")
  ap.add_argument("--frame", default="ECEF")
  ap.add_argument("--order", type=int, default=4)
  ap.add_argument("--theta", type=float, default=0.93)
  ap.add_argument("--P0", type=float, default=1e6)
  args = ap.parse_args()
  run_ewrls_on_csv(args.in_csv, args.out_csv, args.frame, args.order, args.theta, args.P0)