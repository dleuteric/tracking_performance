# file: gpm_triangulation_icrf.py
# Purpose: 2A (triangulation) + 2B (noise model & MC check) in ICRF.
# Units: km, seconds, radians.

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Config ----------
LOS_DIR   = Path("exports/stk_exports/OUTPUT_LOS_VECTORS/HGV_00001")
TARGET_ID = LOS_DIR.name  # e.g., 'HGV_00001'
OBS_IDS   = None             # if None -> infer from LOS filenames
TIME_TOL_S = 1e-3            # max |Δt| allowed when joining LOS and ephem
SIGMA_LOS = 1e-3             # rad (≈1 mrad default)
DO_MONTE_CARLO = True
MC_SAMPLES = 200

# ---------- CSV expectations ----------
# LOS CSV header (per your sample): Time, Observer, Target, Frame, ux, uy, uz
# EPH CSV header (typical): Time, x_km, y_km, z_km, [vx_kmps, vy_kmps, vz_kmps] (vel not required here)

def _read_los_files() -> dict[str, pd.DataFrame]:
    los_files = sorted(LOS_DIR.glob(f"LOS_OBS_*_to_{TARGET_ID}_icrf.csv"))
    print(f"[GPM] Searching LOS files in {LOS_DIR.resolve()} with pattern 'LOS_OBS_*_to_{TARGET_ID}_icrf.csv'")
    if not los_files:
        # Try to discover candidates to help debugging
        any_files = sorted(LOS_DIR.glob("LOS_OBS_*_to_*_icrf.csv"))
        candidates = sorted({p.name.split("_to_")[1].split("_icrf")[0] for p in any_files})
        raise FileNotFoundError(
            f"No LOS files found under {LOS_DIR} for target '{TARGET_ID}'. "
            f"Looked for pattern 'LOS_OBS_*_to_{TARGET_ID}_icrf.csv'. "
            f"Candidates found here: {candidates}"
        )
    out = {}
    for f in los_files:
        df = pd.read_csv(f)
        # Normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        # Parse time robustly (string -> datetime64[ns] -> seconds via epoch)
        df["Time"] = pd.to_datetime(df[cols.get("time", "Time")], utc=True)
        df["ux"] = df[cols.get("ux", "ux")].astype(float)
        df["uy"] = df[cols.get("uy", "uy")].astype(float)
        df["uz"] = df[cols.get("uz", "uz")].astype(float)
        # infer OBS id from filename
        obs_tag = f.name.split("_")[2]  # LOS_OBS_001_to_... -> "001"
        out[obs_tag] = df[["Time", "ux", "uy", "uz"]].copy()
    print(f"[GPM] Loaded {len(out)} LOS streams: {sorted(out.keys())}")
    return out

def _read_ephem_files(obs_ids: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for obs in obs_ids:
        f = EPH_DIR / f"OBS_{obs}_ephem.csv"
        if not f.exists():
            raise FileNotFoundError(f"Missing ephemeris for OBS {obs}: {f}")
        df = pd.read_csv(f)
        cols = {c.lower().strip(): c for c in df.columns}
        df["Time"] = pd.to_datetime(df[cols.get("time", "Time")], utc=True)
        # attempt common position column names
        xk = cols.get("x_km") or cols.get("x") or "x_km"
        yk = cols.get("y_km") or cols.get("y") or "y_km"
        zk = cols.get("z_km") or cols.get("z") or "z_km"
        df.rename(columns={xk:"x_km", yk:"y_km", zk:"z_km"}, inplace=True)
        out[obs] = df[["Time","x_km","y_km","z_km"]].copy()
    return out

def _nearest_join(los_df: pd.DataFrame, eph_df: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    # Merge exact first; if not exact, nearest within tolerance
    merged = pd.merge_asof(
        los_df.sort_values("Time"),
        eph_df.sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_s)
    )
    return merged.dropna()

def _P(u: np.ndarray) -> np.ndarray:
    # Projection matrix onto plane perpendicular to u (unit)
    return np.eye(3) - np.outer(u, u)

def triangulate_epoch(rs: list[np.ndarray], us: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Solve (sum P_i) x = sum P_i r_i. Returns x_hat and cond(A)."""
    A = np.zeros((3,3))
    b = np.zeros(3)
    for r,u in zip(rs,us):
        P = _P(u)
        A += P
        b += P @ r
    # Condition check
    w, _ = np.linalg.eig(A)
    condA = (np.max(np.abs(w)) / np.min(np.abs(w))) if np.min(np.abs(w))>0 else np.inf
    x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
    return x_hat, condA

def covariance_geo(x_hat: np.ndarray, rs: list[np.ndarray], us: list[np.ndarray], sigma_los: float) -> np.ndarray:
    """Sigma ≈ sigma^2 * (sum (1/rho_i^2) * P_i)^-1, rho_i = ||x_hat - r_i||."""
    G = np.zeros((3,3))
    for r,u in zip(rs,us):
        rho = np.linalg.norm(x_hat - r)
        if rho <= 0:
            continue
        G += (1.0 / (rho*rho)) * _P(u)
    # Robust inverse
    # If geometry is degenerate, add tiny Tikhonov
    if np.linalg.cond(G) > 1e12:
        G = G + 1e-12*np.eye(3)
    Sigma = (sigma_los**2) * np.linalg.inv(G)
    return Sigma

def cep50_from_cov2d(Sigma: np.ndarray) -> float:
    """CEP50 (km) from 2x2 horizontal covariance (Sigma[0:2,0:2]); Rayleigh approx."""
    Sxy = Sigma[:2,:2]
    # Equivalent 2D variance: sigma_eq^2 ≈ 0.5*trace(Sxy)
    sigma_eq = np.sqrt(0.5*np.trace(Sxy))
    return 1.1774 * sigma_eq  # CEP50 ≈ 1.1774 * sigma_eq

def perturb_los(u: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply small-angle rotation with std=sigma around a random axis ⟂ u."""
    # Draw a random vector, make it orthogonal to u
    v = rng.normal(size=3)
    v -= (v @ u) * u
    n = np.linalg.norm(v)
    if n < 1e-12:
        # fall back to a fixed orthonormal if unlucky
        v = np.array([u[1], -u[0], 0.0])
        n = np.linalg.norm(v)
    w = v / n  # unit ⟂ u
    alpha = rng.normal(scale=sigma)  # small angle
    u_pert = u + alpha * w
    return u_pert / np.linalg.norm(u_pert)

def run_once(obs_pick: list[str] | None = None):
    los_map = _read_los_files()
    obs_list = obs_pick or (sorted(los_map.keys()) if OBS_IDS is None else OBS_IDS)
    eph_map = _read_ephem_files(obs_list)

    # Build per-epoch stacks by intersection of available times across selected observers
    # Strategy: align each observer's LOS with its ephem; then inner-join times across observers.
    aligned = {}
    for obs in obs_list:
        df = _nearest_join(los_map[obs], eph_map[obs], TIME_TOL_S)
        df["obs"] = obs
        aligned[obs] = df[["Time","obs","ux","uy","uz","x_km","y_km","z_km"]]

    # common time grid (inner)
    common = None
    for obs in obs_list:
        t = aligned[obs][["Time"]].drop_duplicates()
        common = t if common is None else common.merge(t, on="Time", how="inner")
    if common is None or common.empty:
        raise RuntimeError("No common timestamps across selected observers within tolerance.")

    rows = []
    rng = np.random.default_rng(12345)
    for tstamp in common["Time"].values:
        rs, us = [], []
        for obs in obs_list:
            row = aligned[obs].loc[aligned[obs]["Time"]==tstamp].iloc[0]
            r = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=float)
            u = np.array([row["ux"], row["uy"], row["uz"]], dtype=float)
            # normalize defensively
            u = u / np.linalg.norm(u)
            rs.append(r); us.append(u)

        x_hat, condA = triangulate_epoch(rs, us)
        Sigma = covariance_geo(x_hat, rs, us, SIGMA_LOS)
        cep50_km = cep50_from_cov2d(Sigma)

        # Optional MC check
        mc_cep = np.nan
        if DO_MONTE_CARLO:
            X = []
            for _ in range(MC_SAMPLES):
                us_p = [perturb_los(u, SIGMA_LOS, rng) for u in us]
                x_p, _ = triangulate_epoch(rs, us_p)
                X.append(x_p)
            X = np.vstack(X)
            # empirical covariance
            S_emp = np.cov(X.T)
            mc_cep = cep50_from_cov2d(S_emp)

        rows.append({
            "Time": pd.Timestamp(tstamp),
            "xhat_x_km": x_hat[0], "xhat_y_km": x_hat[1], "xhat_z_km": x_hat[2],
            "Sigma_xx": Sigma[0,0], "Sigma_yy": Sigma[1,1], "Sigma_zz": Sigma[2,2],
            "Sigma_xy": Sigma[0,1], "Sigma_xz": Sigma[0,2], "Sigma_yz": Sigma[1,2],
            "CEP50_km_analytic": cep50_km,
            "CEP50_km_MC": mc_cep,
            "condA": condA,
            "Nsats": len(us)
        })

    out = pd.DataFrame(rows).sort_values("Time")
    out_dir = Path("exports") / "triangulation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"xhat_geo_{len(obs_list)}sats.csv", index=False)
    print(f"[OK] Wrote {len(out)} epochs -> {out_dir / f'xhat_geo_{len(obs_list)}sats.csv'}")
    # quick sanity print
    print(out[["Time","CEP50_km_analytic","CEP50_km_MC","condA"]].head())

if __name__ == "__main__":
    # Example: automatically picks all observers present in the LOS folder.
    # For a first test, restrict to two observers with good geometry, e.g., ["001","004"].
    run_once(obs_pick=None)