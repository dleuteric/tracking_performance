# geometry/triangulate_icrf.py
# Purpose: 2A (triangulation) + 2B (noise model & MC check) working in ICRF.
# Inputs: STK LOS files (ICRF) and observer ephemerides (ICRF) from exports/stk_exports.
# Outputs: CSV with per-epoch pseudo-position, analytic covariance, optional MC validation.
# Units: km, km/s, radians, UTC timestamps.

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import time

# --- Time normalization helper ---

def _norm_time_utc(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Parse to UTC and round to nearest second, then sort. Silences FutureWarning for 'S'."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.round("s")
    return df.dropna(subset=[col]).sort_values(col)

# --------------------------
# Config (adjust as needed)
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_ID    = "HGV_00010"                  # folder name under OUTPUT_LOS_VECTORS
LOS_DIR      = PROJECT_ROOT / "exports" / "stk_exports" / "OUTPUT_LOS_VECTORS" / TARGET_ID
EPH_DIR      = PROJECT_ROOT / "exports" / "stk_exports" / "OUTPUT_EPHEM"
OUT_DIR      = PROJECT_ROOT / "exports" / "triangulation"

# Minimum observers required for a geometric fix
N_OBSERVERS  = 2

# Path to target truth ephemeris (OEM)
OEM_PATH = PROJECT_ROOT / "exports" / "target_exports" / "OUTPUT_OEM" / f"{TARGET_ID}.oem"

# TIME_TOL_S   = 31.0  # commented out, not needed

# Noise model
SIGMA_LOS    = 250e-6     # rad
DO_MONTE_CARLO = True
MC_SAMPLES     = 50
MC_SEED        = 12345

# Visibility gate (optional). Set VISIBILITY_GATE=False if STK already applied access filtering.
VISIBILITY_GATE   = False
ELEV_MASK_DEG     = 5.0    # used only if VISIBILITY_GATE=True
USE_OCCLUSION_CHECK = True # used only if VISIBILITY_GATE=True
R_EARTH_KM        = 6378.137

# Performance / diagnostics knobs
MC_EVERY            = 20     # compute MC CEP only every Nth epoch (set 1 for every epoch)
BETA_PAIRS_MODE     = "summary"  # "full" | "summary" | "none"
BETA_SUMMARY_K      = 5      # when summary, include top-K widest and top-K narrowest pairs
PROGRESS_EVERY      = 100    # print progress every N epochs
# --------------------------
# Helpers: header normalization & file readers
# --------------------------
_whitespace_re = re.compile(r"\s+")

def _normalize_stk_headers(cols: list[str]) -> list[str]:
    unit_map = {
        "time (utcg)": "time",
        "x (km)": "x_km", "y (km)": "y_km", "z (km)": "z_km",
        "vx (km/s)": "vx_kmps", "vy (km/s)": "vy_kmps", "vz (km/s)": "vz_kmps",
    }
    cleaned = [_whitespace_re.sub(" ", c.strip().lower()) for c in cols]
    has_km_pos  = any(c in cleaned for c in ("x (km)", "y (km)", "z (km)"))
    has_kmps_vel= any(c in cleaned for c in ("vx (km/s)", "vy (km/s)", "vz (km/s)"))
    out = []
    for c in cleaned:
        if c in unit_map:
            out.append(unit_map[c]); continue
        m = re.match(r"^(.*?)\s*\(.*\)$", c)
        if m:
            base = m.group(1).strip()
            if base in ("x","y","z") and has_km_pos:
                out.append(f"{base}_km"); continue
            if base in ("vx","vy","vz") and has_kmps_vel:
                out.append(f"{base}_kmps"); continue
            out.append(base); continue
        if c in ("x","y","z") and has_km_pos:
            out.append(f"{c}_km"); continue
        if c in ("vx","vy","vz") and has_kmps_vel:
            out.append(f"{c}_kmps"); continue
        out.append(c)
    return out

def read_stk_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", nrows=nrows)
    raw = list(df.columns)
    df.columns = _normalize_stk_headers(raw)
    return df

def read_oem_ccsds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("META"):
                continue
            if line[0].isdigit() or line[0] == '2':  # time line
                parts = line.split()
                if len(parts) < 7:
                    continue
                epoch = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x,y,z,vx,vy,vz = map(float, parts[1:7])
                except Exception:
                    continue
                records.append((epoch, x, y, z, vx, vy, vz))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"])
    return df.sort_values("time").set_index("time")

def _exact_join(los_df: pd.DataFrame, eph_df: pd.DataFrame) -> pd.DataFrame:
    j = pd.merge(los_df, eph_df, on="time", how="inner")
    return j.dropna()

def interp_ephem_to_times(eph: pd.DataFrame, times_utc: pd.Series) -> pd.DataFrame:
    """
    Interpolate ephemeris (x_km, y_km, z_km) to the provided UTC times using time-based interpolation.
    Inputs:
      eph: DataFrame with columns ["time","x_km","y_km","z_km"] (UTC tz-aware)
      times_utc: pandas Series/Index of UTC timestamps (tz-aware)
    Returns:
      DataFrame indexed by the provided times with columns ["x_km","y_km","z_km"].
      Rows outside eph time span are dropped.
    """
    if eph.empty or len(times_utc) == 0:
        return pd.DataFrame(columns=["x_km","y_km","z_km"]).set_index(pd.DatetimeIndex([], tz="UTC"))

    # Ensure monotonic time index for interpolation
    e = eph.copy()
    if "time" in e.columns:
        e = e.set_index("time")
    e = e.sort_index()

    # Keep only within eph span to avoid extrapolation
    t0, t1 = e.index.min(), e.index.max()
    times = pd.to_datetime(times_utc, utc=True)
    mask = (times >= t0) & (times <= t1)
    times_in = times[mask]
    if len(times_in) == 0:
        return pd.DataFrame(columns=["x_km","y_km","z_km"]).set_index(pd.DatetimeIndex([], tz="UTC"))

    # Reindex to desired times and interpolate each column in time
    e_re = e.reindex(e.index.union(times_in)).sort_index()
    e_int = e_re.interpolate(method="time")[ ["x_km","y_km","z_km"] ]
    out = e_int.loc[times_in]
    out.index.name = "time"
    return out

# --- Interpolate LOS to times helper ---
def interp_los_to_times(los: pd.DataFrame, times_utc: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    """Interpolate LOS components (ux,uy,uz) to provided UTC times using time interpolation.
    Returns DataFrame indexed by the provided times with columns ux,uy,uz (unit-normalized).
    Out-of-span times are dropped.
    """
    if los.empty or len(times_utc) == 0:
        return pd.DataFrame(columns=["ux","uy","uz"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    L = los.copy()
    if "time" in L.columns:
        L = L.set_index("time")
    L = L.sort_index()

    # Only interpolate within LOS span
    t0, t1 = L.index.min(), L.index.max()
    times = pd.to_datetime(times_utc, utc=True)
    mask = (times >= t0) & (times <= t1)
    times_in = times[mask]
    if len(times_in) == 0:
        return pd.DataFrame(columns=["ux","uy","uz"]).set_index(pd.DatetimeIndex([], tz="UTC"))

    Lre = L.reindex(L.index.union(times_in)).sort_index()
    U = Lre.interpolate(method="time")[ ["ux","uy","uz"] ]
    U = U.loc[times_in]
    # re-normalize rows to unit vectors
    arr = U.to_numpy(float)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    arr = arr / n
    U.loc[:, ["ux","uy","uz"]] = arr
    U.index.name = "time"
    return U

# --------------------------
# Geometry core
# --------------------------

def _P(u: np.ndarray) -> np.ndarray:
    """Projection onto plane ⟂ u (u must be unit)."""
    return np.eye(3) - np.outer(u, u)

def enforce_los_toward_solution(x: np.ndarray, rs: list[np.ndarray], us: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Ensure each LOS points roughly from sat toward the solution x. If u·(x-r)<0, flip u. Returns (us_new, flips)."""
    us_new = []
    flips = 0
    for r,u in zip(rs,us):
        lam = float(u.dot(x - r))
        if lam < 0:
            u = -u; flips += 1
        us_new.append(u)
    return us_new, flips

def triangulate_epoch(rs: list[np.ndarray], us: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Solve (Σ P_i) x = Σ P_i r_i. Return x̂ and cond(A)."""
    A = np.zeros((3,3)); b = np.zeros(3)
    for r,u in zip(rs,us):
        P = _P(u); A += P; b += P @ r
    w = np.linalg.eigvals(A)
    condA = (np.max(np.abs(w)) / np.min(np.abs(w))) if np.min(np.abs(w))>0 else np.inf
    x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
    return x_hat, float(condA)

def covariance_geo(x_hat: np.ndarray, rs: list[np.ndarray], us: list[np.ndarray], sigma_los: float) -> np.ndarray:
    """Σ ≈ σ² (Σ (1/ρ_i²) P_i)^{-1}, ρ_i = ||x̂ - r_i||."""
    G = np.zeros((3,3))
    for r,u in zip(rs,us):
        rho = np.linalg.norm(x_hat - r)
        if rho <= 0: continue
        G += (1.0/(rho*rho)) * _P(u)
    # tiny Tikhonov if nearly singular
    if np.linalg.cond(G) > 1e12:
        G = G + 1e-12*np.eye(3)
    return (sigma_los**2) * np.linalg.inv(G)

# --- Baseline angle helper ---
from itertools import combinations

def baseline_angles_deg(us: list[np.ndarray], obs_ids: list[str],
                        mode: str = "summary", k: int = 5) -> tuple[float, float, float, str]:
    """
    Compute pairwise baseline angles between LOS unit vectors.
    Returns (beta_min_deg, beta_max_deg, beta_mean_deg, pairs_str)
    where pairs_str encodes angles as "idA-idB:angle" joined by '|'.
    mode: "full" returns all pairs; "summary" returns k smallest and k largest; "none" returns empty string.
    """
    if len(us) < 2:
        return (np.nan, np.nan, np.nan, "")
    pairs = []
    for (i, j) in combinations(range(len(us)), 2):
        ui = us[i]; uj = us[j]
        c = float(np.clip(ui.dot(uj), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))
        pairs.append((ang, obs_ids[i], obs_ids[j]))
    betas = np.array([p[0] for p in pairs], dtype=float)
    beta_min = float(np.min(betas))
    beta_max = float(np.max(betas))
    beta_mean = float(np.mean(betas))

    if mode == "none":
        return beta_min, beta_max, beta_mean, ""

    if mode == "full":
        labels = [f"{a}-{b}:{ang:.2f}" for ang, a, b in pairs]
        return beta_min, beta_max, beta_mean, "|".join(labels)

    # summary mode: top-k smallest and largest
    idx_sorted = np.argsort(betas)
    idx_take = list(idx_sorted[:k]) + list(idx_sorted[-k:]) if len(idx_sorted) > k else list(idx_sorted)
    labels = [f"{pairs[i][1]}-{pairs[i][2]}:{pairs[i][0]:.2f}" for i in idx_take]
    return beta_min, beta_max, beta_mean, "|".join(labels)

def is_visible(r_km: np.ndarray, u_hat: np.ndarray, elev_mask_deg: float, use_occlusion: bool) -> bool:
    """Return True if LOS from satellite position r_km along u_hat is above elevation mask
    and (optionally) not Earth-occluded. Elevation is asin(u·up), where up = r/|r|.
    """
    # elevation
    up = r_km / np.linalg.norm(r_km)
    elev_rad = np.arcsin(float(np.clip(u_hat.dot(up), -1.0, 1.0)))
    if np.degrees(elev_rad) < elev_mask_deg:
        return False
    if use_occlusion:
        rs2 = float(r_km.dot(r_km))
        proj = float(r_km.dot(u_hat))
        d_min = np.sqrt(max(rs2 - proj*proj, 0.0))
        if d_min <= R_EARTH_KM:
            return False
    return True

def cep50_from_cov2d(Sigma: np.ndarray) -> float:
    Sxy = Sigma[:2,:2]
    sigma_eq = np.sqrt(0.5*np.trace(Sxy))
    return 1.1774 * sigma_eq  # km

def perturb_los(u: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    v -= (v @ u) * u
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.array([u[1], -u[0], 0.0]); n = np.linalg.norm(v)
    w = v / n
    alpha = rng.normal(scale=sigma)
    u_p = u + alpha*w
    return u_p / np.linalg.norm(u_p)

# --------------------------
# Data plumbing
# --------------------------

def _discover_observers(los_dir: Path) -> list[str]:
    files = sorted(los_dir.glob(f"LOS_OBS_*_to_{TARGET_ID}_icrf.csv"))
    obs_ids = sorted({p.name.split("_")[2] for p in files})
    return obs_ids

def _load_los_for_obs(obs: str) -> pd.DataFrame:
    f = LOS_DIR / f"LOS_OBS_{obs}_to_{TARGET_ID}_icrf.csv"
    df = read_stk_csv(f)
    req = ("time","ux","uy","uz")
    if not all(c in df.columns for c in req):
        raise ValueError(f"LOS {f} missing required columns {req}; got {df.columns}")
    df = df[["time","ux","uy","uz"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def _load_eph_for_obs(obs: str) -> pd.DataFrame:
    f = EPH_DIR / f"OBS_{obs}_ephem.csv"
    df = read_stk_csv(f)
    req = ("time","x_km","y_km","z_km")
    if not all(c in df.columns for c in req):
        raise ValueError(f"EPHEM {f} missing required columns {req}; got {df.columns}")
    df = df[["time","x_km","y_km","z_km"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

# --------------------------
# Main
# --------------------------

def main():
    print(f"[PATH] LOS_DIR: {LOS_DIR}")
    print(f"[PATH] EPH_DIR: {EPH_DIR}")
    assert LOS_DIR.is_dir(), f"Missing {LOS_DIR}"
    assert EPH_DIR.is_dir(), f"Missing {EPH_DIR}"

    all_obs = _discover_observers(LOS_DIR)
    if not all_obs:
        raise FileNotFoundError(f"No LOS files under {LOS_DIR} for target {TARGET_ID}")
    print(f"[DISC] Observers available: {all_obs}")

    truth_df = read_oem_ccsds(OEM_PATH)
    if truth_df.empty:
        raise RuntimeError(f"[TRUTH] OEM not found or empty: {OEM_PATH}")
    # Normalize OEM times to exact seconds and set as index
    truth_df = _norm_time_utc(truth_df.reset_index(), col="time").set_index("time")
    print(f"[TRUTH] Loaded OEM: {len(truth_df)} epochs from {OEM_PATH.name}")

    obs_list = all_obs  # use all; gate per-epoch with N_OBSERVERS
    print(f"[USE ] Observers selected: {obs_list}")

    master_index = truth_df.index
    print(f"[SYNC] Master epochs from OEM: {len(master_index)}")
    print(f"[CFG ] VISIBILITY_GATE={VISIBILITY_GATE} (mask={ELEV_MASK_DEG} deg, occlusion={USE_OCCLUSION_CHECK})")
    aligned = {}
    for obs in obs_list:
        los = _load_los_for_obs(obs)
        eph = _load_eph_for_obs(obs)
        # Normalize times to whole seconds
        los = _norm_time_utc(los, col="time")
        eph = _norm_time_utc(eph, col="time")
        # Trim to OEM window
        t0, t1 = truth_df.index.min(), truth_df.index.max()
        los = los[(los["time"] >= t0) & (los["time"] <= t1)]
        eph = eph[(eph["time"] >= t0) & (eph["time"] <= t1)]

        # Interpolate both EPHEM and LOS to the OEM master epochs (exact simultaneity)
        eph_int = interp_ephem_to_times(eph, master_index)
        los_int = interp_los_to_times(los, master_index)
        # intersect to avoid extrapolation edges
        common = eph_int.index.intersection(los_int.index)
        if len(common) == 0:
            print(f"[JOIN] OBS {obs}: 0 rows after interpolation to OEM epochs — skipping")
            continue
        aligned_df = pd.DataFrame({
            "ux": los_int.loc[common, "ux"].astype(float),
            "uy": los_int.loc[common, "uy"].astype(float),
            "uz": los_int.loc[common, "uz"].astype(float),
            "x_km": eph_int.loc[common, "x_km"].astype(float),
            "y_km": eph_int.loc[common, "y_km"].astype(float),
            "z_km": eph_int.loc[common, "z_km"].astype(float),
        }, index=common)
        aligned_df.index.name = "time"
        aligned[obs] = aligned_df
        print(f"[JOIN] OBS {obs}: {len(aligned_df)} rows (interpolated to OEM epochs)")

    rng = np.random.default_rng(MC_SEED)
    rows = []
    kept = skipped = 0
    t_start = time.time()
    for idx, tstamp in enumerate(master_index):
        rs, us, used_obs = [], [], []
        for obs in obs_list:
            df_obs = aligned.get(obs)
            if df_obs is None:
                continue
            if tstamp in df_obs.index:
                row = df_obs.loc[tstamp]
                r = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=float)
                u = np.array([row["ux"], row["uy"], row["uz"]], dtype=float)
                u = u / np.linalg.norm(u)
                if VISIBILITY_GATE:
                    if not is_visible(r, u, ELEV_MASK_DEG, USE_OCCLUSION_CHECK):
                        continue  # gated out by elevation/occlusion
                rs.append(r); us.append(u); used_obs.append(obs)
        if len(us) < N_OBSERVERS:
            skipped += 1
            continue

        if (idx % PROGRESS_EVERY) == 0:
            elapsed = time.time() - t_start
            print(f"[PROG] epoch {idx+1}/{len(master_index)} | kept={kept} skipped={skipped} | elapsed={elapsed:.1f}s")

        kept += 1
        x_hat, condA = triangulate_epoch(rs, us)
        us_fixed, flips = enforce_los_toward_solution(x_hat, rs, us)
        if flips > 0:
            x_hat, condA = triangulate_epoch(rs, us_fixed)
            us = us_fixed
        # Baseline angle metrics (geometry hooks)
        beta_min_deg, beta_max_deg, beta_mean_deg, beta_pairs_deg = baseline_angles_deg(
            us, used_obs, mode=BETA_PAIRS_MODE, k=BETA_SUMMARY_K)

        Sigma = covariance_geo(x_hat, rs, us, SIGMA_LOS)
        cep50_km = cep50_from_cov2d(Sigma)

        # Diagnostics
        lambdas = np.array([u.dot(x_hat - r) for r,u in zip(rs,us)])
        perp = np.array([np.linalg.norm(_P(u) @ (x_hat - r)) for r,u in zip(rs,us)])
        lam_med = float(np.median(lambdas))
        perp_mean = float(np.mean(perp))

        # Truth residuals (direct index lookup on OEM)
        xt = truth_df.loc[tstamp, ["x_km","y_km","z_km"]].to_numpy(dtype=float)
        diff = x_hat - xt
        err_x, err_y, err_z = diff.tolist()
        err_norm = float(np.linalg.norm(diff))

        # Optional MC check (downsampled by MC_EVERY)
        mc_cep = np.nan
        if DO_MONTE_CARLO and (idx % MC_EVERY == 0):
            X = []
            for _ in range(MC_SAMPLES):
                us_p = [perturb_los(u, SIGMA_LOS, rng) for u in us]
                x_p, _ = triangulate_epoch(rs, us_p)
                X.append(x_p)
            X = np.vstack(X)
            S_emp = np.cov(X.T)
            mc_cep = cep50_from_cov2d(S_emp)

        rows.append({
            "time": pd.Timestamp(tstamp),
            "xhat_x_km": x_hat[0], "xhat_y_km": x_hat[1], "xhat_z_km": x_hat[2],
            "Sigma_xx": Sigma[0,0], "Sigma_yy": Sigma[1,1], "Sigma_zz": Sigma[2,2],
            "Sigma_xy": Sigma[0,1], "Sigma_xz": Sigma[0,2], "Sigma_yz": Sigma[1,2],
            "CEP50_km_analytic": cep50_km,
            "CEP50_km_MC": mc_cep,
            "condA": condA,
            "Nsats": len(us),
            "lambda_med_km": lam_med,
            "perp_residual_mean_km": perp_mean,
            "obs_used_csv": ",".join(used_obs),
            "err_x_km": err_x, "err_y_km": err_y, "err_z_km": err_z, "err_norm_km": err_norm,
            "beta_min_deg": beta_min_deg,
            "beta_max_deg": beta_max_deg,
            "beta_mean_deg": beta_mean_deg,
            "beta_pairs_deg": beta_pairs_deg,
        })
    print(f"[GATE] N_OBSERVERS>={N_OBSERVERS} | kept {kept} epochs, skipped {skipped}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"xhat_geo_{len(obs_list)}sats_{TARGET_ID}.csv"
    if not rows:
        print(f"[WARN] No epochs satisfied N_OBSERVERS>={N_OBSERVERS}. Nothing to write.")
        return
    df_out = pd.DataFrame(rows).sort_values("time")
    df_out["time"] = pd.to_datetime(df_out["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    df_out.to_csv(out_path, index=False)

    print(f"[OK ] Wrote {len(df_out)} epochs -> {out_path}")
    print(df_out[["time","CEP50_km_analytic","CEP50_km_MC","condA"]].head())

if __name__ == "__main__":
    main()