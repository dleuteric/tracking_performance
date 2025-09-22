# geometry/triangulate_icrf.py
# Clean triangulator (ICRF):
# - Reads LOS (unit vectors) and observer ephemerides from STK exports
# - Reads target truth OEM (ICRF) and uses it ONLY to gate visibility via Earth occlusion
# - Aligns everything to OEM epochs by time interpolation (no nearest joins)
# - Per epoch: keep only observers whose LOS ray to the TRUE target is NOT Earth-occluded
# - If Nsats >= N_OBSERVERS, solve least-squares intersection, compute analytic Σ_geo, CEP50
# - (Optional) Monte Carlo angular noise check at a stride
# Units: km, km/s, radians, UTC timestamps

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re, time
import os, uuid
from datetime import datetime
from itertools import combinations
from typing import List, Dict, Tuple

# Config loader import
from config.loader import load_config

# --------------------------
# Config (from YAML)
# --------------------------
CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()

# Paths
LOS_BASE_DIR = (PROJECT_ROOT / CFG["paths"]["los_root"]).resolve()
EPH_DIR      = (PROJECT_ROOT / CFG["paths"]["ephem_root"]).resolve()
OEM_DIR      = (PROJECT_ROOT / CFG["paths"]["oem_root"]).resolve()
OUT_DIR      = (PROJECT_ROOT / CFG["paths"]["triangulation_out"]).resolve()

# Run ID (single source of truth from loader)
RUN_ID  = CFG["project"]["run_id"]
RUN_DIR = (OUT_DIR / RUN_ID)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Logging (from YAML)
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)

_log("INFO", f"[RUN ] Triangulation run id: {RUN_ID}")
_log("INFO", f"[OUT ] Triangulation directory: {RUN_DIR}")

# Gating & performance from YAML
R_EARTH_KM     = float(CFG["geometry"]["earth_radius_km"])  # km
N_OBSERVERS    = int(CFG["geometry"]["min_observers"])     # >=2
SIGMA_LOS      = float(CFG["gpm_measurement"]["los_noise_rad"])  # rad
BETA_PAIRS_MODE= "summary" if CFG.get("geometry",{}).get("beta_pairs_mode","all")=="kbest" else "summary"
BETA_SUMMARY_K = int(CFG.get("geometry",{}).get("beta_summary_k", 5))
DO_MONTE_CARLO = True
MC_SAMPLES     = 50
MC_EVERY       = 20
MC_SEED        = 12345
PROGRESS_EVERY = 100

# Run mode
LOOP_ALL_TARGETS = True   # we iterate targets discovered under LOS_BASE_DIR that also have OEM
TARGET_ID        = "HGV_00007"  # used only if LOOP_ALL_TARGETS=False

# Helper: get paths for a target
def target_paths(target_id: str) -> tuple[Path, Path]:
    los_dir = LOS_BASE_DIR / target_id
    oem_path = OEM_DIR / f"{target_id}.oem"
    return los_dir, oem_path

# --------------------------
# IO helpers
# --------------------------
_whitespace_re = re.compile(r"\s+")

def _normalize_stk_headers(cols: List[str]) -> List[str]:
    unit_map = {
        "time (utcg)": "time",
        "x (km)": "x_km", "y (km)": "y_km", "z (km)": "z_km",
        "vx (km/s)": "vx_kmps", "vy (km/s)": "vy_kmps", "vz (km/s)": "vz_kmps",
    }
    cleaned = [_whitespace_re.sub(" ", c.strip().lower()) for c in cols]
    has_km_pos  = any(c in cleaned for c in ("x (km)", "y (km)", "z (km)"))
    out = []
    for c in cleaned:
        if c in unit_map: out.append(unit_map[c]); continue
        m = re.match(r"^(.*?)\s*\(.*\)$", c)
        if m:
            base = m.group(1).strip()
            if base in ("x","y","z") and has_km_pos:
                out.append(f"{base}_km"); continue
            out.append(base); continue
        if c in ("x","y","z") and has_km_pos:
            out.append(f"{c}_km"); continue
        out.append(c)
    return out

def read_stk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = _normalize_stk_headers(list(df.columns))
    return df

def read_oem_ccsds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OEM not found: {path}")
    recs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("META"): continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x,y,z,vx,vy,vz = map(float, parts[1:7])
                except Exception:
                    continue
                recs.append((t,x,y,z,vx,vy,vz))
    if not recs:
        raise RuntimeError(f"No state lines in OEM {path}")
    df = pd.DataFrame(recs, columns=["time", "x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps"])
    s = pd.to_datetime(df["time"], errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC")
    else:
        s = s.dt.tz_localize("UTC")
    df["time"] = s
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    return df

def _norm_time_utc(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors="coerce")  # niente utc=True qui
    # se è tz-aware → convert, altrimenti localize
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC")
    else:
        s = s.dt.tz_localize("UTC")
    s = s.dt.round("s")
    out = df.copy()
    out[col] = s
    return out.dropna(subset=[col]).sort_values(col)

def _ensure_utc_index(idx) -> pd.DatetimeIndex:
    """
    Converte qualunque indice/serie temporale in DatetimeIndex UTC:
    - Se tz-aware → tz_convert("UTC")
    - Se naive     → tz_localize("UTC")
    """
    # accetta Series, DatetimeIndex o array-like
    s = pd.to_datetime(idx, errors="coerce")
    if isinstance(s, pd.Series):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC")
        else:
            s = s.dt.tz_localize("UTC")
        return pd.DatetimeIndex(s)
    else:
        # DatetimeIndex
        if getattr(s, "tz", None) is not None:
            s = s.tz_convert("UTC")
        else:
            s = s.tz_localize("UTC")
        return pd.DatetimeIndex(s)

def _to_epoch_s(ts_like) -> np.ndarray:
    # accetta Series/Index con o senza tz, restituisce float seconds
    idx = pd.to_datetime(ts_like, errors="coerce")
    if isinstance(idx, pd.Series):
        idx = idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx.dt.tz_convert("UTC")
        return (idx.view("int64") / 1e9).to_numpy()
    else:
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return (idx.view("int64") / 1e9)

def _nearest_dt_seconds(t_ref_s: np.ndarray, t_grid_s: np.ndarray) -> np.ndarray:
    # per ogni t_ref, trova il vicino in t_grid e ritorna Δt = t_ref - t_near
    idx = np.searchsorted(t_grid_s, t_ref_s)
    idx = np.clip(idx, 1, len(t_grid_s) - 1)
    prev = t_grid_s[idx - 1]
    nxt  = t_grid_s[idx]
    t_near = np.where(np.abs(t_ref_s - prev) <= np.abs(t_ref_s - nxt), prev, nxt)
    return t_ref_s - t_near

def _dt_stats(label: str, dt: np.ndarray):
    if dt.size == 0:
        _log("WARN", f"[DT  ] {label}: series vuota")
        return
    _log("INFO", f"[DT  ] {label}: min={dt.min():.3f}s  med={np.median(dt):.3f}s  max={dt.max():.3f}s")

def interp_ephem_to_times(eph: pd.DataFrame, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    # porta l'ephemeris su indice 'time' in UTC
    if "time" in eph.columns:
        eph = eph.set_index("time")
    eph = eph.sort_index()
    eph.index = _ensure_utc_index(eph.index)

    # normalizza i tempi richiesti
    times = _ensure_utc_index(times_utc)
    # tieni solo i tempi dentro lo span dell'ephemeris
    t0, t1 = eph.index.min(), eph.index.max()
    times = times[(times >= t0) & (times <= t1)]
    if len(times) == 0:
        return pd.DataFrame(columns=["x_km", "y_km", "z_km"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )

    # reindex + interpolate su tempo
    e = eph.reindex(eph.index.union(times)).sort_index()
    e = e.interpolate(method="time")[["x_km", "y_km", "z_km"]]
    out = e.loc[times]
    out.index.name = "time"
    return out


def interp_los_to_times(los: pd.DataFrame, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    # porta la LOS su indice 'time' in UTC
    if "time" in los.columns:
        los = los.set_index("time")
    los = los.sort_index()
    los.index = _ensure_utc_index(los.index)

    # normalizza i tempi richiesti
    times = _ensure_utc_index(times_utc)
    # tieni solo i tempi dentro lo span della LOS
    t0, t1 = los.index.min(), los.index.max()
    times = times[(times >= t0) & (times <= t1)]
    if len(times) == 0:
        return pd.DataFrame(columns=["ux", "uy", "uz"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )

    # reindex + interpolate su tempo
    L = los.reindex(los.index.union(times)).sort_index()
    L = L.interpolate(method="time")[["ux", "uy", "uz"]]

    # rinormalizza i versori
    arr = L.to_numpy(float)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    arr = arr / n
    L.loc[:, ["ux", "uy", "uz"]] = arr

    out = L.loc[times]
    out.index.name = "time"
    return out

def _score_pair(u_i, u_j, elev_i=None, elev_j=None, cfg=None):
    # β in deg
    beta = np.degrees(np.arccos(np.clip(np.abs(np.dot(u_i, u_j)), -1.0, 1.0)))
    # optional cap and min-beta gate
    beta_eff = min(beta, cfg.beta_cap_deg)
    if beta < cfg.min_beta_deg:
        return -np.inf, beta  # reject
    # optional same-plane penalty (until we wire real plane IDs)
    if cfg.prefer_cross_plane and _same_plane_heuristic(i, j):
        beta_eff = max(0.0, beta_eff - cfg.same_plane_penalty_deg)
    s = np.sin(np.radians(beta_eff))
    if cfg.use_elev_weight and (elev_i is not None) and (elev_j is not None):
        s *= max(0.0, np.sin(np.radians(elev_i))) * max(0.0, np.sin(np.radians(elev_j)))
    return s, beta
# --------------------------
# Geometry core
# --------------------------

def P_perp(u: np.ndarray) -> np.ndarray:
    return np.eye(3) - np.outer(u,u)

def triangulate_epoch(rs: List[np.ndarray], us: List[np.ndarray]) -> tuple[np.ndarray, float]:
    A = np.zeros((3,3)); b = np.zeros(3)
    for r,u in zip(rs,us):
        P = P_perp(u); A += P; b += P @ r
    w = np.linalg.eigvals(A)
    condA = (np.max(np.abs(w)) / np.min(np.abs(w))) if np.min(np.abs(w))>0 else np.inf
    x_hat = np.linalg.lstsq(A,b,rcond=None)[0]
    return x_hat, float(condA)

def covariance_geo(x_hat: np.ndarray, rs: List[np.ndarray], us: List[np.ndarray], sigma_los: float) -> np.ndarray:
    G = np.zeros((3,3))
    for r,u in zip(rs,us):
        rho = np.linalg.norm(x_hat - r)
        if rho<=0: continue
        G += (1.0/(rho*rho)) * P_perp(u)
    if np.linalg.cond(G) > 1e12:
        G += 1e-12*np.eye(3)
    return (sigma_los**2) * np.linalg.inv(G)

def cep50_from_cov2d(Sigma: np.ndarray) -> float:
    Sxy = Sigma[:2,:2]
    sigma_eq = float(np.sqrt(0.5*np.trace(Sxy)))
    return 1.1774 * sigma_eq

# Earth occlusion using TRUE target position

def earth_blocks_ray(r_sat: np.ndarray, x_true: np.ndarray, R_earth: float = R_EARTH_KM) -> bool:
    """Return True if the ray from satellite to TRUE target intersects Earth.
    We check the minimum-distance point along the ray segment [0, rho_true].
    """
    d = x_true - r_sat
    rho = float(np.linalg.norm(d))
    if rho <= 0:  # degenerate
        return True
    u = d / rho
    rs2 = float(r_sat.dot(r_sat))
    proj = float(r_sat.dot(u))
    # closest point to Earth center along the ray occurs at lambda* = -proj
    lam_star = -proj
    if lam_star <= 0:
        # closest point is at the satellite itself; cannot be occluded (sat is above Earth)
        return False
    if lam_star >= rho:
        # closest point is beyond the target; Earth doesn't block segment before target
        return False
    d_min2 = rs2 - proj*proj
    return d_min2 <= R_earth*R_earth

# Vectorized Earth-occlusion test
def earth_blocks_ray_vec(r_sat_arr: np.ndarray, x_true_arr: np.ndarray, R_earth: float = R_EARTH_KM) -> np.ndarray:
    """Vectorized Earth-occlusion test.
    Inputs are (N,3) arrays; returns boolean array 'blocked' of shape (N,).
    We test the ray segment from sat to TRUE target as in earth_blocks_ray().
    """
    d = x_true_arr - r_sat_arr                       # (N,3)
    rho = np.linalg.norm(d, axis=1)                  # (N,)
    # Avoid divide by zero
    mask_pos = rho > 0
    u = np.zeros_like(d)
    u[mask_pos] = (d[mask_pos].T / rho[mask_pos]).T
    rs2 = np.einsum('ij,ij->i', r_sat_arr, r_sat_arr)  # |r|^2
    proj = np.einsum('ij,ij->i', r_sat_arr, u)         # r·u
    lam_star = -proj
    # Segment tests
    before = lam_star <= 0
    beyond = lam_star >= rho
    d_min2 = rs2 - proj*proj
    blocked = (~before) & (~beyond) & (d_min2 <= R_earth*R_earth)
    # If rho==0, mark as blocked to be safe
    blocked |= ~mask_pos
    return blocked

# Baseline metrics

def baseline_angles_deg(us: List[np.ndarray], ids: List[str], mode: str = "summary", k: int = 5) -> tuple[float,float,float,str]:
    if len(us) < 2:
        return (np.nan,np.nan,np.nan,"")
    pairs = []
    for (i,j) in combinations(range(len(us)),2):
        c = float(np.clip(us[i].dot(us[j]), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))
        pairs.append((ang, ids[i], ids[j]))
    betas = np.array([p[0] for p in pairs], float)
    bmin, bmax, bmean = float(betas.min()), float(betas.max()), float(betas.mean())
    if mode == "none": return bmin,bmax,bmean, ""
    if mode == "full":
        lab = [f"{a}-{b}:{ang:.2f}" for ang,a,b in pairs]
        return bmin,bmax,bmean, "|".join(lab)
    idx = np.argsort(betas)
    take = list(idx[:k]) + list(idx[-k:]) if len(idx)>k else list(idx)
    lab = [f"{pairs[i][1]}-{pairs[i][2]}:{pairs[i][0]:.2f}" for i in take]
    return bmin,bmax,bmean, "|".join(lab)

# --------------------------
# Data plumbing
# --------------------------

def discover_observers(los_dir: Path) -> List[str]:
    files = sorted(los_dir.glob(f"LOS_OBS_*_to_*_icrf.csv"))
    # Extract observer names for files matching the current target
    obs_set = set()
    for p in files:
        # File pattern: LOS_OBS_{obs}_to_{target_id}_icrf.csv
        parts = p.name.split("_")
        if len(parts) >= 6 and parts[0] == "LOS" and parts[1] == "OBS" and parts[3] == "to":
            obs = parts[2]
            obs_set.add(obs)
    return sorted(obs_set)

def load_los(los_dir: Path, target_id: str, obs: str) -> pd.DataFrame:
    f = los_dir / f"LOS_OBS_{obs}_to_{target_id}_icrf.csv"
    df = read_stk_csv(f)
    need = ("time","ux","uy","uz")
    if not all(c in df.columns for c in need):
        raise ValueError(f"LOS {f} missing columns {need}; got {df.columns}")
    df = df[["time","ux","uy","uz"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_eph(obs: str) -> pd.DataFrame:
    f = EPH_DIR / f"OBS_{obs}_ephem.csv"
    df = read_stk_csv(f)
    need = ("time","x_km","y_km","z_km")
    if not all(c in df.columns for c in need):
        raise ValueError(f"EPHEM {f} missing columns {need}; got {df.columns}")
    df = df[["time","x_km","y_km","z_km"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# --------------------------
# Main
# --------------------------

def run_for_target(target_id: str):
    los_dir, oem_path = target_paths(target_id)
    _log("INFO", f"[TGT ] {target_id}")
    _log("DEBUG", f"[PATH] LOS_DIR: {los_dir}")
    _log("DEBUG", f"[PATH] EPH_DIR: {EPH_DIR}")
    assert los_dir.is_dir() and EPH_DIR.is_dir()

    obs_all = discover_observers(los_dir)
    if not obs_all:
        raise FileNotFoundError(f"No LOS files under {los_dir}")
    _log("DEBUG", f"[DISC] Observers discovered: {obs_all}")

    truth = read_oem_ccsds(oem_path)
    truth = _norm_time_utc(truth.reset_index(), col="time").set_index("time")
    master_index = truth.index
    _log("INFO", f"[TRUTH] OEM epochs: {len(master_index)} from {oem_path.name}")

    # CSV di diagnostica Δt per l’intera run
    dt_diag_rows = []

    # Build aligned per-observer tables on the master OEM epochs
    aligned: Dict[str, pd.DataFrame] = {}
    for obs in obs_all:
        los = _norm_time_utc(load_los(los_dir, target_id, obs), col="time")
        eph = _norm_time_utc(load_eph(obs), col="time")
        # trim to OEM span
        t0, t1 = master_index.min(), master_index.max()
        los = los[(los["time"] >= t0) & (los["time"] <= t1)]
        eph = eph[(eph["time"] >= t0) & (eph["time"] <= t1)]

        # --- Δt GREZZO tra LOS e EPHEM (come da checker) ---
        t_los_raw = _to_epoch_s(los["time"])
        t_eph_raw = _to_epoch_s(eph["time"])
        dt_raw = _nearest_dt_seconds(t_los_raw, t_eph_raw)
        _dt_stats(f"{obs} raw(LOS vs EPH)", dt_raw)

        # salva qualche riga per report
        if dt_raw.size:
            dt_diag_rows.append({
                "observer": obs,
                "phase": "raw",
                "dt_min_s": float(np.min(dt_raw)),
                "dt_med_s": float(np.median(dt_raw)),
                "dt_max_s": float(np.max(dt_raw)),
                "n": int(dt_raw.size),
            })

        # Interpolate both to OEM times, then reindex to master_index (full length)
        E = interp_ephem_to_times(eph, master_index).reindex(master_index)
        U = interp_los_to_times(los, master_index).reindex(master_index)

        # --- Δt POST-ALLINEAMENTO (su OEM master_index) ---
        t_oem = _to_epoch_s(master_index)
        # Nota: E e U sono già reindicizzati su master_index
        t_e = t_oem[E[["x_km", "y_km", "z_km"]].notna().all(axis=1).to_numpy()]
        t_u = t_oem[U[["ux", "uy", "uz"]].notna().all(axis=1).to_numpy()]

        # confronta entrambe con i tempi OEM stessi (devono stare a 0, entro numerica)
        dt_e = t_e - t_oem[:len(t_e)]
        dt_u = t_u - t_oem[:len(t_u)]
        _dt_stats(f"{obs} post(EPH on OEM)", dt_e)
        _dt_stats(f"{obs} post(LOS on OEM)", dt_u)

        if dt_e.size:
            dt_diag_rows.append({
                "observer": obs, "phase": "post_eph",
                "dt_min_s": float(np.min(dt_e)),
                "dt_med_s": float(np.median(dt_e)),
                "dt_max_s": float(np.max(dt_e)),
                "n": int(dt_e.size),
            })
        if dt_u.size:
            dt_diag_rows.append({
                "observer": obs, "phase": "post_los",
                "dt_min_s": float(np.min(dt_u)),
                "dt_med_s": float(np.median(dt_u)),
                "dt_max_s": float(np.max(dt_u)),
                "n": int(dt_u.size),
            })

        # Valid epochs are where both ephem and LOS exist
        pos_ok = E[["x_km","y_km","z_km"]].notna().all(axis=1)
        los_ok = U[["ux","uy","uz"]].notna().all(axis=1)
        valid_mask = (pos_ok & los_ok).to_numpy()
        if not valid_mask.any():
            _log("DEBUG", f"[JOIN] OBS {obs}: 0 valid rows on OEM epochs — skip")
            continue

        # Build full‑length df aligned to master_index; fill values, and precompute visibility
        df = pd.DataFrame({
            "x_km": E["x_km"].astype(float),
            "y_km": E["y_km"].astype(float),
            "z_km": E["z_km"].astype(float),
            "ux":   U["ux"].astype(float),
            "uy":   U["uy"].astype(float),
            "uz":   U["uz"].astype(float),
        }, index=master_index)
        df.index.name = "time"

        # Earth occlusion only on valid epochs; others are not visible
        xt_common = truth.loc[master_index, ["x_km","y_km","z_km"]].to_numpy(float)[valid_mask]
        r_common  = df.loc[master_index, ["x_km","y_km","z_km"]].to_numpy(float)[valid_mask]
        blocked   = earth_blocks_ray_vec(r_common, xt_common, R_EARTH_KM)
        vis = np.zeros(len(master_index), dtype=bool)
        vis[valid_mask] = ~blocked
        df["vis"] = vis

        vis_count = int(vis.sum())
        if vis_count == 0:
            _log("DEBUG", f"[VIS ] OBS {obs}: 0 visible epochs after occlusion — drop")
            continue

        aligned[obs] = df
        _log("DEBUG", f"[JOIN] OBS {obs}: {vis_count} visible rows (full index {len(df)})")

        if dt_diag_rows:
            dt_csv = RUN_DIR / f"dt_diagnostics_{target_id}.csv"
            pd.DataFrame(dt_diag_rows).to_csv(dt_csv, index=False)
            _log("INFO", f"[DT  ] Salvato Δt diagnostics → {dt_csv}")

    if not aligned:
        raise RuntimeError("No observers aligned to OEM epochs.")

    usable_obs = sorted(aligned.keys())
    _log("INFO", f"[USE ] Observers after occlusion gate: {len(usable_obs)} used")
    _log("DEBUG", f"[USE ] List: {usable_obs}")
    _log("DEBUG", f"[SAVE] Output dir: {OUT_DIR}")
    _log("DEBUG", f"[SAVE] Run-stamped dir: {RUN_DIR}")

    # Triangulation loop with Earth-occlusion gate using TRUE target
    rng = np.random.default_rng(MC_SEED)
    rows = []
    kept = skipped = 0
    t_start = time.time()

    for k, tstamp in enumerate(master_index):
        # TRUE target position
        xt = truth.loc[tstamp, ["x_km","y_km","z_km"]].to_numpy(float)
        rs: List[np.ndarray] = []
        us: List[np.ndarray] = []
        used_ids: List[str] = []

        for obs, df in aligned.items():
            # fast positional access (all dfs are reindexed to master_index)
            if not bool(df["vis"].iloc[k]):
                continue
            row = df.iloc[k]
            r = np.array([row["x_km"], row["y_km"], row["z_km"]], float)
            u = np.array([row["ux"], row["uy"], row["uz"]], float)
            rs.append(r); us.append(u); used_ids.append(obs)

        if len(us) < N_OBSERVERS:
            skipped += 1
            if LOG_LEVEL == "DEBUG" and (k % PROGRESS_EVERY)==0:
                _log("DEBUG", f"[PROG] epoch {k+1}/{len(master_index)} | kept={kept} skipped={skipped}")
            continue

        x_hat, condA = triangulate_epoch(rs, us)
        # re-enforce LOS direction toward solution
        fixed = []
        flips = 0
        for r,u in zip(rs,us):
            lam = float(u.dot(x_hat - r))
            if lam < 0: u = -u; flips += 1
            fixed.append(u)
        if flips:
            x_hat, condA = triangulate_epoch(rs, fixed)
            us = fixed

        Sigma = covariance_geo(x_hat, rs, us, SIGMA_LOS)
        cep50 = cep50_from_cov2d(Sigma)
        bmin,bmax,bmean,bpairs = baseline_angles_deg(us, used_ids, mode=BETA_PAIRS_MODE, k=BETA_SUMMARY_K)

        # Errors vs TRUE
        e = x_hat - xt
        err_x,err_y,err_z = e.tolist(); err_norm = float(np.linalg.norm(e))

        # Optional MC CEP (downsampled)
        mc_cep = np.nan
        if DO_MONTE_CARLO and (k % MC_EVERY == 0):
            Xp = []
            for _ in range(MC_SAMPLES):
                ups = []
                for u in us:
                    # small-angle perturbation orthogonal to u
                    v = rng.normal(size=3); v -= (v@u)*u
                    n = np.linalg.norm(v) or 1.0
                    w = v/n
                    alpha = rng.normal(scale=SIGMA_LOS)
                    up = (u + alpha*w); up /= np.linalg.norm(up)
                    ups.append(up)
                x_p, _ = triangulate_epoch(rs, ups)
                Xp.append(x_p)
            Xp = np.vstack(Xp)
            Semp = np.cov(Xp.T)
            mc_cep = cep50_from_cov2d(Semp)

        rows.append({
            "time": pd.Timestamp(tstamp),
            "xhat_x_km": x_hat[0], "xhat_y_km": x_hat[1], "xhat_z_km": x_hat[2],
            "Sigma_xx": Sigma[0,0], "Sigma_yy": Sigma[1,1], "Sigma_zz": Sigma[2,2],
            "Sigma_xy": Sigma[0,1], "Sigma_xz": Sigma[0,2], "Sigma_yz": Sigma[1,2],
            "CEP50_km_analytic": cep50,
            "CEP50_km_MC": mc_cep,
            "condA": condA,
            "Nsats": len(us),
            "obs_used_csv": ",".join(used_ids),
            "beta_min_deg": bmin, "beta_max_deg": bmax, "beta_mean_deg": bmean, "beta_pairs_deg": bpairs,
            "err_x_km": err_x, "err_y_km": err_y, "err_z_km": err_z, "err_norm_km": err_norm,
        })

        kept += 1
        if LOG_LEVEL == "DEBUG" and (k % PROGRESS_EVERY)==0:
            elapsed = time.time() - t_start
            _log("DEBUG", f"[PROG] epoch {k+1}/{len(master_index)} | kept={kept} skipped={skipped} | Nsats={len(us)} | elapsed={elapsed:.1f}s")

    _log("INFO", f"[GATE] N_OBSERVERS>={N_OBSERVERS} | kept {kept} epochs, skipped {skipped}")

    # ensure stamped dir exists
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values("time")
    if out.empty:
        _log("WARN", "[WARN] No epochs kept after Earth-occlusion gating.")
        return

    ts = pd.to_datetime(out["time"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_localize("UTC")
    out["time"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    out_path = RUN_DIR / f"xhat_geo_{target_id}.csv"
    out.to_csv(out_path, index=False)
    _log("INFO", f"[OK ] Wrote {len(out)} epochs -> {out_path}")


# Main driver
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOOP_ALL_TARGETS:
        # Find targets that have both LOS folder and OEM file
        cand_targets = sorted([p.name for p in LOS_BASE_DIR.iterdir() if p.is_dir() and p.name.startswith("HGV_")])
        run_targets = []
        for tid in cand_targets:
            _, oem_path = target_paths(tid)
            if oem_path.exists():
                run_targets.append(tid)
        if not run_targets:
            raise FileNotFoundError(f"No targets with LOS and OEM found under {LOS_BASE_DIR} & {OEM_DIR}")
        _log("INFO", f"[RUN ] Looping over {len(run_targets)} targets")
        _log("DEBUG", f"[RUN ] Targets: {run_targets}")
        for tid in run_targets:
            try:
                run_for_target(tid)
            except Exception as e:
                _log("ERROR", f"[ERR ] Target {tid}: {e}")
    else:
        run_for_target(TARGET_ID)

    # Write a simple manifest for the triangulation run
    try:
        manifest = RUN_DIR / "manifest.txt"
        with manifest.open("w") as mf:
            mf.write(f"triangulation_run_id={RUN_ID}\n")
            mf.write(f"generated_utc={datetime.utcnow().isoformat()}Z\n")
            # Echo key config knobs that affected this run
            mf.write(f"earth_radius_km={R_EARTH_KM}\n")
            mf.write(f"min_observers={N_OBSERVERS}\n")
            mf.write(f"sigma_los_rad={SIGMA_LOS}\n")
            mf.write(f"beta_mode={BETA_PAIRS_MODE}\n")
            mf.write(f"beta_summary_k={BETA_SUMMARY_K}\n")
            if LOOP_ALL_TARGETS:
                mf.write("targets=\n")
                for p in sorted(RUN_DIR.glob("xhat_geo_*.csv")):
                    mf.write(f"  - {p.stem.split('xhat_geo_')[-1]}\n")
        _log("INFO", f"[MAN ] Wrote {manifest}")
    except Exception as e:
        _log("WARN", f"[WARN] Could not write triangulation manifest: {e}")

if __name__ == "__main__":
    main()