# geometry/triangulate_icrf.py
# Triangulator (ICRF) — STK-agnostic
# -----------------------------------------------------------------------------
# What it does now:
# 1) Reads observer ephemerides (ICRF) and target truth OEM (ICRF).
# 2) Interpolates both to OEM epochs; computes OUR LOS: u = (x_true - r_obs)/|.|.
# 3) Saves per-observer LOS CSVs under exports/geometry/los/<RUN_ID>/<TARGET>/.
# 4) Applies Earth occlusion (using TRUE target geometry) to gate visible epochs.
# 5) Triangulates per-epoch with all visible observers (>= min_observers),
#    computes analytic Σ_geo and CEP50, and writes xhat CSVs.
# Notes:
# - Units: km, km/s, radians, UTC timestamps
# - No STK LOS dependency; minimal logging; robust UTC handling.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import re, time

# Config loader import
from config.loader import load_config

# =========================
# Config & paths
# =========================
CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()

EPH_DIR       = (PROJECT_ROOT / CFG["paths"]["ephem_root"]).resolve()           # observers ephem (CSV)
OEM_DIR       = (PROJECT_ROOT / CFG["paths"]["oem_root"]).resolve()             # truth OEM (CCSDS)
TRI_OUT_DIR   = (PROJECT_ROOT / CFG["paths"]["triangulation_out"]).resolve()    # triangulation products
LOS_EXPORT_DIR= (PROJECT_ROOT / "exports" / "geometry" / "los").resolve()      # our LOS exports

RUN_ID  = CFG["project"]["run_id"]
RUN_DIR = (TRI_OUT_DIR / RUN_ID)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Tunables
R_EARTH_KM      = float(CFG["geometry"]["earth_radius_km"])     # km
MIN_OBS         = int(CFG["geometry"]["min_observers"])        # >=2
SIGMA_LOS       = float(CFG["gpm_measurement"]["los_noise_rad"])# rad (for Σ_geo)
DO_MONTE_CARLO  = False                                            # off by default (keep lean)
MC_SAMPLES      = 50
MC_EVERY        = 20
MC_SEED         = 12345

# =========================
# Small logging helper
# =========================
LEVEL = str(CFG.get("logging",{}).get("level","INFO")).upper()
LEVELS={"DEBUG":10,"INFO":20,"WARN":30,"ERROR":40}

def _log(level:str,msg:str):
    if LEVELS.get(level,20) >= LEVELS.get(LEVEL,20):
        print(msg)

_log("INFO", f"[RUN ] Triangulation run id: {RUN_ID}")
_log("INFO", f"[OUT ] Triangulation directory: {RUN_DIR}")

# =========================
# IO helpers
# =========================
_ws_rx = re.compile(r"\s+")

def _norm_cols(cols: List[str]) -> List[str]:
    m = {
        "time (utcg)":"time",
        "x (km)":"x_km","y (km)":"y_km","z (km)":"z_km",
        "vx (km/s)":"vx_kmps","vy (km/s)":"vy_kmps","vz (km/s)":"vz_kmps",
    }
    c = [_ws_rx.sub(" ", s.strip().lower()) for s in cols]
    has_km = any(s in c for s in ("x (km)","y (km)","z (km)"))
    out=[]
    for s in c:
        if s in m: out.append(m[s]); continue
        mm=re.match(r"^(.*?)\s*\(.*\)$", s)
        if mm:
            base=mm.group(1).strip()
            if base in ("x","y","z") and has_km: out.append(f"{base}_km"); continue
            out.append(base); continue
        if s in ("x","y","z") and has_km: out.append(f"{s}_km"); continue
        out.append(s)
    return out

def read_csv_generic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = _norm_cols(list(df.columns))
    return df

def read_oem_ccsds(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OEM not found: {path}")
    recs=[]
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s=raw.strip()
            if not s or s.startswith("#") or s.startswith("META"): continue
            parts=s.split()
            if len(parts)>=7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x,y,z,vx,vy,vz = map(float, parts[1:7])
                except Exception:
                    continue
                recs.append((t,x,y,z,vx,vy,vz))
    if not recs: raise RuntimeError(f"No state lines in OEM {path}")
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"])\
           .dropna(subset=["time"]).sort_values("time")
    s = pd.to_datetime(df["time"], errors="coerce")
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    df["time"]=s
    return df.set_index("time")

# =========================
# Time utilities
# =========================

def _ensure_utc_index(idx) -> pd.DatetimeIndex:
    s = pd.to_datetime(idx, errors="coerce")
    if isinstance(s, pd.Series):
        s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
        return pd.DatetimeIndex(s)
    s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
    return pd.DatetimeIndex(s)

# =========================
# Geometry core
# =========================

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

def earth_blocks_ray_vec(r_sat_arr: np.ndarray, x_true_arr: np.ndarray, R_earth: float = R_EARTH_KM) -> np.ndarray:
    d = x_true_arr - r_sat_arr
    rho = np.linalg.norm(d, axis=1)
    mask_pos = rho > 0
    u = np.zeros_like(d)
    u[mask_pos] = (d[mask_pos].T / rho[mask_pos]).T
    rs2 = np.einsum('ij,ij->i', r_sat_arr, r_sat_arr)
    proj = np.einsum('ij,ij->i', r_sat_arr, u)
    lam_star = -proj
    before = lam_star <= 0
    beyond = lam_star >= rho
    d_min2 = rs2 - proj*proj
    blocked = (~before) & (~beyond) & (d_min2 <= R_earth*R_earth)
    blocked |= ~mask_pos
    return blocked

# =========================
# Data plumbing
# =========================

def discover_targets() -> List[str]:
    return sorted([p.stem for p in OEM_DIR.glob("HGV_*.oem")])

def discover_observers() -> List[str]:
    return sorted([re.sub(r"^OBS_","", p.stem).replace("_ephem","") for p in EPH_DIR.glob("OBS_*_ephem.csv")])

def load_observer_ephem(obs: str) -> pd.DataFrame:
    f = EPH_DIR / f"OBS_{obs}_ephem.csv"
    df = read_csv_generic(f)
    need = ("time","x_km","y_km","z_km")
    if not all(c in df.columns for c in need):
        raise ValueError(f"EPHEM {f} missing columns {need}; got {df.columns}")
    df = df["time x_km y_km z_km".split()].copy()
    s = pd.to_datetime(df["time"], errors="coerce")
    s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    df["time"] = s
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# Interpolate ephemeris to requested UTC times

def interp_ephem_to_times(eph: pd.DataFrame, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    e = eph.set_index("time").sort_index()
    e.index = _ensure_utc_index(e.index)
    times = _ensure_utc_index(times_utc)
    t0, t1 = e.index.min(), e.index.max()
    times = times[(times >= t0) & (times <= t1)]
    if len(times) == 0:
        return pd.DataFrame(columns=["x_km","y_km","z_km"], index=pd.DatetimeIndex([], tz="UTC"))
    ee = e.reindex(e.index.union(times)).sort_index().interpolate(method="time")["x_km y_km z_km".split()]
    out = ee.loc[times]
    out.index.name = "time"
    return out

# =========================
# Core per-target routine
# =========================

def run_for_target(target_id: str, observers: List[str]) -> None:
    # Truth & master time
    oem_path = OEM_DIR / f"{target_id}.oem"
    truth = read_oem_ccsds(oem_path)                     # index=UTC time
    master_idx = truth.index
    _log("INFO", f"[TGT ] {target_id} | epochs={len(master_idx)}")

    # Build aligned per-observer tables on master epochs; compute LOS from truth
    aligned: Dict[str, pd.DataFrame] = {}
    X_true = truth.loc[master_idx, ["x_km","y_km","z_km"]].to_numpy(float)   # (N,3)

    for obs in observers:
        eph = load_observer_ephem(obs)
        E = interp_ephem_to_times(eph, master_idx).reindex(master_idx)
        pos_ok = E[["x_km","y_km","z_km"]].notna().all(axis=1).to_numpy()
        if not pos_ok.any():
            continue
        R = E.to_numpy(float)                             # (N,3)
        d = X_true - R                                    # (N,3)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n==0] = 1.0
        U = d / n                                         # unit LOS vectors (our LOS)
        blocked = earth_blocks_ray_vec(R[pos_ok], X_true[pos_ok], R_EARTH_KM)
        vis = np.zeros(len(master_idx), dtype=bool)
        vis[pos_ok] = ~blocked
        df = pd.DataFrame({
            "x_km": E["x_km"].astype(float),
            "y_km": E["y_km"].astype(float),
            "z_km": E["z_km"].astype(float),
            "ux":   U[:,0],
            "uy":   U[:,1],
            "uz":   U[:,2],
            "vis":  vis,
        }, index=master_idx)
        df.index.name = "time"
        if vis.any():
            aligned[obs] = df

    if not aligned:
        _log("WARN", f"[SKIP] {target_id}: no visible epochs after gating.")
        return

    # Export our LOS per observer
    exp_dir = (LOS_EXPORT_DIR / RUN_ID / target_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    for obs, df in aligned.items():
        out = df[["ux","uy","uz","vis"]].reset_index()
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (exp_dir / f"LOS_OBS_{obs}_to_{target_id}_icrf.csv").write_text(out.to_csv(index=False))
    # Combined long-form (optional)
    long_rows=[]
    for obs, df in aligned.items():
        long_rows.append(pd.DataFrame({
            "time": df.index,
            "observer": obs,
            "ux": df["ux"].to_numpy(float),
            "uy": df["uy"].to_numpy(float),
            "uz": df["uz"].to_numpy(float),
            "vis": df["vis"].to_numpy(bool),
        }))
    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        long_df["time"] = pd.to_datetime(long_df["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (exp_dir / f"LOS_ALL_to_{target_id}_icrf.csv").write_text(long_df.to_csv(index=False))
    _log("INFO", f"[SAVE] LOS → {exp_dir}")

    # Triangulation on our LOS (aligned)
    rows=[]; kept=skipped=0
    for k, tstamp in enumerate(master_idx):
        xt = truth.loc[tstamp, ["x_km","y_km","z_km"]].to_numpy(float)
        rs: List[np.ndarray] = []; us: List[np.ndarray] = []; used: List[str] = []
        for obs, df in aligned.items():
            if not bool(df["vis"].iloc[k]):
                continue
            r = df.iloc[k][["x_km","y_km","z_km"]].to_numpy(float)
            u = df.iloc[k][["ux","uy","uz"]].to_numpy(float)
            rs.append(r); us.append(u); used.append(obs)
        if len(us) < MIN_OBS:
            skipped += 1
            continue
        x_hat, condA = triangulate_epoch(rs, us)
        # enforce LOS direction toward solution
        fixed=[]; flips=0
        for r,u in zip(rs,us):
            lam = float(u.dot(x_hat - r))
            if lam < 0: u = -u; flips += 1
            fixed.append(u)
        if flips:
            x_hat, condA = triangulate_epoch(rs, fixed)
            us = fixed
        Sigma = covariance_geo(x_hat, rs, us, SIGMA_LOS)
        cep50 = cep50_from_cov2d(Sigma)
        e = x_hat - xt
        rows.append({
            "time": pd.Timestamp(tstamp),
            "xhat_x_km": x_hat[0], "xhat_y_km": x_hat[1], "xhat_z_km": x_hat[2],
            "Sigma_xx": Sigma[0,0], "Sigma_yy": Sigma[1,1], "Sigma_zz": Sigma[2,2],
            "Sigma_xy": Sigma[0,1], "Sigma_xz": Sigma[0,2], "Sigma_yz": Sigma[1,2],
            "CEP50_km_analytic": cep50,
            "Nsats": len(us),
            "obs_used_csv": ",".join(used),
            "err_x_km": float(e[0]), "err_y_km": float(e[1]), "err_z_km": float(e[2]),
            "err_norm_km": float(np.linalg.norm(e)),
            "condA": condA,
        })
        kept += 1

    out = pd.DataFrame(rows).sort_values("time")
    if out.empty:
        _log("WARN", f"[WARN] {target_id}: no epochs kept after gating.")
        return
    ts = pd.to_datetime(out["time"], errors="coerce")
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    out["time"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out_path = RUN_DIR / f"xhat_geo_{target_id}.csv"
    out.to_csv(out_path, index=False)
    _log("INFO", f"[OK ] Wrote {len(out)} epochs → {out_path}")

# =========================
# Main
# =========================

def main():
    TRI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    targets = discover_targets()
    obs = discover_observers()
    if not targets:
        raise FileNotFoundError(f"No targets OEM in {OEM_DIR}")
    if not obs:
        raise FileNotFoundError(f"No observer ephemerides in {EPH_DIR}")

    _log("INFO", f"[RUN ] Looping over {len(targets)} targets | observers={len(obs)}")
    for tid in targets:
        try:
            run_for_target(tid, obs)
        except Exception as e:
            _log("ERROR", f"[ERR ] Target {tid}: {e}")

    # Manifest
    try:
        mf = RUN_DIR / "manifest.txt"
        with mf.open("w") as f:
            f.write(f"triangulation_run_id={RUN_ID}\n")
            f.write(f"generated_utc={datetime.utcnow().isoformat()}Z\n")
            f.write(f"earth_radius_km={R_EARTH_KM}\n")
            f.write(f"min_observers={MIN_OBS}\n")
            f.write(f"sigma_los_rad={SIGMA_LOS}\n")
            f.write("targets=\n")
            for p in sorted(RUN_DIR.glob("xhat_geo_*.csv")):
                f.write(f"  - {p.stem.split('xhat_geo_')[-1]}\n")
        _log("INFO", f"[MAN ] Wrote {mf}")
    except Exception as e:
        _log("WARN", f"[WARN] Could not write manifest: {e}")

if __name__ == "__main__":
    main()
