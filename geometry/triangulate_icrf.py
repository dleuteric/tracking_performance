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

# --------------------------
# Config
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Base directories
LOS_BASE_DIR = PROJECT_ROOT / "exports" / "stk_exports" / "OUTPUT_LOS_VECTORS"
EPH_DIR      = PROJECT_ROOT / "exports" / "stk_exports" / "OUTPUT_EPHEM"
OEM_DIR      = PROJECT_ROOT / "exports" / "target_exports" / "OUTPUT_OEM"
OUT_DIR      = PROJECT_ROOT / "exports" / "triangulation"


# ---- Constellation param inference ----
def _infer_constellation_params_from_ephem(eph_dir: Path, r_earth_km: float) -> tuple[int, float, float]:
    """Infer (Nsats, mean altitude [km], mean inclination [deg]) from available OBS ephemerides.
    - Nsats: number of ephemeris files matching OBS_***_ephem.csv that are readable
    - Altitude: median of |r|-R_earth over first ~10 epochs per sat, averaged across sats
    - Inclination: arccos(h_z/|h|) from r×v at first usable epoch per sat, averaged across sats
    Fallback: raise on total failure; caller can catch and use env/defaults.
    """
    eph_files = sorted(eph_dir.glob("OBS_*_ephem.csv"))
    if not eph_files:
        raise FileNotFoundError(f"No ephemeris files in {eph_dir}")

    inc_list = []
    alt_list = []
    good = 0
    for f in eph_files:
        try:
            df = read_stk_csv(f)
            cols = {c.lower(): c for c in df.columns}
            # Normalize time & columns
            needed_pos = [c for c in ["time","x_km","y_km","z_km"] if c in df.columns]
            if len(needed_pos) < 4:
                # try raw x,y,z
                if all(c in df.columns for c in ["time","x","y","z"]):
                    df = df.rename(columns={"x":"x_km","y":"y_km","z":"z_km"})
                else:
                    continue
            df = df[["time","x_km","y_km","z_km"] + [c for c in ["vx_kmps","vy_kmps","vz_kmps"] if c in df.columns]].copy()
            df = _norm_time_utc(df, col="time")
            if len(df) < 2:
                continue
            r = df.loc[:, ["x_km","y_km","z_km"]].to_numpy(float)
            # altitude estimate over first up-to-10 rows
            k = min(10, len(r))
            alt = np.median(np.linalg.norm(r[:k], axis=1) - r_earth_km)

            # velocity: prefer provided; else finite difference
            if all(c in df.columns for c in ["vx_kmps","vy_kmps","vz_kmps"]):
                v = df.loc[:, ["vx_kmps","vy_kmps","vz_kmps"]].to_numpy(float)
            else:
                # central difference for rows 0..2 if possible
                t = pd.to_datetime(df["time"]).astype("int64")/1e9  # seconds
                t = t.to_numpy(float)
                if len(df) >= 3 and (t[2] != t[0]):
                    v = (r[2] - r[0]) / (t[2]-t[0])
                    v = np.vstack([v, v])  # shape guard
                else:
                    # fallback: forward diff on first two
                    if t[1] == t[0]:
                        continue
                    v = np.vstack([ (r[1]-r[0])/(t[1]-t[0]), (r[1]-r[0])/(t[1]-t[0]) ])

            h = np.cross(r[0], v[0])
            hn = np.linalg.norm(h)
            if hn == 0:
                continue
            inc = np.degrees(np.arccos(np.clip(h[2]/hn, -1.0, 1.0)))

            inc_list.append(inc)
            alt_list.append(alt)
            good += 1
        except Exception:
            continue

    if good == 0:
        raise RuntimeError("Could not infer constellation params from ephemerides")

    nsats = good
    alt_km = float(np.mean(alt_list))
    inc_deg = float(np.mean(inc_list))
    return nsats, alt_km, inc_deg

def _get_run_params() -> tuple[int,float,float]:
    # Try inference first
    try:
        ns, alt, inc = _infer_constellation_params_from_ephem(EPH_DIR, R_EARTH_KM)
        print(f"[INF ] Inferred constellation: Nsats~{ns}, alt~{alt:.1f} km, inc~{inc:.2f} deg")
        return ns, alt, inc
    except Exception as e:
        print(f"[INF ] Inference failed ({e}); falling back to env/defaults")
        # Fallback to environment or defaults
        try:
            ns = int(os.environ.get("NSATS", "48"))
        except Exception:
            ns = 48
        try:
            alt = float(os.environ.get("ALT_KM", "550"))
        except Exception:
            alt = 550.0
        try:
            inc = float(os.environ.get("INC_DEG", "53"))
        except Exception:
            inc = 53.0
        return ns, alt, inc

NSATS, ALT_KM, INC_DEG = _get_run_params()
_run_uuid = uuid.uuid4().hex[:8]
_run_time = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
# Round altitude to nearest km and inclination to nearest tenth for stable names
alt_name = int(round(ALT_KM))
inc_name = ("%.1f" % INC_DEG).rstrip("0").rstrip(".")
RUN_ID = os.environ.get("RUN_ID") or f"{_run_time}_{NSATS}sat_{alt_name}km_{inc_name}deg_{_run_uuid}"
RUN_DIR = OUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"[RUN ] Triangulation run id: {RUN_ID}")
print(f"[OUT ] Triangulation directory: {RUN_DIR}")

# Run mode
LOOP_ALL_TARGETS = True   # if True, run all targets found under LOS_BASE_DIR (and with matching OEM)
TARGET_ID        = "HGV_00007"  # used when LOOP_ALL_TARGETS=False

# Helper: get paths for a target
def target_paths(target_id: str) -> tuple[Path, Path]:
    """Return LOS directory for target and OEM path."""
    los_dir = LOS_BASE_DIR / target_id
    oem_path = OEM_DIR / f"{target_id}.oem"
    return los_dir, oem_path

# Gating & performance
R_EARTH_KM   = 6378.137
N_OBSERVERS  = 2          # minimum simultaneous observers
SIGMA_LOS    = 150e-6       # rad (default 1 mrad)
DO_MONTE_CARLO = True
MC_SAMPLES     = 50
MC_EVERY       = 20       # compute MC every N epochs
MC_SEED        = 12345
BETA_PAIRS_MODE= "summary"  # "full" | "summary" | "none"
BETA_SUMMARY_K = 5
PROGRESS_EVERY = 100

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
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"])\
           .dropna(subset=["time"]).sort_values("time")
    return df.set_index("time")

def _norm_time_utc(df: pd.DataFrame, col: str="time") -> pd.DataFrame:
    with pd.option_context('mode.chained_assignment', None):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.round("s")
    return df.dropna(subset=[col]).sort_values(col)

# --------------------------
# Interpolation helpers
# --------------------------

def interp_ephem_to_times(eph: pd.DataFrame, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    if "time" in eph.columns: eph = eph.set_index("time")
    eph = eph.sort_index()
    t0,t1 = eph.index.min(), eph.index.max()
    times = pd.to_datetime(times_utc, utc=True)
    times = times[(times>=t0)&(times<=t1)]
    if len(times)==0:
        return pd.DataFrame(columns=["x_km","y_km","z_km"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    e = eph.reindex(eph.index.union(times)).sort_index()
    e = e.interpolate(method="time")[ ["x_km","y_km","z_km"] ]
    out = e.loc[times]
    out.index.name = "time"
    return out

def interp_los_to_times(los: pd.DataFrame, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    if "time" in los.columns: los = los.set_index("time")
    los = los.sort_index()
    t0,t1 = los.index.min(), los.index.max()
    times = pd.to_datetime(times_utc, utc=True)
    times = times[(times>=t0)&(times<=t1)]
    if len(times)==0:
        return pd.DataFrame(columns=["ux","uy","uz"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    L = los.reindex(los.index.union(times)).sort_index()
    L = L.interpolate(method="time")[ ["ux","uy","uz"] ]
    # normalize
    arr = L.to_numpy(float)
    n = np.linalg.norm(arr, axis=1, keepdims=True); n[n==0]=1.0
    arr = arr/n
    L.loc[:,["ux","uy","uz"]] = arr
    out = L.loc[times]
    out.index.name = "time"
    return out

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
    print(f"[TGT ] {target_id}")
    print(f"[PATH] LOS_DIR: {los_dir}")
    print(f"[PATH] EPH_DIR: {EPH_DIR}")
    assert los_dir.is_dir() and EPH_DIR.is_dir()

    obs_all = discover_observers(los_dir)
    if not obs_all:
        raise FileNotFoundError(f"No LOS files under {los_dir}")
    print(f"[DISC] Observers discovered: {obs_all}")

    truth = read_oem_ccsds(oem_path)
    truth = _norm_time_utc(truth.reset_index(), col="time").set_index("time")
    master_index = truth.index
    print(f"[TRUTH] OEM epochs: {len(master_index)} from {oem_path.name}")

    # Build aligned per-observer tables on the master OEM epochs
    aligned: Dict[str, pd.DataFrame] = {}
    for obs in obs_all:
        los = _norm_time_utc(load_los(los_dir, target_id, obs), col="time")
        eph = _norm_time_utc(load_eph(obs), col="time")
        # trim to OEM span
        t0,t1 = master_index.min(), master_index.max()
        los = los[(los["time"]>=t0)&(los["time"]<=t1)]
        eph = eph[(eph["time"]>=t0)&(eph["time"]<=t1)]
        # interpolate both to OEM times
        E = interp_ephem_to_times(eph, master_index)
        U = interp_los_to_times(los, master_index)
        common = E.index.intersection(U.index)
        if len(common)==0:
            print(f"[JOIN] OBS {obs}: 0 rows on OEM epochs — skip")
            continue
        df = pd.DataFrame({
            "x_km": E.loc[common, "x_km"].astype(float),
            "y_km": E.loc[common, "y_km"].astype(float),
            "z_km": E.loc[common, "z_km"].astype(float),
            "ux":   U.loc[common, "ux"].astype(float),
            "uy":   U.loc[common, "uy"].astype(float),
            "uz":   U.loc[common, "uz"].astype(float),
        }, index=common)
        df.index.name = "time"
        # Pre-compute visibility vs TRUE target (Earth occlusion). Store as column 'vis'.
        xt_common = truth.loc[common, ["x_km","y_km","z_km"]].to_numpy(float)
        r_common  = df[["x_km","y_km","z_km"]].to_numpy(float)
        blocked   = earth_blocks_ray_vec(r_common, xt_common, R_EARTH_KM)
        df["vis"] = ~blocked
        vis_count = int(df["vis"].sum())
        if vis_count == 0:
            print(f"[VIS ] OBS {obs}: 0 visible epochs after occlusion — drop")
            continue
        aligned[obs] = df
        print(f"[JOIN] OBS {obs}: {len(df)} rows (interpolated) | visible={vis_count}")

    if not aligned:
        raise RuntimeError("No observers aligned to OEM epochs.")

    usable_obs = sorted(aligned.keys())
    print(f"[USE ] Observers after occlusion gate: {usable_obs}")
    print(f"[SAVE] Output dir: {OUT_DIR}")
    print(f"[SAVE] Run-stamped dir: {RUN_DIR}")

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
            if tstamp not in df.index:
                continue
            if ("vis" in df.columns) and (not bool(df.at[tstamp, "vis"])):
                continue
            row = df.loc[tstamp]
            r = np.array([row["x_km"], row["y_km"], row["z_km"]], float)
            u = np.array([row["ux"], row["uy"], row["uz"]], float)
            rs.append(r); us.append(u); used_ids.append(obs)

        if len(us) < N_OBSERVERS:
            skipped += 1
            if (k % PROGRESS_EVERY)==0:
                print(f"[PROG] epoch {k+1}/{len(master_index)} | kept={kept} skipped={skipped}")
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
        if (k % PROGRESS_EVERY)==0:
            elapsed = time.time() - t_start
            print(f"[PROG] epoch {k+1}/{len(master_index)} | kept={kept} skipped={skipped} | Nsats={len(us)} | elapsed={elapsed:.1f}s")

    print(f"[GATE] N_OBSERVERS>={N_OBSERVERS} | kept {kept} epochs, skipped {skipped}")

    # ensure stamped dir exists
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values("time")
    if out.empty:
        print("[WARN] No epochs kept after Earth-occlusion gating.")
        return
    out["time"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out_path = RUN_DIR / f"xhat_geo_{target_id}.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK ] Wrote {len(out)} epochs -> {out_path}")


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
        print(f"[RUN ] Looping over {len(run_targets)} targets: {run_targets}")
        for tid in run_targets:
            try:
                run_for_target(tid)
            except Exception as e:
                print(f"[ERR ] Target {tid}: {e}")
    else:
        run_for_target(TARGET_ID)

    # Write a simple manifest for the triangulation run
    try:
        manifest = RUN_DIR / "manifest.txt"
        with manifest.open("w") as mf:
            mf.write(f"triangulation_run_id={RUN_ID}\n")
            mf.write(f"generated_utc={datetime.utcnow().isoformat()}Z\n")
            mf.write(f"nsats={NSATS}\n")
            mf.write(f"alt_km={ALT_KM:.3f}\n")
            mf.write(f"inc_deg={INC_DEG:.3f}\n")
            if LOOP_ALL_TARGETS:
                mf.write("targets=\n")
                for p in sorted(RUN_DIR.glob("xhat_geo_*.csv")):
                    mf.write(f"  - {p.stem.split('xhat_geo_')[-1]}\n")
        print(f"[MAN ] Wrote {manifest}")
    except Exception as e:
        print(f"[WARN] Could not write triangulation manifest: {e}")

if __name__ == "__main__":
    main()