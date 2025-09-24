# geometry/triangulate_icrf.py
# Clean triangulator — ICRF
#
# What it does (no STK LOS anywhere):
# 1) Reads observer ephemerides (ICRF) and target OEM truth (ICRF).
# 2) Aligns both to the OEM epochs and computes OUR LOS = (x_true − r_obs)/||⋅||.
# 3) Gates Earth occlusion (ray from obs to TRUE target).
# 4) If >= N_OBSERVERS visible at an epoch, solves LS intersection, computes Σ_geo, CEP50.
# 5) Computes baseline angles β across the used observers (min / max / mean + label of extreme pairs).
# 6) Exports per‑observer LOS (aligned, with visibility) under exports/geometry/los/<RUN_ID>/<TARGET>/.
# 7) Writes triangulated series under exports/triangulation/<RUN_ID>/xhat_geo_<TARGET>.csv
#
# Units: km, km/s, radians, UTC timestamps (OEM is the master clock).

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations

# --------------------------
# Config
# --------------------------
from config.loader import load_config
CFG = load_config()
ROOT = Path(CFG["project"]["root"]).resolve()

EPH_DIR       = (ROOT / CFG["paths"]["ephem_root"]).resolve()                      # observers ephemerides (ICRF)
OEM_DIR       = (ROOT / CFG["paths"]["oem_root"]).resolve()                        # target truth OEM
LOS_OUT_ROOT  = (ROOT / "exports" / "geometry" / "los").resolve()                  # our LOS export
TRI_OUT_ROOT  = (ROOT / CFG["paths"]["triangulation_out"]).resolve()               # triangulation export
RUN_ID        = CFG["project"]["run_id"]
TRI_RUN_DIR   = (TRI_OUT_ROOT / RUN_ID)

TRI_RUN_DIR.mkdir(parents=True, exist_ok=True)

# knobs
R_EARTH_KM   = float(CFG["geometry"]["earth_radius_km"])  # km
N_OBS_MIN    = int(CFG["geometry"]["min_observers"])      # >= 2
SIGMA_LOS    = float(CFG["gpm_measurement"]["los_noise_rad"])  # rad

# reproducible noise for LOS perturbations used in triangulation
NOISE_SEED   = int(CFG.get("geometry", {}).get("los_noise_seed", 12345))
_rng = np.random.default_rng(NOISE_SEED)

def _perturb_los(u: np.ndarray, sigma_rad: float) -> np.ndarray:
    """
    Apply a small random rotation to unit vector u with std dev sigma_rad (radians).
    The rotation axis is random and orthogonal to u.
    """
    if sigma_rad <= 0:
        return u
    v = _rng.normal(size=3)
    v = v - (v @ u) * u
    n = np.linalg.norm(v)
    if n == 0:
        return u
    w = v / n  # unit orthogonal direction
    # rotate u by angle alpha around axis w (in plane span{u,w})
    alpha = _rng.normal(scale=sigma_rad)
    u_rot = (np.cos(alpha) * u) + (np.sin(alpha) * w)
    # ensure unit length
    return u_rot / np.linalg.norm(u_rot)

# --------------------------
# Simple log
# --------------------------

def log(msg: str) -> None:
    print(msg)

log(f"[RUN ] Triangulation run id: {RUN_ID}")
log(f"[OUT ] Triangulation directory: {TRI_RUN_DIR}")

# --------------------------
# IO helpers
# --------------------------

def _norm_time_utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC")
    else:
        t = t.dt.tz_localize("UTC")
    return t.dt.round("s")


def read_oem_icrf(oem_path: Path) -> pd.DataFrame:
    """Read CCSDS OEM (ICRF/J2000) — returns df indexed by UTC time with x,y,z [km]."""
    recs = []
    with oem_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("META"):  # skip headers/meta
                continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t  = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x, y, z = map(float, parts[1:4])
                except Exception:
                    continue
                recs.append((t, x, y, z))
    if not recs:
        raise RuntimeError(f"No state lines in OEM {oem_path}")
    df = pd.DataFrame(recs, columns=["time", "x_km", "y_km", "z_km"]).dropna()
    df["time"] = _norm_time_utc(df["time"])  # round to s
    return df.set_index("time").sort_index()


def read_ephem_icrf(ephem_csv: Path) -> pd.DataFrame:
    """Observer ephemeris CSV with at least: time, x_km, y_km, z_km (headers may have units)."""
    df = pd.read_csv(ephem_csv, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names: str) -> str:
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"Missing columns in {ephem_csv}")
    tcol = pick("time", "time (utcg)")
    xcol = pick("x_km", "x (km)")
    ycol = pick("y_km", "y (km)")
    zcol = pick("z_km", "z (km)")
    out = df[[tcol, xcol, ycol, zcol]].rename(columns={tcol: "time", xcol: "x_km", ycol: "y_km", zcol: "z_km"})
    out["time"] = _norm_time_utc(out["time"])  # to UTC index seconds
    return out.dropna().sort_values("time").reset_index(drop=True)


def interp_to(times: pd.DatetimeIndex, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Interpolate df[cols] to given UTC times (only inside support)."""
    base = df.set_index("time").sort_index()
    t0, t1 = base.index.min(), base.index.max()
    use = times[(times >= t0) & (times <= t1)]
    if use.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=cols)
    tmp = base.reindex(base.index.union(use)).sort_index().interpolate(method="time")[cols]
    out = tmp.loc[use]
    out.index.name = "time"
    return out

# --------------------------
# Geometry
# --------------------------

def P_perp(u: np.ndarray) -> np.ndarray:
    return np.eye(3) - np.outer(u, u)


def triangulate_epoch(rs: List[np.ndarray], us: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    A = np.zeros((3, 3)); b = np.zeros(3)
    for r, u in zip(rs, us):
        P = P_perp(u); A += P; b += P @ r
    w = np.linalg.eigvals(A)
    cond = (np.max(np.abs(w)) / max(1e-15, np.min(np.abs(w))))
    xhat = np.linalg.lstsq(A, b, rcond=None)[0]
    return xhat, float(cond)


def covariance_geo(xhat: np.ndarray, rs: List[np.ndarray], us: List[np.ndarray], sigma_los_rad: float) -> np.ndarray:
    G = np.zeros((3, 3))
    for r, u in zip(rs, us):
        rho = np.linalg.norm(xhat - r)
        if rho <= 0:
            continue
        G += (1.0 / (rho * rho)) * P_perp(u)
    if np.linalg.cond(G) > 1e12:
        G += 1e-12 * np.eye(3)
    return (sigma_los_rad ** 2) * np.linalg.inv(G)


def cep50_from_cov2d(S: np.ndarray) -> float:
    Sxy = S[:2, :2]
    sigma_eq = float(np.sqrt(0.5 * np.trace(Sxy)))
    return 1.1774 * sigma_eq


def earth_blocks_ray(r_obs: np.ndarray, x_true: np.ndarray, R: float = R_EARTH_KM) -> bool:
    d = x_true - r_obs
    rho = float(np.linalg.norm(d))
    if rho <= 0:
        return True
    u = d / rho
    rs2 = float(r_obs.dot(r_obs))
    proj = float(r_obs.dot(u))
    lam = -proj
    if lam <= 0 or lam >= rho:
        return False
    dmin2 = rs2 - proj * proj
    return dmin2 <= R * R


def beta_angles_deg(us: List[np.ndarray], ids: List[str]) -> Tuple[float, float, float, str]:
    if len(us) < 2:
        return (np.nan, np.nan, np.nan, "")
    pairs = []
    for i, j in combinations(range(len(us)), 2):
        c = float(np.clip(us[i].dot(us[j]), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))
        pairs.append((ang, ids[i], ids[j]))
    betas = np.array([p[0] for p in pairs])
    bmin, bmax, bmean = float(betas.min()), float(betas.max()), float(betas.mean())
    # label only the extremes to keep csv compact
    i_min = int(np.argmin(betas)); i_max = int(np.argmax(betas))
    lab = f"min:{pairs[i_min][1]}-{pairs[i_min][2]}={pairs[i_min][0]:.2f}|max:{pairs[i_max][1]}-{pairs[i_max][2]}={pairs[i_max][0]:.2f}"
    return bmin, bmax, bmean, lab

# --------------------------
# Core per-target
# --------------------------

def run_for_target(target_id: str) -> None:
    log(f"[TGT ] {target_id}")
    oem_path = OEM_DIR / f"{target_id}.oem"
    truth = read_oem_icrf(oem_path)  # index=time UTC
    times = truth.index

    # discover observers (by ephemeris files OBS_*.csv)
    obs_files = sorted(EPH_DIR.glob("OBS_*_ephem.csv"))
    obs_ids = [p.stem.split("OBS_")[-1].split("_ephem")[0] for p in obs_files]
    if not obs_ids:
        raise FileNotFoundError(f"No observer ephemerides in {EPH_DIR}")

    # align and build LOS per observer
    aligned: Dict[str, pd.DataFrame] = {}
    for oid, f in zip(obs_ids, obs_files):
        eph = read_ephem_icrf(f)
        E = interp_to(times, eph, ["x_km", "y_km", "z_km"]).reindex(times)
        # compute LOS towards true target
        X = truth.loc[E.index, ["x_km", "y_km", "z_km"]].to_numpy(float)
        R = E[["x_km", "y_km", "z_km"]].to_numpy(float)
        d = X - R
        n = np.linalg.norm(d, axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            U = np.divide(d, n, out=np.zeros_like(d), where=(n > 0))
        # visibility gate (Earth occlusion)
        vis = np.array([not earth_blocks_ray(r, x) for r, x in zip(R, X)], dtype=bool)
        df = pd.DataFrame({
            "x_km": R[:, 0], "y_km": R[:, 1], "z_km": R[:, 2],
            "ux": U[:, 0], "uy": U[:, 1], "uz": U[:, 2],
            "vis": vis,
        }, index=times)
        df.index.name = "time"
        aligned[oid] = df

    # export OUR LOS (per observer + long-form)
    los_dir = (LOS_OUT_ROOT / RUN_ID / target_id)
    los_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for oid, df in aligned.items():
        out = df.reset_index()[["time", "ux", "uy", "uz", "vis"]].copy()
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (los_dir / f"LOS_OBS_{oid}_to_{target_id}_icrf.csv").write_text(out.to_csv(index=False))
        long_rows.append(out.assign(observer=oid))
    if long_rows:
        long_df = pd.concat(long_rows, ignore_index=True)
        long_df = long_df[["time", "observer", "ux", "uy", "uz", "vis"]]
        (los_dir / f"LOS_ALL_to_{target_id}_icrf.csv").write_text(long_df.to_csv(index=False))
    log(f"[SAVE] LOS → {los_dir}")

    # triangulation loop on OEM epochs
    rows = []
    for k, t in enumerate(times):
        rs: List[np.ndarray] = []
        us: List[np.ndarray] = []
        ids: List[str] = []
        for oid, df in aligned.items():
            row = df.iloc[k]
            if not bool(row["vis"]):
                continue
            r = np.array([row["x_km"], row["y_km"], row["z_km"]], float)
            u = np.array([row["ux"], row["uy"], row["uz"]], float)
            rs.append(r); us.append(u); ids.append(oid)
        if len(us) < N_OBS_MIN:
            continue

        # --- Inject angular noise on LOS before solving (simulates measurement error) ---
        us = [_perturb_los(u, SIGMA_LOS) for u in us]

        # solve with noisy LOS
        xhat, condA = triangulate_epoch(rs, us)
        # enforce directions (towards solution)
        fixed = []
        flips = 0
        for r, u in zip(rs, us):
            lam = float(u.dot(xhat - r))
            if lam < 0:
                u = -u; flips += 1
            fixed.append(u)
        if flips:
            xhat, condA = triangulate_epoch(rs, fixed)
            us = fixed
        S = covariance_geo(xhat, rs, us, SIGMA_LOS)
        cep = cep50_from_cov2d(S)
        bmin, bmax, bmean, blab = beta_angles_deg(us, ids)
        xt = truth.loc[t, ["x_km", "y_km", "z_km"]].to_numpy(float)
        e = xhat - xt
        rows.append({
            "time": pd.Timestamp(t),
            "xhat_x_km": xhat[0], "xhat_y_km": xhat[1], "xhat_z_km": xhat[2],
            "Sigma_xx": S[0, 0], "Sigma_yy": S[1, 1], "Sigma_zz": S[2, 2],
            "Sigma_xy": S[0, 1], "Sigma_xz": S[0, 2], "Sigma_yz": S[1, 2],
            "CEP50_km_analytic": cep,
            "condA": condA,
            "Nsats": len(us),
            "obs_used_csv": ",".join(ids),
            "beta_min_deg": bmin, "beta_max_deg": bmax, "beta_mean_deg": bmean,
            "beta_pairs_deg": blab,
            "err_x_km": float(e[0]), "err_y_km": float(e[1]), "err_z_km": float(e[2]),
            "err_norm_km": float(np.linalg.norm(e)),
        })

    out = pd.DataFrame(rows).sort_values("time")
    if out.empty:
        log("[WARN] No epochs kept (visibility gate)")
        return
    ts = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out["time"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    TRI_RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRI_RUN_DIR / f"xhat_geo_{target_id}.csv"
    out.to_csv(out_path, index=False)
    log(f"[OK ] Wrote {len(out)} epochs → {out_path}")


# --------------------------
# Main
# --------------------------

def main() -> None:
    # choose targets by available OEM
    oems = sorted(OEM_DIR.glob("HGV_*.oem"))
    if not oems:
        raise FileNotFoundError(f"No OEM files in {OEM_DIR}")
    log(f"[RUN ] Looping over {len(oems)} targets | observers={len(list(EPH_DIR.glob('OBS_*_ephem.csv')))}")
    for p in oems:
        try:
            run_for_target(p.stem)
        except Exception as e:
            log(f"[ERR ] Target {p.stem}: {e}")
    # manifest (lightweight)
    man = TRI_RUN_DIR / "manifest.txt"
    man.write_text(
        "\n".join([
            f"triangulation_run_id={RUN_ID}",
            f"generated_utc={datetime.utcnow().isoformat()}Z",
            f"earth_radius_km={R_EARTH_KM}",
            f"min_observers={N_OBS_MIN}",
            f"sigma_los_rad={SIGMA_LOS}",
        ])
    )
    log(f"[MAN ] Wrote {man}")


if __name__ == "__main__":
    main()
