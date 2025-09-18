# filtering/adapter.py
"""
Adapter that turns GPM triangulation CSV outputs into a measurement stream
for the Filtering Module. No new STK runs required.

Inputs (from triangulation):
  exports/triangulation/<RUN_ID>/xhat_geo_HGV_XXXXX.csv

Each CSV is expected to contain columns (as produced by triangulate_icrf):
  time (UTC string), xhat_x_km, xhat_y_km, xhat_z_km,
  Sigma_xx, Sigma_yy, Sigma_zz, Sigma_xy, Sigma_xz, Sigma_yz,
  CEP50_km_analytic, Nsats, beta_min_deg, beta_mean_deg, condA, obs_used_csv, ...

Outputs:
- find_run_id(): detect RUN_ID (env or newest folder)
- find_tri_csvs(run_id): map target_id -> triangulation CSV Path
- load_measurements(csv_path): list of dicts {t,z,R,meta}
- load_truth_oem(oem_path): optional helper to read OEM truth into a DataFrame

All units remain km, km^2, UTC timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# --- Config-driven paths and logging ---
from config.loader import load_config

CFG = load_config()
from pathlib import Path as _P
PROJECT_ROOT = _P(CFG["project"]["root"]).resolve()

# Resolve paths from YAML (absolute if needed)
_DEF = CFG["paths"]
TRI_BASE_DIR = (_P(_DEF["triangulation_out"]) if _P(_DEF["triangulation_out"]).is_absolute() else (PROJECT_ROOT / _DEF["triangulation_out"]) ).resolve()
OEM_DIR       = (_P(_DEF["oem_root"])          if _P(_DEF["oem_root"]).is_absolute()          else (PROJECT_ROOT / _DEF["oem_root"]) ).resolve()

# Logging gate (INFO default)
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)

# -----------------------------
# Run discovery
# -----------------------------

def find_run_id() -> str:
    """Return RUN_ID from env or detect the newest run folder under TRI_BASE_DIR."""
    env_id = os.environ.get("RUN_ID")
    if env_id:
        run_dir = TRI_BASE_DIR / env_id
        if run_dir.is_dir():
            return env_id
        # If env points to a non-existing folder, fall back to detection
        _log("WARN", f"[ADPT] RUN_ID={env_id} not found under {TRI_BASE_DIR}; falling back to latest run")
    candidates = [p for p in TRI_BASE_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found in {TRI_BASE_DIR}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def find_tri_csvs(run_id: str) -> Dict[str, Path]:
    """Return mapping target_id -> triangulation CSV path for the given RUN_ID.

    Expects files like: exports/triangulation/<RUN_ID>/xhat_geo_HGV_00010.csv
    """
    run_dir = TRI_BASE_DIR / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out: Dict[str, Path] = {}
    for p in sorted(run_dir.glob("xhat_geo_*.csv")):
        m = re.search(r"(HGV_\d+)", p.name)
        if not m:
            continue
        tgt = m.group(1)
        out[tgt] = p
    if not out:
        raise FileNotFoundError(f"No triangulation CSVs in {run_dir}")
    _log("INFO", f"[ADPT] Found {len(out)} triangulation files for run {run_id}")
    return out

# -----------------------------
# Data containers
# -----------------------------

@dataclass
class Measurement:
    t: pd.Timestamp               # UTC-aware
    z_km: np.ndarray              # (3,)
    R_km2: np.ndarray             # (3,3)
    meta: dict                    # free-form (Nsats, beta, condA, obs_used_csv, etc.)

# -----------------------------
# OEM truth reader (optional)
# -----------------------------

def load_truth_oem(oem_path: Path) -> pd.DataFrame:
    """Read CCSDS OEM into DataFrame indexed by UTC time.
    Columns: x_km, y_km, z_km, vx_kmps, vy_kmps, vz_kmps
    """
    if not oem_path.exists():
        raise FileNotFoundError(f"OEM not found: {oem_path}")
    recs: List[Tuple[pd.Timestamp, float, float, float, float, float, float]] = []
    with oem_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("META"):
                continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x,y,z,vx,vy,vz = map(float, parts[1:7])
                except Exception:
                    continue
                recs.append((t,x,y,z,vx,vy,vz))
    if not recs:
        raise RuntimeError(f"No usable state lines in OEM {oem_path}")
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"])\
           .dropna(subset=["time"]).sort_values("time")
    return df.set_index("time")

# -----------------------------
# Triangulation → measurements
# -----------------------------

_REQUIRED_SIGMA_COLS = [
    "Sigma_xx", "Sigma_yy", "Sigma_zz", "Sigma_xy", "Sigma_xz", "Sigma_yz"
]

def _read_triangulation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    # enforce presence of required columns
    missing = [c for c in [
        "xhat_x_km","xhat_y_km","xhat_z_km", *_REQUIRED_SIGMA_COLS
    ] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _row_to_measurement(row: pd.Series) -> Measurement:
    z = np.array([row["xhat_x_km"], row["xhat_y_km"], row["xhat_z_km"]], dtype=float)
    Sxx = float(row["Sigma_xx"]); Syy = float(row["Sigma_yy"]); Szz = float(row["Sigma_zz"])
    Sxy = float(row["Sigma_xy"]); Sxz = float(row["Sigma_xz"]); Syz = float(row["Sigma_yz"])
    R = np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]], dtype=float)
    # Symmetrize and fix tiny negatives (num noise)
    R = 0.5 * (R + R.T)
    # If any eigenvalues are negative tiny, nudge
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 0.0)
    R = (V * w) @ V.T
    wanted_keys = [
        "Nsats","beta_min_deg","beta_mean_deg","beta_max_deg","condA",
        "CEP50_km_analytic","CEP50_km_MC","obs_used_csv"
    ]
    present_keys = [c for c in wanted_keys if c in row.index]
    meta = {k: row[k] for k in present_keys}
    # Derived flag (non-destructive): conservative bad-geometry marker
    bad_geom = False
    condA = float(meta.get("condA", np.nan))
    beta_min = float(meta.get("beta_min_deg", np.nan))
    ns = int(meta.get("Nsats", 0)) if pd.notna(meta.get("Nsats", np.nan)) else 0
    if (not np.isnan(condA) and condA > 1e8) or (not np.isnan(beta_min) and beta_min < 10) or ns < 2:
        bad_geom = True
    meta["bad_geom"] = bad_geom
    # Ensure UTC-aware timestamp without conflicting tz args
    tval = pd.Timestamp(row["time"])  # may already be tz-aware
    if tval.tzinfo is None:
        tval = tval.tz_localize("UTC")
    else:
        tval = tval.tz_convert("UTC")
    return Measurement(t=tval, z_km=z, R_km2=R, meta=meta)


def load_measurements(tri_csv: Path) -> List[Measurement]:
    """Convert a triangulation CSV to a list of Measurement objects (ICRF).

    The adapter does not drop bad epochs; it tags them via meta['bad_geom'].
    """
    df = _read_triangulation_csv(tri_csv)
    out: List[Measurement] = []
    for _, row in df.iterrows():
        out.append(_row_to_measurement(row))
    # quick smoke print
    if out:
        _log("INFO", f"[ADPT] {tri_csv.name}: {len(out)} measurements | t0={out[0].t} tN={out[-1].t} | Nsats0={out[0].meta.get('Nsats')} bad_geom0={out[0].meta.get('bad_geom')}")
    return out

# -----------------------------
# Convenience: load all targets for a run
# -----------------------------

def load_all_targets(run_id: Optional[str]=None) -> Dict[str, List[Measurement]]:
    rid = run_id or find_run_id()
    tri = find_tri_csvs(rid)
    by_target: Dict[str, List[Measurement]] = {}
    for tgt, path in tri.items():
        by_target[tgt] = load_measurements(path)
    return by_target

# -----------------------------
# Debug entrypoint
# -----------------------------
if __name__ == "__main__":
    rid = find_run_id()
    tri_map = find_tri_csvs(rid)
    # just load the first
    tgt, p = next(iter(tri_map.items()))
    _log("INFO", f"[ADPT] Loading {tgt} from {p}")
    ms = load_measurements(p)
    # print first 3
    for m in ms[:3]:
        _log("DEBUG", f"{m.t} {m.z_km} diagR={np.diag(m.R_km2)} Nsats={m.meta.get('Nsats')} bad={m.meta.get('bad_geom')}")
