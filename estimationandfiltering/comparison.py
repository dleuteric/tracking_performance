# estimationandfiltering/compare_estimators.py
# One-click comparator:
#  - (optionally) generate LEGACY (CV-KF) and EW-RLS tracks for a given RUN_ID by calling the orchestrator
#  - compare A (legacy) vs B (EW-RLS), optionally vs truth, and save RMSE/CEP metrics
#
# Usage (Run in PyCharm, no shell gymnastics):
#   - Set RUN_ID in args or leave blank to auto-detect latest triangulation run
#   - Choose --generate both|legacy|ewrls|none (default both)
#
# Outputs: CSV with per-target metrics, and pretty print summary.

from __future__ import annotations

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ==============================
# Path helpers & config loading
# ==============================
def project_root() -> Path:
    # <repo_root>/estimationandfiltering/compare_estimators.py -> <repo_root>
    return Path(__file__).resolve().parents[1]

def load_cfg():
    # import config.loader with safe pathing
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from config.loader import load_config
    return load_config()

CFG = load_cfg()
ROOT = project_root()
PATHS = CFG["paths"]
TRI_DIR        = (ROOT / PATHS["triangulation_out"]).resolve()
TRACKS_DIR     = (ROOT / PATHS["tracks_out"]).resolve()
TRUTH_DIR      = (ROOT / PATHS.get("truth_out", "exports/truth")).resolve()  # optional

# ==============================
# I/O utilities
# ==============================
def _latest_run_id(base_dir: Path) -> str:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run folders in {base_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name

def _tri_run_has_csvs(run_id: str) -> bool:
    run_dir = TRI_DIR / run_id
    return run_dir.is_dir() and any(run_dir.glob("xhat_geo_*.csv"))

# ==============================
# Orchestrator runners (no shell)
# ==============================
def _run_orchestrator(run_id: str, use_ewrls: bool, generate_only_estimation: bool = True, continue_on_error: bool = True) -> None:
    """
    Call orchestrator/main_pipeline.py with ENV vars, from Python.
    - use_ewrls=True  -> EW-RLS estimator
      use_ewrls=False -> legacy CV-KF forward filter
    - generate_only_estimation=True -> skip heavy steps (triangulation/plots) and reuse existing RUN_ID
    """
    main_py = ROOT / "orchestrator" / "main_pipeline.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Cannot find orchestrator at {main_py}")

    env = os.environ.copy()
    # shared
    env["RUN_ID"] = str(run_id)
    env["USE_EWRLS"] = "1" if use_ewrls else "0"
    env["RUN_FWD"] = "1"  # run estimation
    env["CONT_ERR"] = "1" if continue_on_error else "0"

    if generate_only_estimation:
        # Skip tri + plots; reuse triangulation outputs for the given RUN_ID
        env["RUN_TRI"]  = "0"
        env["RUN_GPLT"] = "0"
        env["RUN_FPLT"] = "0"
        env["REUSE_TRI"] = "1"
    # else: respect YAML for a full run (rare here)

    # call Python on the orchestrator
    cmd = [sys.executable, str(main_py)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        # bubble up a compact error
        raise RuntimeError(f"Orchestrator failed (USE_EWRLS={use_ewrls}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

# ==============================
# Track readers and comparators
# ==============================
def _read_track(csv_path: str):
    df = pd.read_csv(csv_path)
    # time
    t = None
    for c in ("t_s","time_s","epoch_s","time","t","epoch"):
        if c in df.columns:
            # if iso strings, convert to seconds
            if df[c].dtype == object:
                t = pd.to_datetime(df[c], utc=True, errors="coerce").astype("int64") / 1e9
            else:
                t = df[c].astype(float).to_numpy()
            break
    if t is None:
        raise ValueError(f"No time column found in {csv_path}")

    # position (prefer *_icrf_km, fallback to x_km)
    def pick3(prefix):
        cands = [f"{prefix}_icrf_km", f"{prefix}_ecef_km", f"{prefix}_km", prefix]
        for cand in cands:
            if cand in df.columns: return df[cand].astype(float).to_numpy()
        return None

    x = pick3("x"); y = pick3("y"); z = pick3("z")
    if x is None or y is None or z is None:
        raise ValueError(f"Missing position columns in {csv_path}")

    # velocity optional
    def pickv(prefix):
        cands = [f"v{prefix}_icrf_kms", f"v{prefix}_ecef_kms", f"v{prefix}_kms", f"{prefix}_kms"]
        for cand in cands:
            if cand in df.columns: return df[cand].astype(float).to_numpy()
        return None
    vx, vy, vz = pickv("x"), pickv("y"), pickv("z")

    return dict(t=t, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

def _align(a, b):
    # align on common timestamps (no interpolation for simplicity)
    ta, tb = a["t"], b["t"]
    ra = np.round(ta, 6); rb = np.round(tb, 6)
    common = np.intersect1d(ra, rb)
    if common.size == 0:
        raise ValueError("No common timestamps between tracks.")
    def sel(d, r):
        mask = np.isin(np.round(d["t"], 6), r)
        return {k: v[mask] if hasattr(v, "__len__") else v for k, v in d.items()}
    return sel(a, common), sel(b, common)

def _errors(est, ref):
    ex = est["x"] - ref["x"]
    ey = est["y"] - ref["y"]
    ez = est["z"] - ref["z"]
    eH = np.sqrt(ex**2 + ey**2)
    e3 = np.sqrt(ex**2 + ey**2 + ez**2)
    return ex, ey, ez, eH, e3

def _rmse(arr):
    return float(np.sqrt(np.mean(arr**2))) if arr.size else np.nan

def _cep(percent, eH):
    if eH.size == 0: return np.nan
    return float(np.percentile(eH, percent))

def compare_pair(track_a_csv: str, track_b_csv: str, truth_csv: str | None = None):
    """Return metrics dict comparing A vs B; if truth provided, errors are to truth.
       Units: km for position errors; CEP in km.
    """
    A = _read_track(track_a_csv)
    B = _read_track(track_b_csv)

    if truth_csv:
        T = _read_track(truth_csv)
        A, T = _align(A, T)
        B, T = _align(B, T)
        exA, eyA, ezA, eHA, e3A = _errors(A, T)
        exB, eyB, ezB, eHB, e3B = _errors(B, T)
    else:
        A, B = _align(A, B)
        exA, eyA, ezA, eHA, e3A = _errors(A, B)     # A relative to B
        exB, eyB, ezB, eHB, e3B = _errors(B, A)     # B relative to A (negated)

    return {
        "N": A["x"].size,
        "A_RMSE_x_km": _rmse(exA), "A_RMSE_y_km": _rmse(eyA), "A_RMSE_z_km": _rmse(ezA),
        "A_RMSE_3D_km": _rmse(e3A), "A_CEP50_km": _cep(50, eHA), "A_CEP90_km": _cep(90, eHA),
        "B_RMSE_x_km": _rmse(exB), "B_RMSE_y_km": _rmse(eyB), "B_RMSE_z_km": _rmse(ezB),
        "B_RMSE_3D_km": _rmse(e3B), "B_CEP50_km": _cep(50, eHB), "B_CEP90_km": _cep(90, eHB),
    }

def compare_run(run_id: str, tracks_dir: str, legacy_glob="*_track_icrf_forward.csv",
                ewrls_glob="ewrls_tracks_*.csv", truth_dir: str | None = None, out_csv: str | None = None):
    """Loop all targets in RUN_ID and write a summary CSV."""
    run_tracks = Path(tracks_dir) / run_id
    A_files = sorted(run_tracks.glob(legacy_glob))   # legacy KF (A)
    B_files = sorted(run_tracks.glob(ewrls_glob))    # EW-RLS (B)
    # map by target id (suffix)
    def key(p): return p.stem.split("_")[-1]  # HGV_00001 from *_HGV_00001
    A_map = {key(p): p for p in A_files}
    B_map = {p.stem.replace("ewrls_tracks_",""): p for p in B_files}
    targets = sorted(set(A_map) & set(B_map))
    rows = []
    for tgt in targets:
        truth_csv = None
        if truth_dir:
            cand = Path(truth_dir) / run_id / f"{tgt}_truth.csv"
            truth_csv = str(cand) if cand.exists() else None
        m = compare_pair(str(A_map[tgt]), str(B_map[tgt]), truth_csv)
        m["run_id"] = run_id; m["target"] = tgt
        m["A_file"] = A_map[tgt].name; m["B_file"] = B_map[tgt].name
        rows.append(m)
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df

# ==============================
# Driver: generate + compare
# ==============================
def run_from_here(run_id: Optional[str] = None,
                  generate: str = "both",   # both|legacy|ewrls|none
                  tracks_dir: Optional[str] = None,
                  out_csv: Optional[str] = None,
                  truth_dir: Optional[str] = None,
                  quiet: bool = False):
    """
    End-to-end: ensure tracks exist (optionally generate) then compare.
    """
    if run_id is None:
        run_id = _latest_run_id(TRI_DIR)
    if not _tri_run_has_csvs(run_id):
        raise FileNotFoundError(f"No triangulation CSVs for RUN_ID '{run_id}' in {TRI_DIR/run_id}")

    # optional generation
    if generate in ("legacy", "both"):
        _run_orchestrator(run_id, use_ewrls=False, generate_only_estimation=True, continue_on_error=True)
    if generate in ("ewrls", "both"):
        _run_orchestrator(run_id, use_ewrls=True,  generate_only_estimation=True, continue_on_error=True)

    # compare
    tracks_base = tracks_dir or str(TRACKS_DIR)
    truth_base  = truth_dir  or (str(TRUTH_DIR) if TRUTH_DIR.exists() else None)
    out_path    = out_csv or (ROOT / "exports" / "metrics" / f"ewrls_vs_legacy_{run_id}.csv")
    df = compare_run(
        run_id=run_id,
        tracks_dir=tracks_base,
        legacy_glob="*_track_icrf_forward.csv",
        ewrls_glob="ewrls_tracks_*.csv",
        truth_dir=truth_base,
        out_csv=str(out_path)
    )
    if not quiet:
        # human summary
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(df)
            print("\nSummary:\n", df.describe(include="all"))
        print(f"\n[OK] Saved metrics → {out_path}")
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate (optional) and compare Legacy KF vs EW-RLS tracks.")
    ap.add_argument("--run_id", default=None, help="RUN_ID to use; if omitted, latest triangulation run is used.")
    ap.add_argument("--generate", default="both", choices=["both","legacy","ewrls","none"], help="What to generate before comparing.")
    ap.add_argument("--tracks_dir", default=None, help="Tracks base dir; defaults to CFG paths.tracks_out")
    ap.add_argument("--truth_dir", default=None, help="Truth base dir (optional).")
    ap.add_argument("--out_csv", default=None, help="Where to save metrics CSV; default exports/metrics/ewrls_vs_legacy_<RUN_ID>.csv")
    ap.add_argument("--quiet", action="store_true", help="Suppress printing tables to console.")
    args = ap.parse_args()

    run_from_here(
        run_id=args.run_id,
        generate=args.generate,
        tracks_dir=args.tracks_dir,
        out_csv=args.out_csv,
        truth_dir=args.truth_dir,
        quiet=args.quiet
    )