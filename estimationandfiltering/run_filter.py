"""
Orchestrator for Filtering Module (forward pass).
- Detects RUN_ID (ENV or newest triangulation run via adapter)
- Loads ALL targets' measurements from triangulation (ICRF)
- Runs a forward KF (NCA 9-state) and prints minimal diagnostics
- Writes CSVs with ICRF [r,v,a] to exports/tracks/<RUN_ID>/HGV_xxxxx_track_icrf_forward.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Robust imports whether launched as module or script
try:
    from estimationandfiltering.adapter import find_run_id, find_tri_csvs, load_measurements
    from estimationandfiltering.kf_cv_nca import run_forward
    from estimationandfiltering.models import JERK_PSD
    from estimationandfiltering.frames import icrf_to_ecef
except Exception:
    import sys, pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from estimationandfiltering.adapter import find_run_id, find_tri_csvs, load_measurements
    from estimationandfiltering.kf_cv_nca import run_forward
    from estimationandfiltering.models import JERK_PSD
    from estimationandfiltering.frames import icrf_to_ecef

# --- Config-driven paths and logging ---
from config.loader import load_config
CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()

# Resolve output dir from YAML
_paths = CFG["paths"]
TRACK_BASE_DIR = (Path(_paths["tracks_out"]) if Path(_paths["tracks_out"]).is_absolute() else (PROJECT_ROOT / _paths["tracks_out"]) ).resolve()

G0_KMPS2 = 0.00980665  # 1 g in km/s^2

# Logging gate
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    # RUN_ID: prefer ENV; if invalid/nonexistent, fall back to latest triangulation run
    run_id = os.environ.get("RUN_ID") or find_run_id()
    try:
        tri_map = find_tri_csvs(run_id)  # dict[target_id] = csv_path
    except FileNotFoundError:
        # fallback: adopt latest triangulation run and retry once
        alt_run = find_run_id()
        if alt_run != run_id:
            run_id = alt_run
            os.environ["RUN_ID"] = run_id
            tri_map = find_tri_csvs(run_id)
        else:
            raise

    _log("INFO", f"[RUN ] RUN_ID={run_id}")
    _log("INFO", f"[TGT ] targets: {list(tri_map.keys())}")

    # Save dir for tracks
    out_dir = TRACK_BASE_DIR / run_id
    _ensure_dir(out_dir)

    # Optional truncation for quick smoke tests
    max_epochs_env = os.environ.get("MAX_EPOCHS")
    kmax = int(max_epochs_env) if max_epochs_env else None

    for tgt, tri_csv in tri_map.items():
        _log("INFO", f"[TGT ] {tgt} <- {tri_csv.name}")
        meas = load_measurements(tri_csv)
        if kmax:
            meas = meas[:kmax]
            _log("INFO", f"[CFG ] Truncated to first {kmax} epochs via MAX_EPOCHS")

        # Forward KF
        hist = run_forward(meas_list=meas, qj=JERK_PSD, P0_scale=1.0)

        # Summaries
        upd = sum(1 for h in hist if h.get("did_update", False))
        tot = len(hist)
        _log("INFO", f"[SUM ] {tgt}: updates={upd}/{tot}")
        nis_head = [h.get("nis", np.nan) for h in hist[:10]]
        _log("DEBUG", f"[NIS ] head: {np.array(nis_head)}")

        # Save forward ICRF track
        out_path = out_dir / f"{tgt}_track_icrf_forward.csv"
        rows = []
        for h in hist:
            t = pd.Timestamp(h["t"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            x = h["x"]
            meta = h.get("meta", {})
            rows.append({
                "time": t,
                "x_km": x[0], "y_km": x[1], "z_km": x[2],
                "vx_kmps": x[3], "vy_kmps": x[4], "vz_kmps": x[5],
                "ax_kmps2": x[6], "ay_kmps2": x[7], "az_kmps2": x[8],
                "ax_g": x[6] / G0_KMPS2, "ay_g": x[7] / G0_KMPS2, "az_g": x[8] / G0_KMPS2,
                "nis": h.get("nis", np.nan),
                "did_update": bool(h.get("did_update", False)),
                "R_inflation": h.get("R_inflation", np.nan),
                "Nsats": meta.get("Nsats", np.nan),
                "beta_min_deg": meta.get("beta_min_deg", np.nan),
                "beta_mean_deg": meta.get("beta_mean_deg", np.nan),
                "condA": meta.get("condA", np.nan),
            })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        _log("INFO", f"[SAVE] {tgt}: wrote forward ICRF track -> {out_path}")

        # Save forward ECEF track (same timestamps)
        df_icrf = pd.DataFrame(rows)
        times = df_icrf['time']
        r_i = df_icrf[['x_km','y_km','z_km']].to_numpy(float)
        v_i = df_icrf[['vx_kmps','vy_kmps','vz_kmps']].to_numpy(float)
        a_i = df_icrf[['ax_kmps2','ay_kmps2','az_kmps2']].to_numpy(float)
        r_e, v_e, a_e = icrf_to_ecef(times, r_i, v_i, a_i)
        df_ecef = pd.DataFrame({
            'time': times,
            'x_km': r_e[:,0], 'y_km': r_e[:,1], 'z_km': r_e[:,2],
            'vx_kmps': v_e[:,0], 'vy_kmps': v_e[:,1], 'vz_kmps': v_e[:,2],
            'ax_kmps2': a_e[:,0], 'ay_kmps2': a_e[:,1], 'az_kmps2': a_e[:,2],
            'ax_g': a_e[:,0] / G0_KMPS2, 'ay_g': a_e[:,1] / G0_KMPS2, 'az_g': a_e[:,2] / G0_KMPS2,
            'nis': df_icrf['nis'],
            'did_update': df_icrf['did_update'],
            'R_inflation': df_icrf['R_inflation'],
            'Nsats': df_icrf['Nsats'],
            'beta_min_deg': df_icrf['beta_min_deg'],
            'beta_mean_deg': df_icrf['beta_mean_deg'],
            'condA': df_icrf['condA'],
        })
        out_ecef = out_dir / f"{tgt}_track_ecef_forward.csv"
        df_ecef.to_csv(out_ecef, index=False)
        _log("INFO", f"[SAVE] {tgt}: wrote forward ECEF track -> {out_ecef}")


if __name__ == "__main__":
    main()