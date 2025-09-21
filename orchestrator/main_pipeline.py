# File: orchestrator/main_pipeline.py
"""
Config-driven end-to-end orchestrator for ez-SMAD.

Steps: Triangulation → Geometry plots → Estimation (EW-RLS default / legacy CV-KF)
       → Filter plots → Interactive 3D (optional)
"""
from __future__ import annotations

import os, sys, json, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# --------------- Config loader ---------------
try:
    from config.loader import load_config
except Exception:
    import pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config

CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()

# Paths
_paths = CFG["paths"]
TRI_DIR        = (PROJECT_ROOT / _paths["triangulation_out"]).resolve()
PLOTS_GEOM_DIR = (PROJECT_ROOT / _paths["geom_plots_out"]).resolve()
TRACKS_DIR     = (PROJECT_ROOT / _paths["tracks_out"]).resolve()
PLOTS_FILT_DIR = (PROJECT_ROOT / _paths["filter_plots_out"]).resolve()

# --------------- Logging ---------------
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if extra:
            try:
                msg += f" | {json.dumps(extra, default=str)}"
            except Exception:
                msg += f" | {extra}"
        print(f"[{ts}] [{level:5s}] {msg}")

# --------------- Flags / ENV ---------------
FLAGS = {
    "RUN_TRIANGULATION": True,
    "RUN_GEOM_PLOTS": True,
    "RUN_FILTER_FORWARD": True,     # estimation on by default
    "USE_EWRLS": True,              # EW-RLS replaces legacy CV-KF
    "RUN_FILTER_PLOTS": None,
    "RUN_INTERACTIVE_3D": None,
    "REUSE_TRIANGULATION": None,
    "FORCE_RUN_ID": None,
    "CONTINUE_ON_ERROR": False,
}
_env = os.environ.get
def _env_bool(name: str, default: None | bool = None) -> None | bool:
    v = _env(name)
    if v is None: return default
    v = v.strip().lower()
    if v in ("1","true","yes","on"): return True
    if v in ("0","false","no","off"): return False
    return default

FLAGS["RUN_TRIANGULATION"]  = _env_bool("RUN_TRI",  FLAGS["RUN_TRIANGULATION"])  if FLAGS["RUN_TRIANGULATION"]  is None else FLAGS["RUN_TRIANGULATION"]
FLAGS["RUN_GEOM_PLOTS"]     = _env_bool("RUN_GPLT", FLAGS["RUN_GEOM_PLOTS"])     if FLAGS["RUN_GEOM_PLOTS"]     is None else FLAGS["RUN_GEOM_PLOTS"]
FLAGS["RUN_FILTER_FORWARD"] = _env_bool("RUN_FWD",  FLAGS["RUN_FILTER_FORWARD"]) if FLAGS["RUN_FILTER_FORWARD"] is None else FLAGS["RUN_FILTER_FORWARD"]
FLAGS["USE_EWRLS"]          = _env_bool("USE_EWRLS",FLAGS["USE_EWRLS"])          if FLAGS["USE_EWRLS"]          is None else FLAGS["USE_EWRLS"]
FLAGS["RUN_FILTER_PLOTS"]   = _env_bool("RUN_FPLT", FLAGS["RUN_FILTER_PLOTS"])   if FLAGS["RUN_FILTER_PLOTS"]   is None else FLAGS["RUN_FILTER_PLOTS"]
FLAGS["RUN_INTERACTIVE_3D"] = _env_bool("RUN_I3D",  FLAGS["RUN_INTERACTIVE_3D"]) if FLAGS["RUN_INTERACTIVE_3D"] is None else FLAGS["RUN_INTERACTIVE_3D"]
FLAGS["REUSE_TRIANGULATION"]= _env_bool("REUSE_TRI",FLAGS["REUSE_TRIANGULATION"])if FLAGS["REUSE_TRIANGULATION"]is None else FLAGS["REUSE_TRIANGULATION"]
FLAGS["CONTINUE_ON_ERROR"]  = _env_bool("CONT_ERR", FLAGS["CONTINUE_ON_ERROR"])  if FLAGS["CONTINUE_ON_ERROR"]  is None else FLAGS["CONTINUE_ON_ERROR"]
FLAGS["FORCE_RUN_ID"]       = _env("RUN_ID", FLAGS["FORCE_RUN_ID"]) if FLAGS["FORCE_RUN_ID"] is None else FLAGS["FORCE_RUN_ID"]

# --------------- Helpers ---------------
def _ensure_pkg_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for extra in ("src", "packages"):
        p = PROJECT_ROOT / extra
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

def _validate_directories():
    for d in (TRI_DIR, PLOTS_GEOM_DIR, TRACKS_DIR, PLOTS_FILT_DIR):
        d.mkdir(parents=True, exist_ok=True)
        _log("DEBUG", f"Ensured directory: {d}")

def _latest_run_id(base_dir: Path) -> str:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs: raise FileNotFoundError(f"No run folders in {base_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name

def _tri_run_has_csvs(run_id: str) -> bool:
    run_dir = TRI_DIR / run_id
    if not run_dir.is_dir(): return False
    csvs = list(run_dir.glob("xhat_geo_*.csv"))
    if csvs: _log("DEBUG", f"Found {len(csvs)} triangulation CSVs in {run_id}")
    return bool(csvs)

def _resolve_reusable_run_id(preferred: str | None) -> str:
    if preferred and _tri_run_has_csvs(preferred):
        return preferred
    candidates = [p for p in TRI_DIR.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if any(cand.glob("xhat_geo_*.csv")):
            return cand.name
    raise FileNotFoundError(f"No triangulation CSVs found under {TRI_DIR}")

def _get_run_metadata(run_id: str) -> Dict[str, Any]:
    parts = run_id.split('_')
    if len(parts) >= 5:
        return {"timestamp": parts[0], "n_sats": parts[1], "altitude": parts[2], "inclination": parts[3], "hash": parts[4]}
    return {"run_id": run_id}

# --------------- Steps ---------------
def step_triangulation():
    _ensure_pkg_path()
    _log("DEBUG", f"sys.path[0:3]={sys.path[0:3]}")
    try:
        import geometry.triangulate_icrf as tri
        _log("INFO", "[STEP] Triangulation starting...")
        t0 = datetime.now()
        tri.main()
        _log("INFO", f"[STEP] Triangulation ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Triangulation failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite triangulation error")
            return False
        raise

def step_geom_plots(run_id: str):
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import geometry.plot_suite as plotg
        _log("INFO", "[STEP] Geometry plots starting...", _get_run_metadata(run_id))
        t0 = datetime.now(); plotg.main()
        _log("INFO", f"[STEP] Geometry plots ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Geometry plots failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite geometry plots error")
            return False
        raise

def step_filter_forward(run_id: str):
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.run_filter as rfil
        _log("INFO", "[STEP] Filtering (forward) starting...", _get_run_metadata(run_id))
        t0 = datetime.now(); rfil.main()
        _log("INFO", f"[STEP] Filtering (forward) ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Filtering forward failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite filter error")
            return False
        raise

def step_filter_plots(run_id: str):
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Filter plots starting...", _get_run_metadata(run_id))
        t0 = datetime.now(); pfp.main()
        _log("INFO", f"[STEP] Filter plots ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Filter plots failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite filter plots error")
            return False
        raise

def step_interactive_3d(run_id: str):
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Interactive 3D starting...")
        t0 = datetime.now(); pfp.main()
        _log("INFO", f"[STEP] Interactive 3D ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("WARN", f"[SKIP] Interactive 3D not available: {e}")
        return False

def step_ewrls(run_id: str):
    """Run EW-RLS on ALL triangulated xhat_geo_*.csv for this run."""
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        from estimationandfiltering.ew_rls import run_ewrls_on_csv
        import pandas as pd
        _log("INFO", "[STEP] EW-RLS estimation starting...", _get_run_metadata(run_id))
        t0 = datetime.now()

        tri_dir = TRI_DIR / run_id
        tri_list = sorted(tri_dir.glob("xhat_geo_*.csv"))
        if not tri_list:
            raise FileNotFoundError(f"No triangulation CSVs in {tri_dir}")

        out_dir = (TRACKS_DIR / run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame = str(CFG.get("frames", {}).get("triangulation_frame", "ECEF"))
        theta = float(CFG.get("estimation", {}).get("ewrls_theta", 0.93))
        order = int(CFG.get("estimation", {}).get("ewrls_order", 4))

        n_ok, n_fail = 0, 0
        for tri_csv in tri_list:
            try:
                tgt = tri_csv.stem.replace("xhat_geo_", "")
                out_csv = str(out_dir / f"ewrls_tracks_{tgt}.csv")
                run_ewrls_on_csv(str(tri_csv), out_csv=out_csv, frame=frame, order=order, theta=theta)
                n_ok += 1
                # Write compatibility file expected by plots: *_track_icrf_forward.csv
                comp_csv = out_dir / f"{tgt}_track_icrf_forward.csv"
                try:
                    df_out = pd.read_csv(out_csv)
                    # Ensure 'time' column exists (ISO8601 UTC) for plots expecting it
                    if "time" not in df_out.columns and "t_s" in df_out.columns:
                        try:
                            df_out["time"] = pd.to_datetime(df_out["t_s"], unit="s", utc=True).dt.strftime(
                                "%Y-%m-%dT%H:%M:%S.%fZ")
                        except Exception:
                            pass
                    # If columns are generic (x_km, vx_kms), provide ICRF-aliases for plotting
                    rename_map = {
                        "x_km": "x_icrf_km", "y_km": "y_icrf_km", "z_km": "z_icrf_km",
                        "vx_kms": "vx_icrf_kms", "vy_kms": "vy_icrf_kms", "vz_kms": "vz_icrf_kms",
                    }
                    # Only rename if target columns are absent
                    for src, dst in list(rename_map.items()):
                        if src in df_out.columns and dst not in df_out.columns:
                            df_out[dst] = df_out[src]
                    # Ensure time column is named as plots expect
                    if "t_s" not in df_out.columns:
                        # try common fallbacks
                        for c in ("time_s","t","time","epoch_s","epoch"):
                            if c in df_out.columns:
                                df_out["t_s"] = df_out[c]; break
                    # Mark frame for clarity (do not transform coords here)
                    if "frame" not in df_out.columns:
                        df_out["frame"] = frame
                    df_out.to_csv(comp_csv, index=False)
                except Exception as ce:
                    _log("WARN", f"[EW-RLS] Could not write compatibility file {comp_csv.name}: {ce}")
            except Exception as fe:
                n_fail += 1
                _log("ERROR", f"[EW-RLS] Failed on {tri_csv.name}: {fe}")
                if not FLAGS["CONTINUE_ON_ERROR"]:
                    raise

        _log("INFO", f"[STEP] EW-RLS ✓ wrote {n_ok} file(s), {n_fail} failed, in {(datetime.now()-t0).total_seconds():.1f}s → {out_dir}")
        return n_ok > 0
    except Exception as e:
        _log("ERROR", f"[ERR ] EW-RLS failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite EW-RLS error")
            return False
        raise

# --------------- Main ---------------
def main():
    start = datetime.now()
    _log("INFO", "=" * 60)
    _log("INFO", "PWSA Tracking Layer Pipeline Starting")
    _log("INFO", f"Project root: {PROJECT_ROOT}")
    _log("INFO", "=" * 60)

    _validate_directories()

    run_tri = FLAGS["RUN_TRIANGULATION"]  if FLAGS["RUN_TRIANGULATION"]  is not None else bool(CFG.get("orchestrator", {}).get("run_triangulation", True))
    run_gplt= FLAGS["RUN_GEOM_PLOTS"]     if FLAGS["RUN_GEOM_PLOTS"]     is not None else bool(CFG.get("orchestrator", {}).get("run_geom_plots", True))
    run_fwd = FLAGS["RUN_FILTER_FORWARD"] if FLAGS["RUN_FILTER_FORWARD"] is not None else bool(CFG.get("orchestrator", {}).get("run_filter_forward", False))
    run_fplt= FLAGS["RUN_FILTER_PLOTS"]   if FLAGS["RUN_FILTER_PLOTS"]   is not None else bool(CFG.get("orchestrator", {}).get("run_filter_plots", False))
    run_i3d = FLAGS["RUN_INTERACTIVE_3D"] if FLAGS["RUN_INTERACTIVE_3D"] is not None else bool(CFG.get("orchestrator", {}).get("run_interactive_3d", False))
    reuse_tri = FLAGS["REUSE_TRIANGULATION"] if FLAGS["REUSE_TRIANGULATION"] is not None else True

    _log("INFO", "Pipeline Configuration:")
    _log("INFO", f"  • Triangulation    : {'RUN' if run_tri else 'SKIP/REUSE'}")
    _log("INFO", f"  • Geometry Plots   : {'RUN' if run_gplt else 'SKIP'}")
    _log("INFO", f"  • Filter Forward   : {'RUN' if run_fwd else 'SKIP'}")
    _log("INFO", f"  • Use EW-RLS       : {'YES' if FLAGS['USE_EWRLS'] else 'NO'}")
    _log("INFO", f"  • Filter Plots     : {'RUN' if run_fplt else 'SKIP'}")
    _log("INFO", f"  • Interactive 3D   : {'RUN' if run_i3d else 'SKIP'}")
    _log("INFO", f"  • Continue on Error: {'YES' if FLAGS['CONTINUE_ON_ERROR'] else 'NO'}")

    if run_fplt and run_i3d:
        _log("DEBUG", "[RUN ] interactive 3D handled by filter plots; skipping explicit step")
        run_i3d = False

    run_id = str(CFG["project"].get("run_id")) if CFG["project"].get("run_id") else None
    if FLAGS["FORCE_RUN_ID"]:
        run_id = str(FLAGS["FORCE_RUN_ID"])
        _log("INFO", f"[RUN ] Forced RUN_ID from ENV/flags: {run_id}")

    completed = []

    # 1) Triangulation or reuse
    if run_tri:
        if step_triangulation():
            completed.append("triangulation")
        run_id = _latest_run_id(TRI_DIR)
        os.environ["RUN_ID"] = run_id
        _log("INFO", f"[RUN ] Using RUN_ID: {run_id}")
    else:
        if reuse_tri:
            run_id = _resolve_reusable_run_id(run_id)
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Reusing triangulation RUN_ID: {run_id}")
        else:
            if not run_id:
                raise RuntimeError("RUN_ID undefined and reuse disabled; set RUN_ID or enable triangulation.")
            if not _tri_run_has_csvs(run_id):
                raise FileNotFoundError(f"Triangulation RUN_ID has no CSVs: {TRI_DIR / run_id}")
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Using pre-set RUN_ID: {run_id}")

    # 2) Geometry plots
    if run_gplt and step_geom_plots(run_id):
        completed.append("geom_plots")

    # 3) Estimation
    if run_fwd:
        if FLAGS["USE_EWRLS"]:
            if step_ewrls(run_id):
                completed.append("ewrls_forward")
        else:
            if step_filter_forward(run_id):
                completed.append("filter_forward")

    # 4) Filter plots
    if run_fplt and step_filter_plots(run_id):
        completed.append("filter_plots")

    # 5) Interactive 3D
    if run_i3d and step_interactive_3d(run_id):
        completed.append("interactive_3d")

    # Summary
    total = (datetime.now() - start).total_seconds()
    _log("INFO", "=" * 60)
    _log("INFO", f"[OK  ] Pipeline finished in {total:.1f}s")
    _log("INFO", f"       Completed steps: {', '.join(completed) if completed else 'None'}")
    _log("INFO", f"       RUN_ID: {run_id}")
    _log("INFO", f"       Triangulation: {TRI_DIR / run_id}")
    if "geom_plots" in completed: _log("INFO", f"       Geom plots   : {PLOTS_GEOM_DIR / run_id}")
    if ("filter_forward" in completed) or ("ewrls_forward" in completed):
        _log("INFO", f"       Tracks       : {TRACKS_DIR / run_id}")
    if ("filter_plots" in completed) or ("interactive_3d" in completed):
        _log("INFO", f"       Filter plots : {PLOTS_FILT_DIR / run_id}")
    _log("INFO", "=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("WARN", "\n[ABORT] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        _log("ERROR", f"\n[FATAL] Pipeline failed: {e}")
        sys.exit(1)