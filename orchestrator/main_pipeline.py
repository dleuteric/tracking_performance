# File: orchestrator/main_pipeline.py
"""
Config-driven end-to-end orchestrator for ez-SMAD.

Steps: Triangulation → Geometry plots → Estimation (KF +/or EW-RLS)
       → Filter plots → Interactive 3D (optional)
"""
from __future__ import annotations

import os, sys, json, traceback
import subprocess
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

# --------------- Config globals (populated in main) ---------------
CFG = None
PROJECT_ROOT = None
TRI_DIR = None
PLOTS_GEOM_DIR = None
TRACKS_DIR = None
PLOTS_FILT_DIR = None

# --------------- Logging ---------------
LOG_LEVEL = "INFO"
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
    "USE_KF": True,                 # <— NEW: run legacy CV-KF (run_filter)
    "USE_EWRLS": True,              # EW-RLS in cascade after KF (if enabled)
    "RUN_FILTER_PLOTS": None,
    "RUN_INTERACTIVE_3D": None,
    "REUSE_TRIANGULATION": False,
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

FLAGS["RUN_TRIANGULATION"]   = _env_bool("RUN_TRI",  FLAGS["RUN_TRIANGULATION"])   if FLAGS["RUN_TRIANGULATION"]   is None else FLAGS["RUN_TRIANGULATION"]
FLAGS["RUN_GEOM_PLOTS"]      = _env_bool("RUN_GPLT", FLAGS["RUN_GEOM_PLOTS"])      if FLAGS["RUN_GEOM_PLOTS"]      is None else FLAGS["RUN_GEOM_PLOTS"]
FLAGS["RUN_FILTER_FORWARD"]  = _env_bool("RUN_FWD",  FLAGS["RUN_FILTER_FORWARD"])  if FLAGS["RUN_FILTER_FORWARD"]  is None else FLAGS["RUN_FILTER_FORWARD"]
FLAGS["USE_KF"]              = _env_bool("USE_KF",   FLAGS["USE_KF"])              if FLAGS["USE_KF"]              is None else FLAGS["USE_KF"]
FLAGS["USE_EWRLS"]           = _env_bool("USE_EWRLS",FLAGS["USE_EWRLS"])           if FLAGS["USE_EWRLS"]           is None else FLAGS["USE_EWRLS"]
FLAGS["RUN_FILTER_PLOTS"]    = _env_bool("RUN_FPLT", FLAGS["RUN_FILTER_PLOTS"])    if FLAGS["RUN_FILTER_PLOTS"]    is None else FLAGS["RUN_FILTER_PLOTS"]
FLAGS["RUN_INTERACTIVE_3D"]  = _env_bool("RUN_I3D",  FLAGS["RUN_INTERACTIVE_3D"])  if FLAGS["RUN_INTERACTIVE_3D"]  is None else FLAGS["RUN_INTERACTIVE_3D"]
FLAGS["REUSE_TRIANGULATION"] = _env_bool("REUSE_TRI",FLAGS["REUSE_TRIANGULATION"]) if FLAGS["REUSE_TRIANGULATION"] is None else FLAGS["REUSE_TRIANGULATION"]
FLAGS["CONTINUE_ON_ERROR"]   = _env_bool("CONT_ERR", FLAGS["CONTINUE_ON_ERROR"])   if FLAGS["CONTINUE_ON_ERROR"]   is None else FLAGS["CONTINUE_ON_ERROR"]
FLAGS["FORCE_RUN_ID"]        = _env("RUN_ID", FLAGS["FORCE_RUN_ID"]) if FLAGS["FORCE_RUN_ID"] is None else FLAGS["FORCE_RUN_ID"]

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
    global CFG
    _ensure_pkg_path()
    _log("DEBUG", f"sys.path[0:3]={sys.path[0:3]}")
    try:
        import geometry.triangulate_icrf as tri
        _log("INFO", "[STEP] Triangulation starting...")
        # Log LOS and EPH sources explicitly
        los_root = None
        eph_root = None
        try:
            los_root = CFG["paths"].get("los_root")
        except Exception:
            pass
        try:
            eph_root = CFG["paths"].get("ephem_root")
        except Exception:
            pass
        if los_root is not None:
            _log("INFO", f"[CFG ] LOS source : {los_root}")
        else:
            _log("INFO", "[CFG ] LOS source : (default STK location)")
        if eph_root is not None:
            _log("INFO", f"[CFG ] EPH source : {eph_root}")
        else:
            _log("INFO", "[CFG ] EPH source : (default STK location)")
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
    """Legacy CV-KF pipeline (run_filter)."""
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.run_filter as rfil
        _log("INFO", "[STEP] KF (legacy) filtering starting...", _get_run_metadata(run_id))
        t0 = datetime.now(); rfil.main()
        _log("INFO", f"[STEP] KF filtering ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] KF filtering failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite KF error")
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
    """Run EW-RLS (ICRF) using ewrls_icrf_tracks.py on all triangulated targets for this run."""
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.ewrls_icrf_tracks as ewmod
        _log("INFO", "[STEP] EW-RLS (ICRF) starting...", _get_run_metadata(run_id))
        t0 = datetime.now()
        # ewrls_icrf_tracks.run() discovers RUN_ID from env if needed
        ewmod.run()
        _log("INFO", f"[STEP] EW-RLS (ICRF) ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] EW-RLS (ICRF) failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite EW-RLS error")
            return False
        raise


# --- New: Comparator step ---
def step_compare(run_id: str, config_path: str | None):
    """Run the new comparator to align KF vs EW-RLS vs truth and export metrics/plots."""
    _ensure_pkg_path(); os.environ["RUN_ID"] = run_id
    try:
        comp_py = PROJECT_ROOT / "estimationandfiltering" / "comparator_new.py"
        if not comp_py.exists():
            raise FileNotFoundError(f"Missing comparator: {comp_py}")
        _log("INFO", "[STEP] Comparator starting...", _get_run_metadata(run_id))
        t0 = datetime.now()
        cmd = [sys.executable, str(comp_py), "--run_id", run_id, "--out_dir", PLOTS_FILT_DIR
               ]
        if config_path:
            cmd.extend(["--config", str(config_path)])
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
        if proc.returncode != 0:
            _log("ERROR", f"[ERR ] Comparator stderr:\n{proc.stderr}")
            _log("ERROR", f"[ERR ] Comparator stdout:\n{proc.stdout}")
            raise RuntimeError("Comparator failed")
        _log("INFO", f"[STEP] Comparator ✓ completed in {(datetime.now()-t0).total_seconds():.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Comparator failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite comparator error")
            return False
        raise

# --------------- Main ---------------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ez-SMAD main pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    return parser.parse_args()

def main(config_path: str | None = None):
    global CFG, PROJECT_ROOT, TRI_DIR, PLOTS_GEOM_DIR, TRACKS_DIR, PLOTS_FILT_DIR, LOG_LEVEL
    # 1. Load config (from param, env, or default)
    if config_path is None:
        config_path = os.environ.get("PIPELINE_CFG")
    CFG = load_config(config_path)
    # 2. Set up paths from config
    PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()
    _paths = CFG["paths"]
    TRI_DIR        = (PROJECT_ROOT / _paths["triangulation_out"]).resolve()
    PLOTS_GEOM_DIR = (PROJECT_ROOT / _paths["geom_plots_out"]).resolve()
    TRACKS_DIR     = (PROJECT_ROOT / _paths["tracks_out"]).resolve()
    PLOTS_FILT_DIR = (PROJECT_ROOT / _paths["filter_plots_out"]).resolve()
    LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()

    start = datetime.now()
    _log("INFO", "=" * 60)
    _log("INFO", "PWSA Tracking Layer Pipeline Starting")
    _log("INFO", f"Project root: {PROJECT_ROOT}")
    _log("INFO", "=" * 60)

    _validate_directories()

    run_tri = FLAGS["RUN_TRIANGULATION"]   if FLAGS["RUN_TRIANGULATION"]   is not None else bool(CFG.get("orchestrator", {}).get("run_triangulation", True))
    run_gplt= FLAGS["RUN_GEOM_PLOTS"]      if FLAGS["RUN_GEOM_PLOTS"]      is not None else bool(CFG.get("orchestrator", {}).get("run_geom_plots", True))
    run_fwd = FLAGS["RUN_FILTER_FORWARD"]  if FLAGS["RUN_FILTER_FORWARD"]  is not None else bool(CFG.get("orchestrator", {}).get("run_filter_forward", True))
    run_fplt= FLAGS["RUN_FILTER_PLOTS"]    if FLAGS["RUN_FILTER_PLOTS"]    is not None else bool(CFG.get("orchestrator", {}).get("run_filter_plots", True))
    run_i3d = FLAGS["RUN_INTERACTIVE_3D"]  if FLAGS["RUN_INTERACTIVE_3D"]  is not None else bool(CFG.get("orchestrator", {}).get("run_interactive_3d", False))
    reuse_tri = FLAGS["REUSE_TRIANGULATION"] if FLAGS["REUSE_TRIANGULATION"] is not None else False

    use_kf = bool(FLAGS.get("USE_KF", True))
    use_ew = bool(FLAGS.get("USE_EWRLS", True))

    _log("INFO", f"Flags: reuse_tri={reuse_tri} | use_kf={use_kf} | use_ewrls={use_ew}")

    _log("INFO", "Pipeline Configuration:")
    _log("INFO", f"  • Triangulation    : {'RUN' if run_tri else 'SKIP/REUSE'}")
    _log("INFO", f"  • Geometry Plots   : {'RUN' if run_gplt else 'SKIP'}")
    _log("INFO", f"  • Estimation (KF)  : {'ON'  if (run_fwd and FLAGS['USE_KF']) else 'OFF'}")
    _log("INFO", f"  • Estimation (EW)  : {'ON'  if (run_fwd and FLAGS['USE_EWRLS']) else 'OFF'}")
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

    # --- STEP 1: read + log σ_LOS from config (aligned to YAML) ---
    gpm_cfg = CFG.get("gpm_measurement", {}) or {}
    geom_cfg = CFG.get("geometry", {}) or {}

    # Primary: gpm_measurement.los_noise_rad  (this is how it's named in pipeline.yaml)
    if "los_noise_rad" in gpm_cfg:
        los_sigma = float(gpm_cfg["los_noise_rad"])
    # Fallback: geometry.los_sigma (if someone sets it there)
    elif "los_sigma" in geom_cfg:
        los_sigma = float(geom_cfg["los_sigma"])
    else:
        _log("ERROR", "[CFG ] Missing σ_LOS: expected 'gpm_measurement.los_noise_rad' (rad) "
                      "or fallback 'geometry.los_sigma' (rad).")
        raise KeyError("los_sigma")

    _log("INFO", f"[CFG ] σ_LOS = {los_sigma:.6e} rad (source: "
                 f"{'gpm_measurement.los_noise_rad' if 'los_noise_rad' in gpm_cfg else 'geometry.los_sigma'})")

    # 1) Triangulation or reuse
    if run_tri:
        if step_triangulation():
            completed.append("triangulation")
        run_id = _latest_run_id(TRI_DIR)
        os.environ["RUN_ID"] = run_id
        _log("INFO", f"[RUN ] Using RUN_ID: {run_id}")
        # save los_sigma for this run
        run_root = TRI_DIR / run_id
        (run_root / "los_sigma_rad.txt").write_text(f"{los_sigma:.9e}\n")
        # Write manifest json
        manifest = {
            "los_sigma_rad": los_sigma,
            "los_source": CFG["paths"].get("los_root"),
            "eph_source": CFG["paths"].get("ephem_root"),
            "config_file": config_path if config_path is not None else "(default loader)"
        }
        with open(run_root / "run_config_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        if reuse_tri:
            run_id = _resolve_reusable_run_id(run_id)
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Reusing triangulation RUN_ID: {run_id}")
            # save los_sigma also in reuse case
            run_root = TRI_DIR / run_id
            (run_root / "los_sigma_rad.txt").write_text(f"{los_sigma:.9e}\n")
            # Write manifest json
            manifest = {
                "los_sigma_rad": los_sigma,
                "los_source": CFG["paths"].get("los_root"),
                "eph_source": CFG["paths"].get("ephem_root"),
                "config_file": config_path if config_path is not None else "(default loader)"
            }
            with open(run_root / "run_config_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        else:
            if not run_id:
                raise RuntimeError("RUN_ID undefined and reuse disabled; set RUN_ID or enable triangulation.")
            if not _tri_run_has_csvs(run_id):
                raise FileNotFoundError(f"Triangulation RUN_ID has no CSVs: {TRI_DIR / run_id}")
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Using pre-set RUN_ID: {run_id}")
            # save los_sigma also in preset case
            run_root = TRI_DIR / run_id
            (run_root / "los_sigma_rad.txt").write_text(f"{los_sigma:.9e}\n")
            # Write manifest json
            manifest = {
                "los_sigma_rad": los_sigma,
                "los_source": CFG["paths"].get("los_root"),
                "eph_source": CFG["paths"].get("ephem_root"),
                "config_file": config_path if config_path is not None else "(default loader)"
            }
            with open(run_root / "run_config_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    completed = completed  # keep list

    # 2) Estimation in cascade: KF → EW-RLS
    if run_fwd:
        if FLAGS["USE_KF"]:
            if step_filter_forward(run_id):
                completed.append("kf_forward")
        if FLAGS["USE_EWRLS"]:
            if step_ewrls(run_id):
                completed.append("ewrls_forward")

    # 3) Comparator (align KF vs EW-RLS vs Truth; export metrics/overlay)
    if step_compare(run_id, config_path):
        completed.append("compare")

    # 4) Geometry plots (optional – after having all products)
    if run_gplt and step_geom_plots(run_id):
        completed.append("geom_plots")

    # 5) Filter plots (optional)
    if run_fplt and step_filter_plots(run_id):
        completed.append("filter_plots")

    # 6) Interactive 3D (optional)
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
    if ("kf_forward" in completed) or ("ewrls_forward" in completed):
        _log("INFO", f"       Tracks       : {TRACKS_DIR / run_id}")
    if ("filter_plots" in completed) or ("interactive_3d" in completed):
        _log("INFO", f"       Filter plots : {PLOTS_FILT_DIR / run_id}")
    _log("INFO", "=" * 60)

if __name__ == "__main__":
    try:
        args = parse_args()
        main(config_path=args.config)
    except KeyboardInterrupt:
        _log("WARN", "\n[ABORT] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        _log("ERROR", f"\n[FATAL] Pipeline failed: {e}")
        sys.exit(1)