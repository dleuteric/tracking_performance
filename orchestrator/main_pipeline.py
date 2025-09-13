# File: orchestrator/main_pipeline.py
"""
Config‑driven end‑to‑end orchestrator.

- Reads paths/flags from config/pipeline.yaml via config.loader
- Ensures a single RUN_ID is propagated to all steps
- Allows reusing an existing triangulation run (skip regen)
- Safe to run partial chains (triangulation only; plots only; filter only)
"""
from __future__ import annotations

import os
import sys
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# ---------------- Config loader ----------------
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

# Resolve paths from YAML (relative to project root if needed)
_paths = CFG["paths"]
TRI_DIR = (PROJECT_ROOT / _paths["triangulation_out"]).resolve()
PLOTS_GEOM_DIR = (PROJECT_ROOT / _paths["geom_plots_out"]).resolve()
TRACKS_DIR = (PROJECT_ROOT / _paths["tracks_out"]).resolve()
PLOTS_FILT_DIR = (PROJECT_ROOT / _paths["filter_plots_out"]).resolve()

# ---------------- Logging ----------------
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _log(level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
    """Enhanced logging with optional metadata"""
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if extra:
            msg += f" | {json.dumps(extra, default=str)}"
        print(f"[{timestamp}] [{level:5s}] {msg}")


# ---------------- Quick flags (optional overrides) ----------------
# Override YAML at runtime without editing the config.
# Set to True/False to force, or None to use YAML defaults.
FLAGS = {
    "RUN_TRIANGULATION": True,  # if False -> reuse existing triangulation
    "RUN_GEOM_PLOTS": True,
    "RUN_FILTER_FORWARD": None,
    "RUN_FILTER_PLOTS": None,
    "RUN_INTERACTIVE_3D": None,
    "REUSE_TRIANGULATION": None,  # when RUN_TRIANGULATION is False: adopt latest triangulation RUN_ID
    "FORCE_RUN_ID": None,  # string to pin a specific run, bypassing latest
    "CONTINUE_ON_ERROR": False,  # Continue pipeline even if non-critical steps fail
}

_env = os.environ.get


def _env_bool(name: str, default: None | bool = None) -> None | bool:
    v = _env(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"): return True
    if v in ("0", "false", "no", "off"): return False
    return default


# Map env to FLAGS if provided (FLAGS take precedence when not None)
FLAGS["RUN_TRIANGULATION"] = _env_bool("RUN_TRI", FLAGS["RUN_TRIANGULATION"]) if FLAGS["RUN_TRIANGULATION"] is None else \
FLAGS["RUN_TRIANGULATION"]
FLAGS["RUN_GEOM_PLOTS"] = _env_bool("RUN_GPLT", FLAGS["RUN_GEOM_PLOTS"]) if FLAGS["RUN_GEOM_PLOTS"] is None else FLAGS[
    "RUN_GEOM_PLOTS"]
FLAGS["RUN_FILTER_FORWARD"] = _env_bool("RUN_FWD", FLAGS["RUN_FILTER_FORWARD"]) if FLAGS[
                                                                                       "RUN_FILTER_FORWARD"] is None else \
FLAGS["RUN_FILTER_FORWARD"]
FLAGS["RUN_FILTER_PLOTS"] = _env_bool("RUN_FPLT", FLAGS["RUN_FILTER_PLOTS"]) if FLAGS["RUN_FILTER_PLOTS"] is None else \
FLAGS["RUN_FILTER_PLOTS"]
FLAGS["RUN_INTERACTIVE_3D"] = _env_bool("RUN_I3D", FLAGS["RUN_INTERACTIVE_3D"]) if FLAGS[
                                                                                       "RUN_INTERACTIVE_3D"] is None else \
FLAGS["RUN_INTERACTIVE_3D"]
FLAGS["REUSE_TRIANGULATION"] = _env_bool("REUSE_TRI", FLAGS["REUSE_TRIANGULATION"]) if FLAGS[
                                                                                           "REUSE_TRIANGULATION"] is None else \
FLAGS["REUSE_TRIANGULATION"]
FLAGS["CONTINUE_ON_ERROR"] = _env_bool("CONT_ERR", FLAGS["CONTINUE_ON_ERROR"]) if FLAGS[
                                                                                      "CONTINUE_ON_ERROR"] is None else \
FLAGS["CONTINUE_ON_ERROR"]
FLAGS["FORCE_RUN_ID"] = _env("RUN_ID", FLAGS["FORCE_RUN_ID"]) if FLAGS["FORCE_RUN_ID"] is None else FLAGS[
    "FORCE_RUN_ID"]


# ---------------- Helpers ----------------

def _ensure_pkg_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _validate_directories():
    """Ensure all required directories exist"""
    dirs = [TRI_DIR, PLOTS_GEOM_DIR, TRACKS_DIR, PLOTS_FILT_DIR]
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            _log("DEBUG", f"Created directory: {d}")


def _latest_run_id(base_dir: Path) -> str:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run folders in {base_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name


def _tri_run_has_csvs(run_id: str) -> bool:
    run_dir = TRI_DIR / run_id
    if not run_dir.is_dir():
        return False
    csv_files = list(run_dir.glob("xhat_geo_*.csv"))
    if csv_files:
        _log("DEBUG", f"Found {len(csv_files)} triangulation CSVs in {run_id}")
    return bool(csv_files)


def _resolve_reusable_run_id(preferred: str | None) -> str:
    """Return a valid triangulation RUN_ID with CSVs.
    Order: preferred (if valid) else newest run with CSVs.
    Raises FileNotFoundError if none exist.
    """
    if preferred and _tri_run_has_csvs(preferred):
        return preferred
    # pick newest run that actually has CSVs
    candidates = [p for p in TRI_DIR.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if any(cand.glob("xhat_geo_*.csv")):
            return cand.name
    raise FileNotFoundError(f"No triangulation CSVs found under {TRI_DIR}")


def _get_run_metadata(run_id: str) -> Dict[str, Any]:
    """Extract metadata from RUN_ID string"""
    # Format: YYYYMMDDTHHMMSSZ_Nsats_Xkm_Ydeg_hash
    parts = run_id.split('_')
    if len(parts) >= 5:
        return {
            "timestamp": parts[0],
            "n_sats": parts[1],
            "altitude": parts[2],
            "inclination": parts[3],
            "hash": parts[4]
        }
    return {"run_id": run_id}


# ---------------- Steps ----------------

def step_triangulation():
    _ensure_pkg_path()
    try:
        import geometry.triangulate_icrf as tri
        _log("INFO", "[STEP] Triangulation starting...")
        start_time = datetime.now()
        tri.main()
        duration = (datetime.now() - start_time).total_seconds()
        _log("INFO", f"[STEP] Triangulation ✓ completed in {duration:.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Triangulation failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite triangulation error")
            return False
        raise


def step_geom_plots(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import geometry.plot_suite as plotg
        _log("INFO", "[STEP] Geometry plots starting...", _get_run_metadata(run_id))
        start_time = datetime.now()
        plotg.main()
        duration = (datetime.now() - start_time).total_seconds()
        _log("INFO", f"[STEP] Geometry plots ✓ completed in {duration:.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Geometry plots failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite geometry plots error")
            return False
        raise


def step_filter_forward(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.run_filter as rfil
        _log("INFO", "[STEP] Filtering (forward) starting...", _get_run_metadata(run_id))
        start_time = datetime.now()
        rfil.main()
        duration = (datetime.now() - start_time).total_seconds()
        _log("INFO", f"[STEP] Filtering (forward) ✓ completed in {duration:.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Filtering forward failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite filter error")
            return False
        raise


def step_filter_plots(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Filter plots starting...", _get_run_metadata(run_id))
        start_time = datetime.now()
        pfp.main()
        duration = (datetime.now() - start_time).total_seconds()
        _log("INFO", f"[STEP] Filter plots ✓ completed in {duration:.1f}s")
        return True
    except Exception as e:
        _log("ERROR", f"[ERR ] Filter plots failed: {e}")
        traceback.print_exc()
        if FLAGS["CONTINUE_ON_ERROR"]:
            _log("WARN", "[CONT] Continuing despite filter plots error")
            return False
        raise


def step_interactive_3d(run_id: str):
    """Generate interactive 3D HTMLs using the plots module.
    This is idempotent and safe to run after filter plots.
    """
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Interactive 3D starting...")
        start_time = datetime.now()
        pfp.main()
        duration = (datetime.now() - start_time).total_seconds()
        _log("INFO", f"[STEP] Interactive 3D ✓ completed in {duration:.1f}s")
        return True
    except Exception as e:
        _log("WARN", f"[SKIP] Interactive 3D not available: {e}")
        return False


# ---------------- Main ----------------

def main():
    """Main orchestrator entry point"""
    pipeline_start = datetime.now()
    _log("INFO", "=" * 60)
    _log("INFO", "PWSA Tracking Layer Pipeline Starting")
    _log("INFO", f"Project root: {PROJECT_ROOT}")
    _log("INFO", "=" * 60)

    # Validate directories exist
    _validate_directories()

    # Flags: FLAGS override YAML when not None
    run_tri = FLAGS["RUN_TRIANGULATION"] if FLAGS["RUN_TRIANGULATION"] is not None else bool(
        CFG.get("orchestrator", {}).get("run_triangulation", True))
    run_gplt = FLAGS["RUN_GEOM_PLOTS"] if FLAGS["RUN_GEOM_PLOTS"] is not None else bool(
        CFG.get("orchestrator", {}).get("run_geom_plots", True))
    run_fwd = FLAGS["RUN_FILTER_FORWARD"] if FLAGS["RUN_FILTER_FORWARD"] is not None else bool(
        CFG.get("orchestrator", {}).get("run_filter_forward", False))
    run_fplt = FLAGS["RUN_FILTER_PLOTS"] if FLAGS["RUN_FILTER_PLOTS"] is not None else bool(
        CFG.get("orchestrator", {}).get("run_filter_plots", False))
    run_i3d = FLAGS["RUN_INTERACTIVE_3D"] if FLAGS["RUN_INTERACTIVE_3D"] is not None else bool(
        CFG.get("orchestrator", {}).get("run_interactive_3d", False))
    reuse_tri = FLAGS["REUSE_TRIANGULATION"] if FLAGS["REUSE_TRIANGULATION"] is not None else True

    # Log configuration
    _log("INFO", "Pipeline Configuration:")
    _log("INFO", f"  • Triangulation    : {'RUN' if run_tri else 'SKIP/REUSE'}")
    _log("INFO", f"  • Geometry Plots   : {'RUN' if run_gplt else 'SKIP'}")
    _log("INFO", f"  • Filter Forward   : {'RUN' if run_fwd else 'SKIP'}")
    _log("INFO", f"  • Filter Plots     : {'RUN' if run_fplt else 'SKIP'}")
    _log("INFO", f"  • Interactive 3D   : {'RUN' if run_i3d else 'SKIP'}")
    _log("INFO", f"  • Continue on Error: {'YES' if FLAGS['CONTINUE_ON_ERROR'] else 'NO'}")

    # If filter plots already generate interactive HTMLs, skip explicit i3d step
    if run_fplt and run_i3d:
        _log("DEBUG", "[RUN ] interactive 3D handled by filter plots; skipping explicit step")
        run_i3d = False

    # RUN_ID priority: FORCE_RUN_ID (env/flags) > YAML > latest triangulation (if reusing)
    run_id = str(CFG["project"].get("run_id")) if CFG["project"].get("run_id") else None
    if FLAGS["FORCE_RUN_ID"]:
        run_id = str(FLAGS["FORCE_RUN_ID"])
        _log("INFO", f"[RUN ] Forced RUN_ID from ENV/flags: {run_id}")

    # Track successful steps
    completed_steps = []

    # 1) Triangulation or reuse
    if run_tri:
        if step_triangulation():
            completed_steps.append("triangulation")
        # Adopt the latest triangulation RUN_ID (the one just written)
        run_id = _latest_run_id(TRI_DIR)
        os.environ["RUN_ID"] = run_id
        _log("INFO", f"[RUN ] Using RUN_ID: {run_id}")
    else:
        if reuse_tri:
            # Prefer the forced/YAML run_id if it contains CSVs; otherwise, fallback to latest valid
            try:
                run_id = _resolve_reusable_run_id(run_id)
                os.environ["RUN_ID"] = run_id
                _log("INFO", f"[RUN ] Reusing triangulation RUN_ID: {run_id}")
            except FileNotFoundError as e:
                _log("ERROR", f"[ERR ] {e}")
                raise
        else:
            if not run_id:
                raise RuntimeError("RUN_ID undefined and reuse disabled; set RUN_ID or enable triangulation.")
            if not _tri_run_has_csvs(run_id):
                raise FileNotFoundError(f"Triangulation RUN_ID has no CSVs: {TRI_DIR / run_id}")
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Using pre-set RUN_ID: {run_id}")

    # 2) Geometry plots (GPM)
    if run_gplt:
        if step_geom_plots(run_id):
            completed_steps.append("geom_plots")

    # 3) Filtering forward pass (EFM)
    if run_fwd:
        if step_filter_forward(run_id):
            completed_steps.append("filter_forward")

    # 4) Filter performance / interactive plots
    if run_fplt:
        if step_filter_plots(run_id):
            completed_steps.append("filter_plots")

    # 5) Interactive 3D HTMLs (only if not already handled by filter plots)
    if run_i3d:
        if step_interactive_3d(run_id):
            completed_steps.append("interactive_3d")

    # Final summary
    total_duration = (datetime.now() - pipeline_start).total_seconds()
    _log("INFO", "=" * 60)
    _log("INFO", f"[OK  ] Pipeline finished in {total_duration:.1f}s")
    _log("INFO", f"       Completed steps: {', '.join(completed_steps) if completed_steps else 'None'}")
    _log("INFO", f"       RUN_ID: {run_id}")
    _log("INFO", f"       Triangulation: {TRI_DIR / run_id}")
    if "geom_plots" in completed_steps:
        _log("INFO", f"       Geom plots   : {PLOTS_GEOM_DIR / run_id}")
    if "filter_forward" in completed_steps:
        _log("INFO", f"       Tracks       : {TRACKS_DIR / run_id}")
    if "filter_plots" in completed_steps or "interactive_3d" in completed_steps:
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