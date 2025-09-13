"""
orchestrator/main_pipeline.py

Config-driven end-to-end orchestrator.
- Reads paths/flags from config/pipeline.yaml via config.loader
- Ensures a single RUN_ID is propagated across all steps
- Lets you test just the first chain (Triangulation ➜ Geometry plots)
  by toggling flags in YAML: orchestrator.run_triangulation / run_geom_plots
"""
from __future__ import annotations

import os
import sys
import traceback

from pathlib import Path

# ---------------- Quick flags (optional overrides) ----------------
# Override YAML at runtime without editing the config.
# Set to True/False to force, or None to use YAML defaults.
FLAGS = {
    "RUN_TRIANGULATION": True,     # if False -> reuse existing triangulation
    "RUN_GEOM_PLOTS": None,
    "RUN_FILTER_FORWARD": None,
    "RUN_FILTER_PLOTS": None,
    "RUN_INTERACTIVE_3D": None,
    "REUSE_TRIANGULATION": None,   # when RUN_TRIANGULATION is False: adopt latest triangulation RUN_ID
    "FORCE_RUN_ID": None,          # string to pin a specific run, bypassing latest
}

# Environment overrides (e.g., RUN_TRI=0/1, RUN_ID=..., REUSE_TRI=1)
_env = os.environ.get

def _env_bool(name: str, default: None | bool = None) -> None | bool:
    v = _env(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"): return True
    if v in ("0", "false", "no", "off"): return False
    return default

# Map env to FLAGS if provided
FLAGS["RUN_TRIANGULATION"]  = _env_bool("RUN_TRI", FLAGS["RUN_TRIANGULATION"]) \
                               if FLAGS["RUN_TRIANGULATION"] is None else FLAGS["RUN_TRIANGULATION"]
FLAGS["RUN_GEOM_PLOTS"]      = _env_bool("RUN_GPLT", FLAGS["RUN_GEOM_PLOTS"]) \
                               if FLAGS["RUN_GEOM_PLOTS"] is None else FLAGS["RUN_GEOM_PLOTS"]
FLAGS["RUN_FILTER_FORWARD"]  = _env_bool("RUN_FWD", FLAGS["RUN_FILTER_FORWARD"]) \
                               if FLAGS["RUN_FILTER_FORWARD"] is None else FLAGS["RUN_FILTER_FORWARD"]
FLAGS["RUN_FILTER_PLOTS"]    = _env_bool("RUN_FPLT", FLAGS["RUN_FILTER_PLOTS"]) \
                               if FLAGS["RUN_FILTER_PLOTS"] is None else FLAGS["RUN_FILTER_PLOTS"]
FLAGS["RUN_INTERACTIVE_3D"]  = _env_bool("RUN_I3D", FLAGS["RUN_INTERACTIVE_3D"]) \
                               if FLAGS["RUN_INTERACTIVE_3D"] is None else FLAGS["RUN_INTERACTIVE_3D"]
FLAGS["REUSE_TRIANGULATION"] = _env_bool("REUSE_TRI", FLAGS["REUSE_TRIANGULATION"]) \
                               if FLAGS["REUSE_TRIANGULATION"] is None else FLAGS["REUSE_TRIANGULATION"]
FLAGS["FORCE_RUN_ID"]        = _env("RUN_ID", FLAGS["FORCE_RUN_ID"]) \
                               if FLAGS["FORCE_RUN_ID"] is None else FLAGS["FORCE_RUN_ID"]

# Config loader
try:
    from config.loader import load_config
except Exception:
    # allow running as script
    import pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config

CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()
TRI_DIR = (PROJECT_ROOT / CFG["paths"]["triangulation_out"]).resolve()
PLOTS_GEOM_DIR = (PROJECT_ROOT / CFG["paths"]["geom_plots_out"]).resolve()
TRACKS_DIR = (PROJECT_ROOT / CFG["paths"]["tracks_out"]).resolve()
PLOTS_FILT_DIR = (PROJECT_ROOT / CFG["paths"]["filter_plots_out"]).resolve()

# Logging gate
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)


def _latest_run_id(base_dir: Path) -> str:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run folders in {base_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name


def _ensure_pkg_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def step_triangulation():
    _ensure_pkg_path()
    try:
        import geometry.triangulate_icrf as tri
        _log("INFO", "[STEP] Triangulation…")
        tri.main()
        _log("INFO", "[STEP] Triangulation ✓")
    except Exception as e:
        _log("ERROR", f"[ERR ] Triangulation failed: {e}")
        traceback.print_exc()
        raise


def step_geom_plots(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import geometry.plot_suite as plotg
        _log("INFO", "[STEP] Geometry plots…")
        plotg.main()
        _log("INFO", "[STEP] Geometry plots ✓")
    except Exception as e:
        _log("ERROR", f"[ERR ] Geometry plots failed: {e}")
        traceback.print_exc()
        raise


def step_filter_forward(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.run_filter as rfil
        _log("INFO", "[STEP] Filtering (forward)…")
        rfil.main()
        _log("INFO", "[STEP] Filtering (forward) ✓")
    except Exception as e:
        _log("ERROR", f"[ERR ] Filtering forward failed: {e}")
        traceback.print_exc()
        raise


def step_filter_plots(run_id: str):
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Filter plots…")
        pfp.main()
        _log("INFO", "[STEP] Filter plots ✓")
    except Exception as e:
        _log("ERROR", f"[ERR ] Filter plots failed: {e}")
        traceback.print_exc()
        raise


def step_interactive_3d(run_id: str):
    """Generate interactive 3D HTMLs using the plots module.
    This is idempotent and safe to run after filter plots.
    """
    _ensure_pkg_path()
    os.environ["RUN_ID"] = run_id
    try:
        import estimationandfiltering.plots_tracking_performance as pfp
        _log("INFO", "[STEP] Interactive 3D…")
        pfp.main()  # generates *_3d.html alongside tracking PDFs when Plotly is available
        _log("INFO", "[STEP] Interactive 3D ✓")
    except Exception as e:
        _log("WARN", f"[SKIP] Interactive 3D not available: {e}")


def main():
    # Honor toggles: FLAGS override YAML when not None
    run_tri  = FLAGS["RUN_TRIANGULATION"] \
               if FLAGS["RUN_TRIANGULATION"] is not None \
               else bool(CFG.get("orchestrator", {}).get("run_triangulation", True))
    run_gplt = FLAGS["RUN_GEOM_PLOTS"] \
               if FLAGS["RUN_GEOM_PLOTS"] is not None \
               else bool(CFG.get("orchestrator", {}).get("run_geom_plots", True))
    run_fwd  = FLAGS["RUN_FILTER_FORWARD"] \
               if FLAGS["RUN_FILTER_FORWARD"] is not None \
               else bool(CFG.get("orchestrator", {}).get("run_filter_forward", False))
    run_fplt = FLAGS["RUN_FILTER_PLOTS"] \
               if FLAGS["RUN_FILTER_PLOTS"] is not None \
               else bool(CFG.get("orchestrator", {}).get("run_filter_plots", False))
    run_i3d  = FLAGS["RUN_INTERACTIVE_3D"] \
               if FLAGS["RUN_INTERACTIVE_3D"] is not None \
               else bool(CFG.get("orchestrator", {}).get("run_interactive_3d", False))
    reuse_tri = FLAGS["REUSE_TRIANGULATION"] if FLAGS["REUSE_TRIANGULATION"] is not None else True

    # If filter plots already generate interactive HTMLs, we can skip a separate i3d run
    if run_fplt and run_i3d:
        _log("DEBUG", "[RUN ] interactive 3D handled by filter plots; skipping explicit step")
        run_i3d = False

    # Start with RUN_ID from loader or FORCE_RUN_ID
    run_id = str(CFG["project"].get("run_id")) if CFG["project"].get("run_id") else None
    if FLAGS["FORCE_RUN_ID"]:
        run_id = str(FLAGS["FORCE_RUN_ID"])
        _log("INFO", f"[RUN ] Forced RUN_ID from ENV/flags: {run_id}")

    # 1) Triangulation or reuse
    if run_tri:
        step_triangulation()
        # Adopt the latest triangulation RUN_ID (the one just written)
        run_id = _latest_run_id(TRI_DIR)
        os.environ["RUN_ID"] = run_id
        _log("INFO", f"[RUN ] Using RUN_ID: {run_id}")
    else:
        if run_id:
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Using pre-set RUN_ID: {run_id}")
        elif reuse_tri:
            run_id = _latest_run_id(TRI_DIR)
            os.environ["RUN_ID"] = run_id
            _log("INFO", f"[RUN ] Reusing latest triangulation RUN_ID: {run_id}")
        else:
            raise RuntimeError("RUN_ID undefined and reuse disabled; set RUN_ID or enable triangulation.")

    # 2) Geometry plots (GPM)
    if run_gplt:
        step_geom_plots(run_id)

    # 3) Filtering forward pass (EFM)
    if run_fwd:
        step_filter_forward(run_id)

    # 4) Filter performance plots
    if run_fplt:
        step_filter_plots(run_id)

    # 5) Interactive 3D HTMLs
    if run_i3d:
        step_interactive_3d(run_id)

    _log("INFO", "[OK  ] Pipeline finished.")
    _log("INFO", f"       Triangulation: {TRI_DIR/run_id}")
    if run_gplt:
        _log("INFO", f"       Geom plots   : {PLOTS_GEOM_DIR/run_id}")
    if run_fwd:
        _log("INFO", f"       Tracks       : {TRACKS_DIR/run_id}")
    if run_fplt or run_i3d:
        _log("INFO", f"       Filter plots : {PLOTS_FILT_DIR/run_id}")


if __name__ == "__main__":
    main()