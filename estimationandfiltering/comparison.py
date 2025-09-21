"""
Comparator pipeline: generate → estimate (KF & EW-RLS) → compare → save plots & report

Workflow:
1) Creates a new run (if --fresh) or reuses an existing RUN_ID
2) Generates both legacy KF and EW‑RLS tracks
3) Compares per‑target errors (RMSE_x/y/z, RMSE_3D, CEP50/90)
4) Saves metrics, plots, and a Markdown report in exports/compare/<RUN_ID>/

Usage (PyCharm ▶ Run):
  --fresh --out_dir exports/compare
  --run_id <RUN_ID> --out_dir exports/compare
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- Config & paths -------------------
def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] if here.parent.name == "estimationandfiltering" else here.parent

ROOT = _project_root()
ORCH = ROOT / "orchestrator" / "main_pipeline.py"

TRI_DIR   = ROOT / "exports/triangulation"
TRACKS_DIR= ROOT / "exports/tracks"
TRUTH_DIR = ROOT / "exports/truth"
GEOM_PLOTS_DIR = ROOT / "plots"

# ------------------- Helpers -------------------
def _latest_run_id(base_dir: Path) -> str:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name if runs else None

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------- Orchestrator wrappers -------------------
def _call_orchestrator(env_overrides: Dict[str,str]):
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [sys.executable, str(ORCH)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("Orchestrator failed")

def make_new_run(verbose: bool = True) -> str:
    """Run triangulation (and lightweight plots) to create a new RUN_ID; return that RUN_ID."""
    overrides = {
        "RUN_TRI": "1",
        "RUN_GPLT": "1",   # optional; keep on for quick QA plots
        "RUN_FWD": "0",
        "REUSE_TRI": "0",
        "CONT_ERR": "1",
    }
    _call_orchestrator(overrides)
    run_id = _latest_run_id(TRI_DIR)
    if verbose:
        print(f"[RUN ] New triangulation created → {run_id}")
        print(f"[PATH] Tri dir       : {TRI_DIR / run_id}")
        print(f"[PATH] Geom plots dir: {GEOM_PLOTS_DIR / run_id}")
    return run_id


# ------------------- Track generation -------------------
def generate_tracks(run_id: str, which: str = "both", verbose: bool = True) -> None:
    """Generate legacy KF and/or EW‑RLS tracks for the given RUN_ID, reusing triangulation outputs."""
    if which in ("legacy", "both"):
        _call_orchestrator({
            "RUN_ID": run_id,
            "RUN_TRI": "0",
            "RUN_GPLT": "0",
            "RUN_FWD": "1",
            "USE_EWRLS": "0",
            "REUSE_TRI": "1",
            "CONT_ERR": "1",
        })
        if verbose:
            print("[EST ] Legacy KF tracks ✓")
    if which in ("ewrls", "both"):
        _call_orchestrator({
            "RUN_ID": run_id,
            "RUN_TRI": "0",
            "RUN_GPLT": "0",
            "RUN_FWD": "1",
            "USE_EWRLS": "1",
            "REUSE_TRI": "1",
            "CONT_ERR": "1",
        })
        if verbose:
            print("[EST ] EW‑RLS tracks ✓")

# ------------------- Pipeline driver -------------------
def run_pipeline(run_id: Optional[str] = None,
                 fresh: bool = False,
                 out_dir: Optional[str] = None,
                 which: str = "both",
                 verbose: bool = True) -> Tuple[str, Path, pd.DataFrame]:
    """Full flow per spec: new run (optional) → estimate (KF &/or EW‑RLS) → compare → save outputs.
    Returns (run_id, output_dir, metrics_df).
    """
    # 1) RUN_ID
    if fresh or run_id is None:
        run_id = make_new_run(verbose=verbose)
    else:
        tri_path = TRI_DIR / run_id
        if not tri_path.exists():
            raise FileNotFoundError(f"Triangulation not found for RUN_ID {run_id}: {tri_path}")
        if verbose:
            print(f"[RUN ] Reusing RUN_ID → {run_id}")

    # 2) Estimation (KF +/‑ EW‑RLS)
    generate_tracks(run_id, which=which, verbose=verbose)

    # 3) Comparison
    df = compare_run(run_id, tracks_dir=TRACKS_DIR, truth_dir=TRUTH_DIR)

    # 4) Outputs folder (dedicated)
    base = Path(out_dir) if out_dir else (ROOT / "exports" / "compare")
    out_run = _ensure_dir(base / run_id)
    metrics_csv = out_run / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    if verbose:
        print(f"[OUT ] Metrics → {metrics_csv}")
    write_report(df, out_run, run_id)
    if verbose:
        print(f"[OUT ] Plots & report → {out_run}")

    return run_id, out_run, df

# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate run → estimate (KF & EW‑RLS) → compare → save outputs.")
    ap.add_argument("--run_id", default=None, help="Existing RUN_ID to reuse; omit or use --fresh to create a new one")
    ap.add_argument("--fresh", action="store_true", help="Create a NEW run (triangulation) before estimating")
    ap.add_argument("--which", default="both", choices=["both","legacy","ewrls"], help="Which estimators to run")
    ap.add_argument("--out_dir", default=None, help="Base output dir; default exports/compare/<RUN_ID>")
    ap.add_argument("--quiet", action="store_true", help="Less console noise")
    args = ap.parse_args()

    # No-args default → do everything, visibly
    if args.run_id is None and not args.fresh:
        args.fresh = True
        args.which = "both"
        if not args.out_dir:
            args.out_dir = "exports/compare"
        print("[INFO] No args provided → defaulting to: --fresh --which both --out_dir exports/compare")

    print("[INFO] Repository root:", ROOT)
    print("[INFO] Orchestrator path:", ORCH, "exists=", ORCH.exists())
    print("[INFO] TRI_DIR:", TRI_DIR)
    print("[INFO] TRACKS_DIR:", TRACKS_DIR)

    rid, out_path, df = run_pipeline(
        run_id=args.run_id,
        fresh=args.fresh,
        out_dir=args.out_dir,
        which=args.which,
        verbose=not args.quiet,
    )

    print("[DONE] RUN_ID:", rid)
    print("[DONE] Outputs at:", out_path)
    print("[DONE] Metrics rows:", 0 if df is None else len(df))
