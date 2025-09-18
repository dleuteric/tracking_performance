"""
plots_tracking_performance.py

Interactive 3D visualization (zoom/pan/orbit) of filtered track vs truth.
- Discovers RUN_ID (ENV or newest under exports/tracks from YAML paths)
- For each <TARGET>_track_icrf_forward.csv, overlays OEM truth if available
- Saves an interactive HTML per target to plots_filter/<RUN_ID>/<TARGET>_3d.html

If Plotly is not available, falls back to Matplotlib interactive window.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Optional: Plotly for rich interactivity
try:
    import plotly.graph_objects as go
    from plotly.io import write_image as _plotly_write_image  # will raise if kaleido missing
    HAVE_PLOTLY = True
    HAVE_KALEIDO = True
except Exception:
    try:
        import plotly.graph_objects as go
        HAVE_PLOTLY = True
        HAVE_KALEIDO = False
    except Exception:
        HAVE_PLOTLY = False
        HAVE_KALEIDO = False

# Matplotlib fallback (interactive window)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Config + adapter -------------------------------------------------------
try:
    from config.loader import load_config
    from estimationandfiltering.adapter import load_truth_oem
except Exception:
    import sys, pathlib
    pkg_root = pathlib.Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config
    from estimationandfiltering.adapter import load_truth_oem

CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()
_paths = CFG["paths"]
TRACK_BASE_DIR = (Path(_paths["tracks_out"]) if Path(_paths["tracks_out"]).is_absolute() else (PROJECT_ROOT / _paths["tracks_out"])) .resolve()
PLOT_BASE_DIR  = (Path(_paths["filter_plots_out"]) if Path(_paths["filter_plots_out"]).is_absolute() else (PROJECT_ROOT / _paths["filter_plots_out"])) .resolve()
OEM_DIR        = (Path(_paths["oem_root"]) if Path(_paths["oem_root"]).is_absolute() else (PROJECT_ROOT / _paths["oem_root"])) .resolve()

# Logging gate
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)


# --- Helpers ----------------------------------------------------------------

def _latest_run_id(base_dir: Path) -> str:
    rid = os.environ.get("RUN_ID")
    if rid and (base_dir / rid).is_dir():
        return rid
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run folders in {base_dir}")
    return runs[0].name


def _find_tracks(run_id: str) -> dict[str, Path]:
    run_dir = TRACK_BASE_DIR / run_id
    out: dict[str, Path] = {}
    for p in sorted(run_dir.glob("*_track_icrf_forward.csv")):
        m = re.search(r"(HGV_\d+)", p.name)
        if m:
            out[m.group(1)] = p
    if not out:
        raise FileNotFoundError(f"No *_track_icrf_forward.csv in {run_dir}")
    return out


def _align_truth(track_df: pd.DataFrame, target_id: str) -> pd.DataFrame | None:
    oem_path = OEM_DIR / f"{target_id}.oem"
    if not oem_path.exists():
        return None
    truth = load_truth_oem(oem_path)  # indexed by time
    t_track = pd.to_datetime(track_df['time'], utc=True, errors='coerce')
    truth_aligned = truth.reindex(t_track, method=None)
    return truth_aligned


def _plot_plotly(target_id: str, track_df: pd.DataFrame, truth_df: pd.DataFrame | None, out_html: Path):
    rF = track_df[['x_km','y_km','z_km']].to_numpy(float)
    traces = [
        go.Scatter3d(x=rF[:,0], y=rF[:,1], z=rF[:,2], mode='lines', name='filtered', line=dict(width=4))
    ]
    if truth_df is not None:
        mask = truth_df[['x_km','y_km','z_km']].notna().all(axis=1).to_numpy()
        rT = truth_df[['x_km','y_km','z_km']].to_numpy(float)[mask]
        traces.append(go.Scatter3d(x=rT[:,0], y=rT[:,1], z=rT[:,2], mode='lines', name='truth (OEM)', line=dict(width=3, dash='dash')))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3D track — {target_id}",
        scene=dict(xaxis_title='x [km]', yaxis_title='y [km]', zaxis_title='z [km]', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)
    _log("INFO", f"[HTML] {out_html}")

    # Optional: static PDF snapshot if kaleido available
    if HAVE_KALEIDO:
        try:
            pdf_path = out_html.with_suffix('.pdf')
            fig.write_image(str(pdf_path))
            _log("INFO", f"[PDF ] {pdf_path}")
        except Exception as e:
            _log("WARN", f"[SKIP] PDF snapshot failed (kaleido): {e}")


def _plot_mpl_interactive(target_id: str, track_df: pd.DataFrame, truth_df: pd.DataFrame | None):
    rF = track_df[['x_km','y_km','z_km']].to_numpy(float)
    fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(rF[:,0], rF[:,1], rF[:,2], label='filtered')
    if truth_df is not None:
        mask = truth_df[['x_km','y_km','z_km']].notna().all(axis=1).to_numpy()
        rT = truth_df[['x_km','y_km','z_km']].to_numpy(float)[mask]
        ax.plot(rT[:,0], rT[:,1], rT[:,2], '--', label='truth (OEM)')
    ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
    ax.set_title(f"3D track — {target_id}")
    ax.legend(loc='upper left')
    plt.tight_layout()

    # Save static copies alongside interactive window
    try:
        out_base = PLOT_BASE_DIR / os.environ.get("RUN_ID", "latest")
        out_base.mkdir(parents=True, exist_ok=True)
        png = out_base / f"{target_id}_3d.png"
        pdf = out_base / f"{target_id}_3d.pdf"
        fig.savefig(png, dpi=160, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        _log("INFO", f"[SAVE] {png}")
        _log("INFO", f"[PDF ] {pdf}")
    except Exception as e:
        _log("WARN", f"[SKIP] Could not save static images: {e}")

    plt.show()


# --- Entry point ------------------------------------------------------------

def main():
    run_id = _latest_run_id(TRACK_BASE_DIR)
    tracks = _find_tracks(run_id)
    outdir = PLOT_BASE_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    _log("INFO", f"[INT3D] RUN_ID={run_id} targets={list(tracks.keys())}")
    for tgt, path in tracks.items():
        df = pd.read_csv(path)
        truth = _align_truth(df, tgt)
        out_html = outdir / f"{tgt}_3d.html"
        if HAVE_PLOTLY:
            _plot_plotly(tgt, df, truth, out_html)
        else:
            _log("WARN", "[INT3D] Plotly not available; opening Matplotlib interactive window…")
            _plot_mpl_interactive(tgt, df, truth)


if __name__ == "__main__":
    main()
