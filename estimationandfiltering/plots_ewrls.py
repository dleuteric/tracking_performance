from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# --- Project paths ---
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config/pipeline.yaml"
TRI_ROOT = ROOT / "exports/triangulation"
TRACKS_OUT = ROOT / "exports/tracks_icrf"
PLOTS_OUT = ROOT / "plots_icrf"

# =========================
# Plotting
# =========================

def _set_equal_aspect_3d(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]
    r = max(xr,yr,zr)/2
    ax.set_xlim3d(sum(xlim)/2-r, sum(xlim)/2+r)
    ax.set_ylim3d(sum(ylim)/2-r, sum(ylim)/2+r)
    ax.set_zlim3d(sum(zlim)/2-r, sum(zlim)/2+r)


def plot_3d_vs_truth(tgt: str, est: pd.DataFrame, tru: Dict[str,Any], out_dir: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.2,5.4)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(est["x_km"], est["y_km"], est["z_km"], label="EW‑RLS", lw=2)
    ax.plot(tru["x"], tru["y"], tru["z"], "--", label="truth (ICRF)", lw=1.5)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt}"); ax.legend(); _set_equal_aspect_3d(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"{tgt}_3d.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_errors_4panel(tgt: str, est: pd.DataFrame, tru: Dict[str,Any], out_dir: Path):
    # Interpolate truth to estimator epochs (seconds)
    t_est = est["t_s"].to_numpy(float)
    def interp(comp):
        comp = np.asarray(comp, float)
        return np.interp(t_est, tru["t"], comp, left=comp[0], right=comp[-1])
    Tx, Ty, Tz = interp(tru["x"]), interp(tru["y"]), interp(tru["z"])

    ex = (est["x_km"].to_numpy(float) - Tx) * 1000.0
    ey = (est["y_km"].to_numpy(float) - Ty) * 1000.0
    ez = (est["z_km"].to_numpy(float) - Tz) * 1000.0
    et = np.sqrt(ex**2 + ey**2 + ez**2)
    tt = t_est - t_est[0]

    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222); ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)
    ax1.plot(tt, ex); ax1.set_ylabel("Error X [m]"); ax1.set_xlabel("Time [s]"); ax1.grid(True,alpha=.3)
    ax2.plot(tt, ey); ax2.set_ylabel("Error Y [m]"); ax2.set_xlabel("Time [s]"); ax2.grid(True,alpha=.3)
    ax3.plot(tt, ez); ax3.set_ylabel("Error Z [m]"); ax3.set_xlabel("Time [s]"); ax3.grid(True,alpha=.3)
    ax4.plot(tt, et, lw=2); ax4.set_ylabel("Total error [m]"); ax4.set_xlabel("Time [s]"); ax4.grid(True,alpha=.3)
    fig.suptitle(f"Estimated error vs time — {tgt}")
    fig.tight_layout(rect=[0,0.03,1,0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"{tgt}_errors4.png", dpi=150, bbox_inches="tight"); plt.close(fig)

