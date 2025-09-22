from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Interactive plotting (optional)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

# --- Styling helpers ---
COL_X = "#d62728"  # red
COL_Y = "#2ca02c"  # green
COL_Z = "#1f77b4"  # blue
COL_T = "#9467bd"  # purple total
COL_EST = "#111111"
COL_TRU = "#7f7f7f"


def _rmse3(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> float:
    et2 = ex**2 + ey**2 + ez**2
    return float(np.sqrt(np.nanmean(et2)))


# Plotting helpers

def _set_equal_aspect_3d(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]
    r = max(xr,yr,zr)/2
    ax.set_xlim3d(sum(xlim)/2-r, sum(xlim)/2+r)
    ax.set_ylim3d(sum(ylim)/2-r, sum(ylim)/2+r)
    ax.set_zlim3d(sum(zlim)/2-r, sum(zlim)/2+r)


def plot_3d_vs_truth(tgt: str, est: pd.DataFrame, tru: Dict[str,Any], out_dir: Path):
    """3D trajectory plot: PNG (matplotlib) + HTML (plotly if available)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Matplotlib static PNG
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.5,5.8)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(est["x_km"], est["y_km"], est["z_km"], label="EW‑RLS", lw=2, color=COL_EST)
    ax.plot(tru["x"], tru["y"], tru["z"], "--", label="Truth (ICRF)", lw=1.6, color=COL_TRU)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt}")
    ax.legend(loc="upper left", frameon=False)
    _set_equal_aspect_3d(ax)
    fig.tight_layout()
    fig.savefig(out_dir/f"{tgt}_3d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plotly interactive HTML
    if go is not None:
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=est["x_km"], y=est["y_km"], z=est["z_km"],
                                     mode="lines", name="EW‑RLS",
                                     line=dict(width=5, color=COL_EST)))
        fig3d.add_trace(go.Scatter3d(x=tru["x"], y=tru["y"], z=tru["z"],
                                     mode="lines", name="Truth (ICRF)",
                                     line=dict(width=3, color=COL_TRU, dash="dash")))
        fig3d.update_layout(title=f"3D track — {tgt}",
                            scene=dict(xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]"),
                            margin=dict(l=0,r=0,t=50,b=0))
        html_path = out_dir / f"{tgt}_3d.html"
        fig3d.write_html(str(html_path), include_plotlyjs="cdn")


def plot_errors_4panel(tgt: str, est: pd.DataFrame, tru: Dict[str,Any], out_dir: Path):
    """Errors vs time (X/Y/Z/Total). PNG and optional HTML."""
    out_dir.mkdir(parents=True, exist_ok=True)

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

    rmse_m = _rmse3(ex,ey,ez)

    # Static PNG — paper grade
    fig = plt.figure(figsize=(11,7.5))
    ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222); ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)
    ax1.plot(tt, ex, color=COL_X, lw=1.6); ax1.set_ylabel("Error X [m]"); ax1.set_xlabel("Relative time [s]"); ax1.grid(True,alpha=.35)
    ax2.plot(tt, ey, color=COL_Y, lw=1.6); ax2.set_ylabel("Error Y [m]"); ax2.set_xlabel("Relative time [s]"); ax2.grid(True,alpha=.35)
    ax3.plot(tt, ez, color=COL_Z, lw=1.6); ax3.set_ylabel("Error Z [m]"); ax3.set_xlabel("Relative time [s]"); ax3.grid(True,alpha=.35)
    ax4.plot(tt, et, color=COL_T, lw=2.2, label=f"RMSE3D={rmse_m:,.0f} m"); ax4.set_ylabel("Total error [m]"); ax4.set_xlabel("Relative time [s]"); ax4.grid(True,alpha=.35); ax4.legend(frameon=False)
    fig.suptitle(f"Estimated error vs time — {tgt}")
    fig.tight_layout(rect=[0,0.03,1,0.94])
    fig.savefig(out_dir/f"{tgt}_errors4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Interactive HTML (Plotly)
    if go is not None and make_subplots is not None:
        figi = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                             subplot_titles=("Error X [m]", "Error Y [m]", "Error Z [m]", f"Total error [m] — RMSE3D={rmse_m:,.0f} m"))
        figi.add_trace(go.Scatter(x=tt, y=ex, name="Ex", line=dict(color=COL_X)), row=1, col=1)
        figi.add_trace(go.Scatter(x=tt, y=ey, name="Ey", line=dict(color=COL_Y)), row=2, col=1)
        figi.add_trace(go.Scatter(x=tt, y=ez, name="Ez", line=dict(color=COL_Z)), row=3, col=1)
        figi.add_trace(go.Scatter(x=tt, y=et, name="E total", line=dict(color=COL_T, width=2)), row=4, col=1)
        figi.update_layout(height=900, width=1100, showlegend=False,
                           xaxis4_title_text="Relative time [s]",
                           title_text=f"Estimated error vs time — {tgt}", margin=dict(l=60,r=20,t=60,b=40))
        html_path = out_dir / f"{tgt}_errors4.html"
        figi.write_html(str(html_path), include_plotlyjs="cdn")
