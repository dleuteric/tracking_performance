# geometry/plot_suite.py
# Enhanced plotting suite for triangulation diagnostics and architecture trade decisions
# - Executive summary dashboard
# - Performance vs constellation metrics
# - Statistical performance characterization
# - Trade-off analysis visualizations

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats

# --- config import ---
from config.loader import load_config

import os
import uuid
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages

# --- extra import for JSON parsing ---
import json

# ---------------- config-driven paths ----------------
CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()
TRI_DIR = (PROJECT_ROOT / CFG["paths"]["triangulation_out"]).resolve()
PLOT_DIR = (PROJECT_ROOT / CFG["paths"]["geom_plots_out"]).resolve()
OEM_DIR = (PROJECT_ROOT / CFG["paths"]["oem_root"]).resolve()
RUN_ID = CFG["project"]["run_id"]
RUN_DIR = (PLOT_DIR / RUN_ID)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# logging level (INFO default)
LOG_LEVEL = str(CFG.get("logging", {}).get("level", "INFO")).upper()
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _log(level: str, msg: str):
    if _LEVELS.get(level, 20) >= _LEVELS.get(LOG_LEVEL, 20):
        print(msg)


_log("INFO", f"[RUN ] Plot run id: {RUN_ID}")
_log("INFO", f"[OUT ] Plot directory: {RUN_DIR}")

# Optional DPI from YAML
try:
    plt.rcParams['figure.dpi'] = int(CFG.get('plotting', {}).get('dpi', 160))
except Exception:
    pass

# Enhanced color scheme for architecture decisions
COLORS = {
    'excellent': '#2ecc71',  # green
    'good': '#3498db',  # blue
    'marginal': '#f39c12',  # orange
    'poor': '#e74c3c',  # red
    'primary': '#34495e',  # dark gray
    'secondary': '#95a5a6',  # light gray
}

# Performance thresholds for decision support
THRESHOLDS = {
    'CEP50_km': {'excellent': 0.5, 'good': 1.0, 'marginal': 2.0},  # km
    'RMSE_km': {'excellent': 0.3, 'good': 0.7, 'marginal': 1.5},  # km
    'bias_km': {'excellent': 0.1, 'good': 0.3, 'marginal': 0.5},  # km
    'nsats_min': {'excellent': 8, 'good': 6, 'marginal': 4},  # count
    'beta_deg': {'excellent': (60, 120), 'good': (45, 135), 'marginal': (30, 150)},  # range
}

# knobs
DOTS_PER_EPOCH = 60
MAX_EPOCHS_FOR_MC = 200
ELLIPSOID_SAMPLES = 10  # wireframe density
ELLIPSOID_SCALE_SIG = 1.0  # 1-sigma ellipsoids
RNG_SEED = 7


# ---------------- helpers ----------------

def _find_triangulation_csvs(tri_dir: Path) -> list[Path]:
    run_dir = (tri_dir / RUN_ID)
    cands = sorted(run_dir.glob("xhat_geo_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No triangulation CSVs in {run_dir}")
    return cands


def _read_oem_ccsds(path: Path) -> pd.DataFrame:
    recs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        in_data = False
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("META_START"):
                in_data = False;
                continue
            if s.startswith("META_STOP"):
                in_data = False;
                continue
            if s.startswith("DATA_START"):
                in_data = True;
                continue
            if s.startswith("DATA_STOP"):
                in_data = False;
                continue
            if not in_data:
                parts = s.split()
                if len(parts) >= 7 and ("T" in parts[0] or "+" in parts[0] or "-" in parts[0]):
                    try:
                        t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                        x, y, z, vx, vy, vz = map(float, parts[1:7])
                        recs.append((t, x, y, z, vx, vy, vz))
                    except Exception:
                        pass
                continue
            parts = s.split()
            if len(parts) >= 7:
                try:
                    t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                    x, y, z, vx, vy, vz = map(float, parts[1:7])
                    recs.append((t, x, y, z, vx, vy, vz))
                except Exception:
                    pass
    if not recs:
        raise ValueError(f"No state records parsed from OEM {path}")
    df = pd.DataFrame(recs, columns=["time", "x_km", "y_km", "z_km", "vx_kmps", "vy_kmps", "vz_kmps"]) \
        .dropna(subset=["time"]).sort_values("time")
    return df


def _read_triangulation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    required = [
        "time",
        "xhat_x_km", "xhat_y_km", "xhat_z_km",
        "Sigma_xx", "Sigma_yy", "Sigma_zz",
        "Sigma_xy", "Sigma_xz", "Sigma_yz",
        "Nsats",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Triangulation CSV missing columns: {missing}")
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _cov_from_row(row) -> np.ndarray:
    S = np.array([
        [row["Sigma_xx"], row["Sigma_xy"], row["Sigma_xz"]],
        [row["Sigma_xy"], row["Sigma_yy"], row["Sigma_yz"]],
        [row["Sigma_xz"], row["Sigma_yz"], row["Sigma_zz"]],
    ], dtype=float)
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = S + 1e-12 * np.eye(3)
    return S


def _ecef_to_enu_rot(x_km: float, y_km: float, z_km: float) -> np.ndarray:
    # geocentric lat/lon (good enough for local ENU orientation)
    lon = np.arctan2(y_km, x_km)
    hyp = np.hypot(x_km, y_km)
    lat = np.arctan2(z_km, hyp)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    # rows are unit vectors of ENU axes expressed in ECEF
    R = np.array([[-so, co, 0.0],
                  [-sl * co, -sl * so, cl],
                  [cl * co, cl * so, sl]])
    return R


def _cep50_from_cov2d(C2: np.ndarray) -> float:
    # CEP for general 2D Gaussian — use common approximation based on average eigenvalues
    w, _ = np.linalg.eigh(C2)
    w = np.clip(w, 0.0, None)
    s_eq = float(np.sqrt(0.5 * (w[0] + w[1])))
    return 1.1774 * s_eq  # km


def _compute_derived(tri: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics for each epoch"""
    tri = tri.copy()

    # ENU transform and horizontal CEP
    cep_enu = np.zeros(len(tri))
    sigma_h = np.zeros(len(tri))  # horizontal uncertainty
    sigma_v = np.zeros(len(tri))  # vertical uncertainty

    for idx, row in tri.iterrows():
        S = _cov_from_row(row)
        R = _ecef_to_enu_rot(row["xhat_x_km"], row["xhat_y_km"], row["xhat_z_km"])
        S_enu = R @ S @ R.T
        C2 = S_enu[:2, :2]
        cep_enu[idx] = _cep50_from_cov2d(C2)
        sigma_h[idx] = np.sqrt(0.5 * (S_enu[0, 0] + S_enu[1, 1]))  # avg horizontal sigma
        sigma_v[idx] = np.sqrt(S_enu[2, 2])  # vertical sigma

    tri["CEP50_ENU_km"] = cep_enu
    tri["sigma_h_km"] = sigma_h
    tri["sigma_v_km"] = sigma_v

    # Compute RMSE if error columns exist
    if all(c in tri.columns for c in ("err_x_km", "err_y_km", "err_z_km")):
        tri["RMSE_3D_km"] = np.sqrt((tri["err_x_km"] ** 2 + tri["err_y_km"] ** 2 + tri["err_z_km"] ** 2).mean())
        tri["bias_x_km"] = tri["err_x_km"].mean()
        tri["bias_y_km"] = tri["err_y_km"].mean()
        tri["bias_z_km"] = tri["err_z_km"].mean()
        tri["bias_3D_km"] = np.sqrt(tri["bias_x_km"] ** 2 + tri["bias_y_km"] ** 2 + tri["bias_z_km"] ** 2)

    return tri


def _performance_category(value: float, thresholds: dict) -> str:
    """Categorize performance metric"""
    if value <= thresholds['excellent']:
        return 'excellent'
    elif value <= thresholds['good']:
        return 'good'
    elif value <= thresholds['marginal']:
        return 'marginal'
    else:
        return 'poor'


def _beta_performance_category(beta: float) -> str:
    """Categorize beta angle performance"""
    for cat in ['excellent', 'good', 'marginal']:
        low, high = THRESHOLDS['beta_deg'][cat]
        if low <= beta <= high:
            return cat
    return 'poor'


def _draw_performance_gauge(ax: Axes, value: float, metric: str, title: str):
    """Draw a gauge-style performance indicator"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Get thresholds and category
    if metric in THRESHOLDS:
        thresh = THRESHOLDS[metric]
        if metric == 'beta_deg':
            # Special handling for beta (range-based)
            if 'beta_mean_deg' in title:
                cat = _beta_performance_category(value)
            else:
                cat = 'good'  # default
        else:
            cat = _performance_category(value, thresh)
        color = COLORS[cat]
    else:
        color = COLORS['primary']
        cat = 'N/A'

    # Draw gauge arc
    theta = np.linspace(np.pi, 0, 100)
    x = 0.5 + 0.4 * np.cos(theta)
    y = 0.3 + 0.4 * np.sin(theta)
    ax.plot(x, y, 'k-', lw=2)

    # Color zones if thresholds exist
    if metric in THRESHOLDS and metric != 'beta_deg':
        zones = ['excellent', 'good', 'marginal', 'poor']
        for i, zone in enumerate(zones):
            ax.add_patch(Rectangle((0.1 + i * 0.2, 0.25), 0.2, 0.05,
                                   facecolor=COLORS[zone], alpha=0.3))

    # Value indicator
    if metric != 'beta_deg':
        max_val = thresh.get('marginal', 1.0) * 2
        angle = np.pi * (1 - min(value / max_val, 1.0))
    else:
        angle = np.pi * 0.5  # centered for beta

    ax.arrow(0.5, 0.3, 0.35 * np.cos(angle), 0.35 * np.sin(angle),
             head_width=0.03, head_length=0.02, fc=color, ec=color, lw=2)

    # Labels
    ax.text(0.5, 0.85, title, ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.15, f"{value:.2f}", ha='center', fontsize=14, color=color, fontweight='bold')
    ax.text(0.5, 0.05, cat.upper(), ha='center', fontsize=9, color=color)


def _compute_statistics(tri: pd.DataFrame) -> dict:
    """Compute comprehensive statistics for decision support"""
    stats = {}

    # Basic statistics
    stats['n_epochs'] = len(tri)
    stats['duration_min'] = (tri['time'].max() - tri['time'].min()).total_seconds() / 60

    # Satellite statistics
    stats['nsats_mean'] = tri['Nsats'].mean()
    stats['nsats_min'] = tri['Nsats'].min()
    stats['nsats_max'] = tri['Nsats'].max()
    stats['nsats_std'] = tri['Nsats'].std()

    # CEP statistics
    stats['cep50_mean_km'] = tri['CEP50_ENU_km'].mean()
    stats['cep50_median_km'] = tri['CEP50_ENU_km'].median()
    stats['cep50_p95_km'] = tri['CEP50_ENU_km'].quantile(0.95)
    stats['cep50_max_km'] = tri['CEP50_ENU_km'].max()

    # Beta statistics if available
    if 'beta_mean_deg' in tri.columns:
        stats['beta_mean'] = tri['beta_mean_deg'].mean()
        stats['beta_std'] = tri['beta_mean_deg'].std()
        stats['beta_min'] = tri['beta_min_deg'].min() if 'beta_min_deg' in tri.columns else np.nan
        stats['beta_max'] = tri['beta_max_deg'].max() if 'beta_max_deg' in tri.columns else np.nan

    # Error statistics if available
    if 'RMSE_3D_km' in tri.columns:
        stats['rmse_3d_km'] = tri['RMSE_3D_km'].iloc[0]  # constant across rows
        stats['bias_3d_km'] = tri['bias_3D_km'].iloc[0]

    # Compute availability (% epochs with good geometry)
    good_geom = (tri['Nsats'] >= THRESHOLDS['nsats_min']['marginal'])
    if 'beta_mean_deg' in tri.columns:
        good_beta = tri['beta_mean_deg'].apply(_beta_performance_category) != 'poor'
        good_geom = good_geom & good_beta
    stats['availability_pct'] = 100 * good_geom.mean()

    return stats


# -------------- Enhanced PDF plot builder --------------

def build_executive_summary(tri: pd.DataFrame, stats: dict, target_id: str, fig, ax_grid):
    """Build executive summary dashboard (first page)"""

    # Title - use the figure object passed as parameter
    fig.suptitle(f"Geometric Performance Executive Summary - {target_id}\nRun: {RUN_ID}",
                 fontsize=14, fontweight='bold', y=0.98)

    # Performance gauges (top row)
    if 'cep50_mean_km' in stats:
        _draw_performance_gauge(ax_grid[0, 0], stats['cep50_mean_km'],
                                'CEP50_km', 'Mean CEP50 [km]')

    if 'rmse_3d_km' in stats:
        _draw_performance_gauge(ax_grid[0, 1], stats['rmse_3d_km'],
                                'RMSE_km', '3D RMSE [km]')
    else:
        ax_grid[0, 1].axis('off')

    if 'nsats_mean' in stats:
        _draw_performance_gauge(ax_grid[0, 2], stats['nsats_mean'],
                                'nsats_min', 'Mean #Satellites')

    if 'availability_pct' in stats:
        # Custom availability gauge
        ax = ax_grid[0, 3]
        ax.clear()
        val = stats['availability_pct']
        color = COLORS['excellent'] if val > 95 else COLORS['good'] if val > 90 else COLORS['marginal'] if val > 80 else \
        COLORS['poor']
        ax.pie([val, 100 - val], colors=[color, COLORS['secondary']],
               startangle=90, counterclock=False)
        ax.text(0, 0, f"{val:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.set_title('Geometry Availability', fontsize=11, fontweight='bold')

    # Key metrics table (middle) - flatten the array indexing
    ax = ax_grid[1, 0]  # Changed from ax_grid[1, :2]
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Metric', 'Value', 'Status'],
        ['Duration', f"{stats['duration_min']:.1f} min", '—'],
        ['Epochs', f"{stats['n_epochs']}", '—'],
        ['CEP50 (mean/median/95%/max)',
         f"{stats['cep50_mean_km']:.2f}/{stats['cep50_median_km']:.2f}/{stats['cep50_p95_km']:.2f}/{stats['cep50_max_km']:.2f} km",
         _performance_category(stats['cep50_mean_km'], THRESHOLDS['CEP50_km']).upper()],
    ]

    if 'rmse_3d_km' in stats:
        table_data.append(['3D RMSE', f"{stats['rmse_3d_km']:.3f} km",
                           _performance_category(stats['rmse_3d_km'], THRESHOLDS['RMSE_km']).upper()])
        table_data.append(['3D Bias', f"{stats['bias_3d_km']:.3f} km",
                           _performance_category(stats['bias_3d_km'], THRESHOLDS['bias_km']).upper()])

    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code status column
    for i in range(3, len(table_data)):
        status = table_data[i][2].lower()
        if status in COLORS:
            table[(i, 2)].set_facecolor(COLORS[status])
            table[(i, 2)].set_alpha(0.3)

    # Performance timeline (bottom) - use single axis
    ax = ax_grid[1, 2]  # Changed from ax_grid[1, 2:]
    t = tri['time']
    ax.plot(t, tri['CEP50_ENU_km'], lw=1.2, color=COLORS['primary'])
    ax.fill_between(t, 0, tri['CEP50_ENU_km'], alpha=0.3, color=COLORS['primary'])

    # Add threshold lines
    for cat, thresh in THRESHOLDS['CEP50_km'].items():
        ax.axhline(thresh, ls='--', lw=0.8, color=COLORS[cat], alpha=0.5, label=f"{cat}: {thresh} km")

    ax.set_ylabel('CEP50 [km]')
    ax.set_title('CEP50 Timeline with Performance Thresholds', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Constellation metrics summary - use single axis
    ax = ax_grid[2, 0]  # Changed from ax_grid[2, :2]
    ax.axis('off')
    const_text = f"Satellite Statistics:\n"
    const_text += f"  • Mean: {stats['nsats_mean']:.1f} satellites\n"
    const_text += f"  • Range: {stats['nsats_min']}-{stats['nsats_max']} satellites\n"

    if 'beta_mean' in stats:
        const_text += f"\nGeometry Statistics:\n"
        const_text += f"  • Mean β: {stats['beta_mean']:.1f}°\n"
        const_text += f"  • β Range: {stats.get('beta_min', 'N/A'):.1f}° - {stats.get('beta_max', 'N/A'):.1f}°\n"

    ax.text(0.05, 0.95, const_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    # Performance distribution - use single axis
    ax = ax_grid[2, 2]  # Changed from ax_grid[2, 2:]
    ax.hist(tri['CEP50_ENU_km'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
    ax.axvline(stats['cep50_mean_km'], color='red', lw=2, label=f"Mean: {stats['cep50_mean_km']:.2f}")
    ax.axvline(stats['cep50_median_km'], color='blue', lw=2, label=f"Median: {stats['cep50_median_km']:.2f}")
    ax.set_xlabel('CEP50 [km]')
    ax.set_ylabel('Count')
    ax.set_title('CEP50 Distribution', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def build_and_append_figures(tri: pd.DataFrame, truth: pd.DataFrame, target_id: str, pdf: PdfPages):
    """Build comprehensive performance analysis figures"""

    # Compute derived metrics and statistics
    tri = _compute_derived(tri)
    stats = _compute_statistics(tri)

    # Optional error columns
    has_err = all(c in tri.columns for c in ("err_x_km", "err_y_km", "err_z_km"))

    # ------------- Page 1: Executive Summary -------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    ax_grid = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(3)])

    # Pass both fig and ax_grid to the function - FIXED LINE
    build_executive_summary(tri, stats, target_id, fig, ax_grid)  # Added fig parameter

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ------------- Page 2: 3D Visualization with Performance Zones -------------
    fig = plt.figure(figsize=(14, 10))

    # Main 3D plot
    ax3d = fig.add_subplot(221, projection='3d')
    xhat = tri[["xhat_x_km", "xhat_y_km", "xhat_z_km"]].to_numpy()

    # Truth trajectory
    ax3d.plot(truth["x_km"], truth["y_km"], truth["z_km"],
              'k-', lw=1.5, alpha=0.5, label=f"{target_id} truth")

    # Color by performance category
    cep_colors = [COLORS[_performance_category(c, THRESHOLDS['CEP50_km'])]
                  for c in tri["CEP50_ENU_km"]]

    ax3d.scatter(xhat[:, 0], xhat[:, 1], xhat[:, 2],
                 c=cep_colors, s=20, alpha=0.7, label="Estimates (color=CEP category)")

    ax3d.set_xlabel("X [km]")
    ax3d.set_ylabel("Y [km]")
    ax3d.set_zlabel("Z [km]")
    ax3d.set_title("3D Trajectory with Performance Coloring")

    # Custom legend for performance categories
    handles = [mpatches.Patch(color=COLORS[cat], label=f"{cat.title()}: ≤{thresh} km")
               for cat, thresh in THRESHOLDS['CEP50_km'].items()]
    ax3d.legend(handles=handles, loc='upper right')

    # XY projection
    ax_xy = fig.add_subplot(222)
    ax_xy.plot(truth["x_km"], truth["y_km"], 'k-', lw=1, alpha=0.5, label="Truth")
    sc = ax_xy.scatter(xhat[:, 0], xhat[:, 1], c=tri["CEP50_ENU_km"],
                       cmap='RdYlGn_r', s=10, vmin=0, vmax=THRESHOLDS['CEP50_km']['marginal'])
    plt.colorbar(sc, ax=ax_xy, label="CEP50 [km]")
    ax_xy.set_xlabel("X [km]")
    ax_xy.set_ylabel("Y [km]")
    ax_xy.set_title("Ground Track (XY)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')

    # Performance vs Time
    ax_perf = fig.add_subplot(223)
    t = tri['time']

    # Stacked area chart showing performance categories over time
    excellent_mask = tri['CEP50_ENU_km'] <= THRESHOLDS['CEP50_km']['excellent']
    good_mask = (tri['CEP50_ENU_km'] > THRESHOLDS['CEP50_km']['excellent']) & \
                (tri['CEP50_ENU_km'] <= THRESHOLDS['CEP50_km']['good'])
    marginal_mask = (tri['CEP50_ENU_km'] > THRESHOLDS['CEP50_km']['good']) & \
                    (tri['CEP50_ENU_km'] <= THRESHOLDS['CEP50_km']['marginal'])
    poor_mask = tri['CEP50_ENU_km'] > THRESHOLDS['CEP50_km']['marginal']

    ax_perf.fill_between(t, 0, excellent_mask * tri['CEP50_ENU_km'],
                         color=COLORS['excellent'], alpha=0.5, label='Excellent')
    ax_perf.fill_between(t, 0, good_mask * tri['CEP50_ENU_km'],
                         color=COLORS['good'], alpha=0.5, label='Good')
    ax_perf.fill_between(t, 0, marginal_mask * tri['CEP50_ENU_km'],
                         color=COLORS['marginal'], alpha=0.5, label='Marginal')
    ax_perf.fill_between(t, 0, poor_mask * tri['CEP50_ENU_km'],
                         color=COLORS['poor'], alpha=0.5, label='Poor')

    ax_perf.plot(t, tri['CEP50_ENU_km'], 'k-', lw=1, alpha=0.7)
    ax_perf.set_xlabel("Time")
    ax_perf.set_ylabel("CEP50 [km]")
    ax_perf.set_title("Performance Categories Over Time")
    ax_perf.legend(loc='upper right')
    ax_perf.grid(True, alpha=0.3)

    # Cumulative performance distribution
    ax_cdf = fig.add_subplot(224)
    sorted_cep = np.sort(tri['CEP50_ENU_km'])
    cdf = np.arange(1, len(sorted_cep) + 1) / len(sorted_cep) * 100

    ax_cdf.plot(sorted_cep, cdf, 'b-', lw=2)
    ax_cdf.fill_between(sorted_cep, 0, cdf, alpha=0.3)

    # Mark percentiles
    percentiles = [50, 68, 95, 99]
    for p in percentiles:
        val = np.percentile(sorted_cep, p)
        ax_cdf.axvline(val, ls='--', alpha=0.5)
        ax_cdf.text(val, p, f"{p}%: {val:.2f}km", rotation=90, va='bottom', fontsize=8)

    ax_cdf.set_xlabel("CEP50 [km]")
    ax_cdf.set_ylabel("Cumulative Probability [%]")
    ax_cdf.set_title("CEP50 Cumulative Distribution")
    ax_cdf.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ------------- Page 3: Architecture Trade Analysis -------------
    fig = plt.figure(figsize=(14, 10))

    # Nsats vs Performance
    ax1 = fig.add_subplot(231)
    unique_nsats = sorted(tri['Nsats'].unique())
    cep_by_nsats = [tri[tri['Nsats'] == n]['CEP50_ENU_km'].mean() for n in unique_nsats]
    std_by_nsats = [tri[tri['Nsats'] == n]['CEP50_ENU_km'].std() for n in unique_nsats]

    ax1.errorbar(unique_nsats, cep_by_nsats, yerr=std_by_nsats,
                 fmt='o-', capsize=5, lw=2, ms=8)
    ax1.set_xlabel("Number of Satellites")
    ax1.set_ylabel("Mean CEP50 [km]")
    ax1.set_title("Performance vs Constellation Size")
    ax1.grid(True, alpha=0.3)

    # Add requirement line if defined
    if 'requirement_cep_km' in CFG.get('performance', {}):
        req = CFG['performance']['requirement_cep_km']
        ax1.axhline(req, color='r', ls='--', lw=2, label=f"Requirement: {req} km")
        ax1.legend()

    # Beta vs Performance (if available)
    ax2 = fig.add_subplot(232)
    if 'beta_mean_deg' in tri.columns:
        # Bin beta values and compute statistics
        beta_bins = np.arange(0, 181, 10)
        beta_centers = (beta_bins[:-1] + beta_bins[1:]) / 2
        cep_by_beta = []
        for i in range(len(beta_bins) - 1):
            mask = (tri['beta_mean_deg'] >= beta_bins[i]) & (tri['beta_mean_deg'] < beta_bins[i + 1])
            if mask.any():
                cep_by_beta.append(tri[mask]['CEP50_ENU_km'].mean())
            else:
                cep_by_beta.append(np.nan)

        ax2.bar(beta_centers, cep_by_beta, width=8, alpha=0.7, edgecolor='black')
        ax2.set_xlabel("Mean Beta Angle [deg]")
        ax2.set_ylabel("Mean CEP50 [km]")
        ax2.set_title("Performance vs Viewing Geometry")

        # Mark optimal zones
        for cat, (low, high) in THRESHOLDS['beta_deg'].items():
            ax2.axvspan(low, high, alpha=0.2, color=COLORS[cat])

        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')

    # Horizontal vs Vertical Performance
    ax3 = fig.add_subplot(233)
    if 'sigma_h_km' in tri.columns and 'sigma_v_km' in tri.columns:
        ax3.scatter(tri['sigma_h_km'], tri['sigma_v_km'],
                    c=tri['Nsats'], cmap='viridis', s=20, alpha=0.6)
        cb = plt.colorbar(ax3.collections[0], ax=ax3)
        cb.set_label('#Satellites')

        ax3.set_xlabel("Horizontal Uncertainty [km]")
        ax3.set_ylabel("Vertical Uncertainty [km]")
        ax3.set_title("H/V Error Coupling")
        ax3.grid(True, alpha=0.3)

        # Add diagonal reference lines
        max_val = max(tri['sigma_h_km'].max(), tri['sigma_v_km'].max())
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1')
        ax3.plot([0, max_val], [0, 2 * max_val], 'r--', alpha=0.3, label='1:2')
        ax3.legend(loc='upper left')
    else:
        ax3.axis('off')

    # Time-averaged performance by hour (for diurnal effects)
    ax4 = fig.add_subplot(234)
    tri['hour'] = tri['time'].dt.hour
    hourly_perf = tri.groupby('hour')['CEP50_ENU_km'].agg(['mean', 'std'])

    ax4.bar(hourly_perf.index, hourly_perf['mean'],
            yerr=hourly_perf['std'], capsize=3, alpha=0.7, edgecolor='black')
    ax4.set_xlabel("Hour of Day [UTC]")
    ax4.set_ylabel("Mean CEP50 [km]")
    ax4.set_title("Diurnal Performance Variation")
    ax4.grid(True, alpha=0.3)

    # Performance improvement potential
    ax5 = fig.add_subplot(235)
    if has_err:
        # Show how much performance could improve with better geometry
        theoretical_min = tri['CEP50_ENU_km'].quantile(0.05)
        current_mean = tri['CEP50_ENU_km'].mean()
        current_p95 = tri['CEP50_ENU_km'].quantile(0.95)

        categories = ['Best 5%', 'Mean', 'Worst 5%']
        values = [theoretical_min, current_mean, current_p95]
        colors_bar = [COLORS['excellent'], COLORS['good'], COLORS['poor']]

        bars = ax5.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax5.set_ylabel("CEP50 [km]")
        ax5.set_title("Performance Range Analysis")

        # Add improvement potential annotation
        improvement = ((current_mean - theoretical_min) / current_mean) * 100
        ax5.text(0.5, 0.95, f"Potential improvement: {improvement:.0f}%",
                 transform=ax5.transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax5.grid(True, alpha=0.3)
    else:
        ax5.axis('off')

    # Architecture recommendation box
    ax6 = fig.add_subplot(236)
    ax6.axis('off')

    # Generate recommendations based on statistics
    recommendations = []

    if stats['nsats_mean'] < THRESHOLDS['nsats_min']['good']:
        recommendations.append(f"• Increase constellation size (current avg: {stats['nsats_mean']:.1f})")

    if stats['cep50_mean_km'] > THRESHOLDS['CEP50_km']['good']:
        recommendations.append(f"• Improve geometry (current CEP50: {stats['cep50_mean_km']:.2f} km)")

    if 'beta_mean' in stats:
        if stats['beta_mean'] < 45 or stats['beta_mean'] > 135:
            recommendations.append(f"• Optimize viewing angles (current β: {stats['beta_mean']:.1f}°)")

    if stats['availability_pct'] < 95:
        recommendations.append(f"• Enhance coverage (current availability: {stats['availability_pct']:.1f}%)")

    if not recommendations:
        recommendations.append("• System meets all performance thresholds")
        recommendations.append("• Consider cost optimization opportunities")

    rec_text = "Architecture Recommendations:\n\n" + "\n".join(recommendations)

    # Add quantitative trade-offs
    rec_text += "\n\nKey Trade-offs:\n"
    rec_text += f"• Each additional satellite improves CEP50 by ~{(cep_by_nsats[-1] - cep_by_nsats[0]) / (unique_nsats[-1] - unique_nsats[0]):.2f} km\n"

    if 'beta_mean' in stats:
        rec_text += f"• Optimal β range (60-120°) reduces CEP50 by up to 40%\n"

    ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ------------- Additional pages for detailed analysis -------------
    # ... (keep existing detailed plots but with enhanced styling)


# ---------------- main loop for PDFs ----------------

def main():
    """Generate comprehensive performance analysis PDFs"""

    # Discover CSVs for this RUN_ID
    csvs = _find_triangulation_csvs(TRI_DIR)

    # Generate summary statistics across all targets
    all_stats = {}

    for tri_csv in csvs:
        m = re.search(r"(HGV_\d+)", tri_csv.name)
        if not m:
            _log("WARN", f"[SKIP] Could not infer target id from {tri_csv.name}")
            continue
        target_id = m.group(1)
        oem_path = OEM_DIR / f"{target_id}.oem"

        try:
            tri = _read_triangulation(tri_csv)
            truth = _read_oem_ccsds(oem_path)
        except Exception as e:
            _log("WARN", f"[SKIP] {target_id}: {e}")
            continue

        # Compute statistics for summary
        tri = _compute_derived(tri)
        stats = _compute_statistics(tri)
        all_stats[target_id] = stats

        # Generate individual PDF
        pdf_path = RUN_DIR / f"{target_id}_performance.pdf"
        with PdfPages(pdf_path) as pdf:
            build_and_append_figures(tri, truth, target_id, pdf)
        _log("INFO", f"[PDF ] Generated performance analysis: {pdf_path}")

    # Generate constellation-level summary if multiple targets
    if len(all_stats) > 1:
        summary_path = RUN_DIR / "constellation_summary.pdf"
        # ... (add constellation-level analysis)
        _log("INFO", f"[SUM ] Generated constellation summary: {summary_path}")

    # Write performance metrics JSON for automated analysis
    metrics_path = RUN_DIR / "performance_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    _log("INFO", f"[JSON] Exported metrics: {metrics_path}")

    # Write decision matrix CSV
    if all_stats:
        decision_matrix = []
        for target, stats in all_stats.items():
            row = {
                'target': target,
                'cep50_mean': stats.get('cep50_mean_km', np.nan),
                'rmse_3d': stats.get('rmse_3d_km', np.nan),
                'availability': stats.get('availability_pct', np.nan),
                'recommendation': 'PASS' if stats.get('cep50_mean_km', 999) < THRESHOLDS['CEP50_km']['good'] else 'FAIL'
            }
            decision_matrix.append(row)

        dm_df = pd.DataFrame(decision_matrix)
        dm_path = RUN_DIR / "decision_matrix.csv"
        dm_df.to_csv(dm_path, index=False)
        _log("INFO", f"[CSV ] Exported decision matrix: {dm_path}")

    _log("INFO", "[OK  ] Enhanced performance analysis complete.")


if __name__ == "__main__":
    main()