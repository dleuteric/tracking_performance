# geometry/plot_suite.py
# Full plotting suite for triangulation diagnostics
# - 3D truth + x̂ with CEP coloring and sparse covariance ellipsoids
# - Time series: Nsats, beta_mean_deg, CEP50_ENU, per-axis errors (bias-removed), err_norm
# - Scatter correlations: CEP50_ENU vs beta_mean, |err| vs beta_mean, |err_y| vs Nsats
# - Histograms: per-axis errors (bias-removed)

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.colors import Normalize

# --- config import ---
from config.loader import load_config

import os
import uuid
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages

# ---------------- config-driven paths ----------------
CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()
TRI_DIR      = (PROJECT_ROOT / CFG["paths"]["triangulation_out"]).resolve()
PLOT_DIR     = (PROJECT_ROOT / CFG["paths"]["geom_plots_out"]).resolve()
OEM_DIR      = (PROJECT_ROOT / CFG["paths"]["oem_root"]).resolve()
RUN_ID       = CFG["project"]["run_id"]
RUN_DIR      = (PLOT_DIR / RUN_ID)
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

# knobs
DOTS_PER_EPOCH      = 60
MAX_EPOCHS_FOR_MC   = 200
ELLIPSOID_SAMPLES   = 10  # wireframe density
ELLIPSOID_SCALE_SIG = 1.0 # 1-sigma ellipsoids
RNG_SEED            = 7

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
                in_data = False; continue
            if s.startswith("META_STOP"):
                in_data = False; continue
            if s.startswith("DATA_START"):
                in_data = True; continue
            if s.startswith("DATA_STOP"):
                in_data = False; continue
            if not in_data:
                parts = s.split()
                if len(parts) >= 7 and ("T" in parts[0] or "+" in parts[0] or "-" in parts[0]):
                    try:
                        t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                        x,y,z,vx,vy,vz = map(float, parts[1:7])
                        recs.append((t,x,y,z,vx,vy,vz))
                    except Exception:
                        pass
                continue
            parts = s.split()
            if len(parts) >= 7:
                try:
                    t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                    x,y,z,vx,vy,vz = map(float, parts[1:7])
                    recs.append((t,x,y,z,vx,vy,vz))
                except Exception:
                    pass
    if not recs:
        raise ValueError(f"No state records parsed from OEM {path}")
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"])\
            .dropna(subset=["time"]).sort_values("time")
    return df


def _read_triangulation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    required = [
        "time",
        "xhat_x_km","xhat_y_km","xhat_z_km",
        "Sigma_xx","Sigma_yy","Sigma_zz",
        "Sigma_xy","Sigma_xz","Sigma_yz",
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
        S = S + 1e-12*np.eye(3)
    return S


def _ecef_to_enu_rot(x_km: float, y_km: float, z_km: float) -> np.ndarray:
    # geocentric lat/lon (good enough for local ENU orientation)
    lon = np.arctan2(y_km, x_km)
    hyp = np.hypot(x_km, y_km)
    lat = np.arctan2(z_km, hyp)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    # rows are unit vectors of ENU axes expressed in ECEF
    R = np.array([[-so,          co,          0.0],
                  [-sl*co, -sl*so,  cl],
                  [ cl*co,  cl*so,  sl]])
    return R


def _cep50_from_cov2d(C2: np.ndarray) -> float:
    # CEP for general 2D Gaussian — use common approximation based on average eigenvalues
    w, _ = np.linalg.eigh(C2)
    w = np.clip(w, 0.0, None)
    s_eq = float(np.sqrt(0.5*(w[0] + w[1])))
    return 1.1774 * s_eq  # km


def _draw_cov_ellipsoid(ax: Axes, center: np.ndarray, Sigma: np.ndarray, scale: float = 1.0, n: int = 10):
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 0.0)
    radii = scale * np.sqrt(vals)
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    E = np.stack([radii[0]*xs, radii[1]*ys, radii[2]*zs], axis=0).reshape(3, -1)
    E = (vecs @ E).reshape(3, xs.shape[0], xs.shape[1])
    X = E[0] + center[0]; Y = E[1] + center[1]; Z = E[2] + center[2]
    ax.plot_wireframe(X, Y, Z, linewidth=0.6, rstride=1, cstride=1, alpha=0.3)


def _compute_derived(tri: pd.DataFrame) -> pd.DataFrame:
    # ENU transform and horizontal CEP
    xs = tri[["xhat_x_km","xhat_y_km","xhat_z_km"]].to_numpy()
    cep_enu = np.zeros(len(tri))
    for idx, row in tri.iterrows():
        S = _cov_from_row(row)
        R = _ecef_to_enu_rot(row["xhat_x_km"], row["xhat_y_km"], row["xhat_z_km"])
        S_enu = R @ S @ R.T
        C2 = S_enu[:2, :2]
        cep_enu[idx] = _cep50_from_cov2d(C2)
    tri = tri.copy()
    tri["CEP50_ENU_km"] = cep_enu
    return tri


def _binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray):
    idx = np.digitize(x, bins) - 1
    out = []
    for b in range(len(bins)-1):
        m = idx == b
        if not np.any(m):
            out.append((np.nan, np.nan, np.nan, 0))
            continue
        xb = 0.5*(bins[b]+bins[b+1])
        yb = np.nanmean(y[m])
        ys = np.nanstd(y[m])
        out.append((xb, yb, ys, int(m.sum())))
    arr = np.array(out, dtype=float)
    return arr  # columns: x_bin_center, y_mean, y_std, count


def _scatter_binned(ax: Axes, x, y, nsats=None, bin_deg=0.5, show_points=False, title=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    bins = np.arange(np.nanmin(x)-1e-6, np.nanmax(x)+bin_deg+1e-6, bin_deg)
    stats = _binned_stats(x, y, bins)
    # optional background cloud
    if show_points:
        c = nsats if nsats is not None else 'k'
        ax.scatter(x, y, s=6, alpha=0.3, c=c, cmap='viridis')
    # mean±std as errorbar
    ax.errorbar(stats[:,0], stats[:,1], yerr=stats[:,2], fmt='o-', lw=1.0, ms=3)
    ax.grid(True, linestyle=":", alpha=0.5)
    if title: ax.set_title(title)


def _parse_beta_pairs(row) -> tuple[list[str], dict[tuple[str,str], float]]:
    """Parse 'beta_pairs_deg' like '001-004:95.2|003-004:88.1|...'
    Returns (ids, {(i,j): angle_deg}).
    """
    txt = str(row.get('beta_pairs_deg', ''))
    parts = [p for p in txt.split('|') if p]
    pairs = {}
    ids = set()
    for p in parts:
        try:
            left, ang = p.split(':')
            a,b = left.split('-')
            a=a.strip(); b=b.strip(); ang=float(ang)
            key = (a,b) if a<b else (b,a)
            pairs[key] = ang
            ids.add(a); ids.add(b)
        except Exception:
            continue
    ids = sorted(ids)
    return ids, pairs


def _beta_matrix(ids: list[str], pairs: dict[tuple[str,str], float]) -> np.ndarray:
    n = len(ids)
    M = np.full((n,n), np.nan, float)
    for i in range(n):
        M[i,i] = 0.0
    idx = {s:i for i,s in enumerate(ids)}
    for (a,b), ang in pairs.items():
        if a in idx and b in idx:
            i,j = idx[a], idx[b]
            M[i,j] = M[j,i] = ang
    return M


def _plot_beta_matrix(ax: Axes, row, title="β matrix (deg)"):
    ids, pairs = _parse_beta_pairs(row)
    if len(ids) < 2:
        ax.axis('off'); return
    M = _beta_matrix(ids, pairs)
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=180)
    ax.set_title(title)
    ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=6)
    ax.set_yticklabels(ids, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _score_subset_beta(ids_subset: list[str], pairs: dict[tuple[str,str], float]) -> float:
    # Proxy score: favor angles vicini a 90°, penalizza troppo stretti o opposti
    # score = min(g) + mean(g), g = 1 - |beta-90|/90 clipped to [0,1]
    if len(ids_subset) < 2: return -1e9
    vals = []
    for i in range(len(ids_subset)):
        for j in range(i+1, len(ids_subset)):
            a,b = ids_subset[i], ids_subset[j]
            key = (a,b) if a<b else (b,a)
            ang = pairs.get(key, np.nan)
            if np.isnan(ang):
                continue
            g = 1.0 - min(abs(ang-90.0)/90.0, 1.0)
            vals.append(g)
    if not vals: return -1e9
    vals = np.array(vals)
    return float(np.nanmin(vals) + np.nanmean(vals))


def _greedy_best_subset(ids: list[str], pairs: dict[tuple[str,str], float], m: int) -> list[str]:
    # Start from best pair (closest to 90), then add the element that maximizes score increase
    if len(ids) <= m: return ids
    # best pair
    best_pair = None; best_pair_score = -1e9
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            key = (ids[i], ids[j]) if ids[i]<ids[j] else (ids[j], ids[i])
            ang = pairs.get(key, np.nan)
            if np.isnan(ang):
                continue
            g = 1.0 - min(abs(ang-90.0)/90.0, 1.0)
            if g > best_pair_score:
                best_pair_score = g; best_pair = [ids[i], ids[j]]
    if not best_pair:
        return ids[:m]
    S = best_pair.copy()
    while len(S) < m:
        best_add = None; best_score = -1e9
        rem = [u for u in ids if u not in S]
        for u in rem:
            cand = S + [u]
            sc = _score_subset_beta(cand, pairs)
            if sc > best_score:
                best_score = sc; best_add = u
        if best_add is None:
            break
        S.append(best_add)
    return S



# -------------- PDF plot builder --------------

def build_and_append_figures(tri: pd.DataFrame, truth: pd.DataFrame, target_id: str, pdf: PdfPages):
    # Optional error columns
    has_err = all(c in tri.columns for c in ("err_x_km","err_y_km","err_z_km"))
    # Derived metrics
    tri = _compute_derived(tri)

    # ------------- Figure 1: 3D overview -------------
    rng = np.random.default_rng(RNG_SEED)
    xhat = tri[["xhat_x_km","xhat_y_km","xhat_z_km"]].to_numpy()

    fig = plt.figure(figsize=(11, 8))
    ax: Axes = fig.add_subplot(111, projection='3d')
    ax.plot(truth["x_km"], truth["y_km"], truth["z_km"], lw=1.0, label=f"{target_id} truth (OEM)")
    sc = ax.scatter(xhat[:,0], xhat[:,1], xhat[:,2], s=16, marker='x', c=tri["CEP50_ENU_km"], cmap=cm.viridis, depthshade=False, label="x̂ (colored by CEP50_ENU)")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label("CEP50_ENU [km]")
    # sparse ellipsoids
    step = max(1, len(tri)//12)
    for _, row in tri.iloc[::step].iterrows():
        mu = np.array([row["xhat_x_km"], row["xhat_y_km"], row["xhat_z_km"]], dtype=float)
        S  = _cov_from_row(row)
        _draw_cov_ellipsoid(ax, mu, S, scale=ELLIPSOID_SCALE_SIG, n=ELLIPSOID_SAMPLES)
    # axes
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="upper right")
    ax.set_title("3D: Truth + x̂ colored by CEP50_ENU")
    # equal-ish bounds
    xs, ys, zs = truth["x_km"].to_numpy(), truth["y_km"].to_numpy(), truth["z_km"].to_numpy()
    xr = (xs.min(), xs.max()); yr = (ys.min(), ys.max()); zr = (zs.min(), zs.max())
    centers = np.array([np.mean(xr), np.mean(yr), np.mean(zr)])
    spans = np.array([xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]])
    R = 0.5 * float(spans.max())
    ax.set_xlim(centers[0]-R, centers[0]+R)
    ax.set_ylim(centers[1]-R, centers[1]+R)
    ax.set_zlim(centers[2]-R, centers[2]+R)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ------------- Figure 2: Time series panel -------------
    t = tri["time"]
    nrows = 5 if has_err else 3
    fig2, axes = plt.subplots(nrows, 1, figsize=(11, 2.2*nrows), sharex=True)
    k = 0
    axes[k].plot(t, tri["Nsats"], marker='.', lw=0.8)
    axes[k].set_ylabel("Nsats")
    axes[k].grid(True, linestyle=":", alpha=0.5)
    k += 1
    if all(c in tri.columns for c in ("beta_mean_deg","beta_min_deg","beta_max_deg")):
        axes[k].plot(t, tri["beta_mean_deg"], lw=1.0, label="β_mean")
        axes[k].plot(t, tri["beta_min_deg"], lw=0.8, label="β_min")
        axes[k].plot(t, tri["beta_max_deg"], lw=0.8, label="β_max")
        axes[k].set_ylabel("β [deg]")
        axes[k].legend(loc="upper right")
        axes[k].grid(True, linestyle=":", alpha=0.5)
        k += 1
    axes[k].plot(t, tri["CEP50_ENU_km"], lw=1.0)
    axes[k].set_ylabel("CEP50_ENU [km]")
    axes[k].grid(True, linestyle=":", alpha=0.5)
    k += 1
    if has_err:
        mx, my, mz = tri["err_x_km"].mean(), tri["err_y_km"].mean(), tri["err_z_km"].mean()
        ex = (tri["err_x_km"] - mx).abs(); ey = (tri["err_y_km"] - my).abs(); ez = (tri["err_z_km"] - mz).abs()
        axes[k].plot(t, ex, lw=1.0, label="|err_x| (bias-out)")
        axes[k].plot(t, ey, lw=1.0, label="|err_y| (bias-out)")
        axes[k].plot(t, ez, lw=1.0, label="|err_z| (bias-out)")
        axes[k].set_ylabel("|err| [km]")
        axes[k].legend(loc="upper right")
        axes[k].grid(True, linestyle=":", alpha=0.5)
        k += 1
        axes[k].plot(t, tri.get("err_norm_km", (ex**2+ey**2+ez**2)**0.5), lw=1.0)
        axes[k].set_ylabel("‖err‖ [km]")
        axes[k].grid(True, linestyle=":", alpha=0.5)
        k += 1
    axes[k-1].set_xlabel("time [UTC]")
    fig2.autofmt_xdate()
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    # ------------- Figure 3: Correlations -------------
    fig3, axes3 = plt.subplots(1, 3 if has_err else 2, figsize=(13, 4))
    # CEP vs beta_mean
    if "beta_mean_deg" in tri.columns:
        axes3[0].scatter(tri["beta_mean_deg"], tri["CEP50_ENU_km"], s=12, alpha=0.7)
        axes3[0].set_xlabel("β_mean [deg]")
        axes3[0].set_ylabel("CEP50_ENU [km]")
        axes3[0].grid(True, linestyle=":", alpha=0.5)
    else:
        axes3[0].axis('off')
    # |err| vs beta_mean
    if has_err and "beta_mean_deg" in tri.columns:
        axes3[1].scatter(tri["beta_mean_deg"], (tri["err_x_km"]-tri["err_x_km"].mean()).abs(), s=10, alpha=0.6, label='|err_x|')
        axes3[1].scatter(tri["beta_mean_deg"], (tri["err_y_km"]-tri["err_y_km"].mean()).abs(), s=10, alpha=0.6, label='|err_y|')
        axes3[1].scatter(tri["beta_mean_deg"], (tri["err_z_km"]-tri["err_z_km"].mean()).abs(), s=10, alpha=0.6, label='|err_z|')
        axes3[1].set_xlabel("β_mean [deg]")
        axes3[1].set_ylabel("|err| (bias-out) [km]")
        axes3[1].legend(loc='upper right')
        axes3[1].grid(True, linestyle=":", alpha=0.5)
    else:
        axes3[1].axis('off')
    # |err_y| vs Nsats
    if has_err:
        axes3[2 if has_err else 1].scatter(tri["Nsats"], (tri["err_y_km"]-tri["err_y_km"].mean()).abs(), s=12, alpha=0.7)
        axes3[2 if has_err else 1].set_xlabel("Nsats")
        axes3[2 if has_err else 1].set_ylabel("|err_y| (bias-out) [km]")
        axes3[2 if has_err else 1].grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

    # ------------- Figure 4: Histograms (bias-removed) -------------
    if has_err:
        fig4, ax4 = plt.subplots(1,3, figsize=(12,3.6), sharey=True)
        mx, my, mz = tri["err_x_km"].mean(), tri["err_y_km"].mean(), tri["err_z_km"].mean()
        ax4[0].hist(tri["err_x_km"]-mx, bins=30)
        ax4[0].set_title("err_x (bias-out)")
        ax4[1].hist(tri["err_y_km"]-my, bins=30)
        ax4[1].set_title("err_y (bias-out)")
        ax4[2].hist(tri["err_z_km"]-mz, bins=30)
        ax4[2].set_title("err_z (bias-out)")
        for a in ax4:
            a.grid(True, linestyle=":", alpha=0.4)
            a.set_xlabel("km")
        ax4[0].set_ylabel("count")
        plt.tight_layout()
        pdf.savefig(fig4)
        plt.close(fig4)

    # ------------- Figure 5: Geometry Insights (binned + subset) -------------
    fig5, ax5 = plt.subplots(1, 2, figsize=(11, 4))
    if "beta_mean_deg" in tri.columns:
        _scatter_binned(ax5[0], tri["beta_mean_deg"], tri["CEP50_ENU_km"], nsats=tri["Nsats"], bin_deg=0.5, show_points=False, title="CEP50_ENU vs β_mean (binned)")
        ax5[0].set_xlabel("β_mean [deg]"); ax5[0].set_ylabel("CEP50_ENU [km]")
    else:
        ax5[0].axis('off')

    # Best-subset proxy time series (m=3..5)
    has_pairs = 'beta_pairs_deg' in tri.columns and tri['beta_pairs_deg'].notna().any()
    if has_pairs:
        ms = [3,4,5]
        t = tri['time']
        for msz in ms:
            scores = []
            for _, row in tri.iterrows():
                ids, pairs = _parse_beta_pairs(row)
                if len(ids) < msz:
                    scores.append(np.nan); continue
                Sbest = _greedy_best_subset(ids, pairs, msz)
                sc = _score_subset_beta(Sbest, pairs)
                scores.append(sc)
            ax5[1].plot(t, scores, lw=1.0, label=f"greedy β-proxy, m={msz}")
        ax5[1].set_title("Best-subset proxy score vs time")
        ax5[1].set_ylabel("score (↑ better)"); ax5[1].grid(True, linestyle=":", alpha=0.5)
        ax5[1].legend(loc='upper right')
    else:
        ax5[1].axis('off')
    plt.tight_layout()
    pdf.savefig(fig5); plt.close(fig5)

    # ------------- Figure 6: β-matrix heatmaps for 3 representative epochs -------------
    if has_pairs:
        # pick three epochs: min CEP, median CEP, max CEP
        cep = tri["CEP50_ENU_km"].to_numpy()
        good = np.isfinite(cep)
        if np.any(good):
            idxs = np.where(good)[0]
            i_min = idxs[np.argmin(cep[good])]
            i_med = idxs[len(idxs)//2]
            i_max = idxs[np.argmax(cep[good])]
            picks = [(i_min, 'min CEP'), (i_med, 'median CEP'), (i_max, 'max CEP')]
            fig6, ax6 = plt.subplots(1, 3, figsize=(12, 3.6))
            for k,(ix,lab) in enumerate(picks):
                _plot_beta_matrix(ax6[k], tri.iloc[ix], title=f"β matrix — {lab}")
            plt.tight_layout(); pdf.savefig(fig6); plt.close(fig6)


# ---------------- main loop for PDFs ----------------

def main():
    # Discover CSVs for this RUN_ID
    csvs = _find_triangulation_csvs(TRI_DIR)

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

        pdf_path = RUN_DIR / f"{target_id}.pdf"
        with PdfPages(pdf_path) as pdf:
            build_and_append_figures(tri, truth, target_id, pdf)
        _log("INFO", f"[PDF ] Wrote {pdf_path}")

    # Write a simple manifest of processed targets
    try:
        manifest = RUN_DIR / "manifest.txt"
        with manifest.open("w") as mf:
            mf.write(f"plot_run_id={RUN_ID}\n")
            mf.write(f"generated_utc={datetime.utcnow().isoformat()}Z\n")
            mf.write("files=\n")
            for p in sorted(RUN_DIR.glob("*.pdf")):
                mf.write(f"  - {p.name}\n")
        _log("INFO", f"[MAN ] Wrote {manifest}")
    except Exception as e:
        _log("WARN", f"[WARN] Could not write manifest: {e}")

    _log("INFO", "[OK ] Plot suite PDF generation complete.")


if __name__ == "__main__":
    main()
