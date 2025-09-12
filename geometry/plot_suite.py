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

from exports.triangulation.plot_performance_metrics import TRI_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRI_DIR      = PROJECT_ROOT / "exports" / "triangulation"
TRI_CSV_PATH = TRI_DIR / 'xhat_geo_HGV_00010.csv'  # if None, newest xhat_geo_*sats_*.csv will be used
PLOT_DIR     = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# knobs
DOTS_PER_EPOCH      = 60
MAX_EPOCHS_FOR_MC   = 200
ELLIPSOID_SAMPLES   = 10  # wireframe density
ELLIPSOID_SCALE_SIG = 1.0 # 1-sigma ellipsoids
RNG_SEED            = 7

# ---------------- helpers ----------------

def _find_triangulation_csv(tri_dir: Path) -> Path:
    cands = sorted(tri_dir.glob("xhat_geo_*sats_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No triangulation CSVs in {tri_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


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


# ---------------- main ----------------

def main():
    tri_csv = TRI_CSV_PATH or _find_triangulation_csv(TRI_DIR)
    m = re.search(r"HGV_\d+", tri_csv.name)
    if not m:
        raise RuntimeError(f"Could not infer TARGET_ID from {tri_csv.name}")
    target_id = m.group(0)
    oem_path = PROJECT_ROOT / "exports" / "target_exports" / "OUTPUT_OEM" / f"{target_id}.oem"

    tri = _read_triangulation(tri_csv)
    truth = _read_oem_ccsds(oem_path)

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

    out1 = PLOT_DIR / f"suite_3d_{target_id}.png"
    plt.tight_layout(); fig.savefig(out1, dpi=160)
    print(f"[PLOT] {out1}")

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

    out2 = PLOT_DIR / f"suite_timeseries_{target_id}.png"
    plt.tight_layout(); fig2.savefig(out2, dpi=160)
    print(f"[PLOT] {out2}")

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

    out3 = PLOT_DIR / f"suite_correlations_{target_id}.png"
    plt.tight_layout(); fig3.savefig(out3, dpi=160)
    print(f"[PLOT] {out3}")

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
        out4 = PLOT_DIR / f"suite_hist_{target_id}.png"
        plt.tight_layout(); fig4.savefig(out4, dpi=160)
        print(f"[PLOT] {out4}")

    print("[OK] Plot suite complete.")


if __name__ == "__main__":
    main()
