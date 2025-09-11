# geometry/plot_debug.py
# Purpose: Plot truth trajectory (from OEM), triangulated fixes x̂, and MC clouds from Σ_geo.
# Uses triangulation CSV to infer TARGET_ID and the epoch set.

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRI_DIR      = PROJECT_ROOT / "exports" / "triangulation"
# If you want a specific file, set it here; otherwise newest matching file will be used.
TRI_CSV_PATH = None  # e.g., TRI_DIR / "xhat_geo_36sats_HGV_00001.csv"

# Plot knobs
DOTS_PER_EPOCH      = 100
MAX_EPOCHS_TO_PLOT  = 300
RNG_SEED            = 42
OVERLAY_SATS        = True
MAX_OVERLAY_EPOCHS  = 12
JOIN_TOL_S          = 1.0  # tolerance when matching LOS/EPH to triangulation epochs

# --- Small helpers ---

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
                # accept column-style lines even without DATA_ markers
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
    # parse ISO times
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    need = [
        "time",
        "xhat_x_km","xhat_y_km","xhat_z_km",
        "Sigma_xx","Sigma_yy","Sigma_zz",
        "Sigma_xy","Sigma_xz","Sigma_yz",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Triangulation CSV missing columns: {missing}")
    return df.dropna(subset=["time"]).sort_values("time")


def _cov_from_row(row) -> np.ndarray:
    S = np.array([
        [row["Sigma_xx"], row["Sigma_xy"], row["Sigma_xz"]],
        [row["Sigma_xy"], row["Sigma_yy"], row["Sigma_yz"]],
        [row["Sigma_xz"], row["Sigma_yz"], row["Sigma_zz"]],
    ], dtype=float)
    # Ensure SPD
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = S + 1e-12*np.eye(3)
    return S


def _sample_cloud(mu: np.ndarray, S: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    L = np.linalg.cholesky(S)
    z = rng.normal(size=(3, n))
    pts = mu.reshape(3,1) + L @ z
    return pts.T


def _set_equal_3d(ax: Axes, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    x_range = (xs.min(), xs.max()); y_range = (ys.min(), ys.max()); z_range = (zs.min(), zs.max())
    ranges = np.array([x_range, y_range, z_range], dtype=float)
    centers = ranges.mean(axis=1); spans = ranges[:,1] - ranges[:,0]
    R = 0.5 * float(spans.max() if spans.max() > 0 else 1.0)
    ax.set_xlim(centers[0]-R, centers[0]+R)
    ax.set_ylim(centers[1]-R, centers[1]+R)
    ax.set_zlim(centers[2]-R, centers[2]+R)

# --- Overlay LOS + satellites (optional) ---

def _normalize_cols(cols):
    def norm(c):
        c = re.sub(r"\s+", " ", c.strip().lower())
        m = re.match(r"^(.*?)\s*\((.*?)\)$", c)
        if m:
            base, unit = m.group(1), m.group(2)
            if base in ("x","y","z") and unit.startswith("km"): return f"{base}_km"
            if base == "time" and "utc" in unit: return "time"
            return base
        return c
    return [norm(c) for c in cols]


def _read_stk_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    raw = list(df.columns)
    df.columns = _normalize_cols(raw)
    return df


def _gather_overlay(project_root: Path, target_id: str, tri_times: pd.Series, tol_s: float, max_epochs: int):
    los_dir = project_root / "exports" / "stk_exports" / "OUTPUT_LOS_VECTORS" / target_id
    eph_dir = project_root / "exports" / "stk_exports" / "OUTPUT_EPHEM"
    los_files = sorted(los_dir.glob(f"LOS_OBS_*_to_{target_id}_icrf.csv"))
    obs_ids = sorted({p.name.split("_")[2] for p in los_files})

    tri_times = pd.to_datetime(tri_times, utc=True)
    step = max(1, len(tri_times)//max_epochs)
    epochs = tri_times[::step]

    per_epoch = []
    for t in epochs:
        per = []
        for obs in obs_ids:
            los_path = los_dir / f"LOS_OBS_{obs}_to_{target_id}_icrf.csv"
            eph_path = eph_dir / f"OBS_{obs}_ephem.csv"
            if not los_path.exists() or not eph_path.exists():
                continue
            los = _read_stk_csv(los_path)
            eph = _read_stk_csv(eph_path)
            if not all(c in los.columns for c in ("time","ux","uy","uz")):
                continue
            if not all(c in eph.columns for c in ("time","x_km","y_km","z_km")):
                continue
            los["time"] = pd.to_datetime(los["time"], utc=True, errors="coerce")
            eph["time"] = pd.to_datetime(eph["time"], utc=True, errors="coerce")
            i = los["time"].sub(t).abs().idxmin()
            if pd.isna(i) or abs((los.loc[i, "time"] - t).total_seconds()) > tol_s:
                continue
            j = eph["time"].sub(t).abs().idxmin()
            if pd.isna(j) or abs((eph.loc[j, "time"] - t).total_seconds()) > tol_s:
                continue
            u = los.loc[i, ["ux","uy","uz"]].astype(float).to_numpy()
            r = eph.loc[j, ["x_km","y_km","z_km"]].astype(float).to_numpy()
            u = u / np.linalg.norm(u)
            per.append({"obs": obs, "r": r, "u": u})
        if per:
            per_epoch.append((t, per))
    return per_epoch


# --- Main ---

def main():
    tri_csv = TRI_CSV_PATH or _find_triangulation_csv(TRI_DIR)
    m = re.search(r"HGV_\d+", tri_csv.name)
    if not m:
        raise RuntimeError(f"Could not infer TARGET_ID from {tri_csv.name}")
    target_id = m.group(0)
    oem_path = PROJECT_ROOT / "exports" / "target_exports" / "OUTPUT_OEM" / f"{target_id}.oem"

    print(f"[PATH] Triangulation: {tri_csv}")
    print(f"[PATH] OEM truth   : {oem_path}")
    print(f"[INFO] TARGET_ID inferred: {target_id}")

    tri = _read_triangulation(tri_csv)
    truth = _read_oem_ccsds(oem_path)

    # ranges sanity
    print(f"[RANGE] Truth xyz [km]: x[{truth['x_km'].min():.1f},{truth['x_km'].max():.1f}] y[{truth['y_km'].min():.1f},{truth['y_km'].max():.1f}] z[{truth['z_km'].min():.1f},{truth['z_km'].max():.1f}]")
    print(f"[RANGE] x̂    xyz [km]: x[{tri['xhat_x_km'].min():.1f},{tri['xhat_x_km'].max():.1f}] y[{tri['xhat_y_km'].min():.1f},{tri['xhat_y_km'].max():.1f}] z[{tri['xhat_z_km'].min():.1f},{tri['xhat_z_km'].max():.1f}]")

    # Build MC dots centered on x̂
    rng = np.random.default_rng(RNG_SEED)
    xhat_pts = tri[["xhat_x_km","xhat_y_km","xhat_z_km"]].to_numpy()
    dots_all = []
    take = min(len(tri), MAX_EPOCHS_TO_PLOT)
    for _, row in tri.iloc[:take].iterrows():
        mu = np.array([row["xhat_x_km"], row["xhat_y_km"], row["xhat_z_km"]], dtype=float)
        S  = _cov_from_row(row)
        dots = _sample_cloud(mu, S, DOTS_PER_EPOCH, rng)
        dots_all.append(dots)
    dots_all = np.vstack(dots_all) if dots_all else np.empty((0,3))

    # Optional overlay
    overlays = _gather_overlay(PROJECT_ROOT, target_id, tri["time"], JOIN_TOL_S, MAX_OVERLAY_EPOCHS) if OVERLAY_SATS else []

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax: Axes = fig.add_subplot(111, projection='3d')

    # Truth trajectory
    ax.plot(truth["x_km"], truth["y_km"], truth["z_km"], lw=1.0, label=f"{target_id} truth (OEM)")

    # Triangulated fixes
    if len(xhat_pts):
        ax.scatter(xhat_pts[:,0], xhat_pts[:,1], xhat_pts[:,2], s=14, marker='x', label="triangulated x̂", depthshade=False)

    # MC dots
    if len(dots_all):
        ax.scatter(dots_all[:,0], dots_all[:,1], dots_all[:,2], s=3, alpha=0.12, label="MC samples", depthshade=False)

    # Overlays: satellites and short LOS segments
    for t, per in overlays:
        # nearest triangulation fix at time t
        k = tri["time"].sub(pd.Timestamp(t)).abs().idxmin()
        xhat = tri.loc[k, ["xhat_x_km","xhat_y_km","xhat_z_km"]].to_numpy(dtype=float)
        for item in per:
            r = item["r"]; u = item["u"]
            seg_len = min(300.0, np.linalg.norm(xhat - r))
            p2 = r + seg_len * u
            ax.plot([r[0], p2[0]], [r[1], p2[1]], [r[2], p2[2]], lw=0.8)
            ax.scatter([r[0]],[r[1]],[r[2]], s=12)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="upper right")
    ax.set_title("Truth (OEM) with triangulated fixes and MC clouds")

    _set_equal_3d(ax, truth["x_km"].to_numpy(), truth["y_km"].to_numpy(), truth["z_km"].to_numpy())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
