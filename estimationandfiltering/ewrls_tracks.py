# estimationandfiltering/ewrls_tracks.py (clean standalone)
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths (defaults; override via CLI args) ----------------
ROOT = Path(__file__).resolve().parents[1]
TRI_DIR   = ROOT / "exports/triangulation"
TRUTH_DIR = ROOT / "exports/truth"
OUT_DIR   = ROOT / "exports/tracks"
PLOTS_DIR = ROOT / "plots_ew"

# ---------------- EW-RLS (coordinate-wise polynomial) ----------------
def ewrls_series(y: np.ndarray, lam: float = 0.98, order: int = 2) -> np.ndarray:
    """Exponentially-Weighted RLS on a single series y[k] with polynomial basis [1,t,t^2,...].
    Returns y_hat[k]. t is relative (0..N-1) and mean-centered for numerical stability.
    """
    n = int(y.size)
    t = np.arange(n, dtype=float)
    t = t - t[0]
    t = t - np.mean(t)

    m = order + 1
    theta = np.zeros((m, 1))
    P = 1e6 * np.eye(m)
    yhat = np.zeros(n)

    def phi_row(k: int) -> np.ndarray:
        # Row vector shape (1,m)
        return np.array([t[k] ** i for i in range(m)], dtype=float)[None, :]

    for k in range(n):
        H = phi_row(k)                        # (1,m)
        denom = (lam + (H @ P @ H.T).item())  # scalar
        K = (P @ H.T) / denom                 # (m,1)
        y_hat_k = (H @ theta).item()          # scalar
        err = float(y[k] - y_hat_k)
        theta = theta + K * err               # (m,1)
        P = (P - K @ H @ P) / lam
        yhat[k] = y_hat_k
    return yhat


def ewrls_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, lam: float, order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ewrls_series(x, lam, order), ewrls_series(y, lam, order), ewrls_series(z, lam, order)

# ---------------- I/O helpers ----------------
def latest_run_id(base: Path) -> str:
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs in {base}")
    return runs[0].name


def read_tri(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    # --- time: accept seconds or ISO ---
    if "t_s" in df.columns and np.issubdtype(df["t_s"].dtype, np.number):
        t = df["t_s"].astype(float).to_numpy()
        t_iso = pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        for c in ["time", "epoch", "t", "utc", "timestamp"]:
            if c in df.columns:
                dt = pd.to_datetime(df[c], utc=True, errors="coerce")
                t = dt.astype("int64") / 1e9
                t_iso = dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                break
        else:
            raise ValueError(f"{csv_path.name}: no time column")

    # --- position columns with robust aliasing ---
    def pick_series(prefix: str):
        candidates = [
            f"xhat_{prefix}_km",                 # xhat_x_km, xhat_y_km, xhat_z_km
            f"{prefix}_icrf_km", f"{prefix}_ecef_km", f"{prefix}_geo_km",
            f"{prefix}_km",
            f"{prefix}hat_geo_km", f"{prefix}hat_km", f"{prefix}hat",
            prefix,
        ]
        for cand in candidates:
            if cand in df.columns:
                return df[cand].astype(float).to_numpy()
        # heuristic fallback
        low = [c.lower() for c in df.columns]
        for c, cl in zip(df.columns, low):
            if cl.startswith(prefix) and ("_km" in cl or "icrf" in cl or "ecef" in cl or "geo" in cl):
                try:
                    return df[c].astype(float).to_numpy()
                except Exception:
                    pass
        return None

    x = pick_series("x"); y = pick_series("y"); z = pick_series("z")
    if x is None or y is None or z is None:
        cols = ", ".join(df.columns)
        raise ValueError(f"{csv_path.name}: missing x/y/z columns. Available: [{cols}]")

    return {"t": np.asarray(t), "t_iso": np.asarray(t_iso), "x": x, "y": y, "z": z}


def read_truth(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "t_s" in df.columns and np.issubdtype(df["t_s"].dtype, np.number):
        t = df["t_s"].astype(float).to_numpy()
    else:
        for c in ["time", "epoch", "t", "utc", "timestamp"]:
            if c in df.columns:
                t = pd.to_datetime(df[c], utc=True, errors="coerce").astype("int64") / 1e9
                break
        else:
            raise ValueError(f"{csv_path.name}: no time column")

    def get(colnames: List[str]):
        for c in colnames:
            if c in df.columns:
                return df[c].astype(float).to_numpy()
        raise ValueError(f"{csv_path.name}: missing one of {colnames}")

    x = get(["x_icrf_km", "x_km", "x"]) 
    y = get(["y_icrf_km", "y_km", "y"])
    z = get(["z_icrf_km", "z_km", "z"])
    return {"t": np.asarray(t), "x": x, "y": y, "z": z}

# ---------------- Truth discovery ----------------
def find_truth_file(truth_base: Path, run_id: str, tgt: str) -> Optional[Path]:
    candidates = [
        truth_base / run_id / f"{tgt}_truth.csv",
        truth_base / run_id / f"truth_{tgt}.csv",
        truth_base / run_id / f"{tgt}.csv",
        truth_base / run_id / "targets" / f"{tgt}_truth.csv",
        truth_base / run_id / "targets" / f"{tgt}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ---------------- Plotting ----------------
def plot_3d_vs_truth(tgt: str, est: Dict[str, Any], tru: Dict[str, Any], out_dir: Path):
    ra, rb = np.round(est["t"], 6), np.round(tru["t"], 6)
    common = np.intersect1d(ra, rb)
    if common.size == 0:
        return
    def sel(d):
        m = np.isin(np.round(d["t"], 6), common)
        return {k: (v[m] if hasattr(v, "__len__") else v) for k, v in d.items()}
    A = sel(est); B = sel(tru)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(A["x"], A["y"], A["z"], label="EW-RLS", lw=2)
    ax.plot(B["x"], B["y"], B["z"], "--", label="truth (OEM)", lw=1.5)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt}")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tgt}_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_ew_only(tgt: str, est: Dict[str, Any], out_dir: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(est["x"], est["y"], est["z"], label="EW-RLS", lw=2)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt} (EW-only)")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tgt}_3d_ew_only.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_errors_vs_time(tgt: str, est: Dict[str, Any], tru: Dict[str, Any], out_dir: Path):
    ra, rb = np.round(est["t"], 6), np.round(tru["t"], 6)
    common = np.intersect1d(ra, rb)
    if common.size == 0:
        return
    def sel(d):
        m = np.isin(np.round(d["t"], 6), common)
        return {k: (v[m] if hasattr(v, "__len__") else v) for k, v in d.items()}
    A = sel(est); B = sel(tru)
    ex, ey, ez = A["x"] - B["x"], A["y"] - B["y"], A["z"] - B["z"]
    t = A["t"] - A["t"][0]
    fig = plt.figure(figsize=(7.0, 4.2))
    plt.plot(t, ex, label="ex [km]")
    plt.plot(t, ey, label="ey [km]")
    plt.plot(t, ez, label="ez [km]")
    e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    plt.plot(t, e3, label="e3D [km]", lw=2)
    plt.xlabel("time [s]"); plt.ylabel("error [km]")
    plt.title(f"Errors vs time — {tgt}")
    plt.legend(); plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tgt}_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------------- Driver ----------------
def run(run_id: Optional[str] = None, order: int = 2, theta: float = 0.98,
        tri_dir: Optional[str] = None, truth_dir: Optional[str] = None,
        out_dir: Optional[str] = None, plots_dir: Optional[str] = None):

    tri_base   = Path(tri_dir) if tri_dir else TRI_DIR
    truth_base = Path(truth_dir) if truth_dir else TRUTH_DIR
    out_base   = Path(out_dir) if out_dir else OUT_DIR
    plots_base = Path(plots_dir) if plots_dir else PLOTS_DIR

    if run_id is None:
        run_id = latest_run_id(tri_base)
    tri_run = tri_base / run_id
    out_run = (out_base / run_id); out_run.mkdir(parents=True, exist_ok=True)
    plot_run = (plots_base / run_id); plot_run.mkdir(parents=True, exist_ok=True)

    tri_files = sorted(tri_run.glob("xhat_geo_*.csv"))
    if not tri_files:
        raise FileNotFoundError(f"No triangulation CSVs in {tri_run}")

    print(f"[RUN ] EW-RLS on {len(tri_files)} targets  | run_id={run_id}")
    rows = []
    for tri_csv in tri_files:
        tgt = tri_csv.stem.replace("xhat_geo_", "")
        data = read_tri(tri_csv)
        # estimate EW-RLS (coordinate-wise, staying in the same frame as inputs)
        xh, yh, zh = ewrls_xyz(data["x"], data["y"], data["z"], lam=theta, order=order)

        # save per-target CSV
        out_csv = out_run / f"ewrls_tracks_{tgt}.csv"
        df_out = pd.DataFrame({
            "time_iso": data["t_iso"],
            "t_s": data["t"],
            "x_icrf_km": xh, "y_icrf_km": yh, "z_icrf_km": zh
        })
        df_out.to_csv(out_csv, index=False)
        nrows = len(df_out)
        fsize = out_csv.stat().st_size if out_csv.exists() else 0
        print(f"[SAVE] {out_csv.name}: rows={nrows}, size={fsize} B")

        # plots: always EW-only; add vs-truth if available
        est = {"t": data["t"], "x": xh, "y": yh, "z": zh}
        plot_3d_ew_only(tgt, est, plot_run)
        truth_csv = find_truth_file(truth_base, run_id, tgt)
        if truth_csv is not None:
            try:
                tru = read_truth(truth_csv)
                plot_3d_vs_truth(tgt, est, tru, plot_run)
                plot_errors_vs_time(tgt, est, tru, plot_run)
            except Exception as e:
                print(f"[WARN] Truth parsing/plot failed for {tgt}: {e}")
        else:
            print(f"[INFO] Truth not found for {tgt}; saved EW-only 3D plot.")

        rows.append({"target": tgt, "out_csv": str(out_csv)})

    # index
    df_idx = pd.DataFrame(rows)
    df_idx.to_csv(out_run / "ewrls_index.csv", index=False)
    print(f"[OK  ] EW-RLS tracks saved → {out_run}")
    print(f"[OK  ] Plots (per-target) → {plot_run}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EW-RLS only: build tracks from triangulation and plot vs truth.")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.98)
    ap.add_argument("--tri_dir", default=None)
    ap.add_argument("--truth_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--plots_dir", default=None)
    args = ap.parse_args()

    run(run_id=args.run_id, order=args.order, theta=args.theta,
        tri_dir=args.tri_dir, truth_dir=args.truth_dir,
        out_dir=args.out_dir, plots_dir=args.plots_dir)
