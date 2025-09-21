from __future__ import annotations
import sys, math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
TRI_DIR   = ROOT / "exports/triangulation"
TRUTH_DIR = ROOT / "exports/truth"
OUT_DIR   = ROOT / "exports/tracks"
PLOTS_DIR = ROOT / "plots_ew"

# --- Minimal EW-RLS (coordinate-wise) ---
def ewrls_series(y: np.ndarray, lam: float = 0.98, order: int = 2) -> np.ndarray:
    """EW-RLS su una singola serie y(k) con regressori [1,t,t^2,...].
       Restituisce y_hat(k). t è tempo relativo in secondi, centrato."""
    n = y.size
    t = np.arange(n, dtype=float)
    t = t - t[0]  # 0..n-1
    t = t - np.mean(t)  # centratura per stabilità
    # design row vettoriale
    def phi(k):
        return np.array([t[k]**i for i in range(order+1)], dtype=float)[:, None]  # (m,1)
    m = order + 1
    theta = np.zeros((m,1))
    P = 1e6 * np.eye(m)
    yhat = np.zeros(n)
    for k in range(n):
        H = phi(k).T              # (1,m)
        denom = (lam + (H @ P @ H.T).item())
        K = (P @ H.T) / denom               # (m,1)
        y_hat_k = (H @ theta).item()        # scalar prediction
        err = float(y[k] - y_hat_k)         # scalar innovation
        theta = theta + K * err
        P = (P - K @ H @ P) / lam
        yhat[k] = y_hat_k
    return yhat

def ewrls_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, lam: float, order: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    return ewrls_series(x, lam, order), ewrls_series(y, lam, order), ewrls_series(z, lam, order)

# --- I/O helpers ---
def latest_run_id(base: Path) -> str:
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs: raise FileNotFoundError(f"No runs in {base}")
    return runs[0].name

def read_tri(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    # tempo: accetta ISO o numerico
    if "t_s" in df.columns and np.issubdtype(df["t_s"].dtype, np.number):
        t = df["t_s"].astype(float).to_numpy()
        t_iso = pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        # fallback: qualsiasi colonna tempo in ISO
        for c in ["time","epoch","t","utc","timestamp"]:
            if c in df.columns:
                dt = pd.to_datetime(df[c], utc=True, errors="coerce")
                t = dt.astype("int64")/1e9
                t_iso = dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                break
        else:
            raise ValueError(f"{csv_path.name}: no time column")
    def pick_series(prefix: str):
        # Try explicit common names first
        candidates = [
            # Explicit triangulation outputs
            f"xhat_{prefix}_km",  # e.g., xhat_x_km, xhat_y_km, xhat_z_km
            # Common coordinate names
            f"{prefix}_icrf_km", f"{prefix}_ecef_km", f"{prefix}_geo_km",
            f"{prefix}_km",
            # Older/alt naming
            f"{prefix}hat_geo_km", f"{prefix}hat_km", f"{prefix}hat",
            # Bare prefix last
            prefix,
        ]
        for cand in candidates:
            if cand in df.columns:
                return df[cand].astype(float).to_numpy()
        # Fallback: heuristic search by prefix and unit
        low = [c.lower() for c in df.columns]
        for c, cl in zip(df.columns, low):
            if cl.startswith(prefix) and ("_km" in cl or "icrf" in cl or "ecef" in cl or "geo" in cl):
                try:
                    return df[c].astype(float).to_numpy()
                except Exception:
                    continue
        return None

    x, y, z = pick_series("x"), pick_series("y"), pick_series("z")
    if x is None or y is None or z is None:
        cols = ", ".join(df.columns)
        raise ValueError(f"{csv_path.name}: missing x/y/z columns. Available: [{cols}]")
    return {"t": t, "t_iso": np.asarray(t_iso), "x": x, "y": y, "z": z}

def read_truth(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    # supporta colonne: time_iso / t_s (+ x_icrf_km,y_icrf_km,z_icrf_km)
    if "t_s" in df.columns and np.issubdtype(df["t_s"].dtype, np.number):
        t = df["t_s"].astype(float).to_numpy()
    else:
        for c in ["time","epoch","t","utc","timestamp"]:
            if c in df.columns:
                t = pd.to_datetime(df[c], utc=True, errors="coerce").astype("int64")/1e9
                break
        else:
            raise ValueError(f"{csv_path.name}: no time column")
    def get(colnames: List[str]):
        for c in colnames:
            if c in df.columns: return df[c].astype(float).to_numpy()
        raise ValueError(f"{csv_path.name}: missing {colnames}")
    x = get(["x_icrf_km","x_km","x"])
    y = get(["y_icrf_km","y_km","y"])
    z = get(["z_icrf_km","z_km","z"])
    return {"t": np.asarray(t), "x": x, "y": y, "z": z}

# --- Plotting ---
def plot_3d_vs_truth(tgt: str, est: Dict[str,Any], tru: Dict[str,Any], out_dir: Path):
    # allinea sui timestamp comuni
    ra, rb = np.round(est["t"],6), np.round(tru["t"],6)
    common = np.intersect1d(ra, rb)
    if common.size==0: return
    def sel(d):
        m = np.isin(np.round(d["t"],6), common)
        return {k:(v[m] if hasattr(v,"__len__") else v) for k,v in d.items()}
    A = sel(est); B = sel(tru)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.0,5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(A["x"],A["y"],A["z"], label="EW-RLS", lw=2)
    ax.plot(B["x"],B["y"],B["z"], "--", label="truth (OEM)", lw=1.5)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt}")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tgt}_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_errors_vs_time(tgt: str, est: Dict[str,Any], tru: Dict[str,Any], out_dir: Path):
    ra, rb = np.round(est["t"],6), np.round(tru["t"],6)
    common = np.intersect1d(ra, rb)
    if common.size==0: return
    def sel(d):
        m = np.isin(np.round(d["t"],6), common)
        return {k:(v[m] if hasattr(v,"__len__") else v) for k,v in d.items()}
    A = sel(est); B = sel(tru)
    ex, ey, ez = A["x"]-B["x"], A["y"]-B["y"], A["z"]-B["z"]
    t = (A["t"]-A["t"][0])  # s relativi
    fig = plt.figure(figsize=(7.0,4.2))
    plt.plot(t, ex, label="ex [km]")
    plt.plot(t, ey, label="ey [km]")
    plt.plot(t, ez, label="ez [km]")
    e3 = np.sqrt(ex**2+ey**2+ez**2)
    plt.plot(t, e3, label="e3D [km]", lw=2)
    plt.xlabel("time [s]"); plt.ylabel("error [km]")
    plt.title(f"Errors vs time — {tgt}")
    plt.legend(); plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tgt}_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- Driver ---
def run(run_id: Optional[str]=None, order:int=2, theta:float=0.98,
        tri_dir: Optional[str]=None, truth_dir: Optional[str]=None,
        out_dir: Optional[str]=None, plots_dir: Optional[str]=None):

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
        tgt = tri_csv.stem.replace("xhat_geo_","")
        data = read_tri(tri_csv)
        # stima EW-RLS (coordinate-wise, stesso frame dei CSV: *_icrf_km se presenti)
        xh,yh,zh = ewrls_xyz(data["x"], data["y"], data["z"], lam=theta, order=order)

        # salva CSV separato
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

        # plot vs truth se disponibile
        truth_csv = truth_base / run_id / f"{tgt}_truth.csv"
        if truth_csv.exists():
            tru = read_truth(truth_csv)
            est = {"t": data["t"], "x": xh, "y": yh, "z": zh}
            plot_3d_vs_truth(tgt, est, tru, plot_run)
            plot_errors_vs_time(tgt, est, tru, plot_run)

        rows.append({"target": tgt, "out_csv": str(out_csv)})

    df_idx = pd.DataFrame(rows)
    df_idx.to_csv(out_run / "ewrls_index.csv", index=False)
    print(f"[OK  ] EW-RLS tracks saved → {out_run}")
    print(f"[OK  ] Plots (if truth) → {plot_run}")

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