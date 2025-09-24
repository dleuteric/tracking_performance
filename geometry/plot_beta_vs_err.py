# geometry/plot_beta_vs_err.py
"""
Beta/geometry vs KF error — spike debugger

What it does
------------
Loads:
- Triangulation CSV: exports/triangulation/{run_id}/xhat_geo_{target}.csv
  (has beta_min/max/mean, Nsats, condA, CEP50_km_analytic, etc.)
- KF forward track : exports/tracks/{run_id}/{target}_track_icrf_forward.csv
- Truth OEM        : {paths.oem_root}/{target}.oem

Aligns all on OEM epochs and plots 4 panels to diagnose periodic error spikes:
  1) |err| KF [m] vs time (t−t0)
  2) β_mean [deg] vs time
  3) Nsats used vs time
  4) cond(A) (triangulation LS conditioning) and CEP50 [km]

It also prints quick correlations (Spearman/Pearson) between |err| and
β_mean, Nsats, condA, CEP50 to support the visual inspection.

Usage
-----
python geometry/plot_beta_vs_err.py --run_id YYYYMMDDTHHMMSSZ_... --target HGV_00001
If --run_id is omitted, the latest triangulation run is used.

Notes
-----
- No seaborn, simple matplotlib only.
- Time axis is t − t0 [s] (t0 = first OEM epoch in triangulation CSV).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config loader ----------------
try:
    from config.loader import load_config
except Exception:
    import sys
    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config

CFG = load_config()
ROOT = Path(CFG["project"]["root"]).resolve()
TRI_ROOT = (ROOT / CFG["paths"]["triangulation_out"]).resolve()
TRACKS_ROOT = (ROOT / "exports" / "tracks").resolve()
OEM_ROOT = (ROOT / CFG["paths"]["oem_root"]).resolve()
PLOTS_ROOT = (ROOT / "plots").resolve()

# ---------------- Helpers ----------------
def _latest_run_id(base: Path) -> str:
    runs = [p for p in base.iterdir() if p.is_dir() and p.name[:8].isdigit()]
    if not runs:
        raise FileNotFoundError(f"No runs under {base}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name

def _read_oem_icrf(oem_path: Path) -> pd.DataFrame:
    recs = []
    with oem_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("META"):
                continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x, y, z, vx, vy, vz = map(float, parts[1:7])
                except Exception:
                    continue
                recs.append((t, x, y, z))
    if not recs:
        raise RuntimeError(f"No state lines in OEM {oem_path}")
    df = pd.DataFrame(recs, columns=["time", "x_km", "y_km", "z_km"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time")

def _read_kf_track_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, comment="#")
    # expected columns: time,x_km,y_km,z_km (ICRF)
    # normalize time
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    need = ["time", "x_km", "y_km", "z_km"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"KF track missing columns {missing} in {p.name}")
    return df.dropna(subset=["time"]).sort_values("time")[need]

def _read_tri_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    keep = ["time",
            "beta_min_deg", "beta_max_deg", "beta_mean_deg",
            "Nsats", "condA", "CEP50_km_analytic"]
    present = [c for c in keep if c in df.columns]
    return df.dropna(subset=["time"]).sort_values("time")[present]

def _interp_truth_to(times: pd.Series, oem: pd.DataFrame) -> np.ndarray:
    """Return truth position [km] at 'times' via per-axis time interpolation."""
    oem = oem.sort_values("time")
    t_ref = pd.to_datetime(times, utc=True).astype("int64").to_numpy() / 1e9
    t_oem = oem["time"].astype("int64").to_numpy() / 1e9
    x = np.interp(t_ref, t_oem, oem["x_km"].to_numpy())
    y = np.interp(t_ref, t_oem, oem["y_km"].to_numpy())
    z = np.interp(t_ref, t_oem, oem["z_km"].to_numpy())
    return np.column_stack([x, y, z])

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d*d, axis=1))))

# ---------------- Main logic ----------------
def main(run_id: Optional[str], target: str) -> Path:
    rid = run_id or _latest_run_id(TRI_ROOT)
    tri_csv = TRI_ROOT / rid / f"xhat_geo_{target}.csv"
    kf_csv  = TRACKS_ROOT / rid / f"{target}_track_icrf_forward.csv"
    oem     = OEM_ROOT / f"{target}.oem"
    out_dir = (PLOTS_ROOT / rid)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"beta_vs_err_{target}.png"

    if not tri_csv.exists():
        raise FileNotFoundError(f"triangulation csv missing: {tri_csv}")
    if not kf_csv.exists():
        raise FileNotFoundError(f"KF track csv missing: {kf_csv}")
    if not oem.exists():
        raise FileNotFoundError(f"OEM missing: {oem}")

    tri = _read_tri_csv(tri_csv)
    kf  = _read_kf_track_csv(kf_csv)
    truth = _read_oem_icrf(oem)

    # Align (asof within 1s tolerance) → join tri geometry to KF epochs
    merged = pd.merge_asof(
        kf.sort_values("time"),
        tri.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    )

    # Compute KF error vs truth at KF times
    X_true = _interp_truth_to(merged["time"], truth)
    X_kf   = merged[["x_km", "y_km", "z_km"]].to_numpy(float)
    err_kf_km = np.linalg.norm(X_kf - X_true, axis=1)

    # Time axis (t - t0) in seconds
    t0 = merged["time"].iloc[0]
    tsec = (merged["time"] - t0).dt.total_seconds().to_numpy()

    # Pull geometry channels (may contain NaN if outside tolerance)
    beta_mean = merged["beta_mean_deg"].to_numpy(float)
    Nsats     = merged["Nsats"].to_numpy(float)
    condA     = merged["condA"].to_numpy(float)
    cep_km    = merged["CEP50_km_analytic"].to_numpy(float)

    # --- Plots ---
    fig = plt.figure(figsize=(11.5, 8.0))

    ax1 = fig.add_subplot(411)
    ax1.plot(tsec, err_kf_km*1000.0, label="|err| KF [m]")
    ax1.set_ylabel("position error [m]")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(412, sharex=ax1)
    ax2.plot(tsec, beta_mean, label="β_mean [deg]")
    ax2.set_ylabel("β_mean [deg]")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(413, sharex=ax1)
    ax3.step(tsec, Nsats, where="post", label="Nsats")
    ax3.set_ylabel("Nsats")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(414, sharex=ax1)
    ax4.plot(tsec, condA, label="cond(A)")
    if not np.all(np.isnan(cep_km)):
        ax4b = ax4.twinx()
        ax4b.plot(tsec, cep_km, linestyle="--", label="CEP50 [km]")
        ax4b.set_ylabel("CEP50 [km]")
    ax4.set_xlabel("t − t0 [s]")
    ax4.set_ylabel("cond(A)")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"KF error vs geometry — {target} ({rid})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Quick correlations (printed to console) ---
    try:
        from scipy.stats import pearsonr, spearmanr  # optional
        def _corr(y, x, name):
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 10:
                print(f"[INFO] Not enough data for correlation with {name}.")
                return
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
            print(f"[CORR] |err| vs {name}: pearson r={pr:.3f} (p={pp:.2g}), spearman ρ={sr:.3f} (p={sp:.2g})")
        _corr(err_kf_km, beta_mean, "β_mean [deg]")
        _corr(err_kf_km, Nsats, "Nsats")
        _corr(err_kf_km, condA, "cond(A)")
        if np.isfinite(cep_km).any():
            _corr(err_kf_km, cep_km, "CEP50 [km]")
    except Exception:
        pass

    print(f"[OK ] {out_png}")
    return out_png

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot KF error spikes vs triangulation geometry (β, Nsats, condA, CEP50).")
    ap.add_argument("--run_id", default=None, help="Run id (default: latest triangulation run).")
    ap.add_argument("--target", default="HGV_00001")
    args = ap.parse_args()
    main(args.run_id, args.target)
