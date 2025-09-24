# estimationandfiltering/plot_rmse_components.py
"""
RMSE per asse (x,y,z) — overlay KF vs Triangolazione.
- Legge i path da config.loader.load_config()
- Per ogni target presente nella cartella exports/triangulation/<RUN_ID>:
    * carica verità OEM (ICRF)
    * carica stime triangolazione (xhat_geo_*.csv)
    * carica traccia KF (...._track_icrf_forward.csv)
    * allinea temporalmente su ciascuna serie e calcola RMSE per-asse
    * estrae dagli output di triangolazione: obs_used_csv e beta_*_deg (riassunto)
- Salva in plots/<RUN_ID>/rmse_components_<TARGET>.png
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- config loader, no fronzoli -----------------
try:
    from config.loader import load_config
except Exception:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from config.loader import load_config


# ----------------- IO helpers -----------------
def _parse_oem_icrf(path: Path) -> pd.DataFrame:
    """Parsa OEM CCSDS (ICRF) → DataFrame indicizzato su 'time' UTC con x_km,y_km,z_km."""
    recs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("META"):  # ignora meta e commenti
                continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                try:
                    x, y, z = map(float, parts[1:4])
                except Exception:
                    continue
                recs.append((t, x, y, z))
    if not recs:
        raise RuntimeError(f"Nessuna riga di stato in OEM: {path}")
    df = pd.DataFrame(recs, columns=["time", "x_km", "y_km", "z_km"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df.dropna(subset=["time"]).sort_values("time").set_index("time")


def _read_triangulation_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # colonne attese: time, xhat_x_km, xhat_y_km, xhat_z_km, beta_*_deg, obs_used_csv, ...
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "x_km": df.get("xhat_x_km", np.nan),
        "y_km": df.get("xhat_y_km", np.nan),
        "z_km": df.get("xhat_z_km", np.nan),
        "beta_min_deg": df.get("beta_min_deg", np.nan),
        "beta_mean_deg": df.get("beta_mean_deg", np.nan),
        "beta_max_deg": df.get("beta_max_deg", np.nan),
        "obs_used_csv": df.get("obs_used_csv", ""),
    }, index=t)
    return out.dropna(subset=["x_km", "y_km", "z_km"]).sort_index()


def _read_kf_track(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accetta sia "time" stringa ISO sia epoch secondi
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "t" in df.columns:
        # epoch seconds → datetime
        t = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    else:
        raise KeyError(f"Colonna tempo mancante in {path.name} (attese: 'time' o 't').")
    # nomi pos possibili
    xcol = "x_km" if "x_km" in df else ("xhat_x_km" if "xhat_x_km" in df else None)
    ycol = "y_km" if "y_km" in df else ("xhat_y_km" if "xhat_y_km" in df else None)
    zcol = "z_km" if "z_km" in df else ("xhat_z_km" if "xhat_z_km" in df else None)
    if not all([xcol, ycol, zcol]):
        raise KeyError(f"Mancano colonne posizione in {path.name}")
    out = pd.DataFrame({"x_km": df[xcol], "y_km": df[ycol], "z_km": df[zcol]}, index=t)
    return out.dropna().sort_index()


def _interp_truth(truth: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpola verità sulle epoche 'idx' (solo dentro lo span)."""
    tr = truth.reindex(truth.index.union(idx)).interpolate(method="time")
    out = tr.loc[idx.intersection(truth.index.min().to_timestamp(), truth.index.max().to_timestamp())]
    # Se idx ha valori fuori dallo span, reindex per mantenere la stessa lunghezza (con NaN) e poi dropna:
    out = out.reindex(idx).dropna()
    return out


def _rmse_per_axis(est: pd.DataFrame, truth_on_est: pd.DataFrame) -> Tuple[float,float,float]:
    dx = est["x_km"].loc[truth_on_est.index] - truth_on_est["x_km"]
    dy = est["y_km"].loc[truth_on_est.index] - truth_on_est["y_km"]
    dz = est["z_km"].loc[truth_on_est.index] - truth_on_est["z_km"]
    f = lambda s: float(np.sqrt(np.mean(np.square(s.values))))
    return f(dx), f(dy), f(dz)


def _summarize_geometry(tri_df: pd.DataFrame) -> Tuple[str, str]:
    """Ritorna: obs_summary, beta_summary per legenda."""
    # modal (più frequente) set di obs usati
    if "obs_used_csv" in tri_df:
        mode_obs = tri_df["obs_used_csv"].mode()
        obs_summary = mode_obs.iloc[0] if len(mode_obs) else ""
    else:
        obs_summary = ""
    # beta statistiche (median per robustezza)
    bmin = np.nanmedian(tri_df.get("beta_min_deg", np.nan))
    bavg = np.nanmedian(tri_df.get("beta_mean_deg", np.nan))
    bmax = np.nanmedian(tri_df.get("beta_max_deg", np.nan))
    beta_summary = f"β[min/avg/max]≈ {bmin:.1f}/{bavg:.1f}/{bmax:.1f}°" if np.isfinite([bmin,bavg,bmax]).all() else ""
    return obs_summary, beta_summary


# ----------------- Plot per-target -----------------
def plot_rmse_components_for_target(root: Path, run_id: str, target: str) -> Path:
    cfg = load_config()
    oem_path = Path(cfg["project"]["root"]).resolve() / cfg["paths"]["oem_root"] / f"{target}.oem"
    tri_csv  = root / "exports" / "triangulation" / run_id / f"xhat_geo_{target}.csv"
    kf_csv   = root / "exports" / "tracks" / run_id / f"{target}_track_icrf_forward.csv"
    out_dir  = root / "plots" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png  = out_dir / f"rmse_components_{target}.png"

    truth = _parse_oem_icrf(oem_path)
    tri   = _read_triangulation_csv(tri_csv)
    kf    = _read_kf_track(kf_csv)

    # allinea verità sulle epoche di ciascun stimatore
    truth_on_tri = _interp_truth(truth, tri.index)
    tri = tri.loc[truth_on_tri.index]

    truth_on_kf = _interp_truth(truth, kf.index)
    kf = kf.loc[truth_on_kf.index]

    # RMSE per asse
    rmse_tri = _rmse_per_axis(tri[["x_km","y_km","z_km"]], truth_on_tri)
    rmse_kf  = _rmse_per_axis(kf[["x_km","y_km","z_km"]], truth_on_kf)

    # riassunto geometria per legenda
    obs_summary, beta_summary = _summarize_geometry(tri)

    # --- Plot: barre raggruppate per asse ---
    axes = ["x", "y", "z"]
    x = np.arange(len(axes))
    w = 0.35

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111)
    ax.bar(x - w/2, rmse_tri, width=w, label=f"Triangolazione\n{beta_summary}\nobs: {obs_summary}")
    ax.bar(x + w/2, rmse_kf,  width=w, label="Kalman Filter (forward)")
    ax.set_xticks(x); ax.set_xticklabels(axes)
    ax.set_ylabel("RMSE [km]")
    ax.set_title(f"RMSE per asse — {target}")
    ax.grid(True, axis="y", alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK ] RMSE components → {out_png}")
    return out_png


# ----------------- CLI/main -----------------
def main():
    cfg = load_config()
    ROOT = Path(cfg["project"]["root"]).resolve()
    RUN_ID = cfg["project"]["run_id"]

    tri_dir = ROOT / "exports" / "triangulation" / RUN_ID
    if not tri_dir.exists():
        raise FileNotFoundError(f"Nessuna triangolazione per run_id={RUN_ID}: {tri_dir}")

    # trova i target presenti
    tri_csvs = sorted(tri_dir.glob("xhat_geo_*.csv"))
    if not tri_csvs:
        raise FileNotFoundError(f"Nessun xhat_geo_*.csv in {tri_dir}")

    for p in tri_csvs:
        target = p.stem.replace("xhat_geo_", "")
        try:
            plot_rmse_components_for_target(ROOT, RUN_ID, target)
        except Exception as e:
            print(f"[WARN] {target}: {e}")

if __name__ == "__main__":
    main()