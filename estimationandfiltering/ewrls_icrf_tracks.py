# estimationandfiltering/ewrls_icrf_tracks.py — orchestration (ICRF OEM + EW-RLS)
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# ---- project utils & plotting (tuoi moduli) ----
from tools.utils import (
    ROOT, CFG_PATH, TRI_ROOT, TRACKS_OUT, PLOTS_OUT,
    load_paths, read_oem_icrf, load_run_ewrls, _rot_z, gmst_rad_from_unix,
)
from estimationandfiltering.plots_ewrls import (
    plot_3d_vs_truth, plot_errors_4panel
)

# --------------- helpers ---------------
def latest_run_id(base: Path) -> str:
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs in {base}")
    return runs[0].name

def ecef_to_icrf_xyz(x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Rotate position series ECEF->ICRF with GMST: r_eci ≈ R3(GMST)*r_ecef."""
    R = _rot_z(gmst_rad_from_unix(t_s))
    r = np.column_stack([x_km, y_km, z_km]).astype(float)
    return np.einsum('nij,nj->ni', R, r)


# --------------- main ---------------
def run(run_id: Optional[str]=None, order:int=4, theta:float=0.93,
        config: Optional[str]=None, tri_frame: str = "ICRF"):    # carica path dal config
    paths = load_paths(config or CFG_PATH)
    oem_root: Path = paths["oem_root"]

    # trova run
    if run_id is None:
        run_id = latest_run_id(TRI_ROOT)
    tri_run = TRI_ROOT / run_id

    out_run = (TRACKS_OUT / run_id); out_run.mkdir(parents=True, exist_ok=True)
    plot_run = (PLOTS_OUT / run_id); plot_run.mkdir(parents=True, exist_ok=True)

    # dynamic load dell'EW-RLS pubblico per evitare circular import
    run_ewrls_on_csv = load_run_ewrls()

    tri_files = sorted(tri_run.glob("xhat_geo_*.csv"))
    if not tri_files:
        raise FileNotFoundError(f"No triangulation CSVs in {tri_run}")
    for tri_csv in tri_files:
        tgt = tri_csv.stem.replace("xhat_geo_", "")
        out_csv = out_run / f"ewrls_icrf_{tgt}.csv"

        est_df: pd.DataFrame = run_ewrls_on_csv(
            str(tri_csv), out_csv=str(out_csv),
            frame=tri_frame, order=order, theta=theta
        )

        # --- If triangulation was ECEF, rotate to ICRF before plotting/metrics ---
        if tri_frame.upper() == "ECEF":
            t_s = est_df["t_s"].to_numpy(float)
            r_eci = ecef_to_icrf_xyz(est_df["x_km"], est_df["y_km"], est_df["z_km"], t_s)
            est_df["x_km"], est_df["y_km"], est_df["z_km"] = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
            est_df["frame"] = "ICRF"
            # persistiamo la versione ruotata sullo stesso CSV
            est_df.to_csv(out_csv, index=False)
    print(f"[RUN ] EW-RLS ICRF on {len(tri_files)} targets | run_id={run_id}")
    print(f"[INFO] OEM root (ICRF): {oem_root}")

    index_rows: List[Dict[str,Any]] = []

    for tri_csv in tri_files:
        tgt = tri_csv.stem.replace("xhat_geo_", "")
        out_csv = out_run / f"ewrls_icrf_{tgt}.csv"

        # 1) esegui EW-RLS su triangolazioni (frame='ICRF' solo come label di output)
        est_df: pd.DataFrame = run_ewrls_on_csv(
            str(tri_csv), out_csv=str(out_csv),
            frame="ICRF", order=order, theta=theta
        )
        print(f"[SAVE] {out_csv.name}  rows={len(est_df)}  size={out_csv.stat().st_size} B")

        # 2) leggi truth OEM (ICRF) e plottaggi
        oem_path = oem_root / f"{tgt}.oem"
        if not oem_path.exists():
            print(f"[INFO] OEM not found for {tgt}: {oem_path}")
            index_rows.append({"target": tgt, "est_csv": str(out_csv), "oem": "", "plots": str(plot_run)})
            continue

        tru = read_oem_icrf(oem_path)  # dict: t,t_iso,x,y,z (km), vx,vy,vz (km/s)

        # 3) plot 3D e 4-panel error (interp inside)
        plot_3d_vs_truth(tgt, est_df, tru, plot_run)
        plot_errors_4panel(tgt, est_df, tru, plot_run)

        index_rows.append({"target": tgt, "est_csv": str(out_csv), "oem": str(oem_path), "plots": str(plot_run)})

    # indice run
    pd.DataFrame(index_rows).to_csv(out_run / "index_icrf.csv", index=False)
    print(f"[OK  ] Tracks → {out_run}")
    print(f"[OK  ] Plots  → {plot_run}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EW-RLS vs OEM truth in ICRF — orchestrator (uses tools.utils & plots_ewrls)")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.93)
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--tri_frame", default="ICRF", choices=["ICRF", "ECEF"],
                    help="Frame of triangulation input (before EW-RLS output). If ECEF, rotate to ICRF for comparison.")
    args = ap.parse_args()
    run(run_id=args.run_id, order=args.order, theta=args.theta, config=args.config, tri_frame=args.tri_frame)