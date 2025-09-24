#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

ROOT = Path(__file__).resolve().parents[1]
CFG_OEM = ROOT / "exports" / "target_exports" / "OUTPUT_OEM"
TRI_ROOT = ROOT / "exports" / "triangulation"
KF_ROOT  = ROOT / "exports" / "tracks"

def _sigma_from_run(run_id: str) -> str:
    m = re.search(r"sigma(\d+)\s*urad", run_id, re.IGNORECASE)
    return f"{m.group(1)} µrad" if m else run_id

def read_oem(target: str) -> pd.DataFrame:
    oem = CFG_OEM / f"{target}.oem"
    recs=[]
    with oem.open("r", encoding="utf-8", errors="ignore") as f:
        for s in f:
            s=s.strip()
            if not s or s.startswith("#") or s.startswith("META"): continue
            p=s.split()
            if len(p)>=7 and ("T" in p[0] or ":" in p[0]):
                t=pd.to_datetime(p[0], utc=True, errors="coerce")
                try:
                    x,y,z = map(float, p[1:4])
                except:
                    continue
                recs.append((t,x,y,z))
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km"]).dropna()
    df["time"]=pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()

def read_triang(run_id: str, target: str) -> pd.DataFrame:
    p = TRI_ROOT / run_id / f"xhat_geo_{target}.csv"
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df[["xhat_x_km","xhat_y_km","xhat_z_km"]].rename(
        columns={"xhat_x_km":"x_km","xhat_y_km":"y_km","xhat_z_km":"z_km"}
    )

def read_kf(run_id: str, target: str) -> pd.DataFrame:
    p = KF_ROOT / run_id / f"{target}_track_icrf_forward.csv"
    df = pd.read_csv(p)
    # be permissive on column names
    time_col = "time" if "time" in df.columns else "Time" if "Time" in df.columns else None
    if not time_col:
        raise FileNotFoundError(f"time column not found in {p}")
    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.set_index("time").sort_index()
    for c in ("x_km","y_km","z_km"):
        if c not in df.columns:
            # sometimes upper-case or without _km
            for alt in (c.upper(), c.replace("_km",""), c.capitalize()):
                if alt in df.columns:
                    df = df.rename(columns={alt:c})
                    break
    return df[["x_km","y_km","z_km"]]

def interp_to_truth(est: pd.DataFrame, truth_idx: pd.DatetimeIndex) -> pd.DataFrame:
    # time-based interpolation on each column
    e = est.reindex(est.index.union(truth_idx)).sort_index().interpolate(method="time")
    return e.loc[truth_idx]

def err_norm(est: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    d = est[["x_km","y_km","z_km"]].to_numpy() - truth[["x_km","y_km","z_km"]].to_numpy()
    return pd.Series(np.linalg.norm(d, axis=1), index=truth.index)

def main():
    ap = argparse.ArgumentParser(description="Overlay position error vs time for multiple sigma runs.")
    ap.add_argument("--target", default="HGV_00001")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run IDs like 20250924T211257Z_sigma50urad ...")
    ap.add_argument("--out", default=None, help="Output PNG (optional)")
    args = ap.parse_args()

    truth = read_oem(args.target)

    plt.figure(figsize=(9,5.5))
    ax = plt.gca()

    for run in args.runs:
        label_sigma = _sigma_from_run(run)
        # Triangulation error
        tri = read_triang(run, args.target)
        tri_i = interp_to_truth(tri, truth.index)
        e_tri = err_norm(tri_i, truth)
        ax.plot(e_tri.index, e_tri.values, linestyle="--", linewidth=1.2, label=f"Triang {label_sigma}")

        # KF error
        kf = read_kf(run, args.target)
        kf_i = interp_to_truth(kf, truth.index)
        e_kf = err_norm(kf_i, truth)
        ax.plot(e_kf.index, e_kf.values, linewidth=1.6, label=f"KF {label_sigma}")

    ax.set_ylabel("|position error| [km]")
    ax.set_xlabel("UTC time")
    ax.grid(True, alpha=.3)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out = args.out
    if out is None:
        outdir = ROOT / "plots" / "overlay_sigma"
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"errors_overlay_{args.target}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[OK] Saved → {out}")

if __name__ == "__main__":
    main()