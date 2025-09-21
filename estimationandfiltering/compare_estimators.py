from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import re
import sys

# ---------- Paths / config ----------
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config/pipeline.yaml"

def _expand_paths(paths: Dict[str, str]) -> Dict[str, Path]:
    raw = dict(paths)
    exports_root = Path(raw.get("exports_root", "exports"))
    stk_exports = Path((str(raw.get("stk_exports", "exports/stk_exports"))).format(exports_root=exports_root))
    def _fmt(s: str) -> Path:
        s1 = s.format(exports_root=exports_root, stk_exports=stk_exports)
        return ROOT / s1
    out = {k: _fmt(v) if isinstance(v, str) else v for k, v in raw.items()}
    out["exports_root"] = ROOT / str(exports_root)
    out["stk_exports"] = ROOT / str(stk_exports)
    return out

def load_paths(cfg: Optional[Path] = None) -> Dict[str, Path]:
    cfg_path = Path(cfg) if cfg else CFG_PATH
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    if "paths" not in data:
        raise KeyError(f"'paths' missing in {cfg_path}")
    return _expand_paths(data["paths"])

# ---------- Truth: OEM ICRF ----------
def read_oem_icrf(oem_path: Path) -> Dict[str, Any]:
    times: List[str] = []; rows: List[List[float]] = []; ref=None
    with oem_path.open("r") as f:
        meta=False
        for line in f:
            s=line.strip()
            if not s: continue
            if s=="META_START": meta=True; continue
            if s=="META_STOP":  meta=False; continue
            if meta:
                if s.startswith("REF_FRAME"): ref=s.split("=")[-1].strip()
                continue
            if s[0].isdigit() and "T" in s:
                parts=s.split()
                if len(parts) >= 7:
                    times.append(parts[0]); rows.append([float(p) for p in parts[1:7]])
    if ref is None or ref.upper() not in ("ICRF","J2000","GCRF"):
        raise ValueError(f"OEM not in ICRF/GCRF/J2000: {oem_path}")
    arr=np.asarray(rows,float)
    t = pd.to_datetime(np.asarray(times), utc=True).astype("int64")/1e9
    x,y,z, vx,vy,vz = arr.T
    return {"t": t, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}

# ---------- IO helpers ----------
def read_estimator_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Map columns (assumiamo già km e t_s [s] dal nostro pipeline)
    # Supportiamo anche varianti legacy x/y/z in m.
    def pick(df, cands):
        for c in cands:
            if c in df: return c
        for c in df.columns:
            n=c.lower()
            if any(cc in n for cc in cands): return c
        raise KeyError(f"Missing columns {cands} in {csv_path.name}")
    tcol = pick(df, ["t_s","time_s","t","epoch_s","time"])
    xcol = pick(df, ["x_icrf_km","x_km","x_ecef_km","x","x_icrf_m","x_ecef_m","x_m"])
    ycol = pick(df, ["y_icrf_km","y_km","y_ecef_km","y","y_icrf_m","y_ecef_m","y_m"])
    zcol = pick(df, ["z_icrf_km","z_km","z_ecef_km","z","z_icrf_m","z_ecef_m","z_m"])
    # Units: if looks like meters, convert
    def is_m(col): return str(col).endswith("_m")
    scale = 1000.0 if (is_m(xcol) or is_m(ycol) or is_m(zcol)) else 1.0
    # Time: allow ISO
    t = df[tcol]
    try: t_s = t.astype(float).to_numpy()
    except Exception:
        # if it's ISO strings
        t_s = pd.to_datetime(t, utc=True, errors="coerce")
        if t_s.isna().any():
            t_s = pd.to_datetime(t, errors="coerce")
        if hasattr(t_s, "astype"):
            t_s = t_s.astype("int64").to_numpy()/1e9
        else:  # DatetimeIndex
            t_s = t_s.view("int64")/1e9
    out = pd.DataFrame({
        "t_s": t_s,
        "x_km": df[xcol].astype(float).to_numpy()/scale,
        "y_km": df[ycol].astype(float).to_numpy()/scale,
        "z_km": df[zcol].astype(float).to_numpy()/scale,
    })
    return out.sort_values("t_s").reset_index(drop=True)

def extract_target_id(name: str) -> str:
    # HGV_00001, HGV-00001, ewrls_icrf_HGV_00001, kf_HGV_00001, etc.
    m = re.search(r"(HGV[_-]\d{5})", name, re.IGNORECASE)
    return m.group(1).replace("-", "_").upper() if m else Path(name).stem

# ---------- Metrics ----------
def interp_truth(tru: Dict[str,Any], t_s: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    Tx = np.interp(t_s, tru["t"], tru["x"], left=tru["x"][0], right=tru["x"][-1])
    Ty = np.interp(t_s, tru["t"], tru["y"], left=tru["y"][0], right=tru["y"][-1])
    Tz = np.interp(t_s, tru["t"], tru["z"], left=tru["z"][0], right=tru["z"][-1])
    return Tx,Ty,Tz

def errors_vs_truth(est: pd.DataFrame, tru: Dict[str,Any]) -> Dict[str,np.ndarray]:
    t = est["t_s"].to_numpy(float)
    Tx,Ty,Tz = interp_truth(tru, t)
    ex = (est["x_km"].to_numpy(float) - Tx) * 1000.0
    ey = (est["y_km"].to_numpy(float) - Ty) * 1000.0
    ez = (est["z_km"].to_numpy(float) - Tz) * 1000.0
    et = np.sqrt(ex**2 + ey**2 + ez**2)
    return {"t": t, "ex": ex, "ey": ey, "ez": ez, "et": et}

def metrics_from_errors(err: Dict[str,np.ndarray]) -> Dict[str,float]:
    et = err["et"]
    rmse = float(np.sqrt(np.nanmean(et**2)))
    cep50 = float(np.percentile(et, 50))
    cep90 = float(np.percentile(et, 90))
    return {"RMSE3D_m": rmse, "CEP50_m": cep50, "CEP90_m": cep90}

def rms_between(kf: pd.DataFrame, ew: pd.DataFrame) -> float:
    """RMS 3D distance (m) between KF and EW after time alignment via interpolation."""
    t = kf["t_s"].to_numpy(float)
    def interp_col(df, col): return np.interp(t, df["t_s"].to_numpy(float), df[col].to_numpy(float))
    dx = (kf["x_km"].to_numpy(float) - interp_col(ew, "x_km"))*1000.0
    dy = (kf["y_km"].to_numpy(float) - interp_col(ew, "y_km"))*1000.0
    dz = (kf["z_km"].to_numpy(float) - interp_col(ew, "z_km"))*1000.0
    return float(np.sqrt(np.nanmean(dx*dx + dy*dy + dz*dz)))

# ---------- Plots ----------
COL_TRUTH = "#7f7f7f"
COL_KF    = "#1f77b4"
COL_EW    = "#ff7f0e"
COL_TOT   = "#9467bd"

def _set_equal_aspect_3d(ax):
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]
    r = max(xr,yr,zr)/2
    ax.set_xlim3d(sum(xlim)/2-r, sum(xlim)/2+r)
    ax.set_ylim3d(sum(ylim)/2-r, sum(ylim)/2+r)
    ax.set_zlim3d(sum(zlim)/2-r, sum(zlim)/2+r)

def plot_3d_overlay(tgt: str, kf: Optional[pd.DataFrame], ew: Optional[pd.DataFrame], tru: Dict[str,Any], out_dir: Path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(6.8,5.4)); ax = fig.add_subplot(111, projection="3d")
    if kf is not None:
        ax.plot(kf["x_km"], kf["y_km"], kf["z_km"], color=COL_KF, lw=2.2, label="KF")
    if ew is not None:
        ax.plot(ew["x_km"], ew["y_km"], ew["z_km"], color=COL_EW, lw=2.0, label="EW-RLS")
    ax.plot(tru["x"], tru["y"], tru["z"], "--", color=COL_TRUTH, lw=1.5, label="Truth (ICRF)")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]")
    ax.set_title(f"3D track — {tgt}"); ax.legend(frameon=False, loc="upper left")
    _set_equal_aspect_3d(ax); fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"{tgt}_overlay3d.png", dpi=180, bbox_inches="tight"); plt.close(fig)

def plot_errors_overlay(tgt: str, ekf: Optional[Dict[str,np.ndarray]], eew: Optional[Dict[str,np.ndarray]], out_dir: Path):
    fig = plt.figure(figsize=(11,7.2))
    ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222); ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

    def rel_t(err):
        if err is None: return None
        t = err["t"]; return t - t[0]

    if ekf:
        tk = rel_t(ekf)
        ax1.plot(tk, ekf["ex"], color=COL_KF, alpha=.9, label="KF")
        ax2.plot(tk, ekf["ey"], color=COL_KF, alpha=.9, label="KF")
        ax3.plot(tk, ekf["ez"], color=COL_KF, alpha=.9, label="KF")
        ax4.plot(tk, ekf["et"], color=COL_KF, lw=2, alpha=.9, label="KF")

    if eew:
        te = rel_t(eew)
        ax1.plot(te, eew["ex"], color=COL_EW, alpha=.9, label="EW-RLS")
        ax2.plot(te, eew["ey"], color=COL_EW, alpha=.9, label="EW-RLS")
        ax3.plot(te, eew["ez"], color=COL_EW, alpha=.9, label="EW-RLS")
        ax4.plot(te, eew["et"], color=COL_EW, lw=2, alpha=.9, label="EW-RLS")

    for ax, lab in zip((ax1,ax2,ax3), ("Error X [m]","Error Y [m]","Error Z [m]")):
        ax.set_ylabel(lab); ax.set_xlabel("Relative time [s]"); ax.grid(True, alpha=.3)
    ax4.set_ylabel("Total error [m]"); ax4.set_xlabel("Relative time [s]")
    ax4.grid(True, alpha=.3); ax4.legend(frameon=False)
    fig.suptitle(f"Estimated error vs time — {tgt}")
    fig.tight_layout(rect=[0,0.04,1,0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"{tgt}_errors_overlay.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------- Discovery (robust, no collisions) ----------
def _discover_kf(run_id: str) -> List[Path]:
    # KF legacy: exports/tracks/<RUN_ID>/HGV_*_track_icrf_forward.csv
    base = ROOT / f"exports/tracks/{run_id}"
    return sorted(base.glob("HGV_*_track_icrf_forward.csv"))

def _discover_ew(run_id: str) -> List[Path]:
    # Prefer EW files that are NOT the KF compatibility ones
    base = ROOT / f"exports/tracks/{run_id}"
    ew = sorted(base.glob("ewrls_tracks_HGV_*.csv"))
    if ew:
        return ew
    # Fallback to new icrf folder pattern
    alt = ROOT / f"exports/tracks_icrf/{run_id}"
    return sorted(alt.glob("ewrls_icrf_HGV_*.csv"))

# ---------- Main compare ----------
def compare_run(run_id: str,
                out_dir: Path,
                cfg: Optional[str]=None) -> pd.DataFrame:
    paths = load_paths(cfg)
    oem_root = paths["oem_root"]

    ew_files = _discover_ew(run_id)
    kf_files = _discover_kf(run_id)

    if not ew_files and not kf_files:
        raise FileNotFoundError(f"No estimator CSVs found for run {run_id}")

    # Index by target
    ew_by_tgt = {extract_target_id(p.name): p for p in ew_files}
    kf_by_tgt = {extract_target_id(p.name): p for p in kf_files}

    targets = sorted(set(ew_by_tgt) | set(kf_by_tgt))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: List[Dict[str,Any]] = []

    for tgt in targets:
        oem = oem_root / f"{tgt}.oem"
        if not oem.exists():
            print(f"[WARN] OEM missing for {tgt} → skip plots/metrics")
            continue
        tru = read_oem_icrf(oem)

        kf_df = read_estimator_csv(kf_by_tgt[tgt]) if tgt in kf_by_tgt else None
        ew_df = read_estimator_csv(ew_by_tgt[tgt]) if tgt in ew_by_tgt else None

        # Sanity: warn if exactly identical after alignment
        if kf_df is not None and ew_df is not None:
            try:
                rmsm = rms_between(kf_df, ew_df)
                if rmsm < 1e-6:
                    print(f"[WARN] {tgt}: KF and EW are numerically identical (RMS {rmsm:.3e} m). Check inputs.")
            except Exception as _:
                pass

        # plots
        plot_3d_overlay(tgt, kf_df, ew_df, tru, out_dir)
        ekf = errors_vs_truth(kf_df, tru) if kf_df is not None else None
        eew = errors_vs_truth(ew_df, tru) if ew_df is not None else None
        plot_errors_overlay(tgt, ekf, eew, out_dir)

        # metrics
        row = {"target": tgt}
        if ekf: row.update({f"KF_{k}": v for k,v in metrics_from_errors(ekf).items()})
        if eew: row.update({f"EW_{k}": v for k,v in metrics_from_errors(eew).items()})
        metrics_rows.append(row)

    df = pd.DataFrame(metrics_rows)
    if not df.empty:
        df.to_csv(out_dir / "metrics.csv", index=False)
        # small README
        with (out_dir / "README.md").open("w") as f:
            f.write(f"# Estimator comparison — {run_id}\n\n")
            f.write("**KF (legacy)** vs **EW-RLS (poly RLS)**, truth = OEM ICRF\n\n")
            if not df.empty:
                f.write(df.to_markdown(index=False))
    return df

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compare KF vs EW-RLS against OEM ICRF; overlay plots + metrics.")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to exports/compare/<RUN_ID>")
    ap.add_argument("--config", default=str(CFG_PATH))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / f"exports/compare/{args.run_id}")
    df = compare_run(args.run_id, out_dir, cfg=args.config)
    if df.empty:
        print("[WARN] No metrics computed (no targets?)")
    else:
        # Summary line, robust to missing columns
        kf_mean = df.filter(like="KF_RMSE3D_m").values.mean() if any(c.startswith("KF_RMSE3D_m") for c in df.columns) else float("nan")
        ew_mean = df.filter(like="EW_RMSE3D_m").values.mean() if any(c.startswith("EW_RMSE3D_m") for c in df.columns) else float("nan")
        print(f"[MET ] targets={len(df)} | KF_RMSE3D(m) mean={np.nanmean([kf_mean]):.0f} | EW_RMSE3D(m) mean={np.nanmean([ew_mean]):.0f}")
        print(f"[OK] Saved metrics and plots → {out_dir}")