#!/usr/bin/env python3
"""
Validate triangulation outputs vs LOS uncertainty (σ_LOS).

- Auto-discovers triangulation runs under the path from config.loader.load_config():
    TRI_ROOT = Path(CFG["paths"]["exports_root"]) / "triangulation" / <RUN_ID>
- For each RUN_ID:
    * expects: xhat_geo_<TARGET>.csv    (triangulated pseudo-positions)
               los_sigma_rad.txt        (single-line float in rad)
- Computes per-target errors vs OEM truth (ICRF), aggregates metrics, and plots:
    * summary_metrics.csv
    * scatter_vs_sigma.png
    * time_series_errnorm_<target>.png (overlay across runs)
    * cep_vs_beta_<target>.png

Run examples (from repo root):
    python -m valid.validate_triangulate_icrf
    python -m valid.validate_triangulate_icrf --latest 2
    python -m valid.validate_triangulate_icrf --run_ids 20250922T160046Z_0sats_0km_0deg_0adaaec8 20250922T160148Z_0sats_0km_0deg_99184a99
    python -m valid.validate_triangulate_icrf --runs exports/triangulation/2025.../ exports/triangulation/2025.../
    python -m valid.validate_triangulate_icrf --targets HGV_00001 HGV_00002
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Import config.loader robustly (allow running as module or script)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.loader import load_config  # type: ignore

# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------
@dataclass
class RunInfo:
    path: Path
    name: str
    sigma_los_rad: float  # radians


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _tri_root_from_cfg(cfg: Dict) -> Path:
    exports_root = Path(cfg["paths"]["exports_root"])
    return (ROOT / exports_root / "triangulation").resolve()

def _read_sigma_from_run(run_dir: Path) -> Optional[float]:
    """Primary: los_sigma_rad.txt (single-line float, rad). Fallback: parse '*urad' in name."""
    f = run_dir / "los_sigma_rad.txt"
    if f.exists():
        try:
            return float(f.read_text().strip())
        except Exception:
            pass
    # Fallback: parse “…_<N>urad” in directory name
    m = re.search(r"(\d+(?:\.\d+)?)\s*urad", run_dir.name, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e-6
    return None

def _discover_latest_runs(tri_root: Path, n: int = 2) -> List[Path]:
    """Return the latest N *valid* run directories under tri_root.
    A valid run dir is a directory that:
      - does not start with an underscore (e.g. "_old")
      - contains at least one file matching "xhat_geo_*.csv" directly inside it.
    """
    if not tri_root.exists():
        raise FileNotFoundError(f"Triangulation root not found: {tri_root}")

    def is_valid_run(p: Path) -> bool:
        if not p.is_dir():
            return False
        if p.name.startswith("_"):
            return False
        # must have at least one triangulation CSV at top level
        return any(p.glob("xhat_geo_*.csv"))

    candidates = [p for p in tri_root.iterdir() if is_valid_run(p)]
    if not candidates:
        # Provide a helpful hint of what's there
        names = ", ".join(sorted([p.name for p in tri_root.iterdir() if p.is_dir()]))
        raise RuntimeError(
            "No valid triangulation runs found. "
            f"Looked in: {tri_root}. Subdirs: [{names}]\n"
            "A run dir must contain at least one 'xhat_geo_*.csv'. "
            "Try specifying --runs or --run_ids explicitly."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:n]

def _discover_targets_in_run(run_dir: Path) -> List[str]:
    return sorted([p.stem.split("xhat_geo_")[-1] for p in run_dir.glob("xhat_geo_*.csv")])

def _load_xhat(run_dir: Path, target_id: str) -> pd.DataFrame:
    f = run_dir / f"xhat_geo_{target_id}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing triangulated file: {f}")
    df = pd.read_csv(f)
    # Expect a 'time' column (UTC ISO) from triangulation; enforce index
    if "time" not in df.columns:
        raise ValueError(f"{f} lacks 'time' column")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True).set_index("time")
    return df

def _load_truth_oem_icrf(oem_dir: Path, target_id: str) -> pd.DataFrame:
    f = oem_dir / f"{target_id}.oem"
    if not f.exists():
        raise FileNotFoundError(f"Truth OEM not found: {f}")
    recs = []
    with f.open("r", encoding="utf-8", errors="ignore") as fh:
        meta = False
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s == "META_START":
                meta = True; continue
            if s == "META_STOP":
                meta = False; continue
            if meta:
                continue
            parts = s.split()
            if len(parts) >= 7 and ("T" in parts[0] or ":" in parts[0]):
                t = pd.to_datetime(parts[0], utc=True, errors="coerce")
                if pd.isna(t):
                    continue
                try:
                    x, y, z, vx, vy, vz = map(float, parts[1:7])
                except Exception:
                    continue
                recs.append((t, x, y, z, vx, vy, vz))
    if not recs:
        raise RuntimeError(f"No numeric state lines in OEM {f}")
    df = pd.DataFrame(recs, columns=["time","x_km","y_km","z_km","vx_kmps","vy_kmps","vz_kmps"]) \
         .dropna(subset=["time"]).sort_values("time").set_index("time")
    return df

def _interp_truth_to(df_truth: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    tt = df_truth.reindex(df_truth.index.union(times)).sort_index()
    tt = tt.interpolate(method="time")[["x_km","y_km","z_km"]]
    out = tt.loc[times]
    out.index.name = "time"
    return out

def _rmse(a: np.ndarray, axis=0) -> np.ndarray:
    return np.sqrt(np.nanmean(a*a, axis=axis))

# --------------------------------------------------------------------------------------
# Metrics & plots
# --------------------------------------------------------------------------------------
def compute_metrics_for_run_target(run: RunInfo, target_id: str, oem_dir: Path) -> Tuple[pd.DataFrame, Dict[str,float]]:
    """Return (joined df with errors, metrics dict)."""
    xh = _load_xhat(run.path, target_id)
    tr = _load_truth_oem_icrf(oem_dir, target_id)
    tr_on = _interp_truth_to(tr, xh.index)

    # column mapping for triangulation outputs (robust)
    def pick(df: pd.DataFrame, cands) -> str:
        for c in cands:
            if c in df.columns: return c
        # fuzzy match
        low = [c.lower() for c in df.columns]
        for pat in cands:
            for i,lc in enumerate(low):
                if pat in lc: return df.columns[i]
        raise KeyError(f"Missing {cands} in triangulation file")

    xcol = pick(xh, ["xhat_x_km","x_km","x_icrf_km"])
    ycol = pick(xh, ["xhat_y_km","y_km","y_icrf_km"])
    zcol = pick(xh, ["xhat_z_km","z_km","z_icrf_km"])

    ex = xh[xcol].to_numpy(float) - tr_on["x_km"].to_numpy(float)
    ey = xh[ycol].to_numpy(float) - tr_on["y_km"].to_numpy(float)
    ez = xh[zcol].to_numpy(float) - tr_on["z_km"].to_numpy(float)
    en = np.sqrt(ex*ex + ey*ey + ez*ez)

    joined = xh.copy()
    joined["err_x_km"] = ex
    joined["err_y_km"] = ey
    joined["err_z_km"] = ez
    joined["err_norm_km"] = en

    finite = np.isfinite(en)
    rmse_xyz = _rmse(np.vstack([ex,ey,ez])[:, finite], axis=1)
    rmse_norm = float(_rmse(en[finite]))
    med_cep = float(np.nanmedian(joined.get("CEP50_km_analytic", pd.Series(np.nan, index=joined.index)).to_numpy(float)))
    med_nsat= float(np.nanmedian(joined.get("Nsats", pd.Series(np.nan, index=joined.index)).to_numpy(float)))
    med_beta= float(np.nanmedian(joined.get("beta_mean_deg", pd.Series(np.nan, index=joined.index)).to_numpy(float)))
    med_cond= float(np.nanmedian(joined.get("condA", pd.Series(np.nan, index=joined.index)).to_numpy(float)))

    m = {
        "run": run.name,
        "target": target_id,
        "sigma_los_rad": run.sigma_los_rad,
        "sigma_los_urad": run.sigma_los_rad * 1e6 if np.isfinite(run.sigma_los_rad) else np.nan,
        "rmse_x_km": float(rmse_xyz[0]),
        "rmse_y_km": float(rmse_xyz[1]),
        "rmse_z_km": float(rmse_xyz[2]),
        "rmse_norm_km": rmse_norm,
        "median_CEP50_km": med_cep,
        "median_Nsats": med_nsat,
        "median_beta_mean_deg": med_beta,
        "median_condA": med_cond,
        "n_epochs": int(finite.sum()),
    }
    return joined, m

def plot_time_series_errnorm(per_run_joined: Dict[str,pd.DataFrame], target_id: str, outdir: Path):
    plt.figure(figsize=(10, 4.5))
    for run_name, df in per_run_joined.items():
        plt.plot(df.index.to_pydatetime(), df["err_norm_km"].to_numpy(float), label=run_name)
    plt.title(f"Error norm vs time — {target_id}")
    plt.xlabel("UTC time")
    plt.ylabel("|x̂ − x_true| [km]")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    fn = outdir / f"time_series_errnorm_{target_id}.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()

def plot_scatter_vs_sigma(metrics_df: pd.DataFrame, outdir: Path):
    x = metrics_df["sigma_los_urad"].to_numpy(float)
    y1 = metrics_df["rmse_norm_km"].to_numpy(float)
    y2 = metrics_df["median_CEP50_km"].to_numpy(float)

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(x, y1, marker="o", label="RMSE_norm [km]")
    plt.scatter(x, y2, marker="s", label="median CEP50 [km]")
    # trend lines
    if np.isfinite(x).sum() >= 2 and np.isfinite(y1).sum() >= 2:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        c1 = np.polyfit(x[np.isfinite(x*y1)], y1[np.isfinite(x*y1)], 1)
        plt.plot(xs, np.polyval(c1, xs), alpha=0.6)
    if np.isfinite(x).sum() >= 2 and np.isfinite(y2).sum() >= 2:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        c2 = np.polyfit(x[np.isfinite(x*y2)], y2[np.isfinite(x*y2)], 1)
        plt.plot(xs, np.polyval(c2, xs), alpha=0.6)
    plt.title("Scatter vs LOS uncertainty")
    plt.xlabel("σ_LOS [µrad]")
    plt.ylabel("Distance [km]")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    fn = outdir / "scatter_vs_sigma.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()

def plot_cep_vs_beta(per_run_joined: Dict[str,pd.DataFrame], target_id: str, outdir: Path):
    plt.figure(figsize=(6.5, 4.5))
    for run_name, df in per_run_joined.items():
        if "beta_mean_deg" in df.columns and "CEP50_km_analytic" in df.columns:
            b = df["beta_mean_deg"].to_numpy(float)
            c = df["CEP50_km_analytic"].to_numpy(float)
            mask = np.isfinite(b) & np.isfinite(c)
            plt.scatter(b[mask], c[mask], s=10, alpha=0.45, label=run_name)
    plt.title(f"CEP50 vs baseline angle — {target_id}")
    plt.xlabel("β_mean [deg]")
    plt.ylabel("CEP50_analytic [km]")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    fn = outdir / f"cep_vs_beta_{target_id}.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()

# --------------------------------------------------------------------------------------
# CLI & main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate triangulation vs σ_LOS using paths from config.")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Explicit run directories (…/exports/triangulation/<RUN_ID>). Overrides --run_ids/--latest.")
    ap.add_argument("--run_ids", nargs="*", default=None,
                    help="RUN_ID names to read under triangulation root (from config).")
    ap.add_argument("--latest", type=int, default=2,
                    help="If no runs provided, pick the latest N runs (default 2).")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Optional subset of target IDs; default = intersection across runs.")
    ap.add_argument("--outdir", default="valid/triangulation_checks",
                    help="Output directory for CSV and plots.")
    return ap.parse_args()

def main():
    args = parse_args()
    CFG = load_config()
    tri_root = _tri_root_from_cfg(CFG)
    oem_dir = (ROOT / CFG["paths"]["oem_root"]).resolve()
    outdir = _ensure_outdir((ROOT / args.outdir).resolve())

    # Resolve run directories
    if args.runs:
        run_dirs = [Path(r).resolve() for r in args.runs]
    elif args.run_ids:
        run_dirs = [(tri_root / rid).resolve() for rid in args.run_ids]
    else:
        run_dirs = _discover_latest_runs(tri_root, n=max(2, int(args.latest)))

    # Build RunInfo (skip empty/invalid run dirs defensively)
    runs: List[RunInfo] = []
    for rd in run_dirs:
        if not rd.exists():
            print(f"[WARN] Run dir not found: {rd}")
            continue
        if not any(rd.glob("xhat_geo_*.csv")):
            print(f"[WARN] No xhat_geo_*.csv in {rd} — skipping")
            continue
        sigma = _read_sigma_from_run(rd)
        runs.append(RunInfo(path=rd, name=rd.name, sigma_los_rad=float(sigma) if sigma is not None else np.nan))

    if not runs:
        raise RuntimeError("No usable run directories left after filtering. Aborting.")

    # Determine targets
    if args.targets:
        targets = args.targets
    else:
        sets = []
        for r in runs:
            ts = set(_discover_targets_in_run(r.path))
            if not ts:
                print(f"[WARN] No xhat_geo_*.csv files in {r.path} — skipping this run for target intersection")
                continue
            sets.append(ts)
        if not sets:
            raise RuntimeError("Could not determine targets — no runs with triangulation CSVs found.")
        targets = sorted(list(set.intersection(*sets))) if len(sets) >= 2 else sorted(list(sets[0]))
        if not targets:
            raise RuntimeError("No common targets across runs; pass --targets explicitly.")

    # Per-target metrics + plots
    metrics_rows: List[Dict[str, float]] = []
    for tid in targets:
        per_run_joined: Dict[str, pd.DataFrame] = {}
        for r in runs:
            j, m = compute_metrics_for_run_target(r, tid, oem_dir)
            per_run_joined[r.name] = j
            metrics_rows.append(m)
        plot_time_series_errnorm(per_run_joined, tid, outdir)
        plot_cep_vs_beta(per_run_joined, tid, outdir)

    # Aggregate + summary plots
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = outdir / "summary_metrics.csv"
    metrics_df.sort_values(["target", "sigma_los_urad", "run"]).to_csv(metrics_csv, index=False)
    plot_scatter_vs_sigma(metrics_df, outdir)

    print(f"[OK ] Wrote metrics → {metrics_csv}")
    print(f"[OK ] Plots in → {outdir}")

if __name__ == "__main__":
    main()