# orchestrator/run_parametric_study.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil, subprocess, sys, os, time, datetime as dt
import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "config/pipeline.yaml"

ORCH      = ROOT / "orchestrator" / "main_pipeline.py"
EWRLS_ICRF= ROOT / "estimationandfiltering" / "ewrls_icrf_tracks.py"
COMPARE   = ROOT / "estimationandfiltering" / "compare_estimators.py"

TRI_ROOT  = ROOT / "exports" / "triangulation"

# --------------------------------------------------------------------------------------
# Helpers (cfg, run_id, io)
# --------------------------------------------------------------------------------------
def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_cfg(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def make_run_id(tag: str = "") -> str:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}{('_'+tag) if tag else ''}"

def latest_tri_run_id() -> str:
    runs = [p for p in TRI_ROOT.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No triangulation runs under {TRI_ROOT}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name

# --------------------------------------------------------------------------------------
# Steps (subprocess wrappers)
# --------------------------------------------------------------------------------------
def run_orchestrator(cfg_path: Path, env: Optional[dict]=None) -> str:
    t0 = time.time()
    cmd = [sys.executable, str(ORCH), "--config", str(cfg_path)]
    print(f"[RUN ] Orchestrator → {ORCH.name}\n       cfg={cfg_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    dt_s = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep="\n")
        raise RuntimeError(f"[ERR ] Orchestrator failed in {dt_s:.1f}s")
    rid = latest_tri_run_id()
    print(f"[OK  ] Orchestrator ✓ run_id={rid} | {dt_s:.1f}s")
    return rid

def run_ewrls_icrf(run_id: str, tri_frame: str = "ECEF"):
    if not EWRLS_ICRF.exists():
        raise FileNotFoundError(f"Script not found: {EWRLS_ICRF}")
    t0 = time.time()
    cmd = [sys.executable, str(EWRLS_ICRF), "--run_id", run_id, "--tri_frame", tri_frame]
    print(f"[RUN ] EW-RLS(ICRF)\n       cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt_s = time.time() - t0
    if proc.stdout.strip():
        print(f"[LOG ] EW-RLS stdout:\n{proc.stdout.strip()}\n")
    if proc.stderr.strip():
        print(f"[LOG ] EW-RLS stderr:\n{proc.stderr.strip()}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"[ERR ] EW-RLS failed (rc={proc.returncode}) in {dt_s:.2f}s")
    print(f"[OK  ] EW-RLS(ICRF) ✓ | {dt_s:.2f}s")

def run_compare(run_id: str, out_dir: Path, debug: bool=False) -> pd.DataFrame:
    t0 = time.time()
    cmd = [sys.executable, str(COMPARE), "--run_id", run_id, "--out_dir", str(out_dir)]
    if debug:
        cmd.append("--debug")
    print(f"[RUN ] Compare → {COMPARE.name}\n       run_id={run_id}\n       out={out_dir}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt_s = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise RuntimeError(f"[ERR ] Compare failed in {dt_s:.1f}s")
    print(f"[OK  ] Compare ✓ | {dt_s:.1f}s")
    m = out_dir / "metrics.csv"
    return pd.read_csv(m) if m.exists() else pd.DataFrame()

# --------------------------------------------------------------------------------------
# Summary plot across sigmas
# --------------------------------------------------------------------------------------
def plot_summary(df_all: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    # Aggregate means per sigma
    cols = []
    if "KF_RMSE3D_m" in df_all.columns: cols.append("KF_RMSE3D_m")
    if "EW_RMSE3D_m" in df_all.columns: cols.append("EW_RMSE3D_m")
    agg = df_all.groupby("sigma_urad", as_index=False)[cols].mean() if cols else pd.DataFrame()
    fig = plt.figure(figsize=(6.2,4.4)); ax = fig.add_subplot(111)
    if "KF_RMSE3D_m" in agg.columns:
        ax.plot(agg["sigma_urad"], agg["KF_RMSE3D_m"]/1000.0, "o-", label="KF (mean) [km]")
    if "EW_RMSE3D_m" in agg.columns:
        ax.plot(agg["sigma_urad"], agg["EW_RMSE3D_m"]/1000.0, "s--", label="EW-RLS (mean) [km]")
    ax.set_xlabel("LOS uncertainty σ [µrad]"); ax.set_ylabel("RMSE3D [km]")
    ax.grid(True, alpha=.3); ax.legend(frameon=False)
    fig.tight_layout()
    png = out_dir / "summary_rmse_vs_sigma.png"
    fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)
    return png

# --------------------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------------------
def main(
    sigmas_urad: List[int] = (50, 75, 100, 150, 200, 300),
    tri_frame: str = "ECEF",
    keep_tmp_cfgs: bool = True,
    debug_compare: bool = False,
):
    base_cfg = load_cfg(CFG)
    study_id = make_run_id("LOS_SWEEP")
    study_root = ROOT / "exports" / "studies" / study_id
    study_root.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Study id={study_id} | out={study_root}")

    all_rows: List[pd.DataFrame] = []

    for sigma in sigmas_urad:
        t_sigma = time.time()
        print(f"\n=== σ_LOS = {sigma} µrad ===")

        # 1) clone config e set LOS (µrad → rad)
        cfg2 = base_cfg.copy()
        cfg2.setdefault("geometry", {}).setdefault("measurement_noise", {})
        cfg2["geometry"]["measurement_noise"]["los_sigma"] = float(sigma) * 1e-6
        cfg_run = study_root / f"pipeline_sigma{sigma}.yaml"
        save_cfg(cfg2, cfg_run)
        print(f"[CFG ] los_sigma={cfg2['geometry']['measurement_noise']['los_sigma']:.6e} rad | {cfg_run.name}")

        # 2) orchestrator (triangolazione + KF)
        rid = run_orchestrator(cfg_run)

        # 3) EW-RLS (ICRF) — usa triangolazione di questa run
        run_ewrls_icrf(rid, tri_frame=tri_frame)

        # 4) compare (overlay + metrics)
        compare_dir = study_root / f"compare_{sigma}urad"
        df = run_compare(rid, compare_dir, debug=debug_compare)

        if df.empty:
            print("[WARN] No metrics produced.")
        else:
            df["sigma_urad"] = sigma
            df["run_id"] = rid
            all_rows.append(df)
            kf_mean = df["KF_RMSE3D_m"].mean() if "KF_RMSE3D_m" in df.columns else np.nan
            ew_mean = df["EW_RMSE3D_m"].mean() if "EW_RMSE3D_m" in df.columns else np.nan
            print(f"[MET ] targets={len(df)} | KF_RMSE3D(m) mean={np.nan_to_num(kf_mean):.0f} | EW_RMSE3D(m) mean={np.nan_to_num(ew_mean):.0f}")

        print(f"[DONE] σ_LOS={sigma} µrad ✓ | {time.time()-t_sigma:.1f}s")

    if not all_rows:
        print("[FATAL] No metrics collected. Aborting summary.")
        return

    # 5) Summary finale
    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(study_root / "metrics_all.csv", index=False)
    summary_png = plot_summary(big, study_root)
    print(f"\n[OK  ] Study complete → {study_root}")
    print(f"[FIG ] {summary_png}")

    # opzionale: pulizia cfg temporanei
    if not keep_tmp_cfgs:
        for p in study_root.glob("pipeline_sigma*.yaml"):
            try: p.unlink()
            except Exception: pass

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Parametric LOS study: orchestrator + EW-RLS + compare + summary.")
    ap.add_argument("--sigmas_urad", nargs="+", type=int, default=[50, 300],
                    help="List of LOS sigmas in µrad (will be stored in config as radians).")
    ap.add_argument("--tri_frame", default="ECEF", help="Triangulation frame used before EW-RLS ICRF fit.")
    ap.add_argument("--no_debug_compare", action="store_true", help="Disable extra compare debug.")
    ap.add_argument("--rm_cfgs", action="store_true", help="Remove temporary per-sigma config files at the end.")
    args = ap.parse_args()

    main(
        sigmas_urad=args.sigmas_urad,
        tri_frame=args.tri_frame,
        keep_tmp_cfgs=(not args.rm_cfgs),
        debug_compare=(not args.no_debug_compare),
    )