# orchestrator/run_parametric_study.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess, sys, time, datetime as dt
import pandas as pd
import yaml

ROOT        = Path(__file__).resolve().parents[1]
CFG_PATH    = ROOT / "config/pipeline.yaml"
ORCH_PY     = ROOT / "orchestrator" / "main_pipeline.py"
COMPARE_PY  = ROOT / "estimationandfiltering" / "compare_estimators.py"
TRI_ROOT    = ROOT / "exports" / "triangulation"

def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_cfg(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def make_study_id(tag: str = "LOS_SWEEP") -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ_") + tag

def latest_tri_run_id() -> str:
    runs = [p for p in TRI_ROOT.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No triangulation runs under {TRI_ROOT}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name

def run_orchestrator(cfg_path: Path) -> str:
    t0 = time.time()
    cmd = [sys.executable, str(ORCH_PY), "--config", str(cfg_path)]
    print(f"[RUN ] Orchestrator → {ORCH_PY.name}\n       cfg={cfg_path}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt_s = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise RuntimeError(f"[ERR ] Orchestrator failed in {dt_s:.1f}s")
    rid = latest_tri_run_id()
    print(f"[OK  ] Orchestrator ✓ run_id={rid} | {dt_s:.1f}s")
    return rid

def run_compare(run_id: str, out_dir: Path) -> pd.DataFrame:
    t0 = time.time()
    kf_glob = "exports/tracks/{RUN_ID}/*_track_icrf_forward.csv"
    ew_glob = "exports/tracks/{RUN_ID}/ewrls_tracks_*.csv"
    cmd = [
        sys.executable, str(COMPARE_PY),
        "--run_id", run_id,
        "--out_dir", str(out_dir),
        "--kf_glob", kf_glob,
        "--ew_glob", ew_glob,
        "--config", str(CFG_PATH),
    ]
    print(f"[RUN ] Compare → {COMPARE_PY.name}\n       run_id={run_id}\n       out={out_dir}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    dt_s = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise RuntimeError(f"[ERR ] Compare failed in {dt_s:.1f}s")
    print(f"[OK  ] Compare ✓ | {dt_s:.1f}s")

    met_csv = out_dir / "metrics.csv"
    if not met_csv.exists():
        print(f"[WARN] metrics.csv missing in {out_dir}")
        return pd.DataFrame()
    df = pd.read_csv(met_csv)
    try:
        kf_mean = df["KF_RMSE3D_m"].mean()
        ew_mean = df["EW_RMSE3D_m"].mean()
        print(f"[MET ] targets={len(df)} | KF_RMSE3D(m) mean={kf_mean:.0f} | EW_RMSE3D(m) mean={ew_mean:.0f}")
    except Exception:
        pass
    return df

def plot_summary(df_all: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("KF_RMSE3D_m","EW_RMSE3D_m") if c in df_all.columns]
    agg = df_all.groupby("sigma_urad", as_index=False)[cols].mean() if cols else pd.DataFrame()
    fig = plt.figure(figsize=(6.6,4.8)); ax = fig.add_subplot(111)
    if "KF_RMSE3D_m" in agg:
        ax.plot(agg["sigma_urad"], agg["KF_RMSE3D_m"]/1000.0, "o-", label="KF (mean) [km]")
    if "EW_RMSE3D_m" in agg:
        ax.plot(agg["sigma_urad"], agg["EW_RMSE3D_m"]/1000.0, "s--", label="EW-RLS (mean) [km]")
    ax.set_xlabel("LOS uncertainty σ [µrad]")
    ax.set_ylabel("RMSE3D [km]")
    ax.grid(True, alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    png = out_dir / "summary_rmse_vs_sigma.png"
    fig.savefig(png, dpi=160, bbox_inches="tight"); plt.close(fig)
    return png

def main():
    base_cfg = load_cfg(CFG_PATH)
    sigmas = [50, 300]  # µrad

    study_id = make_study_id("LOS_SWEEP")
    study_root = ROOT / "exports" / "studies" / study_id
    study_root.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Study id={study_id} | out={study_root}")

    all_rows: List[pd.DataFrame] = []

    for sigma in sigmas:
        t_sigma = time.time()
        label = f"{sigma}urad"
        print(f"\n=== σ_LOS = {sigma} µrad ===")

        cfg2 = base_cfg.copy()
        cfg2.setdefault("geometry", {}).setdefault("measurement_noise", {})
        cfg2["geometry"]["measurement_noise"]["los_sigma"] = float(sigma) * 1e-6
        cfg_run = study_root / f"pipeline_sigma{label}.yaml"
        save_cfg(cfg2, cfg_run)
        print(f"[CFG ] los_sigma={cfg2['geometry']['measurement_noise']['los_sigma']:.6e} rad | {cfg_run.name}")

        run_id = run_orchestrator(cfg_run)
        compare_dir = study_root / f"compare_{label}"
        df = run_compare(run_id, compare_dir)

        if df.empty:
            print("[WARN] No metrics produced.")
        else:
            df["sigma_urad"] = float(sigma)
            df["run_id"] = run_id
            all_rows.append(df)

        print(f"[DONE] σ_LOS={sigma} µrad ✓ | {time.time()-t_sigma:.1f}s")

    if not all_rows:
        print("[FATAL] No metrics collected. Aborting summary.")
        return

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(study_root / "metrics_all.csv", index=False)
    summary_png = plot_summary(big, study_root)
    print(f"\n[OK  ] Study complete → {study_root}")
    print(f"[FIG ] {summary_png}")

if __name__ == "__main__":
    main()