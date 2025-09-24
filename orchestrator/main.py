# orchestrator/main.py
from __future__ import annotations
import sys, subprocess
from pathlib import Path

# --- config loader ---
try:
    from config.loader import load_config
except Exception:
    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config


def run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN ] {' '.join(str(c) for c in cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if p.stdout: print(p.stdout, end="")
    if p.stderr: print(p.stderr, end="")
    if p.returncode != 0:
        raise RuntimeError(f"[ERR ] Command failed: {' '.join(map(str, cmd))}")


def ensure_triangulation_outputs(root: Path, run_id: str) -> list[Path]:
    tri_dir = root / "exports" / "triangulation" / run_id
    csvs = sorted(tri_dir.glob("xhat_geo_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nessun xhat_geo_*.csv in {tri_dir}")
    return csvs


def main() -> None:
    CFG = load_config()
    ROOT = Path(CFG["project"]["root"]).resolve()
    RUN_ID = CFG["project"]["run_id"]
    print(f"[INIT] ez-SMAD orchestrator | run_id={RUN_ID}")

    tri_py  = ROOT / "geometry" / "triangulate_icrf.py"
    kf_py   = ROOT / "estimationandfiltering" / "run_filter.py"
    ew_py   = ROOT / "estimationandfiltering" / "ewrls_icrf_tracks.py"
    comp_n  = ROOT / "estimationandfiltering" / "comparator_new.py"
    comp_o  = ROOT / "estimationandfiltering" / "compare_estimators.py"
    beta_py = ROOT / "geometry" / "plot_beta_vs_err.py"
    rmse_py = ROOT / "estimationandfiltering" / "plot_rmse_components.py"
    rpt_py  = ROOT / "tools" / "architecture_performance_report.py"
    plots_dir = ROOT / "plots" / RUN_ID
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Triangolazione (nessun pre-check)
    print("\n[STEP] Triangulation")
    run([sys.executable, str(tri_py)], cwd=ROOT)
    _csvs = ensure_triangulation_outputs(ROOT, RUN_ID)
    print(f"[OK ] Triangulation CSVs: {len(_csvs)}")

    # 2) KF
    print("\n[STEP] Kalman Filter (forward)")
    run([sys.executable, str(kf_py)], cwd=ROOT)

    # 4) Compare
    print("\n[STEP] Compare estimators (KF vs EW-RLS)")
    if comp_n.exists():
        run([sys.executable, str(comp_n), "--run_id", RUN_ID, "--out_dir", str(plots_dir)], cwd=ROOT)
    else:
        run([sys.executable, str(comp_o), "--run_id", RUN_ID, "--out_dir", str(plots_dir),
             "--config", str(ROOT / "config" / "pipeline.yaml")], cwd=ROOT)

    # 5) β vs errore
    if beta_py.exists():
        print("\n[STEP] Plot beta-vs-error")
        run([sys.executable, str(beta_py), "--run_id", RUN_ID], cwd=ROOT)
    else:
        print("[SKIP] geometry/plot_beta_vs_err.py non trovato")

    # 6) RMSE per asse (KF vs Triangolazione)
    if rmse_py.exists():
        print("\n[STEP] Plot RMSE components (KF vs Triangolazione)")
        run([sys.executable, str(rmse_py), "--run_id", RUN_ID], cwd=ROOT)
    else:
        print("[SKIP] estimationandfiltering/plot_rmse_components.py non trovato")

    # 7) Report PDF
    if rpt_py.exists():
        print("\n[STEP] Build architecture performance report (PDF)")
        run([sys.executable, str(rpt_py), "--run_id", RUN_ID], cwd=ROOT)
    else:
        print("[SKIP] tools/architecture_performance_report.py non trovato")

    m_csv = plots_dir / "metrics.csv"
    print(f"\n[OK ] Metrics → {m_csv}" if m_csv.exists() else "\n[WARN] metrics.csv non trovato in plots")

    print(f"[DONE] Run complete ✓ | plots dir: {plots_dir}")


if __name__ == "__main__":
    main()