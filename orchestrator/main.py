# orchestrator/main.py
"""
Orchestratore semplice: Triangulation → KF → EW-RLS → Compare → Beta-vs-Err plot → PDF report
Legge config e run_id da config.loader.load_config() (nessun argomento richiesto).
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path

# --- config loader (senza magie) ---
try:
    from config.loader import load_config
except Exception:
    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config


def sh(cmd: list[str], cwd: Path) -> None:
    """Run a command, crash verbosamente se fallisce."""
    print(f"[RUN ] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def main() -> None:
    CFG = load_config()
    ROOT = Path(CFG["project"]["root"]).resolve()
    RUN_ID = CFG["project"]["run_id"]

    print(f"[INIT] ez-SMAD orchestrator | run_id={RUN_ID}")

    # Percorsi script (tutti relativi alla repo)
    tri_py   = ROOT / "geometry" / "triangulate_icrf.py"
    kf_py    = ROOT / "estimationandfiltering" / "run_filter.py"
    ew_py    = ROOT / "estimationandfiltering" / "ewrls_icrf_tracks.py"
    comp_new = ROOT / "estimationandfiltering" / "comparator_new.py"
    comp_old = ROOT / "estimationandfiltering" / "compare_estimators.py"
    beta_py  = ROOT / "geometry" / "plot_beta_vs_err.py"
    rpt_py   = ROOT / "tools" / "architecture_performance_report.py"

    plots_dir = ROOT / "plots" / RUN_ID
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Triangolazione (genera xhat_geo_* e salva anche le nostre LOS)
    print("\n[STEP] Triangulation")
    sh([sys.executable, str(tri_py)], cwd=ROOT)

    # 2) KF (usa i risultati di triangolazione; legge tutto da config)
    print("\n[STEP] Kalman Filter (forward)")
    sh([sys.executable, str(kf_py)], cwd=ROOT)

    # 3) EW-RLS (ICRF tracks)
    print("\n[STEP] EW-RLS (ICRF tracks)")
    sh([sys.executable, str(ew_py)], cwd=ROOT)

    # 4) Comparator (preferisci comparator_new; fallback su compare_estimators)
    print("\n[STEP] Compare estimators (KF vs EW-RLS)")
    if comp_new.exists():
        sh([sys.executable, str(comp_new),
            "--run_id", RUN_ID, "--out_dir", str(plots_dir), "--config", str(ROOT / "config" / "pipeline.yaml")],
           cwd=ROOT)
    else:
        sh([sys.executable, str(comp_old),
            "--run_id", RUN_ID, "--out_dir", str(plots_dir), "--config", str(ROOT / "config" / "pipeline.yaml")],
           cwd=ROOT)

    # 5) Plot: errore KF vs geometria (β_mean, Nsats, condA, CEP50)
    if beta_py.exists():
        print("\n[STEP] Plot beta-vs-error")
        sh([sys.executable, str(beta_py), "--run_id", RUN_ID], cwd=ROOT)
    else:
        print("[SKIP] plot_beta_vs_err.py non trovato")

    # 5b) RMSE per-asse KF vs Triangolazione
    rmse_comp = ROOT / "estimationandfiltering" / "plot_rmse_components.py"
    if rmse_comp.exists():
        print("\n[STEP] Plot RMSE components (KF vs Triangolazione)")
        sh([sys.executable, str(rmse_comp)], cwd=ROOT)
    else:
        print("[SKIP] plot_rmse_components.py non trovato")

    # 6) PDF report (raccoglie i PNG in plots/{RUN_ID} e crea un PDF unico)
    if rpt_py.exists():
        print("\n[STEP] Build architecture performance report (PDF)")
        sh([sys.executable, str(rpt_py), "--run_id", RUN_ID], cwd=ROOT)
    else:
        print("[SKIP] architecture_performance_report.py non trovato")

    # Dov’è il metrics.csv?
    m1 = plots_dir / "metrics.csv"
    if m1.exists():
        print(f"\n[OK ] Metrics → {m1}")
    else:
        print("\n[WARN] metrics.csv non trovato in plots; controlla l'output del comparator.")

    print(f"[DONE] Run complete ✓ | plots dir: {plots_dir}")


if __name__ == "__main__":
    main()