# orchestrator/run_sigma_sweep_simple.py
"""
Lancia orchestrator/main.py per due run con sigma_LOS = 50 e 500 µrad.
- Scrive temporaneamente config/pipeline.yaml con:
  - gpm_measurement.los_noise_rad = sigma * 1e-6
  - project.run_id = <timestamp>_sigma{X}urad
- Esegue main.py
- Ripristina SEMPRE il pipeline.yaml originale.
"""

from __future__ import annotations
import sys, shutil, subprocess, time
from pathlib import Path
from datetime import datetime
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "pipeline.yaml"
MAIN_PY  = ROOT / "orchestrator" / "main.py"

SIGMAS_URAD = [100.0, 300.0, 450.0]   # <-- qui la lista delle sigma

def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN ] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def _load_yaml(p: Path) -> dict:
    with p.open("r") as f:
        return yaml.safe_load(f)

def _save_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def _make_run_id(tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{tag}"

def main():
    # Backup dell'originale
    backup = CFG_PATH.with_suffix(".bak")
    shutil.copy2(CFG_PATH, backup)
    print(f"[INIT] Backup → {backup.name}")

    try:
        base = _load_yaml(CFG_PATH)

        for s in SIGMAS_URAD:
            print(f"\n=== σ_LOS = {s:.0f} µrad ===")
            cfg = dict(base)  # shallow copy

            # 1) set sigma (in radianti)
            gpm = dict(cfg.get("gpm_measurement", {}))
            gpm["los_noise_rad"] = float(s) * 1e-6
            cfg["gpm_measurement"] = gpm

            # 2) forza un run_id esplicito con la sigma
            proj = dict(cfg.get("project", {}))
            proj["run_id"] = _make_run_id(f"sigma{int(s)}urad")
            cfg["project"] = proj

            # 3) scrivi pipeline.yaml temporaneo
            _save_yaml(cfg, CFG_PATH)
            print(f"[CFG ] gpm_measurement.los_noise_rad={gpm['los_noise_rad']:.6e} | run_id={proj['run_id']}")

            # 4) esegui main.py
            t0 = time.time()
            _run([sys.executable, str(MAIN_PY)], cwd=ROOT)
            print(f"[OK ] main.py ✓ ({time.time()-t0:.1f}s)  | run_id={proj['run_id']}")

        print("\n[DONE] Sweep completato.")

    finally:
        # Ripristino config originale
        shutil.copy2(backup, CFG_PATH)
        print(f"[RESTORE] pipeline.yaml ripristinato dall'originale.")

if __name__ == "__main__":
    main()