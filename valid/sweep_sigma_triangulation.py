#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import subprocess
import sys
import time
import re
import os
import yaml
import shutil

# Constants
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "pipeline.yaml"
CFG_BACKUP = ROOT / "config" / "pipeline.yaml.bak"
ORCH = ROOT / "orchestrator" / "main_pipeline.py"
VALID = ROOT / "valid" / "validate_triangulate_icrf.py"
SIGMAS_URAD = [50,150,300,450]

RUN_ID_RE = re.compile(r'(?:RUN_ID:\s*|run_id=\s*)([0-9A-Za-z_]+)')


# Helpers
def _load_cfg() -> Dict:
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def _save_cfg(cfg: Dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _extract_run_id(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        m = RUN_ID_RE.search(line)
        if m:
            return m.group(1).strip()
    for line in reversed(stdout.splitlines()):
        if "Using RUN_ID:" in line:
            tok = line.split("Using RUN_ID:")[-1].strip()
            if tok:
                return tok
    return ""


def _latest_valid_run_id(tri_root: Path) -> str:
    if not tri_root.exists():
        return ""
    runs = [
        p.name for p in tri_root.iterdir()
        if p.is_dir() and re.match(r'^\d{8}T\d{6}Z_', p.name)
    ]
    runs.sort()
    return runs[-1] if runs else ""


def _run_orchestrator_with_sigma(sigma_urad: float) -> str:
    """
    Run orchestrator with specific sigma by temporarily modifying pipeline.yaml.
    """
    sigma_rad = float(sigma_urad) * 1e-6

    # Backup original config
    print(f"[CFG ] Backing up original config")
    shutil.copy2(CFG_PATH, CFG_BACKUP)

    try:
        # Load and modify config
        cfg = _load_cfg()

        # Set sigma values
        cfg.setdefault("gpm_measurement", {})
        cfg["gpm_measurement"]["los_noise_rad"] = sigma_rad

        cfg.setdefault("geometry", {})
        cfg["geometry"]["los_sigma"] = sigma_rad

        # Force triangulation-only mode
        cfg.setdefault("orchestrator", {})
        cfg["orchestrator"]["run_triangulation"] = True
        cfg["orchestrator"]["run_geom_plots"] = True
        cfg["orchestrator"]["run_filter_forward"] = False
        cfg["orchestrator"]["run_filter_plots"] = False
        cfg["orchestrator"]["run_interactive_3d"] = False

        # Save modified config
        _save_cfg(cfg, CFG_PATH)
        print(f"[CFG ] Updated pipeline.yaml with sigma={sigma_urad}µrad ({sigma_rad:.6e} rad)")

        # Run orchestrator
        cmd = [sys.executable, str(ORCH)]

        # Force environment to disable filters
        env = os.environ.copy()
        env["RUN_TRIANGULATION"] = "1"
        env["RUN_GEOM_PLOTS"] = "1"
        env["RUN_FILTER_FORWARD"] = "0"
        env["RUN_FILTER_PLOTS"] = "0"
        env["RUN_INTERACTIVE_3D"] = "0"
        env["USE_KF"] = "0"
        env["USE_EWRLS"] = "0"
        env["REUSE_TRIANGULATION"] = "0"

        print(f"[RUN ] Orchestrator | sigma={sigma_urad}µrad")
        print(f"[ENV ] Forcing: RUN_FILTER_FORWARD=0, USE_KF=0, USE_EWRLS=0")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0

        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"Orchestrator failed (exit {proc.returncode})")

        print(f"[OK  ] Orchestrator ✓ | {dt:.1f}s")

        # Extract RUN_ID
        run_id = _extract_run_id(proc.stdout)
        if not run_id:
            tri_root = (ROOT / "exports" / "triangulation").resolve()
            run_id = _latest_valid_run_id(tri_root)

        if not run_id:
            raise RuntimeError("Could not recover RUN_ID")

        # Verify sigma was saved correctly
        tri_dir = ROOT / "exports" / "triangulation" / run_id
        if tri_dir.exists():
            sigma_file = tri_dir / "los_sigma_rad.txt"
            if sigma_file.exists():
                saved_sigma = float(sigma_file.read_text().strip())
                print(f"[VER ] Saved sigma: {saved_sigma:.6e} rad (expected {sigma_rad:.6e})")
                if abs(saved_sigma - sigma_rad) > 1e-9:
                    print(f"[WARN] Sigma mismatch! Delta = {abs(saved_sigma - sigma_rad):.6e}")
            else:
                print(f"[WARN] No los_sigma_rad.txt in {tri_dir}")

        return run_id

    finally:
        # Restore original config
        print(f"[CFG ] Restoring original config")
        shutil.copy2(CFG_BACKUP, CFG_PATH)
        CFG_BACKUP.unlink(missing_ok=True)


def _run_validator(run_ids: List[str]) -> None:
    if not run_ids:
        return
    cmd = [sys.executable, str(VALID), "--run_ids", *run_ids]
    print(f"[RUN ] Validator | run_ids={run_ids}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("Validator failed")


# Main
def main() -> None:
    print("=" * 60)
    print("SIGMA SWEEP - TRIANGULATION VALIDATION")
    print("=" * 60)

    run_ids: List[str] = []

    for s in SIGMAS_URAD:
        print(f"\n--- SIGMA = {s} µrad ---")

        try:
            rid = _run_orchestrator_with_sigma(s)
            print(f"[RID ] {s} µrad → run_id={rid}\n")
            run_ids.append(rid)
        except Exception as e:
            print(f"[ERR ] Failed for {s} µrad: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Completed {len(run_ids)} triangulation runs")

    if run_ids:
        print("\nRunning validator...")
        _run_validator(run_ids)

    print("\n[DONE] Sweep finished. Check:")
    print(f"  - Triangulation outputs: exports/triangulation/")
    print(f"  - Validation results: valid/triangulation_checks/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user")
        # Ensure config is restored
        if CFG_BACKUP.exists():
            print("[CFG ] Restoring original config after interrupt")
            shutil.copy2(CFG_BACKUP, CFG_PATH)
            CFG_BACKUP.unlink()
        sys.exit(1)