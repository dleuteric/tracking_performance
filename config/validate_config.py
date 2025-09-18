# config/validate_config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
from loader import load_config
import glob

REQ_PATH_KEYS = [
    "stk_exports","los_root","ephem_root","oem_root",
    "triangulation_out","tracks_out","geom_plots_out","filter_plots_out"
]

def _abs_paths(cfg: Dict) -> Dict[str, Path]:
    root = Path(cfg["project"]["root"]).resolve()
    out: Dict[str, Path] = {}
    for k in REQ_PATH_KEYS:
        p = Path(cfg["paths"][k])
        if not p.is_absolute():
            p = (root / p).resolve()
        out[k] = p
    return out

def main():
    cfg = load_config()
    print("[OK ] YAML loaded.")

    # 1) Resolve all required paths against project.root
    apaths = _abs_paths(cfg)
    print(f"[OK ] project.root = {Path(cfg['project']['root']).resolve()}")

    # 2) Placeholders must have been resolved by loader
    for k in REQ_PATH_KEYS:
        v = cfg["paths"][k]
        assert "{" not in v and "}" not in v, f"Unresolved placeholder in paths.{k}: {v}"
    print("[OK ] Path placeholders resolved.")

    # 3) run_id.template sanity
    tmpl = cfg["run_id"]["template"]
    for key in ["date","nsats","alt_km","inc_deg","hash"]:
        assert "{" + key + "}" in tmpl, f"run_id.template missing {{{key}}}"
    print("[OK ] run_id.template contains required keys.")

    # 4) Types & ranges spot checks
    assert isinstance(cfg["geometry"]["earth_radius_km"], (int,float))
    assert cfg["geometry"]["min_observers"] >= 2
    assert cfg["filter"]["chi2_gate_3dof"] > 0
    print("[OK ] Type/range checks passed.")

    # 5) Existence checks (use absolute paths)
    must_exist = ["stk_exports","ephem_root","oem_root","los_root"]
    for k in must_exist:
        p = apaths[k]
        print(f"[CHK] {k:16s} -> {p} | exists={p.exists()}")

    # 6) Quick content discovery to ensure we can access data
    los_dir = apaths["los_root"]
    if los_dir.exists():
        # Find one target folder and a sample LOS file
        targets = sorted([d.name for d in los_dir.glob("HGV_*") if d.is_dir()])
        print(f"[DISC] Targets under LOS root: {targets[:5]}{'...' if len(targets)>5 else ''}")
        # Sample a target and look for LOS files
        sample_glob = str(los_dir / "HGV_*" / "LOS_OBS_*_to_HGV_*_icrf.csv")
        los_files = glob.glob(sample_glob)
        print(f"[DISC] Sample LOS files found: {len(los_files)}")
        if los_files:
            print(f"       e.g., {los_files[0]}")
    else:
        print("[WARN] LOS root does not exist; copy STK exports or adjust config.paths.los_root")

    print("[DONE] Config validation complete.")

if __name__ == "__main__":
    main()