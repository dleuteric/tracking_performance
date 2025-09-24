from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# --- Project paths ---
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config/pipeline.yaml"
TRI_ROOT = ROOT / "exports/triangulation"
TRACKS_OUT = ROOT / "exports/tracks_icrf"
PLOTS_OUT = ROOT / "plots_icrf"


# =========================
# Safe dynamic loader (avoids circular import)
# =========================


def load_run_ewrls() -> Callable:
    import sys, importlib.util
    mod_path = ROOT / "estimationandfiltering" / "ew_rls.py"
    spec = importlib.util.spec_from_file_location("ezsmad.estimationandfiltering.ew_rls_dyn", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot locate ew_rls module at {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register both under a unique name and the canonical short name to satisfy intra-module references
    sys.modules.setdefault("ezsmad.estimationandfiltering.ew_rls_dyn", mod)
    sys.modules.setdefault("ew_rls", mod)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    if not hasattr(mod, "run_ewrls_on_csv"):
        raise ImportError("ew_rls.run_ewrls_on_csv not found")
    return getattr(mod, "run_ewrls_on_csv")

# =========================
# Config loader (paths)
# =========================

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
    if not data or "paths" not in data:
        raise KeyError(f"'paths' section missing in {cfg_path}")
    return _expand_paths(data["paths"])  # dict[str, Path]

# =========================
# OEM ICRF reader
# =========================

def read_oem_icrf(oem_path: Path) -> Dict[str, Any]:
    """Parse CCSDS OEM v2 with REF_FRAME=ICRF, TIME_SYSTEM=UTC.
    Returns t_epoch[s], t_iso[str], x/y/z [km], vx/vy/vz [km/s].
    """
    times: List[str] = []
    rows: List[List[float]] = []
    ref = None
    with oem_path.open("r") as f:
        meta = False
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "META_START":
                meta = True; continue
            if s == "META_STOP":
                meta = False; continue
            if meta:
                if s.startswith("REF_FRAME"):
                    ref = s.split("=")[-1].strip()
                continue
            if s[0].isdigit() and "T" in s:  # data row
                parts = s.split()
                if len(parts) < 7:
                    continue
                times.append(parts[0])
                vals = [float(p) for p in parts[1:7]]
                rows.append(vals)
    if ref is None or ref.upper() not in ("ICRF","J2000","GCRF"):
        raise ValueError(f"OEM not in ICRF/GCRF/J2000: {oem_path}")
    if not rows:
        raise ValueError(f"OEM has no data: {oem_path}")
    arr = np.asarray(rows, float)
    t_iso = np.asarray(times)
    t = pd.to_datetime(t_iso, utc=True).astype("int64")/1e9
    x,y,z,vx,vy,vz = arr.T
    return {"t": t, "t_iso": t_iso, "x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}


def gmst_rad_from_unix(t_s: np.ndarray) -> np.ndarray:
    """GMST (radians) from UNIX epoch seconds, UTC. Sufficient for km-level alignment."""
    JD = t_s / 86400.0 + 2440587.5
    T = (JD - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3)
    return (gmst_sec % 86400.0) * (2*np.pi / 86400.0)

def _rot_z(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((theta.size, 3, 3))
    R[:,0,0]=c; R[:,0,1]=-s; R[:,1,0]=s; R[:,1,1]=c; R[:,2,2]=1.0
    return R

def ecef_to_icrf_xyz(x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Rotate position series ECEF->ICRF with GMST: r_eci ≈ R3(GMST)*r_ecef."""
    R = _rot_z(gmst_rad_from_unix(t_s))
    r = np.column_stack([x_km, y_km, z_km]).astype(float)
    return np.einsum('nij,nj->ni', R, r)




