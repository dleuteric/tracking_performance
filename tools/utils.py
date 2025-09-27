from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json

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


# =========================
# Ephemerides loader (space_segment outputs)
# =========================
def _ephem_root_candidates() -> List[Path]:
    """Return likely ephemeris root candidates, most specific first."""
    return [
        ROOT / "exports" / "ephemeris",
        ROOT / "flightdynamics" / "exports" / "ephemeris",
    ]

def find_ephemeris_root() -> Path:
    """Locate the ephemeris root containing at least one valid run (with manifest)."""
    for root in _ephem_root_candidates():
        if root.exists() and root.is_dir():
            try:
                for d in root.iterdir():
                    if d.is_dir():
                        man = d / f"manifest_{d.name}.json"
                        if man.exists():
                            return root
            except Exception:
                continue
    raise FileNotFoundError("Ephemeris root not found. Expected exports/ephemeris/<RUN_ID>/manifest_<RUN_ID>.json")

def list_ephemeris_runs(ephem_root: Optional[Path] = None) -> List[Path]:
    """List available run directories (sorted)."""
    base = ephem_root or find_ephemeris_root()
    runs = [p for p in base.iterdir() if p.is_dir() and (p / f"manifest_{p.name}.json").exists()]
    return sorted(runs)

def latest_ephemeris_run(ephem_root: Optional[Path] = None) -> Path:
    runs = list_ephemeris_runs(ephem_root)
    if not runs:
        raise FileNotFoundError("No ephemeris runs found.")
    return runs[-1]

def load_space_segment_manifest(run_dir: Path) -> Dict[str, Any]:
    """Read the manifest_<RUN_ID>.json inside a run directory."""
    run_dir = Path(run_dir)
    man = run_dir / f"manifest_{run_dir.name}.json"
    if not man.exists():
        raise FileNotFoundError(f"Manifest not found: {man}")
    with man.open("r") as f:
        return json.load(f)

def load_space_segment_ephemerides(
    run_dir: Optional[Path] = None,
    *,
    sat_ids: Optional[List[str]] = None,
    regime: Optional[str] = None,
    frame: str = "ECEF",
    parse_time: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load satellite ephemerides exported by flightdynamics/space_segment.py.
    Returns a dict keyed by satellite key (e.g., 'P1_S3'), values are DataFrames with columns:
      ['t_sec','epoch_utc','x_km','y_km','z_km','vx_km_s','vy_km_s','vz_km_s','regime']
    If frame=='ICRF', additional columns ['x_icrf_km','y_icrf_km','z_icrf_km'] are added (positions only).
    Filters:
      - sat_ids: list of keys to include (e.g., ['P1_S1','P1_S2']).
      - regime: 'LEO'|'MEO'|'GEO'|'HEO' (case-insensitive), matched from filename prefix.
    """
    # Resolve run directory
    if run_dir is None:
        run_dir = latest_ephemeris_run()
    run_dir = Path(run_dir)
    manifest = load_space_segment_manifest(run_dir)
    files_map: Dict[str, str] = manifest.get("files", {})

    out: Dict[str, pd.DataFrame] = {}
    regime_u = regime.upper() if regime else None

    for key, rel in files_map.items():
        # Filter by sat id
        if sat_ids is not None and key not in sat_ids:
            continue
        # Filter by regime
        reg = Path(rel).name.split("_", 1)[0].upper() if "_" in Path(rel).name else "UNK"
        if regime_u and reg != regime_u:
            continue

        csv_path = run_dir / rel
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        # Ensure expected columns exist
        req = {"t_sec","epoch_utc","x_km","y_km","z_km","vx_km_s","vy_km_s","vz_km_s"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

        # Parse times
        if parse_time:
            try:
                dt = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
                df["epoch_dt"] = dt
                df["t_unix_s"] = dt.view("int64") / 1e9
            except Exception:
                pass

        # Attach regime
        df["regime"] = reg

        # Optional ICRF position columns
        if frame.upper() == "ICRF":
            if "t_unix_s" not in df.columns:
                dt = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
                df["t_unix_s"] = dt.view("int64") / 1e9
            xeci, yeci, zeci = ecef_to_icrf_xyz(df["x_km"].to_numpy(),
                                                 df["y_km"].to_numpy(),
                                                 df["z_km"].to_numpy(),
                                                 df["t_unix_s"].to_numpy()).T
            df["x_icrf_km"] = xeci
            df["y_icrf_km"] = yeci
            df["z_icrf_km"] = zeci

        out[key] = df

    return out
