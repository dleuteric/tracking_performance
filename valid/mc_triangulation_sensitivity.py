#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
import argparse
import re
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "pipeline.yaml"

# ---------- config ----------
def _load_paths() -> Dict[str, Path]:
    with open(CFG_PATH, "r") as f:
        data = yaml.safe_load(f)
    paths = data["paths"]
    exports_root = Path(paths.get("exports_root", "exports"))
    stk_exports = Path(str(paths.get("stk_exports", "exports/stk_exports")).format(exports_root=exports_root))

    def _fmt(s: str) -> Path:
        return ROOT / str(s).format(exports_root=exports_root, stk_exports=stk_exports)

    out = {k: _fmt(v) if isinstance(v, str) else v for k, v in paths.items()}
    out["exports_root"] = ROOT / str(exports_root)
    out["stk_exports"] = ROOT / str(stk_exports)
    return out

# ---------- helpers ----------
# --- util: estrai OBS_### da df o filename ---
_OBS_RX = re.compile(r"(OBS[_-]\d{3})", re.IGNORECASE)

def _obs_id_from(df: pd.DataFrame | None, path: Path) -> str:
    # 1) dal contenuto (STK LOS ha colonna 'Observer')
    if df is not None:
        for candidate in ["Observer", "observer", "Obs", "obs"]:
            if candidate in df.columns:
                vals = pd.Series(df[candidate].astype(str).str.upper().str.strip()).unique()
                if len(vals) == 1 and _OBS_RX.search(vals[0]):
                    return _OBS_RX.search(vals[0]).group(1).replace("-", "_").upper()
    # 2) dal filename
    m = _OBS_RX.search(path.stem)
    if m:
        return m.group(1).replace("-", "_").upper()
    # fallback: stem upper
    return path.stem.upper()

def _find_time_col(df: pd.DataFrame) -> np.ndarray:
    """
    Ritorna secondi UNIX. Supporta numerici e stringhe (incl. 'Time (UTCG)').
    """
    # 1) numerico
    for c in ["t_s", "time_s", "epoch_s", "t", "time", "epoch"]:
        if c in df.columns:
            try:
                return df[c].astype(float).to_numpy()
            except Exception:
                pass

    # 2) datetime espliciti
    for c in ["Time (UTCG)", "UTC", "UTC Time", "DateUTC", "DatetimeUTC"]:
        if c in df.columns:
            dt = pd.to_datetime(df[c], utc=True, errors="coerce")
            if not dt.isna().all():
                return (dt.astype("int64") / 1e9).to_numpy()  # niente .view

    # 3) fallback euristico
    for c in df.columns:
        n = str(c).lower()
        if any(k in n for k in ["time", "utc", "epoch", "datetime"]):
            dt = pd.to_datetime(df[c], utc=True, errors="coerce")
            if not dt.isna().all():
                return (dt.astype("int64") / 1e9).to_numpy()

    raise KeyError(f"Cannot detect time column in {list(df.columns)}")

def _find_axis_col(df: pd.DataFrame, axis: str, kind: str) -> str:
    """
    Find column for axis 'x'/'y'/'z'.
    kind='los' or 'pos' (position). It looks for flexible names:
    - LOS: ux, u_x, los_x, los_icrf_x, u_icrf_x, etc.
    - POS: x_icrf_km, x_km, x, etc.
    """
    ax = axis.lower()
    cols = list(df.columns)

    def match(name: str) -> bool:
        n = name.lower()
        if ax not in n:
            return False
        if kind == "los":
            return ("u" in n or "los" in n) and not any(s in n for s in ["vx", "vy", "vz"])
        else:
            # position
            return ("x" in n or "y" in n or "z" in n) and not any(s in n for s in ["vx", "vy", "vz", "u", "los"])

    # Strong candidates first
    pri = []
    for c in cols:
        n = c.lower()
        if kind == "los":
            if re.search(fr"(?:^|_)u{ax}\b", n): pri.append(c)
            if re.search(fr"(?:los).*{ax}", n): pri.append(c)
        else:
            if re.search(fr"(?:^|_){ax}(?:_icrf)?(?:_km)?\b", n): pri.append(c)
    if pri:
        # return the most specific (longest)
        return sorted(set(pri), key=len, reverse=True)[0]

    # Fallback: any that matches basic rule
    cands = [c for c in cols if match(c)]
    if cands:
        return sorted(cands, key=len, reverse=True)[0]

    raise KeyError(f"Cannot find {kind} column for axis '{axis}' in {cols}")

# ---------- IO ----------
def _read_oem_icrf(oem_path: Path) -> Dict[str, np.ndarray]:
    times: List[str] = []; rows: List[List[float]] = []; ref=None; meta=False
    for line in oem_path.read_text().splitlines():
        s=line.strip()
        if not s: continue
        if s=="META_START": meta=True; continue
        if s=="META_STOP":  meta=False; continue
        if meta:
            if s.startswith("REF_FRAME"): ref=s.split("=")[-1].strip()
            continue
        if s[0].isdigit() and "T" in s:
            parts=s.split()
            if len(parts)>=7:
                times.append(parts[0]); rows.append([float(p) for p in parts[1:7]])
    if ref is None or ref.upper() not in ("ICRF","J2000","GCRF"):
        raise ValueError(f"OEM not ICRF/GCRF/J2000: {oem_path}")
    arr = np.asarray(rows,float)
    t = pd.to_datetime(np.asarray(times), utc=True).astype("int64")/1e9
    x,y,z,vx,vy,vz = arr.T
    return {"t":t, "x":x, "y":y, "z":z}

def _read_ephem_obs(ephem_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Legge ephemerides osservatori e indicizza con 'OBS_###'.
    """
    files: list[Path] = []
    for pat in ("OBS*.csv", "obs*.csv", "*OBS*icrf*.csv", "*OBS*ecef*.csv"):
        files.extend(sorted(ephem_dir.glob(pat)))

    out: Dict[str, pd.DataFrame] = {}
    for p in files:
        df = pd.read_csv(p, comment="#")
        if df.empty:
            continue
        t = _find_time_col(df)
        cx = _find_axis_col(df, "x", "pos")
        cy = _find_axis_col(df, "y", "pos")
        cz = _find_axis_col(df, "z", "pos")

        obs_id = _obs_id_from(df, p)
        out[obs_id] = pd.DataFrame({
            "t": t,
            "x": df[cx].astype(float).to_numpy(),
            "y": df[cy].astype(float).to_numpy(),
            "z": df[cz].astype(float).to_numpy(),
        })

    if not out:
        raise FileNotFoundError(f"No ephemeris files in {ephem_dir}")

    print(f"[EPH] Observers: {sorted(out.keys())}")
    return out

def _read_los_target(los_tgt_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Legge LOS per un target e le indicizza con 'OBS_###'.
    """
    files: list[Path] = []
    for pat in ("LOS_OBS*.csv", "OBS*.csv", "*LOS*icrf*.csv", "*los*icrf*.csv"):
        files.extend(sorted(los_tgt_dir.glob(pat)))

    out: Dict[str, pd.DataFrame] = {}
    for p in files:
        df = pd.read_csv(p, comment="#")
        if df.empty:
            continue
        t = _find_time_col(df)
        cx = _find_axis_col(df, "x", "los")
        cy = _find_axis_col(df, "y", "los")
        cz = _find_axis_col(df, "z", "los")
        u = np.column_stack([
            df[cx].astype(float).to_numpy(),
            df[cy].astype(float).to_numpy(),
            df[cz].astype(float).to_numpy(),
        ])
        u /= np.linalg.norm(u, axis=1, keepdims=True)

        obs_id = _obs_id_from(df, p)
        out[obs_id] = pd.DataFrame({"t": t, "ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2]})

    if not out:
        raise FileNotFoundError(f"No LOS files in {los_tgt_dir}")

    print(f"[LOS] Observers: {sorted(out.keys())}")
    return out

# ---------- geometry ----------
def _interp_ephem(df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    xi = np.interp(t, df["t"], df["x"])
    yi = np.interp(t, df["t"], df["y"])
    zi = np.interp(t, df["t"], df["z"])
    return np.column_stack([xi,yi,zi])

def _interp_los(df: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    ux = np.interp(t, df["t"], df["ux"])
    uy = np.interp(t, df["t"], df["uy"])
    uz = np.interp(t, df["t"], df["uz"])
    u = np.column_stack([ux,uy,uz])
    return u/np.linalg.norm(u,axis=1,keepdims=True)

def _baseline_angle_deg(r1: np.ndarray, r2: np.ndarray, x_true: np.ndarray) -> float:
    a = x_true - r1; b = x_true - r2
    a/=np.linalg.norm(a); b/=np.linalg.norm(b)
    c = np.clip(np.dot(a,b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _closest_points_two_rays(r1,u1,r2,u2) -> Tuple[np.ndarray, np.ndarray]:
    w0 = r1 - r2
    a = np.dot(u1,u1); b = np.dot(u1,u2); c = np.dot(u2,u2)
    d = np.dot(u1,w0); e = np.dot(u2,w0)
    den = a*c - b*b
    if den < 1e-12:
        s = 0.0
        t = (-e)/c
    else:
        s = (b*e - c*d)/den
        t = (a*e - b*d)/den
    s = max(s,0.0); t = max(t,0.0)
    p1 = r1 + s*u1
    p2 = r2 + t*u2
    return p1,p2

def _noisy_direction(u: np.ndarray, sigma_rad: float, rng: np.random.Generator) -> np.ndarray:
    u = u/np.linalg.norm(u)
    ref = np.array([1.0,0.0,0.0]) if abs(u[0])<0.9 else np.array([0.0,1.0,0.0])
    v = np.cross(u, ref); v /= np.linalg.norm(v)
    w = np.cross(u, v)
    dx = rng.normal(0.0, sigma_rad); dy = rng.normal(0.0, sigma_rad)
    un = u + dx*v + dy*w
    return un/np.linalg.norm(un)

# ---------- MC core ----------
def _pick_pair(obs_names: List[str],
               t: float,
               oem: Dict[str,np.ndarray],
               eph_by_obs: Dict[str,pd.DataFrame],
               policy: str = "closest_to_90") -> Tuple[str,str,float]:
    """
    Ritorna (oi,oj, baseline_deg) secondo 'policy':
      - 'closest_to_90' (default): minimizza |beta-90°|
      - 'max_sin': massimizza sin(beta) (equivalente, ma immune a estremi 0°/180°)
      - 'max_beta': baseline massima (sconsigliato per triangolazione)
    """
    x_true = np.array([
        np.interp(t, oem["t"], oem["x"]),
        np.interp(t, oem["t"], oem["y"]),
        np.interp(t, oem["t"], oem["z"]),
    ])

    best = None
    best_score = -np.inf
    best_beta = np.nan

    for i in range(len(obs_names)):
        for j in range(i+1, len(obs_names)):
            oi, oj = obs_names[i], obs_names[j]
            r1 = _interp_ephem(eph_by_obs[oi], np.array([t]))[0]
            r2 = _interp_ephem(eph_by_obs[oj], np.array([t]))[0]
            beta = _baseline_angle_deg(r1, r2, x_true)  # [deg]

            if policy == "closest_to_90":
                score = -abs(beta - 90.0)
            elif policy == "max_sin":
                score = np.sin(np.radians(beta))
            elif policy == "max_beta":
                score = beta
            else:
                raise ValueError(f"Unknown policy: {policy}")

            if score > best_score:
                best_score = score
                best = (oi, oj)
                best_beta = beta

    return best[0], best[1], float(best_beta)

def mc_rmse_for_sigma(
    tgt: str,
    sigma_rad: float,
    K: int,
    stride: int,
    oem: Dict[str,np.ndarray],
    los_by_obs: Dict[str,pd.DataFrame],
    eph_by_obs: Dict[str,pd.DataFrame],
    rng: np.random.Generator,
    use_truth_los: bool = False,
    per_epoch_bestpair: bool = False,
) -> float:
    t_all = oem["t"]
    t_use = t_all[::stride]
    obs_names = sorted(set(los_by_obs) & set(eph_by_obs))
    if len(obs_names) < 2:
        raise RuntimeError(f"Need at least 2 observers (LOS={list(los_by_obs.keys())}, EPH={list(eph_by_obs.keys())})")

    # coppia iniziale (baseline massima) a metà tempo
    t_mid = t_use[len(t_use)//2]
    oi, oj, base_mid = _pick_pair(obs_names, t_mid, oem, eph_by_obs, policy="closest_to_90")
    print(f"[SEL] initial pair @mid: {oi},{oj} | baseline={base_mid:.2f} deg")

    errs: List[float] = []
    for t in t_use:
        # se richiesto, ricalcolo coppia per-epoca
        if per_epoch_bestpair:
            oi, oj, base_t = _pick_pair(obs_names, t, oem, eph_by_obs, policy="closest_to_90")
        else:
            _, _, base_t = _pick_pair([oi, oj], t, oem, eph_by_obs, policy="closest_to_90")

        x_true = np.array([
            np.interp(t, t_all, oem["x"]),
            np.interp(t, t_all, oem["y"]),
            np.interp(t, t_all, oem["z"]),
        ])
        r1 = _interp_ephem(eph_by_obs[oi], np.array([t]))[0]
        r2 = _interp_ephem(eph_by_obs[oj], np.array([t]))[0]

        if use_truth_los:
            # LOS sintetiche dalla verità — sanity a σ=0 deve dare ≈0
            u1 = (x_true - r1); u1 /= np.linalg.norm(u1)
            u2 = (x_true - r2); u2 /= np.linalg.norm(u2)
        else:
            u1 = _interp_los(los_by_obs[oi], np.array([t]))[0]
            u2 = _interp_los(los_by_obs[oj], np.array([t]))[0]

        acc = 0.0
        if sigma_rad == 0.0 and K > 1:
            # Evitiamo sampling inutile quando non c'è rumore
            K_eff = 1
        else:
            K_eff = K

        for _ in range(K_eff):
            uu1 = _noisy_direction(u1, sigma_rad, rng) if sigma_rad > 0 else u1
            uu2 = _noisy_direction(u2, sigma_rad, rng) if sigma_rad > 0 else u2
            p1,p2 = _closest_points_two_rays(r1,uu1,r2,uu2)
            xhat = 0.5*(p1+p2)
            acc += np.linalg.norm(xhat - x_true)
        errs.append(acc / K_eff)



    return float(np.sqrt(np.mean(np.square(errs))))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Monte-Carlo sensitivity of triangulation vs LOS noise (ICRF)")
    ap.add_argument("--target", default="HGV_00001")
    ap.add_argument("--K", type=int, default=20, help="MC samples per epoch")
    ap.add_argument("--stride", type=int, default=10, help="decimation on OEM epochs")
    ap.add_argument("--sigmas_urad", type=float, nargs="+", default=[0,50,150,300,450])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="valid/triangulation_checks/mc_sensitivity.png")
    ap.add_argument("--truth_los", action="store_true",
                    help="build LOS from truth (u = (x_true-r)/||x_true-r||); σ=0 ⇒ RMSE≈0")
    ap.add_argument("--per_epoch_bestpair", action="store_true",
                    help="reselect best observer pair (max baseline) at every epoch")
    args = ap.parse_args()

    paths = _load_paths()
    oem_path = paths["oem_root"] / f"{args.target}.oem"
    oem = _read_oem_icrf(oem_path)

    los_dir = paths["los_root"] / args.target
    eph_dir = paths["ephem_root"]
    los_by_obs = _read_los_target(los_dir)
    eph_by_obs = _read_ephem_obs(eph_dir)

    # Intersezione osservatori disponibili
    common = sorted(set(los_by_obs) & set(eph_by_obs))
    if len(common) < 2:
        raise RuntimeError(f"Need ≥2 common observers, got {common}")
    # tieni solo i comuni (evita mismatch)
    los_by_obs = {k: los_by_obs[k] for k in common}
    eph_by_obs = {k: eph_by_obs[k] for k in common}
    print(f"[INT] Common observers ({len(common)}): {common[:6]}{'...' if len(common)>6 else ''}")

    rng = np.random.default_rng(args.seed)
    xs=[]; ys=[]

    for s_urad in args.sigmas_urad:
        s_rad = float(s_urad)*1e-6
        rmse = mc_rmse_for_sigma(
            args.target, s_rad, args.K, args.stride,
            oem, los_by_obs, eph_by_obs, rng,
            use_truth_los=args.truth_los,
            per_epoch_bestpair=args.per_epoch_bestpair,
        )
        print(f"[MC ] sigma={s_urad:>4.0f} µrad → RMSE={rmse:7.3f} km")
        xs.append(s_urad); ys.append(rmse)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6,4.6))
    plt.plot(xs, ys, "o-", label=f"RMSE | K={args.K}, stride={args.stride}")
    plt.xlabel("σ_LOS [µrad]"); plt.ylabel("RMSE of |x̂−x_true| [km]")
    plt.grid(True, alpha=.3); plt.legend()
    title_extra = " (truth-LOS)" if args.truth_los else ""
    plt.title(f"Triangulation sensitivity — {args.target}{title_extra}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[OK ] Plot → {out_path}")

if __name__ == "__main__":
    main()