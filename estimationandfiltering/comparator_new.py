#!/usr/bin/env python3
"""
Comparator (new):
- Reads KF tracks from exports/tracks/{RUN_ID}/*_track_icrf_forward.csv (one per target)
- Reads EW-RLS tracks from exports/tracks_icrf/{RUN_ID}/ewrls_icrf_<TARGET>.csv
- Reads truth OEM from paths.oem_root/<TARGET>.oem (ICRF)
- Aligns to estimator timestamps (no nearest-join; time interpolation of truth)
- Computes 3D errors and summary metrics for each target and method
- Writes metrics.csv + simple plots into --out_dir

Usage:
  python -m estimationandfiltering.comparator_new --run_id 2025... --out_dir exports/studies/<study>/compare_new
  (Optional) --targets HGV_00001 HGV_00002 ...

This script does NOT run any filters; it only compares existing outputs.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import re
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Config & paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "pipeline.yaml"


def _load_cfg() -> Dict:
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)


def _paths_from_cfg(cfg: Dict) -> Dict[str, Path]:
    paths = cfg.get("paths", {})
    exports_root = Path(paths.get("exports_root", "exports"))
    # Allow nested placeholders used across repo
    def fmt(p: str) -> Path:
        return Path(str(p).format(exports_root=exports_root))

    out = {
        "exports_root": ROOT / str(exports_root),
        "tracks_root": ROOT / str(fmt(paths.get("tracks_root", "{exports_root}/tracks"))),
        "tracks_icrf_root": ROOT / str(fmt(paths.get("tracks_icrf_root", "{exports_root}/tracks_icrf"))),
        "triang_root": ROOT / str(fmt(paths.get("triangulation_out", "{exports_root}/triangulation"))),
        "oem_root": ROOT / str(fmt(paths.get("oem_root", "exports/target_exports/OUTPUT_OEM"))),
    }
    return out


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
_TIME_CANDS_STR = [
    "time_iso", "time", "Time (UTCG)", "UTC", "DatetimeUTC",
]
_TIME_CANDS_NUM = [
    "t_s", "epoch_s", "time_s", "t",
]


def _to_epoch_s(df: pd.DataFrame) -> np.ndarray:
    # numeric first
    for c in _TIME_CANDS_NUM:
        if c in df.columns:
            try:
                return df[c].astype(float).to_numpy()
            except Exception:
                pass
    # string/datetime
    for c in _TIME_CANDS_STR:
        if c in df.columns:
            dt = pd.to_datetime(df[c], utc=True, errors="coerce")
            if not dt.isna().all():
                return (dt.astype("int64") / 1e9).to_numpy()
    # last chance: heuristic
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["time", "utc", "epoch", "datetime"]):
            dt = pd.to_datetime(df[c], utc=True, errors="coerce")
            if not dt.isna().all():
                return (dt.astype("int64") / 1e9).to_numpy()
    raise KeyError("No time column recognized in estimator CSV")


_OBS_RX = re.compile(r"(HGV[_-]\d{5})", re.IGNORECASE)


def _target_from_name(path: Path) -> Optional[str]:
    m = _OBS_RX.search(path.name)
    return m.group(1).replace("-", "_").upper() if m else None


# Truth OEM reader (ICRF/J2000/GCRF accepted)

def _read_oem_icrf(oem_path: Path) -> Dict[str, np.ndarray]:
    if not oem_path.exists():
        raise FileNotFoundError(f"OEM not found: {oem_path}")
    times: List[str] = []
    rows: List[List[float]] = []
    ref = None
    meta = False
    for raw in oem_path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s == "META_START":
            meta = True
            continue
        if s == "META_STOP":
            meta = False
            continue
        if meta:
            if s.startswith("REF_FRAME"):
                ref = s.split("=")[-1].strip()
            continue
        # state line
        if (s[0].isdigit() and "T" in s) or (":" in s and " " in s and s[0].isdigit()):
            parts = s.split()
            if len(parts) >= 7:
                times.append(parts[0])
                try:
                    rows.append([float(x) for x in parts[1:7]])
                except Exception:
                    pass
    if not rows:
        raise RuntimeError(f"No state lines in OEM {oem_path}")
    if ref and ref.upper() not in ("ICRF", "J2000", "GCRF"):
        raise ValueError(f"OEM frame not inertial/ICRF-like: {ref}")
    arr = np.asarray(rows, float)
    t = pd.to_datetime(np.asarray(times), utc=True).astype("int64") / 1e9
    x, y, z, vx, vy, vz = arr.T
    return {"t": t, "x": x, "y": y, "z": z}


# --------------------------------------------------------------------------------------
# Estimator readers
# --------------------------------------------------------------------------------------

def _read_kf_tracks(tracks_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    files = sorted(tracks_dir.glob("*_track_icrf_forward.csv"))
    for p in files:
        df = pd.read_csv(p, comment="#")
        if df.empty:
            continue
        t = _to_epoch_s(df)
        # prefer explicit ICRF columns if present
        cols_xyz = [
            ("x_icrf_km", "y_icrf_km", "z_icrf_km"),
            ("x_km", "y_km", "z_km"),
        ]
        for cx, cy, cz in cols_xyz:
            if cx in df.columns and cy in df.columns and cz in df.columns:
                xyz = np.column_stack([
                    df[cx].astype(float).to_numpy(),
                    df[cy].astype(float).to_numpy(),
                    df[cz].astype(float).to_numpy(),
                ])
                break
        else:
            raise KeyError(f"Missing position columns in {p}")
        tgt = _target_from_name(p) or "UNKNOWN"
        out[tgt] = pd.DataFrame({"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    return out


def _read_ewrls_tracks(tracks_icrf_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    files = sorted(tracks_icrf_dir.glob("ewrls_icrf_*.csv"))
    for p in files:
        df = pd.read_csv(p, comment="#")
        if df.empty:
            continue
        t = _to_epoch_s(df)
        # strict icrf column names expected from ewrls_icrf_tracks.py
        cols = ("x_km", "y_km", "z_km")
        if not all(c in df.columns for c in cols):
            raise KeyError(f"EW-RLS file missing columns {cols}: {p}")
        xyz = np.column_stack([
            df[cols[0]].astype(float).to_numpy(),
            df[cols[1]].astype(float).to_numpy(),
            df[cols[2]].astype(float).to_numpy(),
        ])
        # target id from filename tail
        m = re.search(r"ewrls_icrf_(HGV[_-]\d{5})", p.name, re.IGNORECASE)
        tgt = m.group(1).replace("-", "_").upper() if m else (_target_from_name(p) or "UNKNOWN")
        out[tgt] = pd.DataFrame({"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _interp_truth(oem: Dict[str, np.ndarray], t: np.ndarray) -> np.ndarray:
    xi = np.interp(t, oem["t"], oem["x"])
    yi = np.interp(t, oem["t"], oem["y"])
    zi = np.interp(t, oem["t"], oem["z"])
    return np.column_stack([xi, yi, zi])


def _errors_vs_truth(est: pd.DataFrame, oem: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    t = est["t"].to_numpy()
    Xhat = est[["x", "y", "z"]].to_numpy(float)
    Xtru = _interp_truth(oem, t)
    err = Xhat - Xtru
    en = np.linalg.norm(err, axis=1)
    return err, en


def _metrics(en_km: np.ndarray) -> Dict[str, float]:
    m = {
        "N": int(en_km.size),
        "RMSE3D_m": float(np.sqrt(np.mean(en_km ** 2)) * 1000.0),
        "P50_m": float(np.percentile(en_km, 50) * 1000.0),
        "P95_m": float(np.percentile(en_km, 95) * 1000.0),
        "MAX_m": float(np.max(en_km) * 1000.0),
    }
    return m


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def _set_equal_3d(ax):
    # Set equal aspect for 3D plots (approx.)
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    r = 0.5 * max(sz)
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)


def plot_overlay_3d(tgt: str, kf: Optional[pd.DataFrame], ew: Optional[pd.DataFrame], oem: Dict[str, np.ndarray], out: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d)
    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_subplot(111, projection='3d')
    # truth (thin)
    Xt = np.column_stack([oem["x"], oem["y"], oem["z"]])
    ax.plot(Xt[:, 0], Xt[:, 1], Xt[:, 2], lw=1, alpha=.5, label="TRUTH OEM")
    if kf is not None and not kf.empty:
        ax.plot(kf["x"], kf["y"], kf["z"], lw=1.4, label="KF")
    if ew is not None and not ew.empty:
        ax.plot(ew["x"], ew["y"], ew["z"], lw=1.4, ls='--', label="EW-RLS")
    ax.set_title(f"ICRF tracks — {tgt}")
    ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
    _set_equal_3d(ax)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_errors_ts(tgt: str, en_kf: Optional[np.ndarray], en_ew: Optional[np.ndarray], t: np.ndarray, out: Path) -> None:
    # Use relative time to avoid huge epoch offsets on the x-axis
    t = np.asarray(t, dtype=float)
    if t.size > 0:
        t_rel = t - float(t[0])
    else:
        t_rel = t

    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(111)
    if en_kf is not None:
        ax.plot(t_rel, en_kf * 1000.0, label="|err| KF [m]")
    if en_ew is not None:
        ax.plot(t_rel, en_ew * 1000.0, label="|err| EW-RLS [m]")
    ax.set_xlabel("t − t0 [s]")
    ax.set_ylabel("position error [m]")
    ax.grid(True, alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare KF vs EW-RLS tracks in ICRF against truth OEM")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--targets", nargs="*", default=None, help="optional subset of target ids (e.g., HGV_00001)")
    args = ap.parse_args()

    cfg = _load_cfg()
    P = _paths_from_cfg(cfg)

    tracks_dir = P["tracks_root"] / args.run_id
    tracks_icrf_dir = P["tracks_icrf_root"] / args.run_id
    out_dir = ROOT / args.out_dir if not args.out_dir.startswith(str(ROOT)) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tracks_dir.exists():
        raise FileNotFoundError(f"KF tracks dir missing: {tracks_dir}")
    if not tracks_icrf_dir.exists():
        print(f"[WARN] EW-RLS tracks dir missing: {tracks_icrf_dir}")

    kf_by_tgt = _read_kf_tracks(tracks_dir)
    ew_by_tgt = _read_ewrls_tracks(tracks_icrf_dir) if tracks_icrf_dir.exists() else {}

    tgt_all = sorted(set(kf_by_tgt) | set(ew_by_tgt))
    if args.targets:
        allow = set([t.upper() for t in args.targets])
        tgt_all = [t for t in tgt_all if t in allow]
    if not tgt_all:
        raise RuntimeError("No targets to compare.")

    rows: List[Dict[str, object]] = []

    for tgt in tgt_all:
        print(f"[TGT ] {tgt}")
        oem_path = P["oem_root"] / f"{tgt}.oem"
        oem = _read_oem_icrf(oem_path)

        kf = kf_by_tgt.get(tgt)
        ew = ew_by_tgt.get(tgt)

        # choose a common time base for plotting errors (union where both exist)
        en_kf = en_ew = None
        t_plot = None

        if kf is not None:
            err_kf, en_kf = _errors_vs_truth(kf, oem)
            m_kf = _metrics(en_kf)
            rows.append({"target": tgt, "method": "KF", **m_kf})
        if ew is not None:
            err_ew, en_ew = _errors_vs_truth(ew, oem)
            m_ew = _metrics(en_ew)
            rows.append({"target": tgt, "method": "EW", **m_ew})

        # plots
        plot_overlay_3d(tgt, kf, ew, oem, out_dir / f"overlay3d_{tgt}.png")
        if kf is not None:
            t_plot = kf["t"].to_numpy()
        elif ew is not None:
            t_plot = ew["t"].to_numpy()
        else:
            continue
        plot_errors_ts(tgt, en_kf, en_ew, t_plot, out_dir / f"errors_ts_{tgt}.png")

    # metrics table
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "metrics.csv", index=False)
        print(f"[OK ] metrics.csv → {out_dir / 'metrics.csv'}")
        # quick pivot summary
        try:
            piv = df.pivot_table(index="target", columns="method", values="RMSE3D_m")
            print(piv)
        except Exception:
            pass
    else:
        print("[WARN] No rows written (no tracks?)")


if __name__ == "__main__":
    main()
