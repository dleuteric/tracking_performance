# valid/check_dt.py
from pathlib import Path
import pandas as pd
import numpy as np
import re
import argparse

ROOT = Path(__file__).resolve().parents[1]
LOS_ROOT = ROOT / "exports/stk_exports/OUTPUT_LOS_VECTORS"
EPH_ROOT = ROOT / "exports/stk_exports/OUTPUT_EPHEM"

OBS_RE = re.compile(r"(OBS[_-]\d+)", re.IGNORECASE)

def _read_times_csv(csv_path: Path, time_col: str = "Time (UTCG)") -> np.ndarray:
    df = pd.read_csv(csv_path, comment="#")
    if time_col not in df.columns:
        raise KeyError(f"{csv_path.name}: missing column '{time_col}'")
    return (pd.to_datetime(df[time_col], utc=True).astype("int64") / 1e9).to_numpy()

def _los_files_for_target(target: str) -> dict[str, Path]:
    d = LOS_ROOT / target
    if not d.exists():
        return {}
    out = {}
    for f in sorted(d.glob("LOS_*_icrf.csv")):
        m = OBS_RE.search(f.name)
        if m:
            out[m.group(1).upper().replace("-", "_")] = f
    return out

def _eph_file_for_obs(obs: str) -> Path | None:
    # Try strict ICRF first
    for pat in (f"{obs}_icrf.csv", f"{obs}.csv", f"*{obs}*icrf*.csv", f"*{obs}*.csv"):
        cand = next(EPH_ROOT.glob(pat), None)
        if cand:
            return cand
    return None

def _nearest_dt(t_ref: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_grid, t_ref)
    idx = np.clip(idx, 1, len(t_grid) - 1)
    prev = t_grid[idx - 1]
    nxt = t_grid[idx]
    nearest = np.where(np.abs(t_ref - prev) <= np.abs(t_ref - nxt), prev, nxt)
    return t_ref - nearest

def main():
    ap = argparse.ArgumentParser(description="Check Δt between LOS and ephemerides per observer.")
    ap.add_argument("--target", default="HGV_00001")
    args = ap.parse_args()

    los_map = _los_files_for_target(args.target)
    if not los_map:
        print(f"[ERR ] No LOS files found in {LOS_ROOT/args.target}")
        listed = list((LOS_ROOT/args.target).glob("*"))
        if listed:
            print("       Present:", ", ".join(p.name for p in listed))
        return

    print(f"[INFO] Target={args.target}")
    print(f"[INFO] Found LOS from observers: {', '.join(sorted(los_map.keys()))}")

    any_ok = False
    for obs, los_path in sorted(los_map.items()):
        eph_path = _eph_file_for_obs(obs)
        if eph_path is None:
            print(f"[WARN] No EPHEM found for {obs} in {EPH_ROOT}")
            continue

        try:
            t_los = _read_times_csv(los_path)
            t_eph = _read_times_csv(eph_path)
        except Exception as e:
            print(f"[WARN] Read failed for {obs}: {e}")
            continue

        if t_los.size == 0 or t_eph.size == 0:
            print(f"[WARN] Empty time series for {obs}")
            continue

        dt = _nearest_dt(t_los, t_eph)
        any_ok = True
        print(f"[OK  ] {obs}: Δt stats [s]  min={dt.min():.3f}  med={np.median(dt):.3f}  max={dt.max():.3f}")
        for k in (0, len(dt)//2, len(dt)-1):
            print(f"       sample k={k:4d}: t_los={t_los[k]:.3f}  t_eph*={t_los[k]-dt[k]:.3f}  Δt={dt[k]:.3f}")

    if not any_ok:
        print("[ERR ] No observer with both LOS and EPHEM could be checked.")
        # Quick inventory to help debugging
        print(f"       EPH_ROOT contents ({EPH_ROOT}):")
        for p in sorted(EPH_ROOT.glob("*.csv"))[:20]:
            print("        -", p.name)

if __name__ == "__main__":
    main()