#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_timings.py (slim)
Allinea ephemeris target/sat a un unico clock di run e scrive SOLO le colonne richieste:

- Satelliti: epoch_utc,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s
- Target:    epoch_utc,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,ax_mps2,ay_mps2,az_mps2

Tempo (semplificato):
- Preferisce 't_sec' (secondi); Δt = t_sec - t0
- Altrimenti usa 'epoch_utc' (datetime UTC); Δt = epoch - epoch0
- epoch_utc(out) = run_start + Δt  (ISO UTC, microsecondi, 'Z')

Run start (priorità): EZSMAD_RUN_START_ISO -> EZSMAD_RUN_ID/--run-id -> now UTC
Percorsi default (relativi alla repo root):
- targets_glob = exports/target_exports/OUTPUT_CSV/HGV_*.csv
- sat_ephem_root = flightdynamics/exports/ephemeris
- out_root = exports/aligned_ephems
"""
from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datetime import datetime, timezone

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# ---------------------------
# Repo root & run start
# ---------------------------

def _find_repo_root(start: Path) -> Path:
    markers = {".git", "exports", "flightdynamics", "tools"}
    best = None
    for p in [start] + list(start.parents):
        try:
            entries = {e.name for e in p.iterdir()}
        except Exception:
            continue
        if entries & markers:
            if {"exports", "flightdynamics"}.issubset(entries):
                return p
            if best is None:
                best = p
    return best or start

def _parse_run_id_to_dt(run_id: str) -> Optional[datetime]:
    if not run_id:
        return None
    m = re.match(r"^(\d{8}T\d{6})Z", run_id) or re.match(r"^(\d{8}T\d{6})", run_id)
    if not m:
        return None
    core = m.group(1)
    try:
        return datetime.strptime(core, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _get_run_start_time(cli_run_id: Optional[str]) -> datetime:
    env_iso = os.environ.get("EZSMAD_RUN_START_ISO")
    if env_iso:
        try:
            dt = pd.to_datetime(env_iso, utc=True)
            if not pd.isna(dt):
                return dt.to_pydatetime()
        except Exception:
            print(f"[WARN] EZSMAD_RUN_START_ISO non parsabile: {env_iso}")
    env_run_id = os.environ.get("EZSMAD_RUN_ID")
    for candidate in (env_run_id, cli_run_id):
        if candidate:
            dt = _parse_run_id_to_dt(candidate)
            if dt:
                return dt
    return _now_utc()

# ---------------------------
# Scoperta ephemeris satelliti
# ---------------------------

def _latest_run_dir(ephem_root: Path) -> Optional[Path]:
    if not ephem_root.exists():
        return None
    run_re = re.compile(r"^\d{8}T\d{6}(Z.*)?")
    run_dirs = [p for p in ephem_root.iterdir() if p.is_dir() and run_re.match(p.name)]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    return run_dirs[0]

def _collect_sat_csvs(ephem_root: Path, run_id: Optional[str]) -> Tuple[List[Path], Optional[Path]]:
    run_dir = None
    if run_id:
        candidates = [p for p in ephem_root.iterdir() if p.is_dir() and p.name.startswith(run_id)]
        if candidates:
            candidates.sort(key=lambda p: p.name, reverse=True)
            run_dir = candidates[0]
    if run_dir is None:
        run_dir = _latest_run_dir(ephem_root)
    if run_dir is None or not run_dir.exists():
        root_csvs = list(ephem_root.glob("*.csv"))
        if root_csvs:
            print(f"[INFO] Nessuna cartella run valida; uso CSV in root: {ephem_root}")
            return root_csvs, None
        print(f"[WARN] Nessuna cartella run valida trovata in {ephem_root}.")
        return [], None
    csvs = list(run_dir.rglob("*.csv"))
    if not csvs:
        print(f"[WARN] Nessun CSV trovato in {run_dir}.")
    return csvs, run_dir

# ---------------------------
# Tempo & salvataggio
# ---------------------------

_SAT_COLS = ["epoch_utc","x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"]
_TGT_COLS = ["epoch_utc","x_m","y_m","z_m","vx_mps","vy_mps","vz_mps","ax_mps2","ay_mps2","az_mps2"]

def _extract_time(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    if "t_s" in df.columns:
        s = pd.to_numeric(df["t_s"], errors="coerce")
        if s.notna().any():
            return "seconds", s.astype(float)
    if "epoch_utc" in df.columns:
        s = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
        if s.notna().any():
            return "datetime", s
    raise ValueError("Colonna tempo non trovata: serve 't_s' o 'epoch_utc'.")

def _align_epoch(df: pd.DataFrame, run_start: datetime) -> pd.DataFrame:
    mode, s = _extract_time(df)
    if mode == "seconds":
        t_rel = (s - float(s.iloc[0])).astype(float)
    else:
        # forza timezone UTC sulla Series
        s = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
        t_rel = (s - s.iloc[0]).dt.total_seconds().astype(float)
    rs = pd.Timestamp(run_start)
    rs = rs.tz_localize("UTC") if rs.tz is None else rs.tz_convert("UTC")
    aligned = rs + pd.to_timedelta(t_rel, unit="s")
    out = df.copy()
    out["epoch_utc"] = pd.Series(aligned).dt.strftime(ISO_FMT)
    return out

def _ensure_dirs(out_root: Path) -> Tuple[Path, Path]:
    sat_dir = out_root / "satellite_epehms"  # naming come richiesto
    tgt_dir = out_root / "targets_epehms"
    sat_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    return sat_dir, tgt_dir

def _save_satellite(df: pd.DataFrame, src: Path, dst_dir: Path) -> Path:
    missing = [c for c in _SAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"mancano colonne sat richieste: {missing}")
    out = df[_SAT_COLS]
    out_path = dst_dir / src.name
    out.to_csv(out_path, index=False)
    return out_path

def _save_target(df: pd.DataFrame, src: Path, dst_dir: Path) -> Path:
    missing = [c for c in _TGT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"mancano colonne target richieste: {missing}")
    out = df[_TGT_COLS]
    out_path = dst_dir / src.name
    out.to_csv(out_path, index=False)
    return out_path

# ---------------------------
# MAIN
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Allinea ephemeris target/sat ad un unico clock di run.")
    parser.add_argument("--targets-glob", default=None,
                        help="Glob CSV target. Default: <repo>/exports/target_exports/OUTPUT_CSV/HGV_*.csv")
    parser.add_argument("--sat-ephem-root", default=None,
                        help="Root ephemeris satelliti. Default: <repo>/flightdynamics/exports/ephemeris")
    parser.add_argument("--run-id", default=None,
                        help="Run ID per selezionare la cartella ephemeris; se assente, usa la più recente.")
    parser.add_argument("--out-root", default=None,
                        help="Output root. Default: <repo>/exports/aligned_ephems")
    args = parser.parse_args(argv)

    # repo root da ENV o autodetect
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("EZSMAD_PROJECT_ROOT")
    repo_root = Path(env_root).expanduser().resolve() if env_root else _find_repo_root(Path(__file__).resolve())

    # defaults
    default_targets_glob = repo_root / "exports/target_exports/OUTPUT_CSV/HGV_*.csv"
    default_sat_root     = repo_root / "flightdynamics" / "exports" / "ephemeris"
    default_out_root     = repo_root / "exports" / "aligned_ephems"

    # normalizza input
    if args.targets_glob:
        tglob_path = Path(args.targets_glob)
        targets_glob = str((repo_root / tglob_path).resolve()) if not tglob_path.is_absolute() else str(tglob_path)
    else:
        targets_glob = str(default_targets_glob)

    if args.sat_ephem_root:
        sat_root = Path(args.sat_ephem_root)
        sat_ephem_root = (repo_root / sat_root).resolve() if not sat_root.is_absolute() else sat_root.resolve()
    else:
        sat_ephem_root = default_sat_root.resolve()

    out_root = (repo_root / Path(args.out_root)).resolve() if args.out_root else default_out_root.resolve()

    # run start
    run_start = _get_run_start_time(args.run_id)
    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] targets_glob: {targets_glob}")
    print(f"[INFO] sat_ephem_root: {sat_ephem_root}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] run_start (UTC): {run_start.isoformat()}")

    sat_out_dir, tgt_out_dir = _ensure_dirs(out_root)

    # targets
    tgt_files = sorted(glob.glob(targets_glob))
    if not tgt_files:
        print(f"[WARN] Nessun target trovato con glob: {targets_glob}")
    else:
        print(f"[INFO] Target CSV trovati: {len(tgt_files)}")
    for f in tgt_files:
        src = Path(f)
        try:
            df = pd.read_csv(src)
            df2 = _align_epoch(df, run_start)
            outp = _save_target(df2, src, tgt_out_dir)
            print(f"[OK] Salvato target allineato: {outp}")
        except Exception as e:
            print(f"[ERR] Target '{src}': {e}")

    # satelliti
    sat_csvs, run_dir = _collect_sat_csvs(Path(sat_ephem_root), args.run_id)
    if not sat_csvs:
        print(f"[WARN] Nessun ephemeris satellite trovato in {sat_ephem_root} (run_id={args.run_id}).")
    else:
        if run_dir:
            print(f"[INFO] Run selezionata: {run_dir}")
        print(f"[INFO] Ephemeris satelliti trovati: {len(sat_csvs)} in {sat_ephem_root}")
    for src in sorted(sat_csvs):
        try:
            df = pd.read_csv(src)
            df2 = _align_epoch(df, run_start)
            outp = _save_satellite(df2, Path(src), sat_out_dir)
            print(f"[OK] Salvato satellite allineato: {outp}")
        except Exception as e:
            print(f"[ERR] Satellite '{src}': {e}")

    print("[DONE] Allineamento completato.")
    return 0

if __name__ == "__main__":
    sys.exit(main())