#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_timings.py
----------------
Allinea gli ephemeris (target + satelliti) su un unico clock "di run" e salva
CSV aggiornati in exports/aligned_ephems/.

Logica:
- Individua il "run clock start":
    1) variabile d'ambiente EZSMAD_RUN_START_ISO (ISO-8601, es. 2025-09-27T18:00:00Z), oppure
    2) variabile d'ambiente EZSMAD_RUN_ID (es. 20250927T154522Z -> convertita a ISO), oppure
    3) ora corrente UTC al lancio dello script.
- Carica i CSV degli ephemeris:
    * target: glob tipo exports/target_exports/OUTPUT_CSV/HGV_*.csv
    * satelliti: dentro flightdynamics/exports/ephemeris/<run_id>/*.csv
      (se --run-id non è passato, prende l'ultima cartella per nome/ordine lessicografico)
- Rileva in modo robusto la colonna temporale:
    * preferenza a colonne datetime (epoch, epoch_iso, time, utc, datetime, UTC, Epoch)
    * in alternativa colonne numeriche (t_s, time_s, t, seconds, Time[s], Time (s))
    * come fallback, Unix seconds se plausibile
- Calcola t_rel_s = tempo relativo (inizio dataset = 0 s)
- Genera epoch_aligned = run_start + t_rel_s (ISO UTC, microsecondi)
- Salva nei path:
    exports/aligned_ephems/satellite_epehms/<base_name.csv>
    exports/aligned_ephems/targets_epehms/<base_name.csv>

Nota: se nei nomi file compaiono LEO/MEO/GEO li preserva; altrimenti mantiene il basename.
"""

from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


def _find_repo_root(start: Path) -> Path:
    """
    Risale le cartelle finché trova una radice plausibile del repo.
    Criteri: presenza di 'exports' e/o 'flightdynamics' e/o '.git'.
    """
    cur = start
    markers = {".git", "exports", "flightdynamics", "tools"}
    for p in [cur] + list(cur.parents):
        try:
            entries = {e.name for e in p.iterdir()}
        except Exception:
            continue
        if entries & markers:
            # Heuristica: preferisci cartella che contiene sia 'exports' che 'flightdynamics'
            if {"exports", "flightdynamics"}.issubset(entries):
                return p
            best = getattr(_find_repo_root, "_best", None)
            if best is None:
                _find_repo_root._best = p  # type: ignore[attr-defined]
    # fallback: primo candidato salvato o la directory corrente
    return getattr(_find_repo_root, "_best", start)


# ---------------------------
# Utility per gestione orari
# ---------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_run_id_to_dt(run_id: str) -> Optional[datetime]:
    """
    Converte un run_id tipo 'YYYYMMDDTHHMMSSZ...' in datetime UTC.
    Accetta anche stringhe con suffissi extra dopo 'Z'.
    """
    if not run_id:
        return None
    m = re.match(r"^(\d{8}T\d{6})Z", run_id)
    if not m:
        # Prova anche senza 'Z'
        m = re.match(r"^(\d{8}T\d{6})", run_id)
    if not m:
        return None
    core = m.group(1)  # es. 20250927T154522
    try:
        dt = datetime.strptime(core, "%Y%m%dT%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _get_run_start_time(cli_run_id: Optional[str]) -> datetime:
    """
    Determina l'istante di start del clock di run:
    - EZSMAD_RUN_START_ISO ha priorità
    - EZSMAD_RUN_ID (o cli_run_id)
    - now UTC
    """
    env_iso = os.environ.get("EZSMAD_RUN_START_ISO")
    if env_iso:
        try:
            # Supporta anche ISO senza 'Z' esplicita
            dt = pd.to_datetime(env_iso, utc=True)
            if pd.isna(dt):
                raise ValueError
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
# Scoperta file satelliti
# ---------------------------

def _latest_run_dir(ephem_root: Path) -> Optional[Path]:
    """
    Restituisce la sottocartella 'più recente' per nome lessicografico
    con nome che sembra un run-id (es. 20250927T162540Z_*).
    Ignora cartelle generiche tipo 'plots'.
    """
    if not ephem_root.exists():
        return None
    dirs = [p for p in ephem_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    run_re = re.compile(r"^\d{8}T\d{6}(Z.*)?")
    run_dirs = [p for p in dirs if run_re.match(p.name)]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    return run_dirs[0]


def _collect_sat_csvs(ephem_root: Path, run_id: Optional[str]) -> List[Path]:
    run_dir: Optional[Path] = None
    if run_id:
        # match esatto o prefisso, privilegiando quello più "grande" lessicograficamente
        candidates = [p for p in ephem_root.iterdir() if p.is_dir() and p.name.startswith(run_id)]
        if candidates:
            candidates.sort(key=lambda p: p.name, reverse=True)
            run_dir = candidates[0]
    if run_dir is None:
        run_dir = _latest_run_dir(ephem_root)
    if run_dir is None or not run_dir.exists():
        # Fallback: CSV direttamente sotto ephem_root
        root_csvs = list(ephem_root.glob("*.csv"))
        if root_csvs:
            print(f"[INFO] Nessuna cartella run valida; uso CSV in root: {ephem_root}")
            return root_csvs
        print(f"[WARN] Nessuna cartella run valida trovata in {ephem_root}.")
        return []
    print(f"[INFO] Run selezionata: {run_dir}")
    csvs = list(run_dir.rglob("*.csv"))
    if not csvs:
        print(f"[WARN] Nessun CSV trovato in {run_dir}.")
    return csvs


# ---------------------------
# Riconoscimento colonna tempo
# ---------------------------

_DT_CANDS = [
    "epoch", "epoch_iso", "epoch_utc", "epochutc", "epoch_utcg",
    "time", "utc", "datetime",
    "utc", "utc_time",
    "Epoch", "DateTime", "EpochUTC"
]
_NUM_CANDS = [
    "t_s", "time_s", "t", "seconds", "time[s]", "time (s)", "elapsed_s",
    "t_sec", "time_sec"
]


def _detect_time_series(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Ritorna ("datetime", series_dt) oppure ("seconds", series_sec) oppure ("unix", series_sec).
    Lancia ValueError se non trova niente di valido.
    Strategia:
      1) match case-insensitive su candidati datetime/numerici più comuni
      2) fallback robusto: prova a fare parse datetime su tutte le colonne object/string
      3) fallback unix-epoch plausibile su colonne numeriche
    """
    # mappa case-insensitive: lower() -> nome originale
    cols_map = {str(c).lower(): c for c in df.columns}

    # 1) datetime-like candidati
    for c_low in _DT_CANDS:
        key = c_low.lower()
        if key in cols_map:
            col = cols_map[key]
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            if s.notna().any():
                return "datetime", s

    # 2) numeric seconds candidati
    for c_low in _NUM_CANDS:
        key = c_low.lower()
        if key in cols_map:
            col = cols_map[key]
            s = df[col]
            # consenti numerici veri o stringhe numeriche
            if np.issubdtype(s.dtype, np.number):
                return "seconds", s.astype(float)
            try:
                s2 = pd.to_numeric(s, errors="coerce")
                if s2.notna().any():
                    return "seconds", s2.astype(float)
            except Exception:
                pass

    # 3) fallback datetime: prova tutte le colonne object/string
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col].dtype):
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            # considera valido se >50% parsabile e almeno 3 valori non NaN
            if s.notna().sum() >= max(3, int(0.5 * len(s))):
                return "datetime", s

    # 4) fallback: unix seconds plausibili su qualsiasi colonna numerica
    for col in df.columns:
        s = df[col]
        if np.issubdtype(s.dtype, np.number):
            s = s.astype(float)
            # intervallo plausibile: 2001-01-01 .. 2099-12-31
            if s.min() > 978307200 and s.max() < 4102444800:
                return "unix", s

    raise ValueError("Colonna tempo non trovata (datetime/seconds/unix).")


def _add_aligned_time_columns(df: pd.DataFrame, run_start: datetime) -> pd.DataFrame:
    mode, s = _detect_time_series(df)

    if mode == "datetime":
        t_rel = (s - s.iloc[0]).dt.total_seconds().astype(float)
    else:
        # "seconds" o "unix": normalizza a 0
        t_rel = (s - float(s.iloc[0])).astype(float)

    epoch_aligned = (pd.to_datetime(run_start) + pd.to_timedelta(t_rel, unit="s")) \
        .dt.tz_convert("UTC").dt.strftime(ISO_FMT)

    out = df.copy()
    out.insert(len(out.columns), "t_rel_s", t_rel.values)
    out.insert(len(out.columns), "epoch_aligned", epoch_aligned.values)
    return out


# ---------------------------
# I/O helpers
# ---------------------------

def _ensure_dirs(out_root: Path) -> Tuple[Path, Path]:
    sat_dir = out_root / "satellite_epehms"
    tgt_dir = out_root / "targets_epehms"
    sat_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    return sat_dir, tgt_dir


def _guess_orbit_prefix(p: Path) -> str:
    name = p.name.upper()
    if "LEO" in name:
        return "LEO_"
    if "MEO" in name:
        return "MEO_"
    if "GEO" in name:
        return "GEO_"
    return ""


def _save_with_prefix(df: pd.DataFrame, src: Path, dst_dir: Path, force_prefix: Optional[str] = None) -> Path:
    prefix = force_prefix if force_prefix is not None else _guess_orbit_prefix(src)
    out_name = prefix + src.name
    out_path = dst_dir / out_name
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------
# MAIN
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Allinea i tempi di ephemeris target/sat su un unico clock di run.")
    parser.add_argument("--targets-glob",
                        default=None,
                        help="Glob dei CSV target. Se non fornito: <repo>/exports/target_exports/OUTPUT_CSV/HGV_*.csv")
    parser.add_argument("--sat-ephem-root",
                        default=None,
                        help="Root ephemeris satelliti. Se non fornito: <repo>/flightdynamics/exports/ephemeris")
    parser.add_argument("--run-id",
                        default=None,
                        help="Run ID per selezionare la cartella ephemeris (se assente, prende la più recente).")
    parser.add_argument("--out-root",
                        default=None,
                        help="Cartella radice per gli output. Se non fornito: <repo>/exports/aligned_ephems")
    args = parser.parse_args(argv)

    # Risolvi repo root: PROIRITÀ env, poi autodetect dal path del file
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("EZSMAD_PROJECT_ROOT")
    if env_root:
        repo_root = Path(env_root).expanduser().resolve()
    else:
        repo_root = _find_repo_root(Path(__file__).resolve()).resolve()

    # Defaults se non passati
    default_targets_glob = repo_root / "exports/target_exports/OUTPUT_CSV/HGV_*.csv"
    default_sat_root     = repo_root / "flightdynamics" / "exports" / "ephemeris"
    default_out_root     = repo_root / "exports" / "aligned_ephems"

    # Normalizza argomenti: se relativi, renderli relativi al repo_root
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

    if args.out_root:
        out_root_arg = Path(args.out_root)
        out_root = (repo_root / out_root_arg).resolve() if not out_root_arg.is_absolute() else out_root_arg.resolve()
    else:
        out_root = default_out_root.resolve()

    print(f"[INFO] repo_root: {repo_root}")
    print(f"[INFO] targets_glob: {targets_glob}")
    print(f"[INFO] sat_ephem_root: {sat_ephem_root}")
    print(f"[INFO] out_root: {out_root}")

    run_start = _get_run_start_time(args.run_id)
    print(f"[INFO] run_start (UTC): {run_start.isoformat()}")

    sat_out_dir, tgt_out_dir = _ensure_dirs(out_root)

    # ---- Targets ----
    tgt_files = sorted(glob.glob(targets_glob))
    if not tgt_files:
        print(f"[WARN] Nessun target trovato con glob: {targets_glob}")
    else:
        print(f"[INFO] Target CSV trovati: {len(tgt_files)}")
    for f in tgt_files:
        src = Path(f)
        try:
            df = pd.read_csv(src)
            df2 = _add_aligned_time_columns(df, run_start)
            outp = _save_with_prefix(df2, src, tgt_out_dir, force_prefix=None)
            print(f"[OK] Salvato target allineato: {outp}")
        except Exception as e:
            print(f"[ERR] Target '{src}': {e}")

    # ---- Satelliti ----
    sat_root = Path(sat_ephem_root)
    sat_csvs = _collect_sat_csvs(sat_root, args.run_id)
    if not sat_csvs:
        print(f"[WARN] Nessun ephemeris satellite trovato in {sat_root} (run_id={args.run_id}).")
    else:
        print(f"[INFO] Ephemeris satelliti trovati: {len(sat_csvs)} in {sat_root}")
    for src in sorted(sat_csvs):
        try:
            df = pd.read_csv(src)
            df2 = _add_aligned_time_columns(df, run_start)
            outp = _save_with_prefix(df2, src, sat_out_dir, force_prefix=None)
            print(f"[OK] Salvato satellite allineato: {outp}")
        except Exception as e:
            print(f"[ERR] Satellite '{src}': {e}")

    print("[DONE] Allineamento completato.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
