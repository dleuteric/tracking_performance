#!/usr/bin/env python3
"""
Robust Constellation Groundtracks Plotter (interactive HTML)

- Reads ephemerides exported by flightdynamics/space_segment.py
  (expects CSV files per satellite with columns: t_sec, epoch_utc, x_km, y_km, z_km, ...)
- Detects the latest run under exports/ephemeris/ unless a run directory is given
- Builds a color‑blind‑friendly interactive HTML (Plotly) with:
    * One color per orbital regime (LEO/MEO/GEO/HEO)
    * Different dashes per regime for readability
    * Small start markers
    * Legend + buttons to toggle per regime
- Outputs to: exports/ephemeris/plots/<RUN_ID>/groundtracks.html

Usage:
  python -m flightdynamics.plot_constellation           # use latest run
  python -m flightdynamics.plot_constellation --run exports/ephemeris/20250927T120000Z
  python -m flightdynamics.plot_constellation --out /tmp/gt.html --max-points 15000

Notes:
- Uses a spherical Earth lat/lon conversion (sufficient for visualization)
- Breaks segments at the dateline to avoid long wrap lines
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Plotly non disponibile. Installa con: pip install plotly"
    )

# ----------------------------
# Helpers
# ----------------------------

def find_ephem_root() -> Path:
    """Return the ephemeris root folder.
    Tries common locations:
      - project root: exports/ephemeris
      - alongside this module (when CWD is flightdynamics): flightdynamics/exports/ephemeris
    """
    candidates = [
        Path('exports/ephemeris'),
        Path(__file__).resolve().parent / 'exports' / 'ephemeris',          # flightdynamics/exports/ephemeris
        Path(__file__).resolve().parent.parent / 'exports' / 'ephemeris',   # project_root/exports/ephemeris
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                # consider it valid if it contains at least one run dir with a matching manifest
                for d in c.iterdir():
                    if d.is_dir():
                        man = d / f'manifest_{d.name}.json'
                        if man.exists():
                            return c
        except Exception:
            continue
    raise FileNotFoundError(
        'Ephemeris root non trovato. Prova a passare --run <path_run> oppure assicurati che esista exports/ephemeris/<RUN_ID>/manifest_<RUN_ID>.json'
    )


def latest_run(ephem_root: Path) -> Path:
    # Accept only folders named like YYYYMMDDTHHMMSSZ that contain their manifest
    ts_pat = re.compile(r'^\d{8}T\d{6}Z$')
    runs = []
    for p in ephem_root.iterdir():
        if not p.is_dir():
            continue
        if ts_pat.match(p.name):
            man = p / f'manifest_{p.name}.json'
            if man.exists():
                runs.append(p)
    if not runs:
        # Fallback: any dir with a self-named manifest
        for p in ephem_root.iterdir():
            if p.is_dir():
                man = p / f'manifest_{p.name}.json'
                if man.exists():
                    runs.append(p)
    if not runs:
        raise FileNotFoundError(f"Nessuna run valida trovata in {ephem_root}. Evitato 'plots' e simili.")
    return sorted(runs)[-1]


def load_manifest(run_dir: Path) -> Dict:
    man = run_dir / f"manifest_{run_dir.name}.json"
    if not man.exists():
        raise FileNotFoundError(f"Manifest non trovato: {man}")
    return json.loads(man.read_text())


def infer_regime_from_filename(filename: str) -> str:
    """Expect filenames like 'LEO_P5_S3.csv' → 'LEO'. Fallback 'UNK'."""
    try:
        return Path(filename).name.split('_', 1)[0].upper()
    except Exception:
        return 'UNK'


def ecef_to_latlon(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y + z * z)
    # Avoid invalid divisions
    r[r == 0] = np.nan
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def break_dateline(lon: np.ndarray, lat: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
    """Insert segment breaks (None) where Δlon jumps across the dateline (>180°).
    Returns list of segments (lon_segs, lat_segs).
    """
    if lon.size == 0:
        return [], []
    lon_segs: List[List[float]] = [[float(lon[0])]]
    lat_segs: List[List[float]] = [[float(lat[0])]]
    for i in range(1, lon.size):
        dlon = abs(lon[i] - lon[i - 1])
        if dlon > 180.0:
            # start new segment
            lon_segs.append([float(lon[i])])
            lat_segs.append([float(lat[i])])
        else:
            lon_segs[-1].append(float(lon[i]))
            lat_segs[-1].append(float(lat[i]))
    return lon_segs, lat_segs


def decimate(arr: np.ndarray, max_points: int) -> np.ndarray:
    if arr.size <= max_points:
        return arr
    step = int(np.ceil(arr.size / max_points))
    return arr[::step]


# Okabe–Ito palette (color‑blind friendly)
REGIME_STYLES = {
    'LEO': dict(color="#0072B2", dash="solid"),   # blue
    'MEO': dict(color="#E69F00", dash="dash"),    # orange
    'GEO': dict(color="#009E73", dash="dot"),     # green
    'HEO': dict(color="#CC79A7", dash="dashdot"), # magenta
}
DEFAULT_STYLE = dict(color="#555555", dash="solid")


# ----------------------------
# Plotting core
# ----------------------------

def build_groundtracks_figure(run_dir: Path, max_points: int = 10000) -> go.Figure:
    manifest = load_manifest(run_dir)
    files_map: Dict[str, str] = manifest.get('files', {})

    fig = go.Figure()
    trace_regimes: List[str] = []

    # Build traces per satellite
    for sat_id, rel in files_map.items():
        csv_path = run_dir / rel
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        required = {'x_km', 'y_km', 'z_km'}
        if not required.issubset(df.columns):
            # Skip malformed file
            continue

        x = df['x_km'].to_numpy(dtype=float)
        y = df['y_km'].to_numpy(dtype=float)
        z = df['z_km'].to_numpy(dtype=float)

        # Decimate evenly to keep file size & rendering OK
        idx = decimate(np.arange(x.size), max_points)
        x, y, z = x[idx], y[idx], z[idx]

        lat, lon = ecef_to_latlon(x, y, z)
        lon_segs, lat_segs = break_dateline(lon, lat)

        regime = infer_regime_from_filename(rel)
        style = REGIME_STYLES.get(regime, DEFAULT_STYLE)

        # Add one line trace per segment for this sat
        for LON, LAT in zip(lon_segs, lat_segs):
            fig.add_trace(go.Scattergeo(
                lon=LON, lat=LAT, mode='lines',
                name=f"{regime} {sat_id}", legendgroup=regime, showlegend=False,
                line=dict(width=1.3, color=style['color'], dash=style['dash']),
                opacity=0.9
            ))
            trace_regimes.append(regime)

        # Start marker
        if lon.size > 0:
            fig.add_trace(go.Scattergeo(
                lon=[float(lon[0])], lat=[float(lat[0])], mode='markers',
                name=f"{regime} start {sat_id}", legendgroup=regime, showlegend=False,
                marker=dict(size=4, color=style['color'], symbol='circle', line=dict(width=0.5, color='black')),
                opacity=0.9
            ))
            trace_regimes.append(regime)

    # Legend entries per regime (one dummy visible line)
    for reg, style in REGIME_STYLES.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode='lines',
            name=reg, legendgroup=reg, showlegend=True,
            line=dict(width=3, color=style['color'], dash=style['dash'])
        ))
        trace_regimes.append(reg)

    # Geo layout
    fig.update_geos(
        showcountries=True, showcoastlines=True, showland=True,
        landcolor="#f5f5f5", coastlinecolor="#888888",
        projection_type="equirectangular",
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lonaxis_dtick=60, lataxis_dtick=30,
    )

    fig.update_layout(
        title=f"Groundtracks – {run_dir.name}",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(title="Regime", orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0)
    )

    # Buttons for regime toggling
    def vis_mask(active: str) -> List[bool]:
        vis: List[bool] = []
        for reg in trace_regimes:
            vis.append(True if active == 'ALL' or reg == active else False)
        return vis

    buttons = [dict(label='ALL', method='update', args=[{'visible': vis_mask('ALL')}])]
    for reg in REGIME_STYLES.keys():
        buttons.append(dict(label=reg, method='update', args=[{'visible': vis_mask(reg)}]))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons', direction='right', x=0.0, y=1.10,
            buttons=buttons, pad=dict(r=5, t=5), showactive=True
        )]
    )

    return fig


def save_html(fig: go.Figure, run_dir: Path, out: Path | None = None) -> Path:
    if out is None:
        out_dir = run_dir.parent / 'plots' / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / 'groundtracks.html'
    fig.write_html(out)
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Interactive groundtracks plotter")
    p.add_argument('--run', type=Path, help='Run directory under exports/ephemeris/')
    p.add_argument('--out', type=Path, help='Output HTML path (optional)')
    p.add_argument('--max-points', type=int, default=10000, help='Max points per satellite after decimation')
    args = p.parse_args()

    # Determine ephemeris root and run directory
    if args.run:
        run_dir = args.run
        # If user accidentally passed plots/<RUN_ID>, map back to ephemeris/<RUN_ID>
        if run_dir.parent.name == 'plots':
            run_dir = run_dir.parent.parent / run_dir.name
    else:
        ephem_root = find_ephem_root()
        run_dir = latest_run(ephem_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory non trovato: {run_dir}")

    fig = build_groundtracks_figure(run_dir, max_points=args.max_points)
    out_path = save_html(fig, run_dir, args.out)
    print(f"[OK] HTML: {out_path}")


if __name__ == '__main__':
    main()
