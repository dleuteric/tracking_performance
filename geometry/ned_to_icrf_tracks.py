#!/usr/bin/env python3
"""
NED/Generic CSV -> ECEF (legacy columns) for ez-SMAD
- First line must encode NED origin: 'NED, lat_deg, lon_deg[, h0_m]'
- Columns mapped to legacy: t_s, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps
  where x_m,y_m,z_m are ECEF [m] (not local!)
"""

from __future__ import annotations
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# WGS-84
A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
EARTH_RADIUS_MEAN = 6371000.0  # for sphere plot only

def _sniff_header_and_sep(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        l1 = fh.readline().strip()
        l2 = fh.readline().strip()
    # detect sep on first non-NED line
    header_line = l2 if l1.upper().startswith('NED') else l1
    counts = {',': header_line.count(','), ';': header_line.count(';'), '\t': header_line.count('\t')}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = None
    skip = 1 if l1.upper().startswith('NED') else 0
    return l1, skip, sep

def _parse_ned_origin(l1: str):
    # Extract floats from the first line (works for comma/semicolon/tab)
    nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', l1)]
    if len(nums) < 2:
        raise ValueError("First line must be like 'NED, lat_deg, lon_deg[, h0_m]'.")
    lat0_deg, lon0_deg = nums[0], nums[1]
    h0_m = nums[2] if len(nums) >= 3 else None
    return np.deg2rad(lat0_deg), np.deg2rad(lon0_deg), h0_m

def _ecef_from_geodetic(lat, lon, h):
    sinφ, cosφ = np.sin(lat), np.cos(lat)
    N = A / np.sqrt(1.0 - E2 * sinφ**2)
    x = (N + h) * cosφ * np.cos(lon)
    y = (N + h) * cosφ * np.sin(lon)
    z = (N * (1.0 - E2) + h) * sinφ
    return np.array([x, y, z])

def _enu_basis_ecef(lat, lon):
    # Columns are unit vectors of ENU in ECEF coords
    sinφ, cosφ = np.sin(lat), np.cos(lat)
    sinλ, cosλ = np.sin(lon), np.cos(lon)
    e_E = np.array([-sinλ,            cosλ,           0.0      ])
    e_N = np.array([-sinφ*cosλ,      -sinφ*sinλ,      cosφ     ])
    e_U = np.array([ cosφ*cosλ,       cosφ*sinλ,      sinφ     ])
    R = np.column_stack((e_E, e_N, e_U))  # E,N,U as columns
    return R  # ECEF = R @ [E,N,U]

def _read_df(path: str, sep, skip):
    df = pd.read_csv(path, sep=sep, engine='python', skiprows=skip)
    if len(df.columns) == 1:
        # retry with comma
        df = pd.read_csv(path, sep=',', engine='python', skiprows=skip)
    return df

def _map_columns(df: pd.DataFrame):
    colmap = {}
    for c in df.columns:
        cl = str(c).strip()
        l = cl.lower()
        if l in ('t','time','time_s','timestamp') or (l.startswith('t') and 'v' not in l and 'a' not in l and l!='tx'):
            colmap[c] = 't_s'
        elif l in ('x','north','n'):
            colmap[c] = 'X'  # local North [m]
        elif l in ('y','east','e'):
            colmap[c] = 'Y'  # local East [m]
        elif l in ('z','down','d'):
            colmap[c] = 'Z'  # local Down [m] (positive down)
        elif l in ('vx','vn','v_n','north_rate'):
            colmap[c] = 'Vx'  # local North rate [m/s]
        elif l in ('vy','ve','v_e','east_rate'):
            colmap[c] = 'Vy'  # local East rate [m/s]
        elif l in ('vz','vd','v_d','down_rate'):
            colmap[c] = 'Vz'  # local Down rate [m/s]
        elif l in ('altitude','alt','h'):
            colmap[c] = 'Altitude'  # absolute height above ellipsoid [m]
        else:
            # keep as is (e.g., Ax,Ay,Az…)
            colmap[c] = cl
    dfr = df.rename(columns=colmap)
    # coerce numeric where relevant
    for k in ('t_s','X','Y','Z','Vx','Vy','Vz','Altitude'):
        if k in dfr.columns:
            dfr[k] = pd.to_numeric(dfr[k], errors='coerce')
    return dfr

def ned_csv_to_ecef_legacy(csv_path: str, out_dir: Path):
    print("="*60)
    print("NED/Generic -> ECEF Converter (legacy columns)")
    print("="*60)

    l1, skip, sep = _sniff_header_and_sep(csv_path)
    lat0, lon0, h0_opt = _parse_ned_origin(l1)

    df_raw = _read_df(csv_path, sep, skip)
    print(f"Loaded {len(df_raw)} rows. Columns: {list(df_raw.columns)}")

    df = _map_columns(df_raw)

    # Decide h0
    if h0_opt is not None:
        h0 = float(h0_opt)
    elif 'Altitude' in df.columns and pd.notna(df['Altitude'].iloc[0]):
        h0 = float(df['Altitude'].iloc[0])
    else:
        h0 = 0.0
    print(f"Origin geodetic: lat={np.rad2deg(lat0):.6f} deg, lon={np.rad2deg(lon0):.6f} deg, h0={h0:.2f} m")

    # Build origin ECEF and ENU basis
    r0_ecef = _ecef_from_geodetic(lat0, lon0, h0)
    R_ENU2ECEF = _enu_basis_ecef(lat0, lon0)

    # Build local ENU offsets
    # E=Y, N=X, U from Altitude (absolute) if present, else U = -Z (Down positive)
    E = df['Y'] if 'Y' in df.columns else pd.Series(np.zeros(len(df)))
    N = df['X'] if 'X' in df.columns else pd.Series(np.zeros(len(df)))
    if 'Altitude' in df.columns:
        U = df['Altitude'] - h0
    elif 'Z' in df.columns:
        U = -df['Z']
    else:
        U = pd.Series(np.zeros(len(df)))

    # Stack ENU and transform to ECEF
    ENU = np.vstack([E.values, N.values, U.values])  # shape (3, N)
    ECEF = (R_ENU2ECEF @ ENU) + r0_ecef.reshape(3,1)
    x_m, y_m, z_m = ECEF[0,:], ECEF[1,:], ECEF[2,:]

    # Velocities (local NED -> ENU -> ECEF). Ignore transport terms.
    if all(k in df.columns for k in ('Vx','Vy','Vz')):
        # NED rates: Vx=Ndot, Vy=Edot, Vz=Ddot
        # ENU rates: [Edot, Ndot, Udot] with Udot = -Ddot
        ENUdot = np.vstack([df['Vy'].values,
                            df['Vx'].values,
                           -df['Vz'].values])
        ECEFdot = R_ENU2ECEF @ ENUdot
        vx_mps, vy_mps, vz_mps = ECEFdot[0,:], ECEFdot[1,:], ECEFdot[2,:]
    else:
        vx_mps = np.zeros_like(x_m)
        vy_mps = np.zeros_like(y_m)
        vz_mps = np.zeros_like(z_m)

    # Time
    t_s = pd.to_numeric(df['t_s'], errors='coerce').fillna(0.0) if 't_s' in df.columns else pd.Series(np.arange(len(x_m), dtype=float))

    # Output dataframe in legacy order (ECEF)
    out = pd.DataFrame({
        't_s': t_s.values,
        'x_m': x_m,
        'y_m': y_m,
        'z_m': z_m,
        'vx_mps': vx_mps,
        'vy_mps': vy_mps,
        'vz_mps': vz_mps,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    out_file = out_dir / f"{stem}_ecef.csv"
    out.to_csv(out_file, index=False)
    print(f"✓ Saved legacy ECEF CSV: {out_file}")

    # Plot quicklook
    _plot_quicklook(out, lat0, lon0, h0, title=f"{stem} (ECEF)")

    return out_file

def _plot_quicklook(df_legacy: pd.DataFrame, lat0, lon0, h0, title="Trajectory"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Earth sphere (for context)
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = EARTH_RADIUS_MEAN * np.outer(np.cos(u), np.sin(v)) / 1000.0
    ys = EARTH_RADIUS_MEAN * np.outer(np.sin(u), np.sin(v)) / 1000.0
    zs = EARTH_RADIUS_MEAN * np.outer(np.ones_like(u), np.cos(v)) / 1000.0
    ax.plot_surface(xs, ys, zs, alpha=0.35, linewidth=0, shade=True)

    x = df_legacy['x_m'] / 1000.0
    y = df_legacy['y_m'] / 1000.0
    z = df_legacy['z_m'] / 1000.0
    t = pd.to_numeric(df_legacy['t_s'], errors='coerce').fillna(0.0)

    ax.plot(x, y, z, '-', linewidth=2, label='Trajectory')
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], s=60, marker='o', label='Start')
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], s=80, marker='*', label='End')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title(title)
    ax.legend()

    # Stats
    r_norm = np.sqrt(x**2 + y**2 + z**2)
    alt_est_km = r_norm - EARTH_RADIUS_MEAN/1000.0
    text = (
        f"Origin: lat={np.rad2deg(lat0):.3f}°, lon={np.rad2deg(lon0):.3f}°, h0={h0:.0f} m\n"
        f"Duration: {t.max():.1f} s, Points: {len(t)}\n"
        f"Alt est (min/max): {alt_est_km.min():.1f}/{alt_est_km.max():.1f} km"
    )
    ax.text2D(0.02, 0.98, text, transform=ax.transAxes, va='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.show()

def main():
    # Default input file (user's test path) if none is provided
    default_in = "/Users/daniele_leuteri/PycharmProjects/ez-SMAD_mk01/exports/target_exports/truth_tracks/ned_test.csv"

    if len(sys.argv) >= 2:
        in_csv = sys.argv[1]
    else:
        print("No input provided on CLI. Using default test file.")
        in_csv = default_in

    if not Path(in_csv).exists():
        print(f"Error: file not found: {in_csv}")
        sys.exit(1)

    # Export in the same folder as the input file
    out_dir = Path(in_csv).parent

    ned_csv_to_ecef_legacy(in_csv, out_dir)

if __name__ == "__main__":
    main()