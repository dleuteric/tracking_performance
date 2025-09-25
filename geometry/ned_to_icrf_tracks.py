#!/usr/bin/env python3
"""
NED to ICRF converter for ez-SMAD
Direct conversion to match legacy CSV format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Constants
EARTH_RADIUS = 6371000  # meters


def ned_to_icrf(ned_file):
    """
    Convert NED CSV to ICRF format matching legacy structure

    Args:
        ned_file: Path to NED CSV file

    Returns:
        DataFrame in legacy format (t_s, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps)
    """

    print(f"Reading file: {ned_file}")

    # Robustly detect header line to skip and delimiter
    with open(ned_file, 'r', encoding='utf-8', errors='ignore') as fh:
        first_line = fh.readline().strip()
        second_line = fh.readline().strip()

    # Skip first line if it looks like a mode tag (e.g., 'NED' or 'NED;')
    skiprows = 1 if first_line.upper().startswith('NED') else 0

    # Heuristic delimiter detection from the (first non-skipped) header line
    header_line = second_line if skiprows == 1 else first_line
    counts = {',': header_line.count(','), ';': header_line.count(';'), '\t': header_line.count('\t')}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        # Fallback to pandas sniffing
        sep = None

    df = pd.read_csv(ned_file, sep=sep, engine='python', skiprows=skiprows)

    print(f"Loaded {len(df)} points")
    print(f"Columns found: {list(df.columns)}")

    # If pandas failed to split columns (single wide column), force comma retry
    if len(df.columns) == 1:
        df = pd.read_csv(ned_file, sep=',', engine='python', skiprows=skiprows)
        print("Re-read with comma separator due to single-column parse.")
        print(f"Columns found (retry): {list(df.columns)}")

    # Map columns to standard names (case-insensitive)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('t', 'time', 'time_s', 'timestamp') or (col_lower.startswith('t') and 's' not in col_lower and 'v' not in col_lower and 'a' not in col_lower):
            col_mapping[col] = 't_s'
        elif 'x' in col_lower and 'v' not in col_lower and 'a' not in col_lower:
            col_mapping[col] = 'x_m'
        elif 'y' in col_lower and 'v' not in col_lower and 'a' not in col_lower:
            col_mapping[col] = 'y_m'
        elif 'z' in col_lower and 'v' not in col_lower and 'a' not in col_lower:
            col_mapping[col] = 'z_m'
        elif 'vx' in col_lower:
            col_mapping[col] = 'vx_mps'
        elif 'vy' in col_lower:
            col_mapping[col] = 'vy_mps'
        elif 'vz' in col_lower:
            col_mapping[col] = 'vz_mps'

    # Rename columns
    df_renamed = df.rename(columns=col_mapping)

    # Coerce known numeric columns to numeric (errors->NaN), then fill NaNs with 0
    for c in ['t_s', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps', 'Altitude']:
        if c in df_renamed.columns:
            df_renamed[c] = pd.to_numeric(df_renamed[c], errors='coerce')

    # Handle NED Z coordinate (positive down) and altitude if present
    if 'Altitude' in df.columns and 'z_m' not in df_renamed.columns:
        # Use altitude as negative Z (since altitude is up, Z in NED is down)
        df_renamed['z_m'] = -df['Altitude'].values
    elif 'z_m' in df_renamed.columns:
        # Invert Z if it's NED (positive down -> negative for up)
        df_renamed['z_m'] = -df_renamed['z_m'].values

    # Add Earth radius to get geocentric coordinates
    # Assuming the trajectory starts near Earth surface
    if 'x_m' in df_renamed.columns:
        df_renamed['x_m'] = df_renamed['x_m'].values + EARTH_RADIUS

    # Invert Z velocity if from NED
    if 'vz_mps' in df_renamed.columns:
        df_renamed['vz_mps'] = -df_renamed['vz_mps'].values

    # Extract only required columns in correct order
    required_cols = ['t_s', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps']
    available_cols = [col for col in required_cols if col in df_renamed.columns]

    output_df = df_renamed[available_cols].copy()

    # Fill missing columns with zeros if needed
    for col in required_cols:
        if col not in output_df.columns:
            output_df[col] = 0.0

    # Ensure correct order
    output_df = output_df[required_cols]

    return output_df


def plot_3d_trajectory(df, title="Trajectory"):
    """
    Create 3D plot with Earth sphere and trajectory
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth as sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v)) / 1000  # km
    y_earth = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v)) / 1000
    z_earth = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) / 1000

    ax.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.4,
                    linewidth=0, antialiased=True)

    # Plot trajectory
    x = df['x_m'] / 1000  # Convert to km
    y = df['y_m'] / 1000
    z = df['z_m'] / 1000

    # Color by time (ensure numeric)
    times = pd.to_numeric(df['t_s'], errors='coerce').fillna(0.0)
    trajectory = ax.plot(x, y, z, 'r-', linewidth=2, label='Trajectory')

    # Plot start and end points
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='green', s=200,
               marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='red', s=200,
               marker='*', label='End', edgecolors='black', linewidth=2)

    # Labels and formatting
    ax.set_xlabel('X [km]', fontsize=12)
    ax.set_ylabel('Y [km]', fontsize=12)
    ax.set_zlabel('Z [km]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Calculate proper view limits
    trajectory_center = [x.mean(), y.mean(), z.mean()]
    trajectory_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())

    plot_range = max(trajectory_range * 0.6, EARTH_RADIUS / 1000 * 1.2)

    ax.set_xlim([trajectory_center[0] - plot_range, trajectory_center[0] + plot_range])
    ax.set_ylim([trajectory_center[1] - plot_range, trajectory_center[1] + plot_range])
    ax.set_zlim([trajectory_center[2] - plot_range, trajectory_center[2] + plot_range])

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add trajectory stats
    altitude = np.sqrt(x ** 2 + y ** 2 + z ** 2) - EARTH_RADIUS / 1000
    ground_range = np.sqrt((x.iloc[-1] - x.iloc[0]) ** 2 + (y.iloc[-1] - y.iloc[0]) ** 2)

    stats_text = f"Max Altitude: {altitude.max():.1f} km\n"
    stats_text += f"Ground Range: {ground_range:.1f} km\n"
    stats_text += f"Duration: {times.max():.1f} s\n"
    stats_text += f"Points: {len(df)}"

    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def main():
    """
    Main conversion routine
    """
    print("=" * 60)
    print("NED to ICRF Trajectory Converter for ez-SMAD")
    print("=" * 60)

    # Input file - change this to your NED file
    input_file = "/Users/daniele_leuteri/PycharmProjects/ez-SMAD_mk01/exports/target_exports/truth_tracks/ned_test.csv"

    # Check if file exists
    if not Path(input_file).exists():
        print(f"Error: File {input_file} not found!")
        print("Available CSV files:")
        for f in Path(".").glob("*.csv"):
            print(f"  - {f}")
        sys.exit(1)

    try:
        # Convert trajectory
        df = ned_to_icrf(input_file)

        # Create output directory
        output_dir = Path("exports/targets/truth_trajectory")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        output_name = Path(input_file).stem
        output_file = output_dir / f"{output_name}_icrf.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")

        # Print first few rows
        print(f"\nFirst 5 rows of output:")
        print(df.head())

        # Create and save 3D plot
        fig = plot_3d_trajectory(df, title=f"Trajectory: {output_name}")
        plot_file = output_dir / f"{output_name}_3d.png"
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to: {plot_file}")

        # Show plot
        plt.show()

        print("\n✓ Conversion complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()