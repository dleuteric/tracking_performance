# main_geometric_analysis.py
"""
Main orchestrator for geometric performance analysis of LEO constellation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from error_propagation import GeometricPerformanceModule


# =====================================================
# DATA LOADERS
# =====================================================

def parse_oem_file(oem_path: Path) -> pd.DataFrame:
    """Parse OEM file to extract target trajectory"""
    print(f"Reading OEM: {oem_path.name}")

    data = []
    with open(oem_path, 'r') as f:
        lines = f.readlines()

    # Find where data starts (after META_STOP)
    data_start = False
    for i, line in enumerate(lines):
        if 'META_STOP' in line:
            data_start = True
            continue

        if data_start:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse data line
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # Parse timestamp - handle the specific format
                    timestamp = pd.to_datetime(parts[0], format='%Y-%m-%dT%H:%M:%S.%f')

                    # Parse position and velocity (in scientific notation)
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    vx = float(parts[4])
                    vy = float(parts[5])
                    vz = float(parts[6])

                    data.append([timestamp, x, y, z, vx, vy, vz])
                except Exception as e:
                    # Debug: uncomment to see parsing errors
                    # print(f"Failed to parse line: {line[:50]}... Error: {e}")
                    continue

    if not data:
        print(f"WARNING: No data parsed from {oem_path}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['time', 'x_km', 'y_km', 'z_km', 'vx_kmps', 'vy_kmps', 'vz_kmps'])
    df.set_index('time', inplace=True)

    print(f"   Successfully parsed {len(df)} epochs")
    return df


def parse_ephemeris_file(eph_path: Path) -> pd.DataFrame:
    """Parse satellite ephemeris CSV file"""
    try:
        # Read CSV, skip any comment lines
        df = pd.read_csv(eph_path, comment='#')

        # Debug: print columns found
        # print(f"Ephemeris columns: {df.columns.tolist()}")

        # Normalize column names - handle variations
        col_map = {
            'Time (UTCG)': 'time',
            'Time(UTCG)': 'time',
            'x (km)': 'x_km',
            'y (km)': 'y_km',
            'z (km)': 'z_km',
            'vx (km/sec)': 'vx_kmps',
            'vy (km/sec)': 'vy_kmps',
            'vz (km/sec)': 'vz_kmps'
        }

        # Map columns
        df.columns = [col.strip() for col in df.columns]  # Remove whitespace
        df.rename(columns=col_map, inplace=True)

        # If still no 'time' column, try first column
        if 'time' not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'time'}, inplace=True)

        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        return df
    except Exception as e:
        print(f"Error parsing ephemeris {eph_path.name}: {e}")
        return pd.DataFrame()


def parse_los_file(los_path: Path) -> pd.DataFrame:
    """Parse LOS unit vectors CSV file"""
    try:
        df = pd.read_csv(los_path, comment='#')

        # Debug: print columns found
        # print(f"LOS columns: {df.columns.tolist()}")

        # Normalize column names
        df.columns = [col.strip() for col in df.columns]

        # Map time column
        time_cols = ['Time (UTCG)', 'Time(UTCG)', 'Time']
        for tcol in time_cols:
            if tcol in df.columns:
                df.rename(columns={tcol: 'time'}, inplace=True)
                break

        # If no time column found, use first column
        if 'time' not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'time'}, inplace=True)

        # Map LOS vector columns (might be x,y,z or ux,uy,uz)
        if 'x' in df.columns:
            df.rename(columns={'x': 'ux', 'y': 'uy', 'z': 'uz'}, inplace=True)

        # Convert time
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        return df
    except Exception as e:
        print(f"Error parsing LOS {los_path.name}: {e}")
        return pd.DataFrame()

def discover_data_files(base_path: Path) -> Dict:
    """Recursively discover all data files in the directory structure"""

    files = {
        'ephemeris': {},
        'los': {},
        'oem': []
    }

    # Find ephemeris files
    eph_dir = base_path / 'stk_exports' / 'OUTPUT_EPHEM'
    if eph_dir.exists():
        for eph_file in eph_dir.glob('OBS_*.csv'):
            # Extract satellite ID from filename (e.g., OBS_001_ephem.csv -> 001)
            match = re.search(r'OBS_(\d+)', eph_file.name)
            if match:
                sat_id = match.group(1)
                files['ephemeris'][sat_id] = eph_file

    # Find LOS files for each target
    los_base = base_path / 'stk_exports' / 'OUTPUT_LOS_VECTORS'
    if los_base.exists():
        for target_dir in los_base.iterdir():
            if target_dir.is_dir():
                target_id = target_dir.name
                files['los'][target_id] = {}

                for los_file in target_dir.glob('LOS_OBS_*_to_*.csv'):
                    match = re.search(r'LOS_OBS_(\d+)_to', los_file.name)
                    if match:
                        sat_id = match.group(1)
                        files['los'][target_id][sat_id] = los_file

    # Find OEM files
    oem_dir = base_path / 'target_exports' / 'OUTPUT_OEM'
    if oem_dir.exists():
        files['oem'] = list(oem_dir.glob('*.oem'))

    return files


# =====================================================
# MAIN ANALYSIS
# =====================================================

def main():
    """Main orchestration function"""

    # Setup paths
    base_path = Path('../../exports_48')  # Adjust if needed
    output_dir = Path('../outputs_geometric_analysis')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GEOMETRIC PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Discover all files
    print("\n1. Discovering data files...")
    files = discover_data_files(base_path)

    print(f"   Found {len(files['ephemeris'])} satellite ephemeris files")
    print(f"   Found {len(files['los'])} targets with LOS data")
    print(f"   Found {len(files['oem'])} OEM trajectory files")

    # Select target to analyze
    if not files['oem']:
        print("ERROR: No OEM files found!")
        return

    # Use first OEM file found (or modify to select specific one)
    oem_file = files['oem'][0]
    target_name = oem_file.stem  # e.g., HGV_00009

    print(f"\n2. Analyzing target: {target_name}")

    # Load target truth trajectory
    target_truth = parse_oem_file(oem_file)
    print(f"   Loaded {len(target_truth)} epochs from OEM")

    # Find corresponding LOS data
    if target_name not in files['los']:
        print(f"ERROR: No LOS data found for target {target_name}")
        return

    los_files = files['los'][target_name]
    print(f"   Found LOS data for {len(los_files)} satellites")

    # Initialize performance module
    print("\n3. Initializing Geometric Performance Module...")
    module = GeometricPerformanceModule(
        sigma_los_mrad=1.0,
        min_elevation_deg=5.0,
        max_range_km=3000.0
    )

    # Process each epoch
    print("\n4. Processing epochs...")
    results = []

    # Sample epochs (process every N-th epoch for speed)
    sample_rate = 10  # Process every 10th epoch
    epochs_to_process = target_truth.index[::sample_rate]

    for i, epoch in enumerate(epochs_to_process[:100]):  # Limit to first 100 for testing
        if i % 10 == 0:
            print(f"   Processing epoch {i + 1}/{len(epochs_to_process[:100])}")

        # Gather satellite data at this epoch
        sat_positions = {}
        los_vectors = {}

        for sat_id in files['ephemeris']:
            # Load ephemeris
            if sat_id not in los_files:
                continue

            eph_df = parse_ephemeris_file(files['ephemeris'][sat_id])
            los_df = parse_los_file(los_files[sat_id])

            # Interpolate to current epoch
            if epoch in eph_df.index:
                sat_positions[sat_id] = eph_df.loc[epoch, ['x_km', 'y_km', 'z_km']].values

            if epoch in los_df.index:
                los_vectors[sat_id] = los_df.loc[epoch, ['ux', 'uy', 'uz']].values

        # Get true target position
        true_pos = target_truth.loc[epoch, ['x_km', 'y_km', 'z_km']].values

        # Process epoch
        epoch_result = module.process_epoch(sat_positions, los_vectors, true_pos)

        # Store results only if valid best_pair exists
        if epoch_result.get('best_pair') is not None:
            bp = epoch_result['best_pair']
            results.append({
                'time': epoch,
                'n_visible': epoch_result['n_visible'],
                'best_pair': f"{bp.pair_ids[0]}-{bp.pair_ids[1]}",
                'baseline_deg': bp.baseline_angle_deg,
                'cep50_km': bp.cep50_km,
                'cep90_km': bp.cep90_km,
                'gdop': bp.gdop,
                'error_3d_km': bp.error_3d if hasattr(bp, 'error_3d') else np.nan
            })
        else:
            print(f"   WARNING: No valid best pair for epoch {epoch}. Skipping.")
            results.append({
                'time': epoch,
                'n_visible': epoch_result.get('n_visible', 0),
                'best_pair': 'None',
                'baseline_deg': np.nan,
                'cep50_km': np.nan,
                'cep90_km': np.nan,
                'gdop': np.nan,
                'error_3d_km': np.nan
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = output_dir / f'geometric_analysis_{target_name}.csv'
    results_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_file}")

    if not results_df.empty:
        print("\nSummary Statistics:")
        print(f"  Mean visible satellites: {results_df['n_visible'].mean():.1f}")
        print(f"  Mean baseline angle: {results_df['baseline_deg'].mean():.1f}°")
        print(f"  Mean CEP50: {results_df['cep50_km'].mean():.2f} km")
        print(f"  Mean CEP90: {results_df['cep90_km'].mean():.2f} km")
        print(f"  Mean GDOP: {results_df['gdop'].mean():.2f}")

        # Plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(results_df.index, results_df['n_visible'])
            axes[0, 0].set_ylabel('N Visible Satellites')
            axes[0, 0].grid(True)

            axes[0, 1].plot(results_df.index, results_df['baseline_deg'])
            axes[0, 1].set_ylabel('Baseline Angle [deg]')
            axes[0, 1].grid(True)

            axes[1, 0].plot(results_df.index, results_df['cep50_km'], label='CEP50')
            axes[1, 0].plot(results_df.index, results_df['cep90_km'], label='CEP90')
            axes[1, 0].set_ylabel('CEP [km]')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            axes[1, 1].plot(results_df.index, results_df['gdop'])
            axes[1, 1].set_ylabel('GDOP')
            axes[1, 1].grid(True)

            plt.suptitle(f'Geometric Performance Analysis - {target_name}')
            plt.tight_layout()

            plot_file = output_dir / f'geometric_analysis_{target_name}.png'
            plt.savefig(plot_file)
            print(f"\nPlot saved to: {plot_file}")
            plt.show()

        except ImportError:
            print("\n(matplotlib not available for plotting)")


if __name__ == "__main__":
    main()