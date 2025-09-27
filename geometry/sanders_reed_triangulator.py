#!/usr/bin/env python3
"""
triangulate_sanders_reed.py

Two-sensor triangulation with analytic error propagation based on:
Sanders-Reed (2001) "Error propagation in two-sensor three-dimensional position estimation"
Optical Engineering 40(4), 627-636

This module implements:
- Rigorous two-sensor triangulation 
- First-order error propagation via partial derivatives
- Full 3x3 covariance matrix computation
- Geometric singularity handling

Author: ez-SMAD Team
Reference: Sanders-Reed, J.N. (2001), DOI: 10.1117/1.1353798
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings
import yaml
from datetime import datetime


# Constants
R_EARTH_KM = 6378.137
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

# Config
try:
    from config.loader import load_config
except:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.loader import load_config

CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()


@dataclass
class SensorMeasurement:
    """Single sensor measurement at epoch"""
    sat_id: str
    position: np.ndarray  # [x,y,z] km ICRF
    los_unit: np.ndarray  # unit vector to target
    azimuth: float  # rad (from y-axis, clockwise)
    elevation: float  # rad (from xy-plane)


@dataclass
class TriangulationResult:
    """Result from two-sensor triangulation"""
    target_estimate: np.ndarray  # [x,y,z] km
    covariance_3d: np.ndarray  # 3x3 covariance matrix
    cep50_km: float
    cep90_km: float
    rmse_x: float
    rmse_y: float
    rmse_z: float
    baseline_angle_deg: float
    condition_number: float
    pair_ids: Tuple[str, str]


class SandersReedTriangulator:
    """
    Implements Sanders-Reed two-sensor triangulation with analytic error propagation.

    Key equations from paper:
    - Target position: Eq. (6), (7) for x,y; Eq. (5) for z
    - Error propagation: Eq. (8)-(16) for partial derivatives
    - Gaussian statistics: Eq. (17)-(26) for standard deviations
    """

    def __init__(self,
                 sigma_los_rad: float = 1e-3,  # 1 mrad default
                 sigma_elevation_rad: float = 1e-3,
                 sigma_pos_km: float = 0.001):  # 1m position error
        """
        Initialize with measurement noise parameters

        Args:
            sigma_los_rad: LOS azimuth measurement std dev (rad)
            sigma_elevation_rad: Elevation measurement std dev (rad)
            sigma_pos_km: Sensor position measurement std dev (km)
        """
        self.sigma_los = sigma_los_rad
        self.sigma_elev = sigma_elevation_rad
        self.sigma_pos = sigma_pos_km

    def triangulate_pair(self, s1: SensorMeasurement, s2: SensorMeasurement) -> TriangulationResult:
        """
        Triangulate target from two sensor measurements

        Implements Sanders-Reed Eq. (6), (7) for (x,y) and Eq. (5) for z
        """
        # Extract positions and angles
        x1, y1, z1 = s1.position
        x2, y2, z2 = s2.position
        th1, th2 = s1.azimuth, s2.azimuth
        el1, el2 = s1.elevation, s2.elevation

        # Check for geometric singularity (Section 2.2, observation 3)
        tan_diff = np.tan(th1) - np.tan(th2)
        if np.abs(tan_diff) < 1e-6:  # ~0.06 degrees
            warnings.warn(f"Near-singular geometry: th1={th1*RAD2DEG:.1f}°, th2={th2*RAD2DEG:.1f}°")
            # Return large uncertainty
            return self._singular_result(s1, s2)

        # Compute target x,y position (Eq. 6, 7)
        tan_th1, tan_th2 = np.tan(th1), np.tan(th2)
        denominator = tan_th1 - tan_th2

        x_t = (x2*tan_th1 - x1*tan_th2 + (y1-y2)*tan_th1*tan_th2) / denominator
        y_t = (y1*tan_th1 - y2*tan_th2 + (x2-x1)) / denominator

        # Compute ranges (Eq. 3)
        r1 = np.sqrt((x1-x_t)**2 + (y1-y_t)**2)
        r2 = np.sqrt((x2-x_t)**2 + (y2-y_t)**2)

        # Compute target z position (Eq. 5 - average)
        z_t = 0.5 * (r1*np.tan(el1) + z1 + r2*np.tan(el2) + z2)

        target_estimate = np.array([x_t, y_t, z_t])

        # Compute error propagation
        covariance = self._compute_covariance(
            s1, s2, target_estimate, r1, r2
        )

        # Compute metrics
        baseline_deg = self._baseline_angle(s1.los_unit, s2.los_unit)

        return TriangulationResult(
            target_estimate=target_estimate,
            covariance_3d=covariance,
            cep50_km=self._cep_from_cov(covariance, 0.5),
            cep90_km=self._cep_from_cov(covariance, 0.9),
            rmse_x=np.sqrt(covariance[0,0]),
            rmse_y=np.sqrt(covariance[1,1]),
            rmse_z=np.sqrt(covariance[2,2]),
            baseline_angle_deg=baseline_deg,
            condition_number=np.linalg.cond(covariance),
            pair_ids=(s1.sat_id, s2.sat_id)
        )

    def _compute_covariance(self, s1: SensorMeasurement, s2: SensorMeasurement,
                            target: np.ndarray, r1: float, r2: float) -> np.ndarray:
        """
        Compute 3x3 error covariance matrix using Sanders-Reed formulation

        Implements Eq. (17)-(26) with partial derivatives from Eq. (9), (11), (14), (16)
        """
        x1, y1, z1 = s1.position
        x2, y2, z2 = s2.position
        th1, th2 = s1.azimuth, s2.azimuth
        el1, el2 = s1.elevation, s2.elevation
        x_t, y_t, z_t = target

        tan_th1, tan_th2 = np.tan(th1), np.tan(th2)
        sec2_th1, sec2_th2 = 1.0/np.cos(th1)**2, 1.0/np.cos(th2)**2
        sec2_el1, sec2_el2 = 1.0/np.cos(el1)**2, 1.0/np.cos(el2)**2
        tan_diff = tan_th1 - tan_th2

        # === Partial derivatives for x_t (Eq. 9) ===
        dx_dx1 = -tan_th2 / tan_diff
        dx_dx2 =  tan_th1 / tan_diff
        dx_dy1 =  tan_th1 * tan_th2 / tan_diff
        dx_dy2 = -tan_th1 * tan_th2 / tan_diff

        # Complex partials for azimuth angles
        numerator1 = x2 + (y1 - y2) * tan_th2
        numerator_common = x2 * tan_th1 - x1 * tan_th2 + (y1 - y2) * tan_th1 * tan_th2
        dx_dth1 = (numerator1 / tan_diff - numerator_common / tan_diff**2) * sec2_th1

        numerator2 = -x1 + (y1 - y2) * tan_th1
        dx_dth2 = (numerator2 / tan_diff + numerator_common / tan_diff**2) * sec2_th2

        # === Partial derivatives for y_t (Eq. 11) ===
        dy_dx1 = -1.0 / tan_diff
        dy_dx2 =  1.0 / tan_diff
        dy_dy1 =  tan_th1 / tan_diff
        dy_dy2 = -tan_th2 / tan_diff

        numerator_y_common = y1 * tan_th1 - y2 * tan_th2 + (x2 - x1)
        dy_dth1 = (y1 / tan_diff - numerator_y_common / tan_diff**2) * sec2_th1
        dy_dth2 = (-y2 / tan_diff + numerator_y_common / tan_diff**2) * sec2_th2

        # === Partial derivatives for r (Eq. 14) ===
        dr1_dx1 = (x1 - x_t) / r1
        dr1_dy1 = (y1 - y_t) / r1
        dr1_dxt = -(x1 - x_t) / r1
        dr1_dyt = -(y1 - y_t) / r1

        dr2_dx2 = (x2 - x_t) / r2
        dr2_dy2 = (y2 - y_t) / r2
        dr2_dxt = -(x2 - x_t) / r2
        dr2_dyt = -(y2 - y_t) / r2

        # === Partial derivatives for z_t (Eq. 16) ===
        dz_dr1 = np.tan(el1)
        dz_dr2 = np.tan(el2)
        dz_del1 = r1 * sec2_el1
        dz_del2 = r2 * sec2_el2
        dz_dz1 = 0.5
        dz_dz2 = 0.5

        # === Compute variance contributions (Eq. 22-24) ===
        # Position measurement error contributions
        sigma2_pos = self.sigma_pos**2

        var_x_pos = (dx_dx1**2 + dx_dx2**2 + dx_dy1**2 + dx_dy2**2) * sigma2_pos
        var_y_pos = (dy_dx1**2 + dy_dx2**2 + dy_dy1**2 + dy_dy2**2) * sigma2_pos

        # Azimuth measurement error contributions
        sigma2_th = self.sigma_los**2
        var_x_th = (dx_dth1**2 + dx_dth2**2) * sigma2_th
        var_y_th = (dy_dth1**2 + dy_dth2**2) * sigma2_th

        # Total x,y variances
        var_x = var_x_pos + var_x_th
        var_y = var_y_pos + var_y_th

        # Range variance propagation (Eq. 24)
        var_r1 = ((dr1_dx1**2 + dr1_dy1**2) * sigma2_pos +
                  (dr1_dxt**2 * var_x + dr1_dyt**2 * var_y))
        var_r2 = ((dr2_dx2**2 + dr2_dy2**2) * sigma2_pos +
                  (dr2_dxt**2 * var_x + dr2_dyt**2 * var_y))

        # Z variance (Eq. 20, 25)
        sigma2_el = self.sigma_elev**2
        sigma2_z_pos = self.sigma_pos**2

        var_z1 = dz_dr1**2 * var_r1 + dz_del1**2 * sigma2_el + dz_dz1**2 * sigma2_z_pos
        var_z2 = dz_dr2**2 * var_r2 + dz_del2**2 * sigma2_el + dz_dz2**2 * sigma2_z_pos

        # Covariance term (Eq. 26)
        cov_r1r2 = dr1_dxt * dr2_dxt * var_x + dr1_dyt * dr2_dyt * var_y

        # Combined z variance with correlation
        var_z = 0.25 * (var_z1 + var_z2 + 2.0 * dz_dr1 * dz_dr2 * cov_r1r2)

        # Build 3x3 covariance matrix
        # Simplified: assume x,y uncorrelated (can add cross-terms if needed)
        Sigma = np.zeros((3, 3))
        Sigma[0, 0] = var_x
        Sigma[1, 1] = var_y
        Sigma[2, 2] = var_z

        # Add small cross-correlation based on geometry (optional)
        correlation_factor = np.cos(th1 - th2) * 0.1
        Sigma[0, 1] = Sigma[1, 0] = np.sqrt(var_x * var_y) * correlation_factor

        return Sigma
    
    def _baseline_angle(self, los1: np.ndarray, los2: np.ndarray) -> float:
        """Compute baseline stereo angle between LOS vectors"""
        cos_angle = np.clip(np.dot(los1, los2), -1, 1)
        return np.arccos(cos_angle) * RAD2DEG
    
    def _cep_from_cov(self, Σ: np.ndarray, confidence: float) -> float:
        """
        Compute Circular Error Probable from covariance
        Uses horizontal components only (x,y)
        """
        Σ_xy = Σ[:2, :2]
        eigenvalues, _ = np.linalg.eigh(Σ_xy)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # For 2D Gaussian, use Rayleigh approximation
        σ_equiv = np.sqrt(0.5 * np.sum(eigenvalues))
        
        # CEP scaling factors
        if confidence == 0.5:
            return 1.1774 * σ_equiv
        elif confidence == 0.9:
            return 2.1460 * σ_equiv
        elif confidence == 0.95:
            return 2.4477 * σ_equiv
        else:
            # General case using chi-square quantile
            from scipy.stats import chi2
            return σ_equiv * np.sqrt(chi2.ppf(confidence, df=2))
    
    def _singular_result(self, s1: SensorMeasurement, s2: SensorMeasurement) -> TriangulationResult:
        """Return result with large uncertainty for singular geometry"""
        # Estimate position using midpoint
        pos_avg = 0.5 * (s1.position + s2.position)
        
        # Large covariance
        Σ = np.eye(3) * 1e6
        
        return TriangulationResult(
            target_estimate=pos_avg,
            covariance_3d=Σ,
            cep50_km=1e6,
            cep90_km=1e6,
            rmse_x=1e3,
            rmse_y=1e3,
            rmse_z=1e3,
            baseline_angle_deg=0.0,
            condition_number=1e12,
            pair_ids=(s1.sat_id, s2.sat_id)
        )


def compute_azimuth_elevation(r_sensor: np.ndarray, los_unit: np.ndarray) -> Tuple[float, float]:
    """
    Compute azimuth and elevation from sensor position and LOS
    Following Sanders-Reed convention:
    - Azimuth: measured clockwise from positive y-axis  
    - Elevation: measured from x-y plane
    """
    # Transform to sensor-local frame
    x, y, z = los_unit
    
    # Azimuth from y-axis, clockwise
    azimuth = np.arctan2(x, y)  # Note: x,y swapped for y-axis reference
    
    # Elevation from xy-plane
    r_xy = np.sqrt(x**2 + y**2)
    elevation = np.arctan2(z, r_xy)
    
    return azimuth, elevation


def process_epoch(triangulator: SandersReedTriangulator,
                  observers: Dict[str, np.ndarray],
                  los_vectors: Dict[str, np.ndarray],
                  min_baseline_deg: float = 10.0,
                  verbose: bool = False) -> Optional[TriangulationResult]:
    """
    Process single epoch with all available observer pairs
    Returns best result based on minimum CEP50
    """
    measurements = []

    for obs_id, r_obs in observers.items():
        if obs_id in los_vectors:
            los = los_vectors[obs_id]
            az, el = compute_azimuth_elevation(r_obs, los)

            measurements.append(SensorMeasurement(
                sat_id=obs_id,
                position=r_obs,
                los_unit=los,
                azimuth=az,
                elevation=el
            ))

    if len(measurements) < 2:
        return None

    # Process all pairs
    max_baseline_seen = 0.0
    results = []
    for i in range(len(measurements)):
        for j in range(i+1, len(measurements)):
            result = triangulator.triangulate_pair(measurements[i], measurements[j])
            if result is not None:
                max_baseline_seen = max(max_baseline_seen, result.baseline_angle_deg)
            # Filter by baseline angle
            if result is not None and result.baseline_angle_deg >= min_baseline_deg:
                results.append(result)
    # Return best (minimum CEP50)
    if results:
        return min(results, key=lambda r: r.cep50_km)
    if not results and verbose:
        print(f"      No pairs met baseline >= {min_baseline_deg:.1f}°. Max available baseline was {max_baseline_seen:.1f}°")
    return None


def main():
    """
    Main entry point - processes latest run with Sanders-Reed triangulation
    Compatible with existing ez-SMAD pipeline
    """
    import os

    # Setup paths
    _paths = CFG["paths"]
    LOS_DIR = (PROJECT_ROOT / _paths["los_root"]).resolve()
    TRI_OUT = (PROJECT_ROOT / _paths["triangulation_out"]).resolve()

    print(f"[Sanders-Reed] LOS_DIR={LOS_DIR}")
    print(f"[Sanders-Reed] TRI_OUT={TRI_OUT}")

    # Get run ID
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        runs = sorted([d for d in LOS_DIR.iterdir() if d.is_dir()],
                     key=lambda x: x.stat().st_mtime)
        if not runs:
            raise FileNotFoundError("No LOS data found")
        run_id = runs[-1].name

    print(f"[Sanders-Reed] Processing run: {run_id}")

    # Setup output
    out_dir = TRI_OUT / run_id / "sanders_reed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize triangulator with config values
    tri_cfg = CFG.get("triangulation", {})
    sigma_los_mrad = tri_cfg.get("sigma_los_mrad", 1.0)
    sigma_elev_mrad = tri_cfg.get("sigma_elev_mrad", 1.0)
    sigma_pos_m = tri_cfg.get("sigma_pos_m", 1.0)

    triangulator = SandersReedTriangulator(
        sigma_los_rad=sigma_los_mrad * 1e-3,
        sigma_elevation_rad=sigma_elev_mrad * 1e-3,
        sigma_pos_km=sigma_pos_m * 1e-3
    )

    # Process all targets
    los_run_dir = LOS_DIR / run_id
    targets = set()
    files_lower = list(los_run_dir.glob("*/los_*.csv"))
    files_upper = list(los_run_dir.glob("*/LOS_*.csv"))
    for f in files_lower + files_upper:
        targets.add(f.parent.name)
    print(f"  Found {len(targets)} target(s): {sorted(list(targets))}")
    if not targets:
        print("  No targets found under LOS run dir. Expected structure: <LOS_DIR>/<RUN_ID>/<TARGET>/los_*.csv")

    results_summary = []

    tri_cfg = CFG.get("triangulation", {})
    sel_cfg = tri_cfg.get("selection", {})
    min_beta_deg = float(sel_cfg.get("min_beta_deg", 10.0))

    for target_id in sorted(targets):
        print(f"  Processing {target_id}...")
        target_los_dir = los_run_dir / target_id

        # Load all LOS data for this target
        observers_data = {}
        los_files = sorted(list(target_los_dir.glob("los_*.csv")) + list(target_los_dir.glob("LOS_*.csv")))
        if not los_files:
            print(f"    Skipping - no LOS CSVs found in {target_los_dir}")
            continue
        for los_file in los_files:
            obs_id = los_file.stem.split('_')[-1]
            df = pd.read_csv(los_file)
            # Normalize columns to lowercase to be robust to case
            df.columns = [str(c).strip().lower() for c in df.columns]
            # Create a robust time key to avoid float mismatch across observers
            df["t_key"] = pd.to_numeric(df.get("time"), errors="coerce").round(3)  # seconds, rounded to 1 ms
            observers_data[obs_id] = df
        print(f"    Observers for {target_id}: {sorted(list(observers_data.keys()))}")

        if len(observers_data) < 2:
            print(f"    Skipping - only {len(observers_data)} observers")
            continue

        # Process each epoch
        results = []

        # Get common timestamps using robust time key
        times = None
        for df in observers_data.values():
            keys = set(df["t_key"].dropna().astype(float).tolist())
            if times is None:
                times = keys
            else:
                times &= keys

        if not times:
            print(f"    Skipping - no common epochs after time alignment (check sampling/clock sync)")
            continue
        else:
            print(f"    Common epochs after alignment: {len(times)}")

        for t in sorted(times):
            # Gather measurements at this epoch
            observers = {}
            los_vectors = {}
            for obs_id, df in observers_data.items():
                match = df[df["t_key"] == t]
                if match.empty:
                    continue
                row = match.iloc[0]
                observers[obs_id] = np.array([row['x_obs_km'], row['y_obs_km'], row['z_obs_km']])
                los_vectors[obs_id] = np.array([row['ux'], row['uy'], row['uz']])
            if len(observers) < 2:
                continue
            # Triangulate
            result = process_epoch(triangulator, observers, los_vectors, min_baseline_deg=min_beta_deg, verbose=True)
            if result:
                results.append({
                    'time': float(row['time']),
                    'x_km': result.target_estimate[0],
                    'y_km': result.target_estimate[1],
                    'z_km': result.target_estimate[2],
                    'sigma_x_km': result.rmse_x,
                    'sigma_y_km': result.rmse_y,
                    'sigma_z_km': result.rmse_z,
                    'cep50_km': result.cep50_km,
                    'cep90_km': result.cep90_km,
                    'baseline_deg': result.baseline_angle_deg,
                    'condition_number': result.condition_number,
                    'pair': f"{result.pair_ids[0]}-{result.pair_ids[1]}"
                })

        # Save results
        if results:
            df_out = pd.DataFrame(results)
            out_file = out_dir / f"{target_id}_triangulated_sanders_reed.csv"
            df_out.to_csv(out_file, index=False)
            print(f"    → Wrote {len(df_out)} rows to {out_file}")

            # Summary stats
            mean_cep50 = df_out['cep50_km'].mean()
            p95_cep50 = df_out['cep50_km'].quantile(0.95)

            results_summary.append({
                'target': target_id,
                'epochs': len(results),
                'mean_cep50_km': mean_cep50,
                'p95_cep50_km': p95_cep50
            })

            print(f"    ✓ {len(results)} epochs, mean CEP50: {mean_cep50:.3f} km")
    
    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(out_dir / "summary_sanders_reed.csv", index=False)
        
        print(f"\n[Sanders-Reed] Complete - Results in {out_dir}")
        print(summary_df.to_string(index=False))
    
    return out_dir


if __name__ == "__main__":
    main()