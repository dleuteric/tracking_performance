# geometric_performance.py
"""
Geometric Performance Module for LEO Constellation Tracking
Based on Sanders-Reed (2001) error propagation for two-sensor triangulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from itertools import combinations
import warnings

# Constants
R_EARTH_KM = 6378.137
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


@dataclass
class SensorMeasurement:
    """Single sensor measurement at a given epoch"""
    sat_id: str
    position: np.ndarray  # [x, y, z] km in ICRF
    los_unit: np.ndarray  # [ux, uy, uz] unit vector to target
    azimuth: float  # rad
    elevation: float  # rad
    range_to_target: float  # km


@dataclass
class GeometricPerformance:
    """Performance metrics for a sensor pair"""
    pair_ids: Tuple[str, str]
    baseline_angle_deg: float  # stereo angle between sensors
    target_estimate: np.ndarray  # [x, y, z] km
    covariance_matrix: np.ndarray  # 3x3 covariance matrix
    cep50_km: float  # Circular Error Probable (50%)
    cep90_km: float  # CEP 90%
    cep95_km: float  # CEP 95%
    rmse_x: float
    rmse_y: float
    rmse_z: float
    gdop: float  # Geometric Dilution of Precision
    condition_number: float


class TriangulationGeometry:
    """
    Implements Sanders-Reed triangulation and error propagation
    for two-sensor angles-only position estimation
    """

    def __init__(self, sigma_los_rad: float = 1e-3, sigma_elevation_rad: float = 1e-3):
        """
        Initialize with measurement noise parameters

        Args:
            sigma_los_rad: Standard deviation of LOS azimuth measurement (rad)
            sigma_elevation_rad: Standard deviation of elevation measurement (rad)
        """
        self.sigma_los = sigma_los_rad
        self.sigma_elev = sigma_elevation_rad

    def compute_azimuth_elevation(self, r_sensor: np.ndarray, los_unit: np.ndarray) -> Tuple[float, float]:
        """
        Compute azimuth and elevation angles from sensor position and LOS

        Following Sanders-Reed convention:
        - Azimuth: measured clockwise from positive y-axis
        - Elevation: measured from x-y plane
        """
        # Project LOS to local horizontal frame at sensor
        r_norm = r_sensor / np.linalg.norm(r_sensor)

        # Local frame: z_local = radial up, x_local = velocity direction proxy
        z_local = r_norm
        x_local = np.array([0, 0, 1]) - z_local * z_local[2]
        x_local /= np.linalg.norm(x_local) if np.linalg.norm(x_local) > 0 else 1
        y_local = np.cross(z_local, x_local)

        # Transform LOS to local frame
        los_local = np.array([
            np.dot(los_unit, x_local),
            np.dot(los_unit, y_local),
            np.dot(los_unit, z_local)
        ])

        # Compute angles
        azimuth = np.arctan2(los_local[0], los_local[1])
        elevation = np.arcsin(np.clip(los_local[2], -1, 1))

        return azimuth, elevation

    def triangulate_pair(self, s1: SensorMeasurement, s2: SensorMeasurement) -> Tuple[np.ndarray, float]:
        """
        Triangulate target position from two sensors using least squares

        Returns:
            target_position: Estimated [x,y,z] position
            condition_number: Numerical conditioning of the solution
        """
        # Build least squares system A*x = b
        # Following Sanders-Reed Eq. (4) adapted for 3D
        P1 = np.eye(3) - np.outer(s1.los_unit, s1.los_unit)  # Projection perpendicular to LOS1
        P2 = np.eye(3) - np.outer(s2.los_unit, s2.los_unit)  # Projection perpendicular to LOS2

        A = P1 + P2
        b = P1 @ s1.position + P2 @ s2.position

        # Solve with condition number check
        eigenvalues = np.linalg.eigvals(A)
        cond = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)) if np.min(np.abs(eigenvalues)) > 0 else np.inf

        try:
            x_hat = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Degenerate case - sensors aligned
            x_hat = 0.5 * (s1.position + s2.position)

        return x_hat, cond

    def compute_error_propagation(self, s1: SensorMeasurement, s2: SensorMeasurement,
                                  x_hat: np.ndarray) -> np.ndarray:
        """
        Compute 3x3 covariance matrix using Sanders-Reed error propagation

        Returns:
            Sigma: 3x3 covariance matrix (km²)
        """
        # Angular separation (key geometry parameter)
        theta_sep = s1.azimuth - s2.azimuth

        # Avoid singularity near 0° and 180°
        if np.abs(np.sin(theta_sep)) < 0.01:  # ~0.6 degrees
            warnings.warn(f"Near-singular geometry: separation angle = {theta_sep * RAD2DEG:.1f}°")
            return 1e6 * np.eye(3)  # Large uncertainty

        # Range from each sensor to estimated target
        r1 = np.linalg.norm(x_hat - s1.position)
        r2 = np.linalg.norm(x_hat - s2.position)

        # Error propagation coefficients (Sanders-Reed Section 2.3)
        # Simplified for equal measurement noise on both sensors

        # Azimuth error coefficient (grows with range and poor geometry)
        k_azimuth = 1.0 / np.abs(np.sin(theta_sep))

        # Position error variance contributions
        # X-Y plane errors (dominated by azimuth uncertainty)
        var_xy = (self.sigma_los ** 2) * ((r1 ** 2 + r2 ** 2) / 2) * (k_azimuth ** 2)

        # Z error (dominated by elevation uncertainty)
        var_z = (self.sigma_elev ** 2) * ((r1 ** 2 + r2 ** 2) / 2)

        # Build covariance matrix
        # Simplified: assume uncorrelated x,y,z errors
        Sigma = np.diag([var_xy, var_xy, var_z])

        # Add geometric factor for off-diagonal terms based on viewing geometry
        cos_sep = np.cos(theta_sep)
        Sigma[0, 1] = Sigma[1, 0] = var_xy * cos_sep * 0.1  # Small correlation

        return Sigma

    def compute_baseline_angle(self, los1: np.ndarray, los2: np.ndarray) -> float:
        """Compute baseline (stereo) angle between two LOS vectors in degrees"""
        cos_angle = np.clip(np.dot(los1, los2), -1, 1)
        return np.arccos(cos_angle) * RAD2DEG

    def compute_gdop(self, covariance: np.ndarray) -> float:
        """Compute Geometric Dilution of Precision from covariance matrix"""
        return np.sqrt(np.trace(covariance))

    def compute_cep(self, covariance: np.ndarray, confidence: float = 0.5) -> float:
        """
        Compute Circular Error Probable for given confidence level

        Args:
            covariance: 2x2 or 3x3 covariance matrix (uses first 2 dimensions)
            confidence: Confidence level (0.5 for CEP50, 0.9 for CEP90, etc.)
        """
        # Extract 2D covariance (x-y plane)
        Sigma_2d = covariance[:2, :2]

        # Compute equivalent circular standard deviation
        sigma_eq = np.sqrt(0.5 * np.trace(Sigma_2d))

        # Scale factor for confidence level (Rayleigh distribution)
        if confidence == 0.5:
            k = 1.1774  # CEP50
        elif confidence == 0.9:
            k = 2.1460  # CEP90
        elif confidence == 0.95:
            k = 2.4477  # CEP95
        else:
            # General formula: k = sqrt(-2 * ln(1 - confidence))
            k = np.sqrt(-2 * np.log(1 - confidence))

        return k * sigma_eq


class GeometricPerformanceModule:
    """
    Main module for evaluating geometric performance of satellite constellation
    """

    def __init__(self, sigma_los_mrad: float = 1.0, min_elevation_deg: float = 5.0,
                 max_range_km: float = 3000.0):
        """
        Initialize performance module

        Args:
            sigma_los_mrad: LOS measurement noise in milliradians
            min_elevation_deg: Minimum elevation angle for visibility
            max_range_km: Maximum sensor range
        """
        self.triangulator = TriangulationGeometry(
            sigma_los_rad=sigma_los_mrad * 1e-3,
            sigma_elevation_rad=sigma_los_mrad * 1e-3
        )
        self.min_elevation = min_elevation_deg
        self.max_range = max_range_km

    def check_visibility(self, r_sat: np.ndarray, los_unit: np.ndarray,
                         r_target: np.ndarray) -> Tuple[bool, str]:
        """
        Comprehensive visibility check

        Returns:
            (is_visible, reason_if_not)
        """
        # Check LOS validity
        los_norm = np.linalg.norm(los_unit)
        if los_norm < 0.99 or np.any(np.isnan(los_unit)):
            return False, "invalid_los"

        # Range check
        range_km = np.linalg.norm(r_target - r_sat)
        if range_km > self.max_range:
            return False, f"range_{range_km:.0f}km"

        # Earth occlusion
        if self._earth_blocks_ray(r_sat, r_target):
            return False, "earth_occluded"

        # Elevation angle check
        _, elevation = self.triangulator.compute_azimuth_elevation(r_sat, los_unit)
        elev_deg = elevation * RAD2DEG
        if elev_deg < self.min_elevation:
            return False, f"low_elev_{elev_deg:.1f}deg"

        return True, "visible"

    def _earth_blocks_ray(self, r_sat: np.ndarray, r_target: np.ndarray) -> bool:
        """Check if Earth blocks the ray from satellite to target"""
        d = r_target - r_sat
        rho = np.linalg.norm(d)
        if rho <= 0:
            return True

        u = d / rho
        proj = np.dot(r_sat, u)

        # Closest approach distance to Earth center
        lam_star = -proj
        if lam_star <= 0 or lam_star >= rho:
            return False

        d_min_sq = np.dot(r_sat, r_sat) - proj * proj
        return d_min_sq <= R_EARTH_KM ** 2

    def process_epoch(self, satellite_positions: Dict[str, np.ndarray],
                      los_vectors: Dict[str, np.ndarray],
                      true_target: Optional[np.ndarray] = None) -> Dict:
        """
        Process single epoch to compute geometric performance

        Args:
            satellite_positions: {sat_id: [x,y,z]} positions in ICRF
            los_vectors: {sat_id: [ux,uy,uz]} unit LOS vectors
            true_target: True target position for error computation

        Returns:
            Dictionary with performance metrics
        """
        # Filter visible satellites
        visible_sats = []
        visibility_stats = {}

        for sat_id in satellite_positions:
            r_sat = satellite_positions[sat_id]
            los = los_vectors.get(sat_id)

            if los is None:
                continue

            # Use true target for visibility if available
            r_tgt = true_target if true_target is not None else r_sat + 1000 * los
            is_vis, reason = self.check_visibility(r_sat, los, r_tgt)

            if is_vis:
                az, el = self.triangulator.compute_azimuth_elevation(r_sat, los)
                visible_sats.append(SensorMeasurement(
                    sat_id=sat_id,
                    position=r_sat,
                    los_unit=los,
                    azimuth=az,
                    elevation=el,
                    range_to_target=np.linalg.norm(r_tgt - r_sat)
                ))

            visibility_stats[reason] = visibility_stats.get(reason, 0) + 1

        if len(visible_sats) < 2:
            return {
                'n_visible': len(visible_sats),
                'visibility_stats': visibility_stats,
                'pairs': []
            }

        # Process all pairs
        pair_results = []
        for s1, s2 in combinations(visible_sats, 2):
            # Triangulate
            x_hat, cond = self.triangulator.triangulate_pair(s1, s2)

            # Compute error covariance
            Sigma = self.triangulator.compute_error_propagation(s1, s2, x_hat)

            # Compute metrics
            baseline = self.triangulator.compute_baseline_angle(s1.los_unit, s2.los_unit)

            perf = GeometricPerformance(
                pair_ids=(s1.sat_id, s2.sat_id),
                baseline_angle_deg=baseline,
                target_estimate=x_hat,
                covariance_matrix=Sigma,
                cep50_km=self.triangulator.compute_cep(Sigma, 0.5),
                cep90_km=self.triangulator.compute_cep(Sigma, 0.9),
                cep95_km=self.triangulator.compute_cep(Sigma, 0.95),
                rmse_x=np.sqrt(Sigma[0, 0]),
                rmse_y=np.sqrt(Sigma[1, 1]),
                rmse_z=np.sqrt(Sigma[2, 2]),
                gdop=self.triangulator.compute_gdop(Sigma),
                condition_number=cond
            )

            # Add true error if available
            if true_target is not None:
                error = x_hat - true_target
                perf.error_x = error[0]
                perf.error_y = error[1]
                perf.error_z = error[2]
                perf.error_3d = np.linalg.norm(error)

            pair_results.append(perf)

        # Find best pair (minimum CEP50)
        if pair_results:
            best_pair = min(pair_results, key=lambda p: p.cep50_km)
        else:
            best_pair = None

        return {
            'n_visible': len(visible_sats),
            'visibility_stats': visibility_stats,
            'pairs': pair_results,
            'best_pair': best_pair,
            'baseline_angles': [p.baseline_angle_deg for p in pair_results]
        }

    def compute_constellation_metrics(self, results_over_time: List[Dict]) -> Dict:
        """
        Aggregate metrics over time for constellation performance assessment
        """
        if not results_over_time:
            return {}

        # Extract time series of best performances
        cep50_series = []
        cep90_series = []
        baseline_series = []
        n_visible_series = []

        for result in results_over_time:
            if result['best_pair']:
                cep50_series.append(result['best_pair'].cep50_km)
                cep90_series.append(result['best_pair'].cep90_km)
                baseline_series.extend(result['baseline_angles'])
            n_visible_series.append(result['n_visible'])

        metrics = {
            'mean_cep50_km': np.mean(cep50_series) if cep50_series else np.nan,
            'median_cep50_km': np.median(cep50_series) if cep50_series else np.nan,
            'p95_cep50_km': np.percentile(cep50_series, 95) if cep50_series else np.nan,
            'mean_cep90_km': np.mean(cep90_series) if cep90_series else np.nan,
            'mean_baseline_deg': np.mean(baseline_series) if baseline_series else np.nan,
            'optimal_baseline_fraction': np.mean(
                [1 for b in baseline_series if 60 <= b <= 120]) if baseline_series else 0,
            'mean_n_visible': np.mean(n_visible_series),
            'availability': np.mean([1 for n in n_visible_series if n >= 2])
        }

        return metrics


# Example usage function
def evaluate_constellation_geometry(sat_ephemeris: pd.DataFrame,
                                    los_data: pd.DataFrame,
                                    target_truth: pd.DataFrame,
                                    config: Dict) -> pd.DataFrame:
    """
    Main evaluation function compatible with existing pipeline

    Args:
        sat_ephemeris: Satellite positions over time
        los_data: LOS measurements
        target_truth: True target trajectory
        config: Configuration parameters

    Returns:
        DataFrame with performance metrics over time
    """
    # Initialize module
    module = GeometricPerformanceModule(
        sigma_los_mrad=config.get('sigma_los_mrad', 1.0),
        min_elevation_deg=config.get('min_elevation_deg', 5.0),
        max_range_km=config.get('max_range_km', 3000.0)
    )

    results = []

    # Process each epoch
    for timestamp in target_truth.index:
        # Gather satellite data at this epoch
        sat_positions = {}
        los_vectors = {}

        for sat_id in sat_ephemeris['sat_id'].unique():
            sat_data = sat_ephemeris[sat_ephemeris['sat_id'] == sat_id]
            if timestamp in sat_data.index:
                sat_positions[sat_id] = sat_data.loc[timestamp, ['x_km', 'y_km', 'z_km']].values

                los_subset = los_data[(los_data['sat_id'] == sat_id) & (los_data.index == timestamp)]
                if not los_subset.empty:
                    los_vectors[sat_id] = los_subset[['ux', 'uy', 'uz']].values[0]

        # Get true target position
        true_pos = target_truth.loc[timestamp, ['x_km', 'y_km', 'z_km']].values

        # Process epoch
        epoch_result = module.process_epoch(sat_positions, los_vectors, true_pos)

        # Store results
        if epoch_result['best_pair']:
            bp = epoch_result['best_pair']
            results.append({
                'time': timestamp,
                'n_visible': epoch_result['n_visible'],
                'best_pair': f"{bp.pair_ids[0]}-{bp.pair_ids[1]}",
                'baseline_deg': bp.baseline_angle_deg,
                'cep50_km': bp.cep50_km,
                'cep90_km': bp.cep90_km,
                'cep95_km': bp.cep95_km,
                'rmse_x_km': bp.rmse_x,
                'rmse_y_km': bp.rmse_y,
                'rmse_z_km': bp.rmse_z,
                'gdop': bp.gdop,
                'error_3d_km': bp.error_3d if hasattr(bp, 'error_3d') else np.nan
            })

    return pd.DataFrame(results)


# Configurazione
config = {
    'sigma_los_mrad': 1.0,      # Rumore LOS in milliradianti
    'min_elevation_deg': 5.0,    # Elevazione minima
    'max_range_km': 3000.0       # Range massimo sensore IR
}
