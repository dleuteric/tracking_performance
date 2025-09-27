"""
Space Segment Module - Keplerian Propagator with J2 Perturbations
Generates satellite ephemeris for multi-layer constellations
Fast vectorized propagation for architecture trade studies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SpaceSegment:
    """Fast Keplerian propagator for constellation ephemeris generation"""

    def __init__(self):
        """Initialize space segment from config"""
        self.earth = config.EARTH_PARAMS
        self.config = config.SPACE_SEGMENT
        self.satellites = []  # List to store all satellite states

        # Initialize constellation from config
        self._initialize_constellation()

    def _initialize_constellation(self):
        """Create initial satellite states from config layers"""
        sat_id = 0

        for layer in self.config['layers']:
            if not layer['enabled']:
                continue

            print(f"Initializing {layer['name']}: {layer['total_sats']} satellites")

            # Convert altitude to semi-major axis
            a_km = self.earth['R_E'] + layer['altitude_km']  # km

            # Generate Walker constellation
            for plane_idx in range(layer['num_planes']):
                # RAAN for this plane (uniform spacing)
                raan_deg = (360.0 / layer['num_planes']) * plane_idx

                for sat_in_plane in range(layer['sats_per_plane']):
                    # Mean anomaly with inter-plane phasing
                    M_deg = (360.0 / layer['sats_per_plane']) * sat_in_plane
                    M_deg += layer['phase_offset_deg'] * plane_idx  # Walker phasing

                    # Create satellite dictionary
                    sat = {
                        'id': f"{layer['name']}_SAT_{sat_id:03d}",
                        'layer': layer['name'],
                        'a_km': a_km,
                        'e': layer['eccentricity'],
                        'i_deg': layer['inclination_deg'],
                        'raan_deg': raan_deg,
                        'omega_deg': layer['arg_perigee_deg'],
                        'M_deg': M_deg % 360.0,  # Initial mean anomaly
                    }

                    # Convert to Cartesian state (ECI)
                    r_eci, v_eci = self._keplerian_to_cartesian(
                        sat['a_km'], sat['e'], sat['i_deg'],
                        sat['raan_deg'], sat['omega_deg'], sat['M_deg']
                    )

                    sat['r_eci'] = r_eci  # km
                    sat['v_eci'] = v_eci  # km/s

                    self.satellites.append(sat)
                    sat_id += 1

        print(f"Total satellites initialized: {len(self.satellites)}")

    def _keplerian_to_cartesian(self, a_km, e, i_deg, raan_deg, omega_deg, M_deg):
        """
        Convert Keplerian elements to Cartesian state vectors (ECI frame)

        Args:
            a_km: Semi-major axis [km]
            e: Eccentricity [-]
            i_deg: Inclination [degrees]
            raan_deg: Right Ascension of Ascending Node [degrees]
            omega_deg: Argument of perigee [degrees]
            M_deg: Mean anomaly [degrees]

        Returns:
            r_eci: Position vector [km] in ECI frame
            v_eci: Velocity vector [km/s] in ECI frame
        """
        # Convert degrees to radians
        i = np.radians(i_deg)
        raan = np.radians(raan_deg)
        omega = np.radians(omega_deg)
        M = np.radians(M_deg)

        # Solve Kepler's equation: M = E - e*sin(E)
        # Newton-Raphson iteration
        E = M  # Initial guess
        for _ in range(10):  # Usually converges in 3-5 iterations
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))

        # True anomaly from eccentric anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + e) * np.sin(E / 2),
            np.sqrt(1 - e) * np.cos(E / 2)
        )

        # Distance from focus
        r = a_km * (1 - e * np.cos(E))

        # Position in orbital plane
        r_orbital = r * np.array([np.cos(nu), np.sin(nu), 0])

        # Velocity in orbital plane (vis-viva equation)
        h = np.sqrt(self.earth['mu'] * a_km * (1 - e ** 2))  # Specific angular momentum
        v_orbital = np.array([
            -self.earth['mu'] / h * np.sin(nu),
            self.earth['mu'] / h * (e + np.cos(nu)),
            0
        ])

        # Rotation matrix from orbital plane to ECI
        # R = R_z(-Ω) * R_x(-i) * R_z(-ω)
        cos_raan, sin_raan = np.cos(raan), np.sin(raan)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_omega, sin_omega = np.cos(omega), np.sin(omega)

        # Combined rotation matrix (orbital to ECI)
        R = np.array([
            [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
             -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
             sin_raan * sin_i],
            [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
             -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
             -cos_raan * sin_i],
            [sin_omega * sin_i,
             cos_omega * sin_i,
             cos_i]
        ])

        # Transform to ECI frame
        r_eci = R @ r_orbital
        v_eci = R @ v_orbital

        return r_eci, v_eci

    def test_initialization(self):
        """Quick test to verify constellation initialization"""
        print(f"\n{'=' * 50}")
        print("CONSTELLATION INITIALIZATION TEST")
        print(f"{'=' * 50}")

        # Summary statistics
        for layer_name in set(sat['layer'] for sat in self.satellites):
            layer_sats = [s for s in self.satellites if s['layer'] == layer_name]
            altitudes = [s['a_km'] - self.earth['R_E'] for s in layer_sats]

            print(f"\n{layer_name}:")
            print(f"  Satellites: {len(layer_sats)}")
            print(f"  Altitude: {np.mean(altitudes):.1f} km")
            print(f"  First satellite ID: {layer_sats[0]['id']}")
            print(f"  Position (ECI): {layer_sats[0]['r_eci']} km")
            print(f"  |r|: {np.linalg.norm(layer_sats[0]['r_eci']):.1f} km")
            print(f"  |v|: {np.linalg.norm(layer_sats[0]['v_eci']):.3f} km/s")


# Test the module
if __name__ == "__main__":
    space_seg = SpaceSegment()
    space_seg.test_initialization()