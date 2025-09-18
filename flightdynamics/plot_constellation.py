"""
Constellation Visualization Tool
Plots satellite orbits in 3D and ground tracks
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from flightdynamics.space_segment import SpaceSegment


def plot_constellation_3d(space_seg):
    """Create 3D visualization of satellite constellation"""

    fig = plt.figure(figsize=(15, 6))

    # ========== 3D Orbit Plot ==========
    ax1 = fig.add_subplot(121, projection='3d')

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = config.EARTH_PARAMS['R_E'] * np.outer(np.cos(u), np.sin(v))
    y_earth = config.EARTH_PARAMS['R_E'] * np.outer(np.sin(u), np.sin(v))
    z_earth = config.EARTH_PARAMS['R_E'] * np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='lightblue')

    # Plot satellites and orbit traces
    colors = plt.cm.Set1(np.linspace(0, 1, 10))

    # Group satellites by plane for better visualization
    planes = {}
    for sat in space_seg.satellites:
        plane_key = f"{sat['layer']}_{sat['raan_deg']:.1f}"
        if plane_key not in planes:
            planes[plane_key] = []
        planes[plane_key].append(sat)

    # Plot each orbital plane
    for idx, (plane_key, sats) in enumerate(planes.items()):
        color = colors[idx % len(colors)]

        # Plot full orbit for first satellite in plane
        sat = sats[0]
        orbit_points = []

        # Generate full orbit (0-360 degrees mean anomaly)
        for M_deg in np.linspace(0, 360, 100):
            r_eci, _ = space_seg._keplerian_to_cartesian(
                sat['a_km'], sat['e'], sat['i_deg'],
                sat['raan_deg'], sat['omega_deg'], M_deg
            )
            orbit_points.append(r_eci)

        orbit_points = np.array(orbit_points)
        ax1.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2],
                 'k-', alpha=0.3, linewidth=0.5)

        # Plot satellites in this plane
        for sat in sats:
            ax1.scatter(sat['r_eci'][0], sat['r_eci'][1], sat['r_eci'][2],
                        color=color, s=50, marker='o', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_zlabel('Z [km]')
    ax1.set_title(f'3D Constellation View\n{len(space_seg.satellites)} Satellites')

    # Equal aspect ratio
    max_range = max([sat['a_km'] for sat in space_seg.satellites])
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])

    # ========== Ground Track Plot ==========
    ax2 = fig.add_subplot(122)

    # Convert to lat/lon for ground track
    for idx, (plane_key, sats) in enumerate(planes.items()):
        color = colors[idx % len(colors)]

        for sat in sats:
            # Calculate sub-satellite point (simplified - assumes spherical Earth)
            r = sat['r_eci']
            lat = np.degrees(np.arcsin(r[2] / np.linalg.norm(r)))
            lon = np.degrees(np.arctan2(r[1], r[0]))

            ax2.scatter(lon, lat, color=color, s=30, marker='o',
                        edgecolor='black', linewidth=0.5)

    # Plot Earth map grid
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-180, 180])
    ax2.set_ylim([-90, 90])
    ax2.set_xlabel('Longitude [deg]')
    ax2.set_ylabel('Latitude [deg]')
    ax2.set_title('Instantaneous Ground Track')
    ax2.set_xticks(np.arange(-180, 181, 60))
    ax2.set_yticks(np.arange(-90, 91, 30))

    plt.tight_layout()
    plt.show()


def plot_orbital_parameters(space_seg):
    """Plot histogram of orbital parameters"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Extract parameters
    altitudes = [sat['a_km'] - config.EARTH_PARAMS['R_E'] for sat in space_seg.satellites]
    inclinations = [sat['i_deg'] for sat in space_seg.satellites]
    raans = [sat['raan_deg'] for sat in space_seg.satellites]
    mean_anomalies = [sat['M_deg'] for sat in space_seg.satellites]
    velocities = [np.linalg.norm(sat['v_eci']) for sat in space_seg.satellites]
    radii = [np.linalg.norm(sat['r_eci']) for sat in space_seg.satellites]

    # Altitude distribution
    axes[0, 0].hist(altitudes, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Altitude [km]')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Altitude Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Inclination distribution
    axes[0, 1].hist(inclinations, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Inclination [deg]')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Inclination Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # RAAN distribution
    axes[0, 2].hist(raans, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('RAAN [deg]')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('RAAN Distribution (Planes)')
    axes[0, 2].grid(True, alpha=0.3)

    # Mean anomaly distribution
    axes[1, 0].hist(mean_anomalies, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Mean Anomaly [deg]')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Mean Anomaly Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Velocity magnitude
    axes[1, 1].hist(velocities, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Velocity [km/s]')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Velocity Magnitude')
    axes[1, 1].grid(True, alpha=0.3)

    # Orbital radius
    axes[1, 2].hist(radii, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Radius [km]')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Orbital Radius')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Constellation Parameters - {len(space_seg.satellites)} Satellites',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize constellation
    print("Initializing constellation...")
    space_seg = SpaceSegment()

    # Create visualizations
    print("Generating plots...")
    plot_constellation_3d(space_seg)
    plot_orbital_parameters(space_seg)

    print("Visualization complete!")