# ============================================
# SPACE SEGMENT CONFIGURATION
# ============================================

# Earth Parameters (WGS84)
EARTH_PARAMS = {
    'mu': 398600.4418,  # km³/s² - Earth gravitational parameter
    'J2': 1.08262668e-3,  # J2 perturbation coefficient
    'R_E': 6378.137,  # km - Earth equatorial radius
    'omega_E': 7.2921159e-5  # rad/s - Earth rotation rate
}

# Multi-layer Constellation Definition
# Each layer can be LEO, MEO, or GEO with different Walker patterns
SPACE_SEGMENT = {
    'layers': [
        {
            'name': 'LEO_Layer_1',
            'enabled': True,
            'type': 'walker',

            # Walker: 24 sats, 6 planes x 4 sats/plane
            'total_sats': 24,
            'num_planes': 6,
            'sats_per_plane': 4,

            # Orbital Elements
            'altitude_km': 850,        # keep current baseline
            'inclination_deg': 53.0,
            'eccentricity': 0.001,
            'arg_perigee_deg': 0.0,

            # Phasing
            'raan_spacing': 'uniform',
            'phase_offset_deg': 0.0,
        },
        {
            'name': 'MEO_Layer_1',
            'enabled': True,
            'type': 'walker',

            # Walker: 15 sats, 3 planes x 5 sats/plane
            'total_sats': 15,
            'num_planes': 3,
            'sats_per_plane': 5,

            # Orbital Elements
            'altitude_km': 8000,
            'inclination_deg': 55.0,
            'eccentricity': 0.001,
            'arg_perigee_deg': 0.0,

            # Phasing
            'raan_spacing': 'uniform',
            'phase_offset_deg': 0.0,
        },
        {
            'name': 'GEO_Layer_1',
            'enabled': True,
            'type': 'walker',

            # Implement 3 GEO sats positioned at ~30°, 60°, 90° ECEF longitude
            # Trick using current initializer (no code changes):
            #   - 3 planes, 1 sat per plane
            #   - RAAN = 0,120,240 deg (from uniform spacing)
            #   - Set arg_perigee = 30 deg and per-plane mean anomaly offset = -90 deg
            #   => resulting angle (RAAN + argp + nu) ≈ 30, 60, 90 deg at t0
            'total_sats': 3,
            'num_planes': 3,
            'sats_per_plane': 1,

            # Orbital Elements for GEO
            'altitude_km': 35786,
            'inclination_deg': 0.0,
            'eccentricity': 0.0,
            'arg_perigee_deg': 30.0,

            # Phasing
            'raan_spacing': 'uniform',
            'phase_offset_deg': -90.0,
        },
        {
            'name': 'HEO_Layer_1',
            'enabled': True,
            'type': 'walker',

            # 4 HEO (Molniya-like): 2 planes x 2 sats
            'total_sats': 4,
            'num_planes': 2,
            'sats_per_plane': 2,

            # Molniya-like elements (semi-major axis via altitude_km = a - R_E)
            'altitude_km': 20184,          # a ≈ 26562 km ⇒ a - R_E ≈ 20184 km
            'inclination_deg': 63.4,
            'eccentricity': 0.74,
            'arg_perigee_deg': 270.0,

            # Phase
            'raan_spacing': 'uniform',
            'phase_offset_deg': 180.0,
        },
    ],

    # Propagation Settings
    'propagation': {
        'method': 'RK4',  # Integration method
        'include_J2': True,  # Toggle J2 perturbation
        'timestep_sec': 1.0,  # Propagation timestep
        'duration_min': 180,  # Mission duration in minutes
    },

    # Output Settings
    'output': {
        'coordinate_frame': 'ECEF',  # 'ECI' or 'ECEF'
        'save_ephemeris': True,
        'ephemeris_dir': 'exports/ephemeris/',
        'file_format': 'csv',  # Export format
    }
}