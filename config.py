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
            'enabled': True,  # Toggle layer on/off for trades
            'type': 'walker',  # 'walker' or 'custom'

            # Walker Delta Pattern: i:T/P/F notation
            # T = total satellites, P = planes, F = phasing parameter
            'total_sats': 24,
            'num_planes': 4,
            'sats_per_plane': 6,  # T/P

            # Orbital Elements (all satellites in layer share these)
            'altitude_km': 550,  # km above Earth surface
            'inclination_deg': 53.0,  # degrees
            'eccentricity': 0.001,
            'arg_perigee_deg': 0.0,  # degrees

            # Phasing
            'raan_spacing': 'uniform',  # 360/P degree spacing
            'phase_offset_deg': 0.0,  # F*360/T inter-plane phasing
        },
        # {
        #     'name': 'MEO_Layer_1',
        #     'enabled': False,
        #     'type': 'walker',
        #     'total_sats': 12,
        #     'num_planes': 3,
        #     'sats_per_plane': 4,
        #     'altitude_km': 8000,
        #     'inclination_deg': 55.0,
        #     'eccentricity': 0.001,
        #     'arg_perigee_deg': 0.0,
        #     'raan_spacing': 'uniform',
        #     'phase_offset_deg': 0.0,
        # }
    ],

    # Propagation Settings
    'propagation': {
        'method': 'RK4',  # Integration method
        'include_J2': True,  # Toggle J2 perturbation
        'timestep_sec': 1.0,  # Propagation timestep
        'duration_min': 30,  # Mission duration in minutes
    },

    # Output Settings
    'output': {
        'coordinate_frame': 'ECEF',  # 'ECI' or 'ECEF'
        'save_ephemeris': True,
        'ephemeris_dir': 'output/ephemeris/',
        'file_format': 'csv',  # Export format
    }
}