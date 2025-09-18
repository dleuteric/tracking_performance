# calculations.py

import numpy as np

def analyze_wfov_design(altitude_km=1000,
                        fov_deg=50,
                        pixel_pitch_um=15,
                        array_size=4096):
    """
    Analyze WFOV design based on real parameters
    Returns a dictionary of optical properties.
    """
    altitude = altitude_km * 1000
    pixel_pitch = pixel_pitch_um * 1e-6
    half_fov_rad = np.radians(fov_deg / 2)

    # Detector width
    detector_width = array_size * pixel_pitch
    focal_length = detector_width / (2 * np.tan(half_fov_rad))

    # GSD
    gsd_center = (pixel_pitch * altitude) / focal_length
    gsd_edge = gsd_center / np.cos(half_fov_rad)

    # IFOV (center/edge)
    ifov_center = (pixel_pitch / focal_length) * 1e6
    ifov_edge = ifov_center / np.cos(half_fov_rad)

    # Coverage
    ground_coverage = 2 * altitude * np.tan(half_fov_rad) / 1000

    # Simple diffraction estimate (SWIR)
    wavelength = 1.5e-6
    min_aperture = 1.22 * wavelength * altitude / gsd_center

    # Choose F/# guidance
    min_f_number = 1 / (2 * np.tan(half_fov_rad))
    suggested_f_number = max(2.9, min_f_number)
    required_aperture = focal_length / suggested_f_number

    return {
        "focal_length_m": focal_length,
        "gsd_center_km": gsd_center / 1000,
        "gsd_edge_km": gsd_edge / 1000,
        "ifov_center_urad": ifov_center,
        "ifov_edge_urad": ifov_edge,
        "ground_coverage_km": ground_coverage,
        "min_aperture_m": min_aperture,
        "required_aperture_m": required_aperture,
        "f_number": suggested_f_number,
        "feasible": True  # You can refine feasibility logic later
    }

