# analysis.py

import numpy as np
from .calculations import analyze_wfov_design

def sweep_fov_range(start=40, stop=60, steps=50,
                    altitude_km=1000, pixel_pitch_um=15, array_size=4096):
    """
    Sweep FOV in degrees from 'start' to 'stop' and return a list of results
    """
    fov_range = np.linspace(start, stop, steps)
    results = []

    for fov in fov_range:
        res = analyze_wfov_design(
            altitude_km=altitude_km,
            fov_deg=fov,
            pixel_pitch_um=pixel_pitch_um,
            array_size=array_size
        )
        results.append(res)

    return fov_range, results

