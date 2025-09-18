# core/mission_requirements.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class SystemRequirements:
    altitude_range_km: tuple = (500, 1500)
    coverage_fractions: tuple = (0.5, 1.0)
    max_gsd: float = 1500
    max_edge_ifov: float = 450
    min_fov_deg: float = 50
    pixel_pitch_m: float = 15e-6
    # etc. ...
    # Detector formats, sensor wavelength, etc.
    formats: Dict[str, int] = None

    def __post_init__(self):
        if self.formats is None:
            self.formats = {
                "1K": 1024,
                "2K": 2048,
                "4K": 4096
            }