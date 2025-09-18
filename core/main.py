# main.py

from core.calculations import analyze_wfov_design
from core.analysis import sweep_fov_range
from core.plots import plot_fov_sweep

def main():
    # 1. Single run with defaults
    result = analyze_wfov_design()
    print("\n=== WFOV Design Analysis ===")
    print(f"Focal Length: {result['focal_length_m']:.3f} m")
    print(f"GSD (center): {result['gsd_center_km']:.3f} km")
    print(f"GSD (edge): {result['gsd_edge_km']:.3f} km")
    print(f"IFOV (center): {result['ifov_center_urad']:.1f} μrad")
    print(f"IFOV (edge): {result['ifov_edge_urad']:.1f} μrad")
    print(f"Ground Coverage: {result['ground_coverage_km']:.1f} km")
    print(f"Required Aperture: {result['required_aperture_m']:.3f} m")
    print(f"F/#: {result['f_number']:.1f}")

    # 2. Parameter sweep across FOV
    fov_range, results = sweep_fov_range(start=40, stop=60, steps=50,
                                         altitude_km=1000,
                                         pixel_pitch_um=15,
                                         array_size=4096)

    # 3. Plot results of the sweep
    plot_fov_sweep(fov_range, results)

if __name__ == "__main__":
    main()