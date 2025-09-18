# plots.py

import matplotlib.pyplot as plt

def plot_fov_sweep(fov_range, results):
    """
    Plot results from the FOV sweep
    """
    gsd_edge = [r['gsd_edge_km'] for r in results]
    ifov_edge = [r['ifov_edge_urad'] for r in results]

    plt.figure(figsize=(12, 6))

    # Left subplot: Edge GSD vs FOV
    plt.subplot(121)
    plt.plot(fov_range, gsd_edge, 'b-')
    plt.grid(True)
    plt.xlabel('FOV (degrees)')
    plt.ylabel('Edge GSD (km)')
    plt.title('Edge GSD vs FOV')

    # Right subplot: Edge IFOV vs FOV
    plt.subplot(122)
    plt.plot(fov_range, ifov_edge, 'r-')
    plt.grid(True)
    plt.xlabel('FOV (degrees)')
    plt.ylabel('Edge IFOV (μrad)')
    plt.title('Edge IFOV vs FOV')

    plt.tight_layout()
    plt.show()