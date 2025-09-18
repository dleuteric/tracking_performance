# footprint_evolution.py  (pseudo–Python 3.10+)
import numpy as np
from pathlib import Path
from hgv_generate_save_plot import propagate, downrange, man, Re

# 1) draw full Monte-Carlo as usual → accepted = [...]
accepted = run_full_monte_carlo(NUM_TRAJ=2000)

# 2) choose snapshot times (e.g. every 10 s)
snap_grid = np.arange(0, 800, 10)  # seconds

footprint_radii = []
for t_snap in snap_grid:
    impact_pts = []
    for lab, T, Y, *_ in accepted:
        # find state closest to snapshot
        idx = np.searchsorted(T, t_snap, side="right") - 1
        if idx < 0 or Y[idx,0] <= Re:   # already on ground
            continue
        y_snap = Y[idx]
        # cheap continuation: two extreme bank schedules
        impacts = []
        for bank_profile in (bank_left, bank_right):   # max lift & min lift
            T2, Y2 = propagate(y_snap, p=dict(m=m_nom, S=S_nom,
                                              alpha=alpha_bal,
                                              bank=bank_profile))
            impacts.append((Y2[-1,1], Y2[-1,2]))  # (lon, lat)
        impact_pts.extend(impacts)

    # 3) compute circle radius that encloses all impact_pts
    lon0, lat0 = np.mean(np.array(impact_pts), axis=0)
    radii_km = [downrange(lon0, lat0, lo, la) for lo, la in impact_pts]
    footprint_radii.append(max(radii_km))

# 4) plot radius shrink vs time
import matplotlib.pyplot as plt
plt.plot(snap_grid, footprint_radii)
plt.xlabel('Elapsed time [s]')
plt.ylabel('Footprint radius [km]')
plt.title('Collapse of reachable impact region')
plt.grid(True); plt.show()