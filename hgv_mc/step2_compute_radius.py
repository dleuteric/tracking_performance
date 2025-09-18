"""
step2_compute_radius.py
-----------------------------------------------------------
Same as before, but each propagated impact point is also
required to satisfy the original down-range window
R_MIN_KM…R_MAX_KM relative to *its own* launch point.
"""

from __future__ import annotations
import sys, pickle, time, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER-TUNABLE
PICKLE       = sys.argv[1] if len(sys.argv) > 1 else "batch.pkl"
SNAP_DT      = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
SNAP_MAX     = float(sys.argv[3]) if len(sys.argv) > 3 else 800.0
SPEC_E_THRESH = 1.0e6                 # J kg-1
OUT_PNG      = "footprint_radius.png"
OUT_CSV      = "footprint_radius.csv"
# ---------------------------------------------------------------------

# physics & helpers
import footprint_evolution as fe
propagate     = fe.propagate
bank_bounds   = (fe.bank_left, fe.bank_right)
alpha_fn      = fe.alpha_bal
m_nom, S_nom  = fe.m_nom, fe.S_nom
downrange     = fe.downrange
Re            = fe.Re
R_MIN_KM, R_MAX_KM = fe.R_MIN_KM, fe.R_MAX_KM      # ← original range gate

with open(PICKLE, "rb") as f:
    accepted = pickle.load(f)
print(f"Loaded {len(accepted)} trajectories from {PICKLE}")

snap_grid = np.arange(0.0, SNAP_MAX + SNAP_DT, SNAP_DT)
footprint_radii = []

def state_at_time(T: np.ndarray, Y: np.ndarray, t_snap: float):
    k = np.searchsorted(T, t_snap, side="right") - 1
    if k < 0 or Y[k, 0] <= Re + 1.0:
        return None
    return Y[k]

t0 = time.time()
for t_snap in snap_grid:
    impacts = []
    for (_, T, Y, _R, lat0_deg, lon0_deg, _tof) in accepted:
        ys = state_at_time(T, Y, t_snap)
        if ys is None:
            continue
        # -------- specific-energy gate ---------------------------------
        h = ys[0] - Re
        v = ys[3]
        if 0.5 * v**2 + 9.80665 * h < SPEC_E_THRESH:
            continue

        # propagate under two extreme banks
        for b_fn in bank_bounds:
            T2, Y2 = propagate(ys, p=dict(m=m_nom, S=S_nom,
                                          alpha=alpha_fn, bank=b_fn))
            lon_f, lat_f = Y2[-1, 1], Y2[-1, 2]          # rad
            # -------- down-range gate (original window) ----------------
            dr_km = downrange(np.deg2rad(lon0_deg),
                              np.deg2rad(lat0_deg),
                              lon_f, lat_f)
            if not (R_MIN_KM <= dr_km <= R_MAX_KM):
                continue
            # store down-range distance relative to *this* launch site
            impacts.append(downrange(np.deg2rad(lon0_deg),
                                      np.deg2rad(lat0_deg),
                                      lon_f, lat_f))

    # ----- footprint radius: maximum residual down-range -----------------
    if impacts:
        rad_km = max(impacts)          # each element already a distance [km]
    else:
        rad_km = np.nan

    footprint_radii.append(rad_km)
    print(f"t = {t_snap:5.1f} s   radius = {rad_km:8.1f} km   "
          f"(impact pts: {len(impacts):3})")

print(f"Finished in {time.time()-t0:.1f} s")

# -------------------- save & plot ------------------------------------------
np.savetxt(OUT_CSV,
           np.column_stack([snap_grid, footprint_radii]),
           header="time_s,radius_km", delimiter=",", fmt="%.2f")

plt.figure(figsize=(7,4))
plt.plot(snap_grid, footprint_radii, marker='o')
plt.xlabel("Elapsed time [s]"); plt.ylabel("Footprint radius [km]")
plt.title("Max residual down‑range vs time")
plt.grid(True); plt.tight_layout(); plt.savefig(OUT_PNG, dpi=150)
print(f"Outputs: {OUT_CSV}, {OUT_PNG}")