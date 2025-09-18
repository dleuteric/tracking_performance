"""
hgv_bleeding_analysis.py
------------------------------------------------------------
Energy‑budget feasibility check for a single launch → target
scenario, including optional manoeuvre schedule.

Usage (quick default):
    python hgv_bleeding_analysis.py --plot

Or simply run the file in PyCharm: the PARAMETERS block at the top is used.

Custom example:
    python hgv_bleeding_analysis.py \
        --lat0 39.65 --lon0 9.65  \
        --latf 34.00 --lonf 18.00 \
        --h0 100e3  --v0 5500     \
        --man Jump-W             \
        --plot

The script:

1. Computes great‑circle baseline range R_GC.
2. Estimates minimum required specific energy
       E_min = g0 * R_GC / (L/D)
3. Propagates the full 6‑DoF point‑mass model (imports *footprint_evolution*),
   applying the requested manoeuvre schedule.
4. Accumulates drag work  ∫ D v dt  →  E_bleed.
5. Reports residual energy  E_res  and verdict.

Author: Hyper HGV SME • July 2025
"""

from __future__ import annotations
import argparse, math, sys, json, pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D proj)

g0 = 9.80665
L_OVER_D = 1.5                              # constant heuristic

# ---------------------------------------------------------------------- #
# USER‑TUNABLE DEFAULTS  (edit here when you don't want to pass CLI flags)
P = dict(
    LAT0 = 52,        # launch latitude  [deg]
    LON0 =  13,        # launch longitude [deg]
    LATF = 41.00,        # target latitude  [deg]
    LONF = 13.00,        # target longitude [deg]
    H0   = 100e3,        # initial altitude [m]
    V0   = 4500.0,       # initial speed    [m/s]
    MAN  = "Bal-No",    # manoeuvre schedule label
    PLOT = True          # whether to produce plots
)
# ---------------------------------------------------------------------- #

import importlib, types
fe = importlib.import_module("footprint_evolution")
fe.NUM_TRAJ = 0          # stop Monte‑Carlo generation
fe.accepted = []         # clear any leftover list

# ---------------------------------------------------------------------- #
#  ISA‑based speed of sound (same helper used in step‑3 script)
def a_sound(h_m: float) -> float:
    if h_m < 11e3:
        T = 288.15 - 0.0065 * h_m
    elif h_m < 20e3:
        T = 216.65
    elif h_m < 32e3:
        T = 216.65 + 0.001 * (h_m - 20e3)
    elif h_m < 47e3:
        T = 228.65 + 0.0028 * (h_m - 32e3)
    else:
        T = 270.0
    return math.sqrt(1.4 * 287.05 * T)

# ---------------------------------------------------------------------- #
def great_circle_km(lat0, lon0, latf, lonf) -> float:
    """Great‑circle distance in km."""
    lo1, la1, lo2, la2 = map(math.radians, [lon0, lat0, lonf, latf])
    c = math.acos(
        np.clip(math.sin(la1) * math.sin(la2) +
                math.cos(la1) * math.cos(la2) * math.cos(lo2 - lo1), -1, 1))
    return fe.Re * c / 1e3

def energy_required(range_km: float) -> float:
    """Min specific energy [J/kg] to cover range_km with constant L/D."""
    return g0 * (range_km * 1e3) / L_OVER_D

# ---------------------------------------------------------------------- #
def propagate_with_drag(y0, alpha_fn, bank_fn, t_max=4000.0):
    """Wrapper around fe.propagate that accumulates drag energy bleed."""
    t, dt, y = 0.0, fe.TIME_STEP, y0.copy()
    E_bleed = 0.0
    while t < t_max and y[0] > fe.Re + 1.0:
        # aerodynamic coefficients
        a = alpha_fn(t, y)
        rho = fe.rho_atm(y[0] - fe.Re)
        CL, CD = fe.aero_coeffs(a)
        q = 0.5 * rho * y[3] ** 2
        D = q * CD * fe.S_nom
        # RK4 integration
        y_new, thdot = fe.rk4(fe.hgv_dot, t, y, dt,
                              dict(alpha=alpha_fn, bank=bank_fn,
                                   m=fe.m_nom, S=fe.S_nom))
        E_bleed += (D / fe.m_nom) * y[3] * dt   # J/kg
        # adaptive dt (same logic as original)
        if abs(thdot) > 0.05:
            dt *= 0.5
            continue
        y, t = y_new, t + dt
        dt = min(dt * 1.2, fe.TIME_STEP)
    return y, E_bleed, t

# ---------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Energy‑margin check for HGV with scheduled manoeuvres.")
    p.add_argument("--lat0", type=float, default=P["LAT0"],
                   help="launch latitude  [deg]  (default 39.65)")
    p.add_argument("--lon0", type=float, default=P["LON0"],
                   help="launch longitude [deg]  (default 9.65)")
    p.add_argument("--latf", type=float, default=P["LATF"],
                   help="target latitude  [deg]  (default 34.0)")
    p.add_argument("--lonf", type=float, default=P["LONF"],
                   help="target longitude [deg] (default 18.0)")
    p.add_argument("--h0",  type=float, default=P["H0"],
                   help="initial altitude [m]")
    p.add_argument("--v0",  type=float, default=P["V0"],
                   help="initial speed [m/s]")
    p.add_argument("--man", "--schedule", default=P["MAN"],
                   help="label of manoeuvre schedule (e.g. 'Jump-W')")
    p.add_argument("--plot", action="store_true", default=P["PLOT"])
    # User can tweak launch/target via --lat0 --lon0 --latf --lonf
    # These act as "user gates" for the analysis.
    args = p.parse_args(argv)


    # --- baseline range & energy ----------------------------------------
    R_gc = great_circle_km(args.lat0, args.lon0, args.latf, args.lonf)
    E_min = energy_required(R_gc)
    print(f"Great‑circle range  R_gc = {R_gc:,.0f} km")
    print(f"Min specific energy E_min = {E_min/1e6:.2f} MJ kg⁻¹ "
          f"(L/D={L_OVER_D})")

    # --- initial energy --------------------------------------------------
    E0 = 0.5 * args.v0 ** 2 + g0 * args.h0
    print(f"Initial energy      E0    = {E0/1e6:.2f} MJ kg⁻¹")
    if E0 < E_min:
        print("❌  Infeasible even without manoeuvres.")
        sys.exit(1)

    # --- pick manoeuvre law ---------------------------------------------
    man_dict = {lab: (a_fn, b_fn) for lab, a_fn, b_fn in fe.man}
    if args.man not in man_dict:
        print(f"Unknown manoeuvre '{args.man}'. Available: {list(man_dict.keys())}")
        sys.exit(2)
    alpha_fn, bank_fn = man_dict[args.man]

    # --- build initial state vector -------------------------------------
    y0 = np.array([fe.Re + args.h0,
                   math.radians(args.lon0),
                   math.radians(args.lat0),
                   args.v0,
                   math.radians(-5.0),      # shallow flight path
                   math.radians(0.0)])      # heading along great‑circle (approx)

    # --- propagate -------------------------------------------------------
    y_end, E_bleed, tof = propagate_with_drag(y0, alpha_fn, bank_fn)
    E_res = E0 - E_bleed
    print(f"Drag bleed energy   E_bleed = {E_bleed/1e6:.2f} MJ kg⁻¹")
    print(f"Residual energy     E_res   = {E_res/1e6:.2f} MJ kg⁻¹")
    print(f"Time of flight      ToF     = {tof:.0f} s")

    verdict = "✅ Feasible — margin!" if E_res >= E_min else "⚠️ Undershoot risk!"
    print(verdict)

    # --- optional quick-look plot ---------------------------------------
    if args.plot:
        # recompute full trajectory (pos history) for plotting
        T_hist, Y_hist = fe.propagate(y0, p=dict(alpha=alpha_fn, bank=bank_fn,
                                                m=fe.m_nom, S=fe.S_nom))
        h_hist = Y_hist[:,0] - fe.Re
        a_hist = np.vectorize(a_sound)(h_hist)
        mach_hist = Y_hist[:,3] / a_hist
        plt.figure(figsize=(7,4))
        plt.plot(T_hist, mach_hist)
        plt.xlabel("Time [s]"); plt.ylabel("Mach [-]")
        plt.title(f"Mach vs time  ({args.man})")
        plt.grid(True); plt.tight_layout()
        out_png = pathlib.Path("bleed_profile.png")
        plt.savefig(out_png, dpi=120)
        print(f"Plot saved to {out_png}")

        # ---- ground track -------------------------------------------------
        lon_deg = np.degrees(Y_hist[:,1])
        lat_deg = np.degrees(Y_hist[:,2])
        plt.figure(figsize=(6,5))
        plt.plot(lon_deg, lat_deg, lw=1.5, color="#0072B2")
        plt.scatter([lon_deg[0]],[lat_deg[0]],color="green",label="Launch")
        plt.scatter([lon_deg[-1]],[lat_deg[-1]],color="red",label="Impact")
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude  [deg]")
        plt.title("Ground track")
        plt.grid(True, linestyle=":")
        plt.legend()
        gt_png = pathlib.Path("ground_track.png")
        plt.tight_layout(); plt.savefig(gt_png, dpi=120)
        print(f"Ground‑track plot saved to {gt_png}")

        # ---- altitude vs down‑range --------------------------------------
        dr_km = [fe.downrange(Y_hist[0,1], Y_hist[0,2], lo, la)
                 for lo, la in zip(Y_hist[:,1], Y_hist[:,2])]
        alt_km = (Y_hist[:,0]-fe.Re)/1e3
        plt.figure(figsize=(6,4))
        plt.plot(dr_km, alt_km, lw=1.5, color="#D55E00")
        plt.xlabel("Down‑range [km]")
        plt.ylabel("Altitude [km]")
        plt.title("Altitude vs down‑range")
        plt.grid(True, linestyle=":")
        ad_png = pathlib.Path("alt_vs_range.png")
        plt.tight_layout(); plt.savefig(ad_png, dpi=120)
        print(f"Alt‑vs‑range plot saved to {ad_png}")

        # ---- energy bleed curve ------------------------------------------
        E_hist = 0.5 * Y_hist[:,3]**2 + g0 * h_hist      # specific energy J/kg
        plt.figure(figsize=(7,4))
        plt.plot(T_hist, E_hist/1e6, color="#009E73")
        plt.xlabel("Time [s]")
        plt.ylabel("Specific energy [MJ/kg]")
        plt.title("Energy bleed vs time")
        plt.grid(True, linestyle=":")
        en_png = pathlib.Path("energy_vs_time.png")
        plt.tight_layout(); plt.savefig(en_png, dpi=120)
        print(f"Energy‑vs‑time plot saved to {en_png}")

        # ---- 3‑D trajectory -------------------------------------------------
        x_ecef, y_ecef, z_ecef = fe.ecef(Y_hist[:,0], Y_hist[:,1], Y_hist[:,2])
        fig3d = plt.figure(figsize=(7,7))
        ax3d = fig3d.add_subplot(111, projection='3d')
        # Earth translucent sphere
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = fe.Re * np.outer(np.cos(u), np.sin(v))
        ys = fe.Re * np.outer(np.sin(u), np.sin(v))
        zs = fe.Re * np.outer(np.ones_like(u), np.cos(v))
        ax3d.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                          color='lightblue', alpha=0.3, linewidth=0, shade=True)
        # trajectory
        ax3d.plot(x_ecef, y_ecef, z_ecef, color="#D55E00", lw=1.5)
        ax3d.set_box_aspect([1,1,1])
        ax3d.set_title("3‑D trajectory vs Earth")
        tri3d_png = pathlib.Path("trajectory_3d.png")
        fig3d.tight_layout(); fig3d.savefig(tri3d_png, dpi=120)
        print(f"3‑D plot saved to {tri3d_png}")

        # ---- remaining range estimate --------------------------------------
        # Using simple E = g0*R/(L/D)  ->  R = E * (L/D) / g0
        R_rem_km = (E_hist * L_OVER_D / g0) / 1e3
        plt.figure(figsize=(7,4))
        plt.plot(T_hist, R_rem_km, color="#CC79A7")
        plt.xlabel("Time [s]")
        plt.ylabel("Remaining range [km]")
        plt.title("Remaining theoretical range vs time")
        plt.grid(True, linestyle=":")
        rr_png = pathlib.Path("remaining_range.png")
        plt.tight_layout(); plt.savefig(rr_png, dpi=120)
        print(f"Remaining‑range plot saved to {rr_png}")

        plt.show()
        print("🖼️  Plots displayed and PNGs saved in current directory.")

if __name__ == "__main__":
    main()