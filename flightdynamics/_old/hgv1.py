"""
HGV propagator v2: dynamic AoA & bank from Zhang et al., IEEE Access 2022
 - Attack‑angle model: Eq.(4–7), p.21096  [balanced→jumping glide transition]
 - C‑turn bank model: text preceding Eq.(8), p.21098  [constant bank]
"""

import numpy as np, matplotlib.pyplot as plt

# -------- Constants --------
Re_km, muE = 6378.137, 398600.4418       # Earth radius [km], GM [km^3/s^2]
rho0, Hs = 1.225, 7.2                    # Simple exponential atmos

# -------- Helper models --------
def rho_atm(h_km):                       # crude density, as before
    return rho0 * np.exp(-np.maximum(h_km, 0)/Hs)

def aero_coeffs(alpha):                  # same toy CL/CD (future: table)
    CL_alpha, CD0, k = 2.0, 0.05, 0.5
    CL = CL_alpha * alpha
    return CL, CD0 + k*CL**2

# -------- Longitudinal control: attack‑angle schedule (Eq.4, p.21096) --------
alpha_max_deg, alpha_ld_deg = 15.0, 2.5  # αmax & αmax(L/D)  – pick reasonable demo values
v1, v2 = 6.0, 3.0                        # km/s speed break‑points (paper Table 2 spans 3–7 km s⁻¹)

def alpha_fn(t, state):
    _, _, _, v, _, _ = state
    if v > v1:
        return np.deg2rad(alpha_max_deg)                     # α = αmax  [Eq.4 case 1]
    elif v < v2:
        return np.deg2rad(alpha_ld_deg)                      # α = αmax(L/D)  [Eq.4 case 3]
    # sinusoidal blend, Eq.(4) case 2
    alpha_mid = 0.5*(alpha_max_deg + alpha_ld_deg)           # Eq.(5)
    alpha_bal = 0.5*(alpha_max_deg - alpha_ld_deg)           # Eq.(6)
    v_mid = 0.5*(v1 + v2)                                    # Eq.(7)
    alpha = alpha_mid + alpha_bal * np.sin(np.pi*(v - v_mid)/(v1 - v2))
    return np.deg2rad(alpha)

# -------- Lateral control: constant C‑turn bank (text before Eq.8, p.21098) --------
bank_deg = 20.0
def bank_fn(t, state): return np.deg2rad(bank_deg)

# -------- Dynamics (Eq.1, p.21096) – identical to v1 snippet --------
def hgv_dot(t, y, p):
    r, lon, lat, v, th, sig = y
    h = r - Re_km
    alpha, bank = p['alpha'](t, y), p['bank'](t, y)
    rho = rho_atm(h); v_m = v*1000
    CL, CD = aero_coeffs(alpha); q = 0.5*rho*v_m**2
    aL = q*CL*p['S']/p['m']/1000;  aD = q*CD*p['S']/p['m']/1000
    g = muE/r**2
    rdot   = v*np.sin(th)
    londot = v*np.cos(th)*np.sin(sig)/(r*np.cos(lat)+1e-9)
    latdot = v*np.cos(th)*np.cos(sig)/r
    vdot   = -aD - g*np.sin(th)
    thdot  = aL*np.cos(bank)/v + v*np.cos(th)/r - g*np.cos(th)/v
    sigdot = aL*np.sin(bank)/(v*np.cos(th)+1e-9) + v*np.cos(th)*np.sin(sig)*np.tan(lat)/r
    return np.array([rdot, londot, latdot, vdot, thdot, sigdot])

# -------- RK4 integrator (unchanged) --------
def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p)
    k2 = fun(t+0.5*h, y+0.5*h*k1, p)
    k3 = fun(t+0.5*h, y+0.5*h*k2, p)
    k4 = fun(t+h, y+h*k3, p)
    return y + h*(k1+2*k2+2*k3+k4)/6

def propagate(t0, tf, dt, y0, p):
    ts, ys = [t0], [y0.copy()]
    t, y = t0, y0.copy()
    while t < tf and y[0] > Re_km:
        y = rk4(hgv_dot, t, y, dt, p); t += dt
        ts.append(t); ys.append(y.copy())
    return np.array(ts), np.vstack(ys)


# ---------- 3‑D plotting + down‑range helper (Zhang Fig.2, p.21097) ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D)

def ecef_xyz(y_arr):
    """Convert state array [r,lon,lat,…] to ECEF km."""
    r, lon, lat = y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def plot_3d_ecef(T, Y):
    x, y, z = ecef_xyz(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=1.2, label='HGV path')
    ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=40, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='r', marker='x', s=40, label='End')
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
    ax.set_title('3‑D ECEF HGV Trajectory')
    ax.legend(); plt.show()

def downrange_km(Y):
    """Great‑circle distance start→end (km)."""
    lat1, lon1 = Y[0, 2], Y[0, 1]
    lat2, lon2 = Y[-1, 2], Y[-1, 1]
    c = np.arccos(np.clip(np.sin(lat1)*np.sin(lat2) +
                          np.cos(lat1)*np.cos(lat2)*np.cos(lon2 - lon1), -1, 1))
    return Re_km * c



# -------- Demo run --------
if __name__ == "__main__":
    pars = dict(m=1000.0, S=2.0, alpha=alpha_fn, bank=bank_fn)
    y0 = np.array([Re_km+100, 0, 0, 7.0, np.deg2rad(-5), np.deg2rad(90)])
    T, Y = propagate(0, 1200, 0.5, y0, pars)                  # 2‑min flight
    alt = Y[:,0]-Re_km; v = Y[:,3]*1000
    print(f"End: t={T[-1]:.1f}s  h={alt[-1]:.1f} km  v={Y[-1,3]:.3f} km/s")
    fig, ax1 = plt.subplots()
    ax1.plot(T, alt); ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Altitude [km]')
    ax2 = ax1.twinx(); ax2.plot(T, np.rad2deg([alpha_fn(0,t) for t in Y]), '--')
    ax2.set_ylabel('Attack angle [deg]')
    plt.title('HGV skip‐glide with C‑turn'); plt.show()

    # --- existing altitude/alpha plot here ---
    plot_3d_ecef(T, Y)                                     # new 3‑D view
    print(f"Approx. down‑range = {downrange_km(Y):.0f} km")
