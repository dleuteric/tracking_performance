"""
Minimal HGV point-mass propagator in the Velocity-Turn-Climb (VTC) frame
based on Zhang et al. (IEEE Access 2022) Eqns. of Motion. Units: km, km/s, rad.
Aerodynamics computed in SI internally and converted back to km/s^2 accelerations.

State vector y = [r_km, lon_rad, lat_rad, v_kms, theta_rad, sigma_rad]
    r      : geocentric radius
    lon    : longitude
    lat    : latitude
    v      : speed magnitude
    theta  : velocity inclination (flight-path angle, +up)
    sigma  : velocity course / heading angle in local horizontal

Controls:
    alpha_fn(t, state) -> attack angle [rad]
    bank_fn(t, state)  -> bank angle [rad]

Next (future steps): replace placeholder aero coeffs, exponential atmos,
and constant control laws with paper's alpha(v) schedule & maneuver logic.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
Re_km = 6378.137                          # Earth equatorial radius [km]
mu_earth_km3_s2 = 398600.4418             # Earth GM [km^3/s^2]

# --- Simple exponential atmosphere (placeholder) ---
def rho_atm_kg_m3(alt_km, rho0=1.225, Hs_km=7.2):
    """Very crude scale-height model; adequate only for code shakeout."""
    return rho0 * np.exp(-np.maximum(alt_km, 0.0) / Hs_km)

# --- Placeholder aerodynamic coefficients ---
def aero_coeffs(alpha_rad):
    """
    Toy model: CL = CL_alpha * alpha; CD = CD0 + k*CL^2.
    Replace with table or fit when available.
    """
    CL_alpha = 2.0
    CD0 = 0.05
    k = 0.5
    CL = CL_alpha * alpha_rad
    CD = CD0 + k * CL**2
    return CL, CD

# --- Default control laws (constant AoA & bank) ---
def alpha_fn(t, state, alpha_deg=5.0):
    return np.deg2rad(alpha_deg)

def bank_fn(t, state, bank_deg=0.0):
    return np.deg2rad(bank_deg)

# --- Dynamics ---
def hgv_derivs(t, y, params):
    """
    Compute time-derivatives dy/dt using Zhang et al. VTC equations.
    params: dict with mass_kg, area_m2, alpha_fn, bank_fn
    """
    r, lon, lat, v, theta, sigma = y
    alt_km = r - Re_km

    # Controls
    alpha = params['alpha_fn'](t, y)
    bank  = params['bank_fn'](t, y)

    # Atmosphere & aero (convert to SI internally)
    rho = rho_atm_kg_m3(alt_km)                 # kg/m^3
    v_mps = v * 1000.0                          # m/s
    CL, CD = aero_coeffs(alpha)
    q = 0.5 * rho * v_mps**2                    # dynamic pressure [N/m^2]
    L = q * CL * params['area_m2']              # Lift [N]
    D = q * CD * params['area_m2']              # Drag [N]
    aL = (L / params['mass_kg']) / 1000.0       # km/s^2
    aD = (D / params['mass_kg']) / 1000.0       # km/s^2

    # Gravity magnitude
    g = mu_earth_km3_s2 / (r**2)                # km/s^2

    # Zhang et al. 6-state EOM (paper Eq. (1))
    r_dot     = v * np.sin(theta)
    lon_dot   = v * np.cos(theta) * np.sin(sigma) / (r * max(np.cos(lat), 1e-8))
    lat_dot   = v * np.cos(theta) * np.cos(sigma) / r
    v_dot     = -aD - g * np.sin(theta)
    theta_dot = (aL * np.cos(bank)) / v + (v * np.cos(theta)) / r - (g * np.cos(theta)) / v
    sigma_dot = (aL * np.sin(bank)) / (v * max(np.cos(theta), 1e-8)) \
                + (v * np.cos(theta) * np.sin(sigma) * np.tan(lat)) / r

    return np.array([r_dot, lon_dot, lat_dot, v_dot, theta_dot, sigma_dot])

# --- RK4 Integrator ---
def rk4_step(fun, t, y, dt, params):
    k1 = fun(t, y, params)
    k2 = fun(t + 0.5*dt, y + 0.5*dt*k1, params)
    k3 = fun(t + 0.5*dt, y + 0.5*dt*k2, params)
    k4 = fun(t + dt,     y + dt*k3,     params)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_hgv(t0, tf, dt, y0, params):
    n = int(np.ceil((tf - t0)/dt)) + 1
    T = np.zeros(n)
    Y = np.zeros((n, len(y0)))
    t, y = t0, np.array(y0, dtype=float)
    for i in range(n):
        T[i], Y[i] = t, y
        if t >= tf or y[0] <= Re_km:  # stop if impact
            T = T[:i+1]; Y = Y[:i+1]
            break
        y = rk4_step(hgv_derivs, t, y, dt, params)
        t += dt
    return T, Y

# --- Self-test / example run ---
if __name__ == "__main__":
    params = dict(mass_kg=1000.0, area_m2=2.0,
                  alpha_fn=alpha_fn, bank_fn=bank_fn)

    # Initial condition: 80 km altitude, equator, eastward, shallow -5 deg glide, 7 km/s.
    y0 = [Re_km + 80.0, 0.0, 0.0, 7.0, np.deg2rad(-5.0), np.deg2rad(90.0)]
    t0, tf, dt = 0.0, 60.0, 0.5  # 60 s demo

    T, Y = integrate_hgv(t0, tf, dt, y0, params)

    alt_km = Y[:,0] - Re_km
    print(f"Final t = {T[-1]:.1f} s, alt = {alt_km[-1]:.2f} km, v = {Y[-1,3]:.3f} km/s")

    # Quick look plot
    plt.figure()
    plt.plot(T, alt_km)
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [km]")
    plt.title("HGV Altitude vs Time (demo)")
    plt.grid(True)
    plt.show()
