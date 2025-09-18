import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plot

# User-defined down-range window (km) - set to ensure >=1000 km
R_MIN_KM, R_MAX_KM = 1000, 3000

# Constants & environment (from baseline)
Re = 6.378e6; mu = 3.986e14; omega_e = 7.292e-5
m_nom, S_nom = 907.0, 0.4839
def rho_atm(h):
    if h < 25e3:   Hs = 7.2e3
    elif h < 50e3: Hs = 6.6e3
    elif h < 75e3: Hs = 6.0e3
    else:          Hs = 5.6e3
    return 1.225 * np.exp(-h / Hs)
def aero_coeffs(a): CL_a, CD0, k = 1.8, 0.05, 0.5; CL = CL_a * a; return CL, CD0 + k * CL**2

# EOM (from baseline)
def hgv_dot(t, y, p):
    r, lon, lat, v, th, sig = y; h = r - Re; a, b = p['alpha'](t, y), p['bank'](t, y)
    rho = rho_atm(h); CL, CD = aero_coeffs(a); q = 0.5 * rho * v**2
    aL = q * CL * p['S'] / p['m']; aD = q * CD * p['S'] / p['m']; g = mu / r**2
    rdot = v * np.sin(th)
    vdot = -aD - g * np.sin(th) + omega_e**2 * r * np.cos(lat) * (np.sin(th) * np.cos(lat) - np.cos(th) * np.sin(sig) * np.sin(lat))
    thdot = (aL * np.cos(b) / v) + (v / r - g / v) * np.cos(th) + 2 * omega_e * np.cos(sig) * np.cos(lat) + (omega_e**2 * r / v) * np.cos(lat) * (np.cos(th) * np.cos(lat) + np.sin(th) * np.sin(sig) * np.sin(lat))
    londot = v * np.cos(th) * np.sin(sig) / (r * np.cos(lat) + 1e-9)
    latdot = v * np.cos(th) * np.cos(sig) / r
    sigdot = (aL * np.sin(b) / (v * np.cos(th) + 1e-9)) + (v / r) * np.cos(th) * np.sin(sig) * np.tan(lat) - 2 * omega_e * (np.tan(th) * np.cos(lat) * np.sin(sig) - np.sin(lat)) + (omega_e**2 * r / (v * np.cos(th) + 1e-9)) * np.sin(sig) * np.cos(lat) * np.sin(lat)
    return np.array([rdot, londot, latdot, vdot, thdot, sigdot])

# RK4 (corrected to return y_new, k1 for adaptive dt)
def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p)
    k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, p)
    k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, p)
    k4 = fun(t + h, y + h * k3, p)
    y_new = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_new, k1  # Now returns both

# Adaptive propagator (from baseline, now matches unpack)
def propagate(t0, tf, dt0, y0, p):
    t, dt, y = t0, dt0, y0.copy(); T, Y = [t0], [y0.copy()]
    while t < tf and y[0] > Re:
        y_new, k1 = rk4(hgv_dot, t, y, dt, p)  # Unpack two values
        if abs(k1[4]) > 0.05: dt *= 0.5; continue  # Adaptive on thdot
        y, t = y_new, t + dt; T.append(t); Y.append(y.copy()); dt = min(dt * 1.2, dt0)
    return np.array(T), np.vstack(Y)

# Downrange great-circle distance (km) (from baseline)
def downrange(lon1, lat1, lon2, lat2):
    c = np.arccos(np.clip(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1), -1, 1))
    return Re * c / 1e3

# Complex maneuver: Jump-Weave (jumping glide + weaving lateral)
def alpha_jump(t, y):
    v_km = y[3] / 1e3; a_max, a_ld, v1, v2 = 15, 2.5, 6, 3
    if v_km > v1: return np.deg2rad(a_max)
    if v_km < v2: return np.deg2rad(a_ld)
    a_mid = 0.5 * (a_max + a_ld); a_amp = 0.5 * (a_max - a_ld); v_mid = 0.5 * (v1 + v2)
    return np.deg2rad(a_mid + a_amp * np.sin(np.pi * (v_km - v_mid) / (v1 - v2)))

def bank_weave(t, y): return np.deg2rad(20 * np.sin(t / 120))  # Sinusoidal for S-weave

# Generate single complex trajectory (tuned for ~1500 km range)
p = dict(m=m_nom, S=S_nom, alpha=alpha_jump, bank=bank_weave)
y0 = np.array([Re + 50e3, 0, 0, 5000, np.deg2rad(-0.05), np.deg2rad(0)])  # Nominal init
T, Y = propagate(0, 2000, 1.0, y0, p)
R = downrange(0, 0, Y[-1, 1], Y[-1, 2])
print(f"Complex Trajectory Range: {R:.0f} km")

# Plot 3D (height vs lon vs lat) and 2D groundtrack
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
h = Y[:, 0] - Re  # m
lon_deg = np.rad2deg(Y[:, 1])
lat_deg = np.rad2deg(Y[:, 2])
ax1.plot(lon_deg, lat_deg, h, 'b-')
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_zlabel('Height (m)')
ax1.set_title('3D Complex Trajectory (Jump-Weave)')
ax2.plot(lon_deg, lat_deg, 'b-')
ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title('2D Groundtrack')
ax2.grid(True)
plt.tight_layout()
plt.show()