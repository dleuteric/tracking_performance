import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants in SI units (paper values)
Re = 6.378e6  # Earth radius (m)
mu = 3.986e14  # GM (m³/s²)
omega_e = 7.29e-5  # Rotation (rad/s)
rho0 = 1.225  # Sea-level density (kg/m³)
Hs = 7200  # Scale height (m)
m_nom = 907.0  # Mass (kg)
S_nom = 0.4839  # Area (m²)


# Aero coeffs (tunable α_max)
def aero_coeffs(alpha, alpha_max):
    CL_alpha = 0.12 * alpha_max / 15.0  # Scale CL with α_max (normalized to 15°)
    CD0, k = 0.05, 0.5
    CL = CL_alpha * alpha
    return CL, CD0 + k * CL ** 2


# Dynamics: Full EOM with rotation
def hgv_dot(t, y, p):
    r, lon, lat, v, th, sig = y
    h = r - Re
    alpha, bank = p['alpha'](t, y, p['alpha_max']), p['bank'](t, y)
    rho = rho0 * np.exp(-h / Hs)
    CL, CD = aero_coeffs(alpha, p['alpha_max'])
    q = 0.5 * rho * v ** 2
    aL = q * CL * p['S'] / p['m']
    aD = q * CD * p['S'] / p['m']
    g = mu / r ** 2

    rdot = v * np.sin(th)
    vdot = -aD - g * np.sin(th) + omega_e ** 2 * r * np.cos(lat) * (
                np.sin(th) * np.cos(lat) - np.cos(th) * np.sin(sig) * np.sin(lat))
    thdot = (aL * np.cos(bank) / v) + (v / r - g / v) * np.cos(th) + 2 * omega_e * np.cos(sig) * np.cos(lat) + (
                omega_e ** 2 * r / v) * np.cos(lat) * (
                        np.cos(th) * np.cos(lat) + np.sin(th) * np.sin(sig) * np.sin(lat))
    londot = (v * np.cos(th) * np.sin(sig)) / (r * np.cos(lat) + 1e-9)
    latdot = (v * np.cos(th) * np.cos(sig)) / r
    sigdot = (aL * np.sin(bank) / (v * np.cos(th) + 1e-9)) + (v / r) * np.cos(th) * np.sin(sig) * np.tan(
        lat) - 2 * omega_e * (np.tan(th) * np.cos(lat) * np.sin(sig) - np.sin(lat)) + (
                         omega_e ** 2 * r / (v * np.cos(th) + 1e-9)) * np.sin(sig) * np.cos(lat) * np.sin(lat)

    return np.array([rdot, londot, latdot, vdot, thdot, sigdot])


# RK4 step
def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p)
    k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, p)
    k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, p)
    k4 = fun(t + h, y + h * k3, p)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Propagator
def propagate(t0, tf, dt, y0, p):
    ts, ys = [t0], [y0.copy()]
    t, y = t0, y0.copy()
    while t < tf and y[0] > Re:
        y = rk4(hgv_dot, t, y, dt, p)
        t += dt
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.vstack(ys)


# Alpha functions (tunable α_max passed)
def alpha_bal(t, state, alpha_max): return np.deg2rad(0.5 * alpha_max)  # Scaled constant


def alpha_jump(t, state, alpha_max):  # Velocity-scheduled
    _, _, _, v, _, _ = state
    v_km = v / 1000
    alpha_ld = 0.17 * alpha_max  # Scaled from 2.5° at α_max=15°
    v1_km, v2_km = 6.0, 3.0
    if v_km > v1_km:
        return np.deg2rad(alpha_max)
    elif v_km < v2_km:
        return np.deg2rad(alpha_ld)
    alpha_mid = 0.5 * (alpha_max + alpha_ld)
    alpha_bal = 0.5 * (alpha_max - alpha_ld)
    v_mid = 0.5 * (v1_km + v2_km)
    alpha = alpha_mid + alpha_bal * np.sin(np.pi * (v_km - v_mid) / (v1_km - v2_km))
    return np.deg2rad(alpha)


# Bank functions (unchanged)
def bank_no(t, state): return 0.0


def bank_left(t, state): return np.deg2rad(-20.0)


def bank_right(t, state): return np.deg2rad(20.0)


def bank_weave(t, state): return np.deg2rad(20.0 * np.sin(t / 200.0))


# Generate N=8 trajectories with downrange tuning
trajs = []
colors = ['b', 'orange', 'y', 'purple', 'g', 'c', 'r', 'navy']
maneuvers = [
    {'name': 'Bal-No', 'alpha_type': 'bal', 'bank_type': 'no'},
    {'name': 'Bal-Left', 'alpha_type': 'bal', 'bank_type': 'left'},
    {'name': 'Bal-Right', 'alpha_type': 'bal', 'bank_type': 'right'},
    {'name': 'Bal-Weave', 'alpha_type': 'bal', 'bank_type': 'weave'},
    {'name': 'Jump-No', 'alpha_type': 'jump', 'bank_type': 'no'},
    {'name': 'Jump-Left', 'alpha_type': 'jump', 'bank_type': 'left'},
    {'name': 'Jump-Right', 'alpha_type': 'jump', 'bank_type': 'right'},
    {'name': 'Jump-Weave', 'alpha_type': 'jump', 'bank_type': 'weave'}
]
for i, man in enumerate(maneuvers):
    p = dict(m=m_nom, S=S_nom)
    # Tune downrange via v0, θ0, α_max (scaled for 1000-5000 km)
    v0 = np.random.uniform(6000, 7000)  # m/s
    theta0 = np.deg2rad(np.random.uniform(-0.1, 0))  # rad
    alpha_max = np.random.uniform(10, 20)  # deg
    p['alpha_max'] = alpha_max
    sig0 = np.deg2rad(np.random.uniform(-15, 15))  # rad
    h0 = np.random.uniform(40e3, 100e3)  # m
    y0 = np.array([Re + h0, 0, 0, v0, theta0, sig0])
    p['alpha'] = alpha_bal if man['alpha_type'] == 'bal' else alpha_jump
    if man['bank_type'] == 'no':
        p['bank'] = bank_no
    elif man['bank_type'] == 'left':
        p['bank'] = bank_left
    elif man['bank_type'] == 'right':
        p['bank'] = bank_right
    else:
        p['bank'] = bank_weave
    _, Y = propagate(0, 1200, 1.0, y0, p)
    lon_end = np.rad2deg(Y[-1, 1]) if Y[-1, 0] <= Re else np.nan
    lat_end = np.rad2deg(Y[-1, 2]) if Y[-1, 0] <= Re else np.nan
    downrange = Re * np.abs(np.deg2rad(lon_end)) / 1000 if not np.isnan(lon_end) else 0  # km, approx
    trajs.append({'name': f"Trajectory {i + 1} ({man['name']})", 'Y': Y, 'color': colors[i], 'downrange': downrange})

# Plot: 3D + 2D groundtrack with downrange labels
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
for traj in trajs:
    h = (traj['Y'][:, 0] - Re)  # m
    lon_deg = np.rad2deg(traj['Y'][:, 1])
    lat_deg = np.rad2deg(traj['Y'][:, 2])
    ax1.plot(lon_deg, lat_deg, h, color=traj['color'], label=f"{traj['name']} ({traj['downrange']:.0f} km)")
    ax1.scatter(lon_deg[::100], lat_deg[::100], h[::100], color=traj['color'], s=10)
    ax2.plot(lon_deg, lat_deg, color=traj['color'], label=f"{traj['name']} ({traj['downrange']:.0f} km)")
    ax2.scatter(lon_deg[::100], lat_deg[::100], color=traj['color'], s=10)
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_zlabel('Height (m)')
ax1.set_title('3D Maneuver Trajectories')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title('2D Groundtracks')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)
plt.tight_layout()
plt.show()