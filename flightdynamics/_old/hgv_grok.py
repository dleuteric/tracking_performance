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


# Aero coeffs (tuned for skips/turns)
def aero_coeffs(alpha):  # Toy CL/CD (rad)
    CL_alpha, CD0, k = 1.8, 0.05, 0.5
    CL = CL_alpha * alpha
    return CL, CD0 + k * CL ** 2


# Dynamics: Full EOM with rotation (from previous)
def hgv_dot(t, y, p):
    r, lon, lat, v, th, sig = y
    h = r - Re
    alpha, bank = p['alpha'](t, y), p['bank'](t, y)
    rho = rho0 * np.exp(-h / Hs)
    CL, CD = aero_coeffs(alpha)
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


# RK4 step (from previous)
def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p)
    k2 = fun(t + 0.5 * h, y + 0.5 * h * k1, p)
    k3 = fun(t + 0.5 * h, y + 0.5 * h * k2, p)
    k4 = fun(t + h, y + h * k3, p)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Propagator (shortened for multiple runs)
def propagate(t0, tf, dt, y0, p):
    ts, ys = [t0], [y0.copy()]
    t, y = t0, y0.copy()
    while t < tf and y[0] > Re:
        y = rk4(hgv_dot, t, y, dt, p)
        t += dt
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.vstack(ys)


# Maneuver types (Fig. 4 combos: long x lat = 8)
maneuvers = [
    {'name': 'Bal-No', 'alpha_type': 'bal', 'bank_type': 'no'},  # 1
    {'name': 'Bal-Left', 'alpha_type': 'bal', 'bank_type': 'left'},  # 2
    {'name': 'Bal-Right', 'alpha_type': 'bal', 'bank_type': 'right'},  # 3
    {'name': 'Bal-Weave', 'alpha_type': 'bal', 'bank_type': 'weave'},  # 4
    {'name': 'Jump-No', 'alpha_type': 'jump', 'bank_type': 'no'},  # 5
    {'name': 'Jump-Left', 'alpha_type': 'jump', 'bank_type': 'left'},  # 6
    {'name': 'Jump-Right', 'alpha_type': 'jump', 'bank_type': 'right'},  # 7
    {'name': 'Jump-Weave', 'alpha_type': 'jump', 'bank_type': 'weave'}  # 8
]


# Alpha functions
def alpha_bal(t, state): return np.deg2rad(8.0)  # Constant for balanced


def alpha_jump(t, state):  # Velocity-scheduled (Eqs. 4-7)
    _, _, _, v, _, _ = state
    v = v / 1000  # to km/s
    alpha_max, alpha_ld = 15.0, 2.5
    v1_km, v2_km = 6.0, 3.0
    if v > v1_km:
        return np.deg2rad(alpha_max)
    elif v < v2_km:
        return np.deg2rad(alpha_ld)
    alpha_mid = 0.5 * (alpha_max + alpha_ld)
    alpha_bal = 0.5 * (alpha_max - alpha_ld)
    v_mid = 0.5 * (v1_km + v2_km)
    alpha = alpha_mid + alpha_bal * np.sin(np.pi * (v - v_mid) / (v1_km - v2_km))
    return np.deg2rad(alpha)


# Bank functions
def bank_no(t, state): return 0.0


def bank_left(t, state): return np.deg2rad(-20.0)  # Negative for left turn


def bank_right(t, state): return np.deg2rad(20.0)


def bank_weave(t, state):  # Sinusoidal for S-shape
    return np.deg2rad(20.0 * np.sin(t / 200.0))  # Period ~1256 s for weave


# Generate trajectories
trajs = []
colors = ['b', 'orange', 'y', 'purple', 'g', 'c', 'r', 'navy']  # Match paper
for i, man in enumerate(maneuvers):
    p = dict(m=m_nom, S=S_nom)
    # Random init per Table 1 (vary h0, v0, th0, sig0)
    h0 = np.random.uniform(40e3, 70e3)  # m
    v0 = np.random.uniform(3000, 6000)  # m/s
    th0 = np.deg2rad(np.random.uniform(-0.1, 0))
    sig0 = np.deg2rad(np.random.uniform(-15, 15))
    y0 = np.array([Re + h0, 0, 0, v0, th0, sig0])
    # Assign controls
    p['alpha'] = alpha_bal if man['alpha_type'] == 'bal' else alpha_jump
    if man['bank_type'] == 'no':
        p['bank'] = bank_no
    elif man['bank_type'] == 'left':
        p['bank'] = bank_left
    elif man['bank_type'] == 'right':
        p['bank'] = bank_right
    else:
        p['bank'] = bank_weave
    _, Y = propagate(0, 1200, 0.5, y0, p)
    trajs.append({'name': f"Trajectory {i + 1} ({man['name']})", 'Y': Y, 'color': colors[i]})

# Plot like Fig. 1: 3D height (m) vs lon (deg) vs lat (deg)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for traj in trajs:
    h = (traj['Y'][:, 0] - Re)  # m
    lon_deg = np.rad2deg(traj['Y'][:, 1])
    lat_deg = np.rad2deg(traj['Y'][:, 2])
    ax.plot(lon_deg, lat_deg, h, color=traj['color'], label=traj['name'])
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_zlabel('High (m)')
ax.set_title('Typical Maneuver Trajectories of HGV')
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()