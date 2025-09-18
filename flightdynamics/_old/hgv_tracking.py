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


# Helper: ECEF from spherical (lon rad, lat rad, h m)
def ecef_from_llh(lon, lat, h):
    N = Re / np.sqrt(1 - 0.00669438 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - 0.00669438) + h) * np.sin(lat)
    return np.array([x, y, z])


# Aero coeffs
def aero_coeffs(alpha):
    CL_alpha, CD0, k = 1.8, 0.05, 0.5
    CL = CL_alpha * alpha
    return CL, CD0 + k * CL ** 2


# Dynamics: Full EOM (unchanged, linking math as before)
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


# Maneuver types (Fig. 4 combos)
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


# Alpha functions
def alpha_bal(t, state): return np.deg2rad(8.0)


def alpha_jump(t, state):
    _, _, _, v, _, _ = state
    v_km = v / 1000
    alpha_max, alpha_ld = 15.0, 2.5
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


# Bank functions
def bank_no(t, state): return 0.0


def bank_left(t, state): return np.deg2rad(-20.0)


def bank_right(t, state): return np.deg2rad(20.0)


def bank_weave(t, state): return np.deg2rad(20.0 * np.sin(t / 200.0))


# Generate trajectories
trajs = []
colors = ['b', 'orange', 'y', 'purple', 'g', 'c', 'r', 'navy']
for i, man in enumerate(maneuvers):
    p = dict(m=m_nom, S=S_nom)
    h0 = np.random.uniform(40e3, 70e3)
    v0 = np.random.uniform(3000, 6000)
    th0 = np.deg2rad(np.random.uniform(-0.1, 0))
    sig0 = np.deg2rad(np.random.uniform(-15, 15))
    y0 = np.array([Re + h0, 0, 0, v0, th0, sig0])
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


# LEO Satellites (dummy positions at t=600 s, circular 1000 km orbit)
def sat_ecef(lon0, lat0, t):
    a = Re + 1000e3  # Semi-major axis (m)
    n = np.sqrt(mu / a ** 3)  # Mean motion (rad/s)
    lon = lon0 + n * t  # Simple Keplerian approx
    lat = lat0
    h = 1000e3
    return ecef_from_llh(lon, lat, h) / 1e3  # km


sat1 = sat_ecef(np.deg2rad(30), np.deg2rad(0), 600)  # Sat 1 at 30°E
sat2 = sat_ecef(np.deg2rad(-30), np.deg2rad(0), 600)  # Sat 2 at 30°W

# Plot: 3D + 2D groundtrack + LOS at t=600 s
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.view_init(elev=30, azim=45)  # Initial view for better perspective
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
for traj in trajs:
    h = (traj['Y'][:, 0] - Re)  # m
    lon_deg = np.rad2deg(traj['Y'][:, 1])
    lat_deg = np.rad2deg(traj['Y'][:, 2])
    # 3D plot
    ax1.plot(lon_deg, lat_deg, h, color=traj['color'], label=traj['name'])
    ax1.scatter(lon_deg[::100], lat_deg[::100], h[::100], color=traj['color'], s=10)
    # 2D groundtrack
    ax2.plot(lon_deg, lat_deg, color=traj['color'], label=traj['name'])
    ax2.scatter(lon_deg[::100], lat_deg[::100], color=traj['color'], s=10)
    # LOS at t=600 s (approx index)
    idx = np.argmin(np.abs(traj['Y'][:, 0] - (Re + 50e3)))  # Near start height
    hgv_pos = ecef_from_llh(traj['Y'][idx, 1], traj['Y'][idx, 2], traj['Y'][idx, 0] - Re) / 1e3
    ax3.plot([sat1[0], hgv_pos[0]], [sat1[1], hgv_pos[1]], [sat1[2], hgv_pos[2]], 'k--',
             label='LOS Sat1' if traj['name'] == 'Trajectory 1 (Bal-No)' else "")
    ax3.plot([sat2[0], hgv_pos[0]], [sat2[1], hgv_pos[1]], [sat2[2], hgv_pos[2]], 'k-.',
             label='LOS Sat2' if traj['name'] == 'Trajectory 1 (Bal-No)' else "")
# Earth in 3D plot (3rd subplot for LOS clarity)
u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
x = Re / 1e3 * np.cos(u) * np.sin(v)
y = Re / 1e3 * np.sin(u) * np.sin(v)
z = Re / 1e3 * np.cos(v)
ax3.plot_surface(x, y, z, color='grey', alpha=0.5)
ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_zlabel('Height (m)')
ax1.set_xlim(-5, 45)
ax1.set_ylim(-5, 5)
ax1.set_zlim(0, 7e4)
ax1.set_title('3D Maneuver Trajectories')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_xlim(-5, 45)
ax2.set_ylim(-5, 5)
ax2.set_title('2D Groundtracks')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)
ax3.set_xlabel('X (km)')
ax3.set_ylabel('Y (km)')
ax3.set_zlabel('Z (km)')
ax3.set_title('LOS at t=600 s')
ax3.legend()
plt.tight_layout()
plt.show()