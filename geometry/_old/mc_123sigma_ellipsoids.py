import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares  # For triangulation

# Params to tweak for intuition (play: change baseline=600 for elongation, sigma_rad=0.002 for larger)
baseline = 1200.0  # km (wider → better angle, smaller ellipsoid)
sigma_rad = 0.001  # rad (larger → scales up variances uniformly)
num_mc = 200  # MC runs (more → smoother cloud/R)
sat1 = np.array([0.0, -300.0, 1000.0])  # Fixed sat1
sat2 = np.array([baseline, 300.0, 1000.0])  # Sat2 varies with baseline
true_target = np.array([2000.0, 0.0, 100.0])  # Fixed mid-arc target
scales = [1, 2, 3]  # Sigma levels to plot (add e.g. 1.73 for ~68% containment)

# Nominal LOS
u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)

# Perturb function
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])
    phi = np.arctan2(unit_vec[1], unit_vec[0])
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)
    phi_noisy = phi + np.random.normal(0, sigma_rad)
    sin_theta = np.sin(theta_noisy)
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])
    return unit_noisy / np.linalg.norm(unit_noisy)

# Triangulate function
def triangulate(sat1, u1, sat2, u2):
    def residual(params):
        t1, t2 = params
        P1 = sat1 + t1 * u1
        P2 = sat2 + t2 * u2
        return P1 - P2
    res = least_squares(residual, [0, 0], method='lm')
    t1_opt, t2_opt = res.x
    P_opt = (sat1 + t1_opt * u1 + sat2 + t2_opt * u2) / 2
    return P_opt

# MC for cloud and R
np.random.seed(42)
rs_mc = np.zeros((num_mc, 3))
for i in range(num_mc):
    u1_pert = perturb_unit_vector(u1_nom, sigma_rad)
    u2_pert = perturb_unit_vector(u2_nom, sigma_rad)
    rs_mc[i] = triangulate(sat1, u1_pert, sat2, u2_pert)
mean_r = np.mean(rs_mc, axis=0)
diffs = rs_mc - mean_r
R = np.dot(diffs.T, diffs) / (num_mc - 1)

# Eigendecomp for base ellipsoid (1-sigma)
eigvals, eigvecs = np.linalg.eig(R)
radii_base = np.sqrt(eigvals)  # Base 1-sigma radii
u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]  # Grid for surface

# Plot cloud + multi-sigma ellipsoids
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rs_mc[:, 0], rs_mc[:, 1], rs_mc[:, 2], c='r', s=5, alpha=0.5, label='MC cloud')  # Points
colors = ['c', 'm', 'y']  # For 1,2,3 sigma
for i, k in enumerate(scales):
    radii = k * radii_base  # Scale radii for k-sigma
    x = radii[0] * np.cos(u) * np.cos(v)  # Principal
    y = radii[1] * np.sin(u) * np.cos(v)
    z = radii[2] * np.sin(v)
    points = np.stack((x.ravel(), y.ravel(), z.ravel()))
    rotated = eigvecs @ points  # Rotate
    ellipsoid = rotated.T + mean_r  # Center
    ax.plot_wireframe(ellipsoid[:, 0].reshape(20, 10), ellipsoid[:, 1].reshape(20, 10), ellipsoid[:, 2].reshape(20, 10),
                      color=colors[i], alpha=0.3, label=f'{k}-sigma ellipsoid')
ax.scatter(true_target[0], true_target[1], true_target[2], c='k', marker='x', label='Truth')  # Target
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.set_title(f'Multi-Sigma Error Ellipsoids (baseline={baseline} km, σ={sigma_rad} rad)')
# Auto-zoom to outermost
max_rad = max(scales) * max(radii_base) * 1.2
ax.set_xlim(mean_r[0] - max_rad, mean_r[0] + max_rad)
ax.set_ylim(mean_r[1] - max_rad, mean_r[1] + max_rad)
ax.set_zlim(mean_r[2] - max_rad, mean_r[2] + max_rad)
ax.legend()
plt.show()