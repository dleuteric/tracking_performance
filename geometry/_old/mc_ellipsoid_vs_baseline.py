import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares  # For triangulation

# Params: baselines to compare (tweak list for more cases)
baselines = [600.0, 1200.0, 1800.0]  # km (small → elongated, large → compact)
sigma_rad = 0.001  # rad fixed
num_mc = 100  # MC per case (balance speed/accuracy)
sat1 = np.array([0.0, -300.0, 1000.0])  # Fixed sat1
true_target = np.array([2000.0, 0.0, 100.0])  # Fixed target
colors = ['r', 'g', 'b']  # For each baseline's ellipsoid

# Functions (perturb, triangulate as before)
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])
    phi = np.arctan2(unit_vec[1], unit_vec[0])
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)
    phi_noisy = phi + np.random.normal(0, sigma_rad)
    sin_theta = np.sin(theta_noisy)
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])
    return unit_noisy / np.linalg.norm(unit_noisy)

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

# Plot setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(true_target[0], true_target[1], true_target[2], c='k', marker='x', label='Target')  # Common center
ax.scatter(sat1[0], sat1[1], sat1[2], c='m', label='Sat1')  # Fixed sat

np.random.seed(42)  # Reproducible across cases
max_rad = 0  # For auto-zoom
for idx, bl in enumerate(baselines):
    sat2 = np.array([bl, 300.0, 1000.0])  # Vary x for baseline
    ax.scatter(sat2[0], sat2[1], sat2[2], c=colors[idx], marker='^', label=f'Sat2 (bl={bl} km)')  # Plot sat2

    # Nominal LOS for this geometry
    u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
    u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)

    # MC for R
    rs_mc = np.zeros((num_mc, 3))
    for i in range(num_mc):
        u1_pert = perturb_unit_vector(u1_nom, sigma_rad)
        u2_pert = perturb_unit_vector(u2_nom, sigma_rad)
        rs_mc[i] = triangulate(sat1, u1_pert, sat2, u2_pert)
    mean_r = np.mean(rs_mc, axis=0)
    diffs = rs_mc - mean_r
    R = np.dot(diffs.T, diffs) / (num_mc - 1)

    # Ellipsoid (1-sigma wireframe)
    eigvals, eigvecs = np.linalg.eig(R)
    radii = np.sqrt(eigvals)
    u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]
    x = radii[0] * np.cos(u) * np.cos(v)
    y = radii[1] * np.sin(u) * np.cos(v)
    z = radii[2] * np.sin(v)
    points = np.stack((x.ravel(), y.ravel(), z.ravel()))
    rotated = eigvecs @ points
    ellipsoid = rotated.T + true_target  # Center at target (approx mean_r ~ target)
    ax.plot_wireframe(ellipsoid[:, 0].reshape(20, 10), ellipsoid[:, 1].reshape(20, 10), ellipsoid[:, 2].reshape(20, 10),
                      color=colors[idx], alpha=0.5, label=f'Ellipsoid (bl={bl} km)')

    max_rad = max(max_rad, max(radii))  # Track largest for zoom

# Axes and title
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.set_title('Parametric Comparison: Error Ellipsoids vs Baseline')
ax.set_xlim(true_target[0] - max_rad*1.5, true_target[0] + max_rad*1.5)
ax.set_ylim(true_target[1] - max_rad*1.5, true_target[1] + max_rad*1.5)
ax.set_zlim(true_target[2] - max_rad*1.5, true_target[2] + max_rad*1.5)
ax.legend()
plt.show()