import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares  # For triangulation

# Example fixed geometry: sat1, sat2, true_target (mid-arc for viz)
sat1 = np.array([0.0, -300.0, 1000.0])  # km
sat2 = np.array([1200.0, 300.0, 1000.0])  # Baseline 1200 km
true_target = np.array([2000.0, 0.0, 100.0])  # Fixed target pos
angular_sigma = 0.00001  # rad
num_mc = 100  # MC runs (cloud size; try 1000 for denser)

# Nominal unit LOS (no noise)
u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)

# Perturb function (as before)
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])  # Polar from z [0, pi]
    phi = np.arctan2(unit_vec[1], unit_vec[0])  # Azimuth [-pi, pi]
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)
    phi_noisy = phi + np.random.normal(0, sigma_rad)
    sin_theta = np.sin(theta_noisy)
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])
    return unit_noisy / np.linalg.norm(unit_noisy)

# Triangulate function (as before)
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

# MC loop: Perturb, triangulate N times for cloud
np.random.seed(4)  # Reproducible
rs_mc = np.zeros((num_mc, 3))
for i in range(num_mc):
    u1_pert = perturb_unit_vector(u1_nom, angular_sigma)  # Fresh noise each run
    u2_pert = perturb_unit_vector(u2_nom, angular_sigma)  # Independent for each sat
    rs_mc[i] = triangulate(sat1, u1_pert, sat2, u2_pert)  # Triangulate noisy LOS

# Compute stats: mean, diffs, R (cov)
mean_r = np.mean(rs_mc, axis=0)  # Cloud center (est pos)
diffs = rs_mc - mean_r  # Deviations
R = np.dot(diffs.T, diffs) / (num_mc - 1)  # Cov matrix (3x3)

# Visualize MC cloud in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rs_mc[:, 0], rs_mc[:, 1], rs_mc[:, 2], c='r', s=5, label='MC pseudo-positions (cloud)')  # Noisy points
ax.scatter(true_target[0], true_target[1], true_target[2], c='k', marker='x', label='True target')  # Truth
ax.scatter(mean_r[0], mean_r[1], mean_r[2], c='g', marker='o', label='Mean est')  # Cloud mean
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.set_title('MC Cloud of Triangulated Positions (N=100)')
ax.legend()
plt.show()

# Print for confirmation
print(f"Mean est: {mean_r}")
print(f"R (cov):\n{R}")