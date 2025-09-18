import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares  # For triangulation optimization

# Parameters for ballistic trajectory (from previous)
dt = 1.0  # s
N = 700  # Steps for full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s
sat_alt = 1000.0  # km
angular_sigma = 1e-3  # rad
baseline = 600.0  # km (tweak to see global effect)
num_mc = 50  # MC per subsample (balance speed)
sub_interval = 100  # Subsample steps for viz (every 100 + ends)

# Propagate trajectory: constant accel in z (gravity), CV in x/y, stop at z<=0
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])  # Gravity accel
    target_vel[k] = target_vel[k-1] + accel_term  # Update vel
    next_pos = target_pos[k-1] + target_vel[k-1] * dt + 0.5 * accel_term * dt  # Update pos
    if next_pos[2] <= 0:  # Ground impact
        impact_step = k
        break
    target_pos[k] = next_pos

# Trim to impact
target_pos = target_pos[:impact_step]
N_actual = impact_step

# Sats fixed
sat1 = np.array([0.0, -300.0, sat_alt])
sat2 = np.array([baseline, 300.0, sat_alt])

# Subsample steps
sub_steps = list(range(0, N_actual, sub_interval)) + [N_actual - 1]

# Perturb unit vector with angular noise
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])  # Polar angle from z [0, pi]
    phi = np.arctan2(unit_vec[1], unit_vec[0])  # Azimuth [-pi, pi]
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)  # Add noise, clip
    phi_noisy = phi + np.random.normal(0, sigma_rad)  # Add noise
    sin_theta = np.sin(theta_noisy)  # Sin for reconstruction
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])  # Cartesian
    return unit_noisy / np.linalg.norm(unit_noisy)  # Normalize

# Triangulate position from two LOS rays (least-squares skew line minimization)
def triangulate(sat1, u1, sat2, u2):
    def residual(params):  # Residual for optimization
        t1, t2 = params
        P1 = sat1 + t1 * u1  # Point on ray1
        P2 = sat2 + t2 * u2  # Point on ray2
        return P1 - P2  # Diff to minimize
    res = least_squares(residual, [0, 0], method='lm')  # Optimize scalars t1,t2
    t1_opt, t2_opt = res.x
    P_opt = (sat1 + t1_opt * u1 + sat2 + t2_opt * u2) / 2  # Midpoint of closest points
    return P_opt

# MC cov estimation at subsample k
def estimate_cov_at_step(k):
    true_pos = target_pos[k]  # True pos at k
    u1_nom = (true_pos - sat1) / np.linalg.norm(true_pos - sat1)  # Nominal unit LOS1
    u2_nom = (true_pos - sat2) / np.linalg.norm(true_pos - sat2)  # Nominal unit LOS2
    rs_mc = np.zeros((num_mc, 3))  # MC positions array
    for i in range(num_mc):
        u1_pert = perturb_unit_vector(u1_nom, angular_sigma)  # Perturb LOS1
        u2_pert = perturb_unit_vector(u2_nom, angular_sigma)  # Perturb LOS2
        rs_mc[i] = triangulate(sat1, u1_pert, sat2, u2_pert)  # Triangulate noisy
    mean_r = np.mean(rs_mc, axis=0)  # Mean est pos
    diffs = rs_mc - mean_r  # Deviations
    R = np.dot(diffs.T, diffs) / (num_mc - 1)  # Sample cov
    return R, mean_r

# Plot trajectory with ellipsoids and principal axes at subsamples
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Truth trajectory')  # Plot arc
ax.scatter(sat1[0], sat1[1], sat1[2], c='b', label='Sat1')  # Sat positions
ax.scatter(sat2[0], sat2[1], sat2[2], c='g', label='Sat2')
for k in sub_steps:
    R, mean_r = estimate_cov_at_step(k)  # Get R at k
    eigvals, eigvecs = np.linalg.eig(R)  # Decomp for principal
    radii = np.sqrt(eigvals)  # 1-sigma radii
    u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]  # Surface grid
    x = radii[0] * np.cos(u) * np.cos(v)  # Principal coords
    y = radii[1] * np.sin(u) * np.cos(v)
    z = radii[2] * np.sin(v)
    points = np.stack((x.ravel(), y.ravel(), z.ravel()))  # Stack for matmul
    rotated = eigvecs @ points  # Rotate to R frame
    ellipsoid = rotated.T + target_pos[k]  # Center at true pos (approx mean_r)
    ax.plot_wireframe(ellipsoid[:, 0].reshape(20, 10), ellipsoid[:, 1].reshape(20, 10), ellipsoid[:, 2].reshape(20, 10),
                      color='c', alpha=0.2)  # Wireframe ellipsoid

    # Overlay semi-axes: sort by length, quiver from center, label lengths
    idx_sort = np.argsort(radii)[::-1]  # Descending for major first
    colors_ax = ['r', 'g', 'b']  # Major red, minor blue
    for i, idx in enumerate(idx_sort):
        dir_ax = eigvecs[:, idx]  # Direction vector
        len_ax = radii[idx]  # Length (1-sigma std dev)
        center = target_pos[k]  # Start quiver at center
        ax.quiver(center[0], center[1], center[2], dir_ax[0]*len_ax, dir_ax[1]*len_ax, dir_ax[2]*len_ax,
                  color=colors_ax[i], arrow_length_ratio=0.1, label=f'Axis {i+1}: {len_ax:.2f} km' if k==sub_steps[0] else None)  # Label once
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Error Ellipsoids with Principal Axes Along Trajectory (baseline=1200 km)')
ax.legend()
plt.show()