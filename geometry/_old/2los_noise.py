import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares  # Added import for triangulation optimization

# Parameters for rudimentary ballistic trajectory: time step, gravity, initial conditions
dt = 1.0  # s
N = 700  # Number of steps to capture full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km at launch
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s (horizontal boost + vertical climb)
sat_alt = 1000.0  # km (LEO altitude for sats)
angular_sigma = 0.001  # rad (~1 mrad angular noise std dev for LOS pointing)

# Propagate trajectory: constant accel in z (gravity), CV in x/y, stop at z<=0
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N  # Default to full N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])  # Accel only in z (gravity)
    target_vel[k] = target_vel[k - 1] + accel_term  # Update velocity
    next_pos = target_pos[k - 1] + target_vel[k - 1] * dt + 0.5 * accel_term * dt  # Tentative position
    if next_pos[2] <= 0:  # Check for ground impact
        impact_step = k
        break  # Stop propagation
    target_pos[k] = next_pos  # Assign if above ground

# Trim arrays to impact
target_pos = target_pos[:impact_step]
target_vel = target_vel[:impact_step]
N_actual = impact_step  # Update N to actual steps

# Updated sat positions as per your change: sat1 at [0, -300, sat_alt], sat2 at [2000, 300, sat_alt]
sat1_pos = np.full((N_actual, 3), [0.0, -300.0, sat_alt])  # Constant position with y-offset
sat2_pos = np.full((N_actual, 3), [2000.0, 300.0, sat_alt])  # Mid-range baseline with y-offset

# Compute nominal LOS unit vectors from sats to target (no noise initially)
los1 = target_pos - sat1_pos  # Vector from sat1 to target (N x 3)
los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]  # Distances
los1_unit = los1 / los1_dist  # Nominal unit LOS vectors

los2 = target_pos - sat2_pos  # Vector from sat2 to target
los2_dist = np.linalg.norm(los2, axis=1)[:, np.newaxis]
los2_unit = los2 / los2_dist  # Nominal unit LOS vectors


# Function to perturb unit vector with angular noise (convert to spherical, add noise to az/el, reconstruct)
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])  # Elevation angle from z-axis [0, pi]
    phi = np.arctan2(unit_vec[1], unit_vec[0])  # Azimuth angle [-pi, pi]

    # Add Gaussian noise to theta and phi (small angle approx, clip to valid range)
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)
    phi_noisy = phi + np.random.normal(0, sigma_rad)  # No clip needed for phi

    # Reconstruct noisy unit vector from noisy spherical coords
    sin_theta = np.sin(theta_noisy)
    unit_noisy = np.array([
        sin_theta * np.cos(phi_noisy),
        sin_theta * np.sin(phi_noisy),
        np.cos(theta_noisy)
    ])
    return unit_noisy / np.linalg.norm(unit_noisy)  # Re-normalize


# Apply noise to LOS unit vectors for each step
np.random.seed(42)  # For reproducibility
los1_unit_noisy = np.zeros((N_actual, 3))
los2_unit_noisy = np.zeros((N_actual, 3))
for k in range(N_actual):
    los1_unit_noisy[k] = perturb_unit_vector(los1_unit[k], angular_sigma)
    los2_unit_noisy[k] = perturb_unit_vector(los2_unit[k], angular_sigma)


# Function to triangulate position from two LOS rays (least-squares minimization of skew line distance)
def triangulate(sat1, u1, sat2, u2):
    def residual(params):
        t1, t2 = params
        P1 = sat1 + t1 * u1
        P2 = sat2 + t2 * u2
        return P1 - P2  # Residual vector (3,)

    res = least_squares(residual, [0, 0], method='lm')  # Optimize t1, t2
    t1_opt, t2_opt = res.x
    P_opt = (sat1 + t1_opt * u1 + sat2 + t2_opt * u2) / 2  # Midpoint of closest points
    return P_opt


# Compute noisy triangulated pseudo-positions for each step
pseudo_pos_noisy = np.zeros((N_actual, 3))
for k in range(N_actual):
    pseudo_pos_noisy[k] = triangulate(sat1_pos[k], los1_unit_noisy[k], sat2_pos[k], los2_unit_noisy[k])

# 3D Plot of truth trajectory and noisy pseudo-positions (no rays for clarity)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Truth trajectory')
ax.plot(pseudo_pos_noisy[:, 0], pseudo_pos_noisy[:, 1], pseudo_pos_noisy[:, 2], 'r.', label='Noisy pseudo-positions')
ax.scatter(sat1_pos[0, 0], sat1_pos[0, 1], sat1_pos[0, 2], c='b', label='Sat1')
ax.scatter(sat2_pos[0, 0], sat2_pos[0, 1], sat2_pos[0, 2], c='g', label='Sat2')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Ballistic Trajectory with Noisy Triangulated Pseudo-Positions')
ax.legend()
plt.show()

# Diagnostic print for pseudo at peak
peak_step = np.argmax(target_pos[:, 2])
print(f"Noisy pseudo at peak (step {peak_step}): {pseudo_pos_noisy[peak_step]}")