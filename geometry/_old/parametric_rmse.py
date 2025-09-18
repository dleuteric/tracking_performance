import numpy as np
import matplotlib.pyplot as plt

# Parameters for ballistic trajectory (fixed for comparison)
dt = 1.0  # s
N = 700  # Steps for full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s
sat_alt = 1000.0  # km
angular_sigma = 0.001  # rad
baselines = [600.0, 1200.0, 1800.0]  # km to compare (tweak list for play)

# Propagate trajectory once (shared across baselines)
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])  # Gravity only in z
    target_vel[k] = target_vel[k-1] + accel_term  # Update velocity
    next_pos = target_pos[k-1] + target_vel[k-1] * dt + 0.5 * accel_term * dt  # Update position
    if next_pos[2] <= 0:  # Ground impact check
        impact_step = k
        break
    target_pos[k] = next_pos

# Trim to impact
target_pos = target_pos[:impact_step]
N_actual = impact_step

# Perturb unit vector with angular noise (Gaussian on spherical coords)
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])  # Polar angle from z-axis [0, pi]
    phi = np.arctan2(unit_vec[1], unit_vec[0])  # Azimuth angle [-pi, pi]
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)  # Add noise, clip range
    phi_noisy = phi + np.random.normal(0, sigma_rad)  # Add noise (no clip needed)
    sin_theta = np.sin(theta_noisy)  # Sin for reconstruction
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])  # Reconstruct Cartesian
    return unit_noisy / np.linalg.norm(unit_noisy)  # Normalize to unit length

# Triangulate pseudo-position (midpoint of shortest segment between skew rays)
def triangulate(sat1, u1, sat2, u2):
    d = sat2 - sat1  # Baseline vector
    cross_u1_u2 = np.cross(u1, u2)  # Cross product for denom
    denom = np.dot(cross_u1_u2, cross_u1_u2)  # Squared norm
    if denom == 0:  # Parallel fallback
        return (sat1 + sat2) / 2
    t1 = np.dot(np.cross(d, u2), cross_u1_u2) / denom  # Scalar along ray1
    t2 = np.dot(np.cross(d, u1), cross_u1_u2) / denom  # Scalar along ray2
    p1 = sat1 + t1 * u1  # Point on ray1
    p2 = sat2 + t2 * u2  # Point on ray2
    return (p1 + p2) / 2  # Midpoint closest approach

# KF setup (3D CA model, state [px vx ax py vy ay pz vz az], observe pos)
F = np.eye(9)  # State transition matrix
for i in range(0, 9, 3):  # Fill for each dim (pos, vel, accel)
    F[i, i+1] = dt  # Pos += dt * vel
    F[i, i+2] = 0.5 * dt**2  # Pos += 0.5 dt^2 * accel
    F[i+1, i+2] = dt  # Vel += dt * accel
H = np.zeros((3,9))  # Observation matrix for [x,y,z]
H[0,0] = 1; H[1,3] = 1; H[2,6] = 1
Q = np.diag([0.01, 0.001, 1e-5, 0.01, 0.001, 1e-5, 0.01, 0.001, 1e-5])  # Process cov (tuned for CA)

# Loop baselines for parametric RMSE overlay
np.random.seed(42)  # Reproducible noise across runs
colors = ['r', 'g', 'b']  # For curves
fig, ax = plt.subplots(figsize=(8, 4))
for idx, bl in enumerate(baselines):
    sat1 = np.array([0.0, -300.0, sat_alt])  # Fixed sat1
    sat2 = np.array([bl, 300.0, sat_alt])  # Vary baseline
    # Nominal LOS vectors
    los1 = target_pos - sat1
    los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]
    los1_unit = los1 / los1_dist
    los2 = target_pos - sat2
    los2_dist = np.linalg.norm(los2, axis=1)[:, np.newaxis]
    los2_unit = los2 / los2_dist
    # Noisy LOS
    los1_unit_noisy = np.zeros((N_actual, 3))
    los2_unit_noisy = np.zeros((N_actual, 3))
    for k in range(N_actual):
        los1_unit_noisy[k] = perturb_unit_vector(los1_unit[k], angular_sigma)  # Perturb sat1 LOS
        los2_unit_noisy[k] = perturb_unit_vector(los2_unit[k], angular_sigma)  # Perturb sat2 LOS
    # Pseudo-positions from noisy LOS
    pseudo_pos_noisy = np.zeros((N_actual, 3))
    for k in range(N_actual):
        pseudo_pos_noisy[k] = triangulate(sat1, los1_unit_noisy[k], sat2, los2_unit_noisy[k])  # Triangulate
    # Fixed R from residuals (empirical cov for this baseline)
    residuals = pseudo_pos_noisy - target_pos
    R = np.cov(residuals.T)  # 3x3 cov matrix
    # KF fusion
    x = np.concatenate((pseudo_pos_noisy[0], [0]*6))  # Init state with first pseudo, zero vel/accel
    P = np.diag([10, 1, 0.1, 10, 1, 0.1, 10, 1, 0.1])  # Init cov (larger for uncertainty)
    filtered_pos = np.zeros((N_actual, 3))
    filtered_pos[0] = pseudo_pos_noisy[0]
    for k in range(1, N_actual):
        x = F @ x  # Predict state
        P = F @ P @ F.T + Q  # Predict cov
        innovation = pseudo_pos_noisy[k] - H @ x  # Innovation (3,)
        S = H @ P @ H.T + R  # Innovation cov (3x3)
        K = P @ H.T @ np.linalg.inv(S)  # Gain (9x3)
        x = x + K @ innovation  # Update state
        P = P - K @ H @ P  # Update cov
        filtered_pos[k] = x[[0,3,6]]  # Extract filtered pos
    # RMSE over time (3D mean)
    rmse = np.sqrt(np.mean((filtered_pos - target_pos)**2, axis=1))
    ax.plot(range(N_actual), rmse, color=colors[idx], label=f'bl={bl} km (mean {np.mean(rmse):.2f} km)')  # Overlay curve

# Plot setup
ax.set_xlabel('Step k')
ax.set_ylabel('RMSE (km)')
ax.set_title('Parametric Filtered RMSE vs Baseline')
ax.legend()
ax.grid(True)
plt.show()