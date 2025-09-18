import numpy as np
import matplotlib.pyplot as plt

# Parameters for ballistic trajectory (from previous)
dt = 1.0  # s
N = 700  # Steps for full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s
sat_alt = 1000.0  # km
angular_sigma = 0.001  # rad

# Propagate trajectory: constant accel in z (gravity), CV in x/y, stop at z<=0
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])  # Gravity in z
    target_vel[k] = target_vel[k-1] + accel_term  # Update velocity
    next_pos = target_pos[k-1] + target_vel[k-1] * dt + 0.5 * accel_term * dt  # Update position
    if next_pos[2] <= 0:  # Ground impact
        impact_step = k
        break
    target_pos[k] = next_pos

# Trim to impact
target_pos = target_pos[:impact_step]
target_vel = target_vel[:impact_step]
N_actual = impact_step

# Sats positions: sat1 at [0, -300, sat_alt], sat2 at [2000, 300, sat_alt]
sat1 = np.array([0.0, -300.0, sat_alt])
sat2 = np.array([2000.0, 300.0, sat_alt])

# Nominal LOS unit vectors
los1 = target_pos - sat1
los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]
los1_unit = los1 / los1_dist
los2 = target_pos - sat2
los2_dist = np.linalg.norm(los2, axis=1)[:, np.newaxis]
los2_unit = los2 / los2_dist

# Perturb unit vector with angular noise
def perturb_unit_vector(unit_vec, sigma_rad):
    theta = np.arccos(unit_vec[2])  # Elevation from z
    phi = np.arctan2(unit_vec[1], unit_vec[0])  # Azimuth
    theta_noisy = np.clip(theta + np.random.normal(0, sigma_rad), 0, np.pi)
    phi_noisy = phi + np.random.normal(0, sigma_rad)
    sin_theta = np.sin(theta_noisy)
    unit_noisy = np.array([sin_theta * np.cos(phi_noisy), sin_theta * np.sin(phi_noisy), np.cos(theta_noisy)])
    return unit_noisy / np.linalg.norm(unit_noisy)

# Apply noise to LOS
np.random.seed(42)
los1_unit_noisy = np.zeros((N_actual, 3))
los2_unit_noisy = np.zeros((N_actual, 3))
for k in range(N_actual):
    los1_unit_noisy[k] = perturb_unit_vector(los1_unit[k], angular_sigma)
    los2_unit_noisy[k] = perturb_unit_vector(los2_unit[k], angular_sigma)

# Triangulate pseudo-positions (fixed definition)
def triangulate(sat1, u1, sat2, u2):
    d = sat2 - sat1
    cross_u1_u2 = np.cross(u1, u2)
    denom = np.dot(cross_u1_u2, cross_u1_u2)
    if denom == 0:
        return (sat1 + sat2) / 2  # Parallel fallback
    t1 = np.dot(np.cross(d, u2), cross_u1_u2) / denom
    t2 = np.dot(np.cross(d, u1), cross_u1_u2) / denom
    p1 = sat1 + t1 * u1
    p2 = sat2 + t2 * u2
    return (p1 + p2) / 2

# Compute noisy pseudo-positions
pseudo_pos_noisy = np.zeros((N_actual, 3))
for k in range(N_actual):
    pseudo_pos_noisy[k] = triangulate(sat1, los1_unit_noisy[k], sat2, los2_unit_noisy[k])

# Empirical R from residuals
residuals = pseudo_pos_noisy - target_pos
R_empirical = np.cov(residuals.T)  # Average R over trajectory

# 3D CA KF: state [px, vx, ax, py, vy, ay, pz, vz, az], H for pos
F = np.eye(9)
for i in range(0, 9, 3):
    F[i, i+1] = dt
    F[i, i+2] = 0.5 * dt**2
    F[i+1, i+2] = dt
H = np.array([[1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]])
Q = np.diag([0.01, 0.001, 1e-5, 0.01, 0.001, 1e-5, 0.01, 0.001, 1e-5])
R = R_empirical

# Initialize KF
x = np.concatenate((pseudo_pos_noisy[0], [0]*6))  # Pos from first pseudo, vel/accel=0
P = np.diag([R[0,0], 1, 0.1, R[1,1], 1, 0.1, R[2,2], 1, 0.1])

# Fuse pseudo into KF
filtered_pos = np.zeros((N_actual, 3))
filtered_pos[0] = pseudo_pos_noisy[0]
for k in range(1, N_actual):
    x = F @ x  # Predict state
    P = F @ P @ F.T + Q  # Predict cov
    innovation = pseudo_pos_noisy[k] - H @ x  # Innovation
    S = H @ P @ H.T + R  # Innovation cov
    K = P @ H.T @ np.linalg.inv(S)  # Gain
    x = x + K @ innovation  # Update state
    P = P - K @ H @ P  # Update cov
    filtered_pos[k] = x[[0,3,6]]  # Extract pos

# Compute RMSE over time (3D average)
rmse = np.sqrt(np.mean((filtered_pos - target_pos)**2, axis=1))

# Plot RMSE evolution
plt.figure(figsize=(8, 4))
plt.plot(range(N_actual), rmse, 'g-', label='3D RMSE')
plt.xlabel('Step k')
plt.ylabel('RMSE (km)')
plt.title('Filtered RMSE Evolution Along Ballistic Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# Diagnostic print for final filtered
print(f"Mean RMSE: {np.mean(rmse):.2f} km")
print(f"Final filtered position [x,y,z]: {filtered_pos[-1]}")