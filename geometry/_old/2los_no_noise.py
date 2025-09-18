import numpy as np
import matplotlib.pyplot as plt

# Parameters for rudimentary ballistic trajectory: time step, gravity, initial conditions
dt = 1.0  # s
N = 700  # Number of steps to capture full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km at launch
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s (horizontal boost + vertical climb)
sat_alt = 1000.0  # km (LEO altitude for sats)

# Propagate trajectory: constant accel in z (gravity), CV in x/y, stop at z<=0
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N  # Default to full N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])  # Accel only in z (gravity)
    target_vel[k] = target_vel[k-1] + accel_term  # Update velocity
    next_pos = target_pos[k-1] + target_vel[k-1] * dt + 0.5 * accel_term * dt  # Tentative position
    if next_pos[2] <= 0:  # Check for ground impact
        impact_step = k
        break  # Stop propagation
    target_pos[k] = next_pos  # Assign if above ground

# Trim arrays to impact
target_pos = target_pos[:impact_step]
target_vel = target_vel[:impact_step]
N_actual = impact_step  # Update N to actual steps

# Simple sat positions: sat1 fixed at [0, 0, sat_alt], sat2 at [2000, 0, sat_alt] for baseline coverage
sat1_pos = np.full((N_actual, 3), [0.0, -300.0, sat_alt])  # Constant position
sat2_pos = np.full((N_actual, 3), [2000.0, 300.0, sat_alt])  # Mid-range baseline

# Compute nominal LOS unit vectors from sats to target (no noise yet)
los1 = target_pos - sat1_pos  # Vector from sat1 to target (N x 3)
los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]  # Distances
los1_unit = los1 / los1_dist  # Unit vectors

los2 = target_pos - sat2_pos  # Vector from sat2 to target
los2_dist = np.linalg.norm(los2, axis=1)[:, np.newaxis]
los2_unit = los2 / los2_dist

# 3D Plot of trajectory, sats, and sample LOS rays at key steps (k=0, peak, impact-1)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Ballistic trajectory')
ax.scatter(sat1_pos[0, 0], sat1_pos[0, 1], sat1_pos[0, 2], c='b', label='Sat1')
ax.scatter(sat2_pos[0, 0], sat2_pos[0, 1], sat2_pos[0, 2], c='g', label='Sat2')

# Add LOS rays at k=0, peak_step, impact_step-1
key_steps = [0, np.argmax(target_pos[:, 2]), impact_step-1]
for k in key_steps:
    ax.plot([sat1_pos[k, 0], target_pos[k, 0]], [sat1_pos[k, 1], target_pos[k, 1]], [sat1_pos[k, 2], target_pos[k, 2]], 'b--')  # LOS from Sat1
    ax.plot([sat2_pos[k, 0], target_pos[k, 0]], [sat2_pos[k, 1], target_pos[k, 1]], [sat2_pos[k, 2], target_pos[k, 2]], 'g--')  # LOS from Sat2

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Ballistic Trajectory with Dual LEO Sat LOS (No Noise)')
ax.legend()
plt.show()

# Diagnostic print for key LOS distances
print(f"LOS1 distance at peak: {los1_dist[np.argmax(target_pos[:, 2])][0]:.2f} km")
print(f"LOS2 distance at impact: {los2_dist[-1][0]:.2f} km")