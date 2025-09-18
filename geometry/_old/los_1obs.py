import numpy as np
import matplotlib.pyplot as plt

# Parameters: time step, target truth trajectory (simplified 1D motion in x at 100 km altitude), sat altitudes
dt = 1.0  # s
N = 100  # Number of steps
target_alt = 100.0  # km (hypersonic threat altitude)
sat_alt = 1000.0  # km (LEO altitude)
true_vx = 0.1  # km/s (target velocity in x)
target_pos = np.zeros((N, 3))  # Target truth positions [x, y, z] km
for k in range(N):
    target_pos[k] = [100.0 + k * dt * true_vx, 0.0, target_alt]  # Linear motion in x, fixed y/z

# Simple sat positions: sat1 fixed at [100, 0, sat_alt], sat2 at [150, 0, sat_alt] for baseline
sat1_pos = np.full((N, 3), [100.0, 0.0, sat_alt])  # Constant position (assume stationary for simplicity)
sat2_pos = np.full((N, 3), [150.0, 0.0, sat_alt])  # Second sat for triangulation baseline ~50 km

# Compute LOS vectors from sat1 to target (unit vectors for direction)
los1 = target_pos - sat1_pos  # Vector from sat to target (N x 3)
los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]  # Distances (N x 1)
los1_unit = los1 / los1_dist  # Unit LOS vectors (N x 3)

# Plot target and sat1 positions with LOS lines for first and last steps (visual check)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Target truth trajectory')
ax.scatter(sat1_pos[0, 0], sat1_pos[0, 1], sat1_pos[0, 2], c='b', label='Sat1 position')
ax.plot([sat1_pos[0, 0], target_pos[0, 0]], [sat1_pos[0, 1], target_pos[0, 1]], [sat1_pos[0, 2], target_pos[0, 2]], 'r--', label='LOS at k=0')
ax.plot([sat1_pos[-1, 0], target_pos[-1, 0]], [sat1_pos[-1, 1], target_pos[-1, 1]], [sat1_pos[-1, 2], target_pos[-1, 2]], 'g--', label='LOS at k=99')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite LOS to Target Trajectory')
ax.legend()
plt.show()