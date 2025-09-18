import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares  # For triangulation optimization

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

# Compute LOS vectors from sats to target (unit vectors for direction, no noise yet)
los1 = target_pos - sat1_pos  # Vector from sat1 to target (N x 3)
los1_dist = np.linalg.norm(los1, axis=1)[:, np.newaxis]  # Distances (N x 1)
los1_unit = los1 / los1_dist  # Unit LOS vectors (N x 3)

los2 = target_pos - sat2_pos  # Vector from sat2 to target (N x 3)
los2_dist = np.linalg.norm(los2, axis=1)[:, np.newaxis]  # Distances (N x 1)
los2_unit = los2 / los2_dist  # Unit LOS vectors (N x 3)


# Function to triangulate position from two LOS rays (least-squares minimization of skew line distance)
def triangulate(sat1, u1, sat2, u2):
    # Objective: Find points P1 = sat1 + t1 * u1, P2 = sat2 + t2 * u2 minimizing ||P1 - P2||
    def residual(params):
        t1, t2 = params
        P1 = sat1 + t1 * u1
        P2 = sat2 + t2 * u2
        return P1 - P2  # Residual vector (3,)

    res = least_squares(residual, [0, 0], method='lm')  # Optimize t1, t2
    t1_opt, t2_opt = res.x
    P_opt = (sat1 + t1_opt * u1 + sat2 + t2_opt * u2) / 2  # Midpoint of closest points
    return P_opt


# Compute triangulated pseudo-positions for each step
pseudo_pos = np.zeros((N, 3))
for k in range(N):
    pseudo_pos[k] = triangulate(sat1_pos[k], los1_unit[k], sat2_pos[k], los2_unit[k])

# 3D Plot of truth, triangulated positions, sats, and sample LOS rays
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Truth trajectory')  # Truth path
ax.plot(pseudo_pos[:, 0], pseudo_pos[:, 1], pseudo_pos[:, 2], 'r--',
        label='Triangulated pseudo-position')  # Pseudo path
ax.scatter(sat1_pos[0, 0], sat1_pos[0, 1], sat1_pos[0, 2], c='b', label='Sat1')  # Sat1 position
ax.scatter(sat2_pos[0, 0], sat2_pos[0, 1], sat2_pos[0, 2], c='g', label='Sat2')  # Sat2 position

# Add sample LOS rays for k=0 and k=99 (scaled for visualization, e.g., to pseudo-pos)
for k in [0, 99]:
    ax.plot([sat1_pos[k, 0], pseudo_pos[k, 0]], [sat1_pos[k, 1], pseudo_pos[k, 1]], [sat1_pos[k, 2], pseudo_pos[k, 2]],
            'b--')  # LOS from Sat1
    ax.plot([sat2_pos[k, 0], pseudo_pos[k, 0]], [sat2_pos[k, 1], pseudo_pos[k, 1]], [sat2_pos[k, 2], pseudo_pos[k, 2]],
            'g--')  # LOS from Sat2

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Triangulation from Dual-Satellite LOS (No Noise)')
ax.legend()
plt.show()

# Diagnostic print for first and last pseudo-positions
print(f"Pseudo-position at k=0: {pseudo_pos[0]}")
print(f"Pseudo-position at k=99: {pseudo_pos[-1]}")