import numpy as np
import matplotlib.pyplot as plt

# Parameters for rudimentary ballistic trajectory: time step, gravity, initial conditions
dt = 1.0  # s
N = 700  # Increased number of steps to capture full arc (test vs 100)
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km at launch
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s (horizontal boost + vertical climb)

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

# Plot trajectory in x-z plane (y=0)
plt.figure(figsize=(8, 4))
plt.plot(target_pos[:impact_step, 0], target_pos[:impact_step, 2], 'k-', label='Ballistic trajectory (x-z)')
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
plt.title('Rudimentary Ballistic Missile Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# Diagnostic print for peak altitude and impact
peak_step = np.argmax(target_pos[:impact_step, 2])
print(f"Peak altitude: {np.max(target_pos[:impact_step, 2]):.2f} km at step {peak_step}")
print(f"Impact at step {impact_step-1} with x-range: {target_pos[impact_step-1, 0]:.2f} km")