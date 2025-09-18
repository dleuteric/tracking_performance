import numpy as np
import matplotlib.pyplot as plt

# Parameters (tweakable: boost duration, thrust accel magnitudes)
dt = 1.0  # s
N = 800  # Increased steps to capture extended arc if needed
g = -0.00981  # km/s² (refined gravity)
thrust_duration = 100  # s (boost phase length)
thrust_accel = np.array([0.07, 0.0, 0.03])  # km/s² [ax, ay, az] during boost (~7g horiz, 3g vert)
initial_pos = np.array([0.0, 0.0, 0.0])  # km
initial_vel = np.array([0.0, 0.0, 0.0])  # km/s (start from rest, build during boost)

# Propagate: boost + coast, stop at z<=0
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N
for k in range(1, N):
    if (k - 1) * dt < thrust_duration:  # Boost phase: thrust + gravity
        accel = thrust_accel + np.array([0.0, 0.0, g])
    else:  # Coast: gravity only
        accel = np.array([0.0, 0.0, g])
    accel_term = accel * dt
    target_vel[k] = target_vel[k - 1] + accel_term  # Update vel
    next_pos = target_pos[k - 1] + target_vel[k - 1] * dt + 0.5 * accel_term * dt  # Position update (trapezoidal integration)
    if next_pos[2] <= 0:  # Ground impact
        impact_step = k
        break
    target_pos[k] = next_pos

# Trim arrays
target_pos = target_pos[:impact_step]
target_vel = target_vel[:impact_step]

# Quick checks (should match your prior run)
peak_z = np.max(target_pos[:, 2])
final_vel_x = target_vel[-1, 0]
print(f"Peak height: {peak_z:.2f} km")  # ~308.72 km
print(f"Impact step: {impact_step}, Final vx: {final_vel_x:.2f} km/s")  # 557, 7.00 km/s

# Plot trajectory (x vs z; y=0 so 2D sufficient)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(target_pos[:, 0], target_pos[:, 2], 'k-', label='Boosted trajectory')
# Mark end of boost with a red dot instead of vertical line
boost_end_idx = int(thrust_duration / dt)
ax.plot(target_pos[boost_end_idx, 0], target_pos[boost_end_idx, 2], 'ro', label='End of boost')
ax.set_xlabel('Downrange X (km)')
ax.set_ylabel('Altitude Z (km)')
ax.set_title('2D Ballistic Trajectory with Boost Phase')
ax.legend()
ax.grid(True)
plt.show()