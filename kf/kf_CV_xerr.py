import numpy as np
import matplotlib.pyplot as plt

# Parameters for 3D CV model: time step, true initial state [px, vx, py, vy, pz, vz], noise
dt = 1.0  # s
true_x0_3d = np.array([100.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # km, km/s (motion in x only for simplicity)
Q_3d = np.diag([0.01, 0.001, 0.01, 0.001, 0.01, 0.001])  # Process noise cov (diagonal for independent axes)
R_3d = np.diag([25.0, 25.0, 25.0])  # Measurement noise cov for 3D position (sigma=5 km isotropic)
N_3d = 100  # Number of steps

# State transition matrix F (6x6 for 3D CV): Identity with dt coupling pos-vel per axis
F_3d = np.eye(6)  # Start with identity
F_3d[0, 1] = dt  # x-pos to x-vel coupling
F_3d[2, 3] = dt  # y-pos to y-vel
F_3d[4, 5] = dt  # z-pos to z-vel

# Observation matrix H (3x6, measures positions only)
H_3d = np.array([
    [1, 0, 0, 0, 0, 0],  # px
    [0, 0, 1, 0, 0, 0],  # py
    [0, 0, 0, 0, 1, 0]  # pz
])

# Generate truth trajectory (3D CV, no process noise for truth)
truth_x_3d = np.zeros((N_3d, 6))
truth_x_3d[0] = true_x0_3d
for k in range(1, N_3d):
    truth_x_3d[k] = F_3d @ truth_x_3d[k - 1]  # Propagate state via matrix multiplication

# Simulate noisy 3D position measurements
np.random.seed(42)  # For reproducibility
z_3d = truth_x_3d[:, [0, 2, 4]] + np.random.randn(N_3d, 3) * np.sqrt(np.diag(R_3d))  # z = H x + noise (N_3d x 3)

# Initialize Kalman for 3D CV
x_hat_3d = np.zeros((N_3d, 6))
P_3d = np.zeros((N_3d, 6, 6))
x_hat_3d[0] = np.concatenate((z_3d[0], [0, 0, 0]))  # Init positions from first meas, vels=0
P_3d[0] = np.diag([R_3d[0, 0], 1, R_3d[1, 1], 1, R_3d[2, 2], 1])  # Init cov: meas var on pos, unc on vel

# Run Kalman for 3D CV (predict and update loop)
for k in range(1, N_3d):
    # Predict step: Propagate state and covariance
    x_minus = F_3d @ x_hat_3d[k - 1]  # Predicted state (6,)
    P_minus = F_3d @ P_3d[k - 1] @ F_3d.T + Q_3d  # Predicted cov (6x6)

    # Update step: Incorporate 3D measurement
    innovation = z_3d[k] - H_3d @ x_minus  # Residual (3,)
    S = H_3d @ P_minus @ H_3d.T + R_3d  # Innovation cov (3x3)
    K = P_minus @ H_3d.T @ np.linalg.inv(S)  # Kalman gain (6x3)
    x_hat_3d[k] = x_minus + K @ innovation  # Corrected state (6,)
    P_3d[k] = P_minus - K @ H_3d @ P_minus  # Corrected cov (6x6)

# Plot x-dimension for simplicity (motion axis; y/z should stay near 0)
plt.figure(figsize=(8, 4))
plt.plot(range(N_3d), truth_x_3d[:, 0], 'k--', label='Truth x-position')
plt.plot(range(N_3d), z_3d[:, 0], 'b-', label='Noisy x-measurements')
plt.plot(range(N_3d), x_hat_3d[:, 0], 'r-', label='Kalman x-position estimate')
plt.xlabel('Time step k')
plt.ylabel('X-position (km)')
plt.title('3D CV Model: X-Dimension Tracking')
plt.legend()
plt.grid(True)
plt.show()

# Diagnostic prints for confirmation
print(f"Truth final position [x,y,z]: {truth_x_3d[-1, [0, 2, 4]]}")
print(f"Final position estimate [x,y,z]: {x_hat_3d[-1, [0, 2, 4]]}")
print(f"Final velocity estimate [vx,vy,vz]: {x_hat_3d[-1, [1, 3, 5]]}")

# After the Kalman loop, add velocity plot for x-dimension (motion axis)
plt.figure(figsize=(8, 4))
plt.plot(range(N_3d), truth_x_3d[:, 1], 'k--', label='Truth x-velocity')
plt.plot(range(N_3d), x_hat_3d[:, 1], 'r-', label='Kalman x-velocity estimate')
plt.xlabel('Time step k')
plt.ylabel('X-velocity (km/s)')
plt.title('3D CV Model: X-Velocity Estimation')
plt.legend()
plt.grid(True)
plt.show()