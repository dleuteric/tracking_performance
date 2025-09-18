import numpy as np
import matplotlib.pyplot as plt

# Section 1: Constant Value Estimation (from Steps 1-4)
# Parameters: true value, noise std dev, number of measurements
mu = 100.0  # km, e.g., target altitude
sigma = 5.0  # km, measurement noise
N = 500  # number of measurements

# Generate noisy measurements
np.random.seed(42)  # for reproducibility
y = mu + sigma * np.random.randn(N)

# Compute sample mean (numerical average)
y_mean = np.mean(y)

# Compute recursive average filter estimates
mu_hat = np.zeros(N)
mu_hat[0] = y[0]  # Initialize with first measurement
for k in range(1, N):
    mu_hat[k] = mu_hat[k - 1] + (1 / (k + 1)) * (y[k] - mu_hat[k - 1])  # Recursive update

# Kalman filter parameters (for constant state)
Q = 0.0  # Process noise variance (zero for constant)
R = sigma ** 2  # Measurement noise variance

# Initialize Kalman estimates and covariance
x_hat_kf = np.zeros(N)
P_kf = np.zeros(N)
x_hat_kf[0] = y[0]
P_kf[0] = R

# Run Kalman filter updates
for k in range(1, N):
    # Predict (identity for constant state)
    x_minus = x_hat_kf[k - 1]
    P_minus = P_kf[k - 1] + Q

    # Update
    K = P_minus / (P_minus + R)  # Kalman gain
    x_hat_kf[k] = x_minus + K * (y[k] - x_minus)
    P_kf[k] = (1 - K) * P_minus

# Section 2: Constant Velocity (CV) Model (from Step 5)
# Parameters for CV model: time step, true initial state, process/measurement noise
dt = 100.0  # s
true_x0 = np.array([100.0, 0.1])  # [p km, v km/s]
Q_cv = np.array([[0.01, 0.0], [0.0, 0.001]])  # Process noise cov (small for near-CV)
R_cv = 0.5  # Measurement noise var (sigma=5 km, position only)
N_cv = 500  # Number of steps

# Generate truth trajectory (CV model, no process noise for truth)
truth_x = np.zeros((N_cv, 2))
truth_x[0] = true_x0
for k in range(1, N_cv):
    truth_x[k, 0] = truth_x[k - 1, 0] + dt * truth_x[k - 1, 1]  # p_k = p_{k-1} + dt v_{k-1}
    truth_x[k, 1] = truth_x[k - 1, 1]  # v_k = v_{k-1}

# Simulate noisy position measurements with consistent seed
np.random.seed(42)  # Align randomness for reproducibility
z_cv = truth_x[:, 0] + np.sqrt(R_cv) * np.random.randn(N_cv)  # z = H x + nu, H=[1 0]

# Initialize Kalman for CV
F = np.array([[1, dt], [0, 1]])  # State transition
H = np.array([[1, 0]])  # Observation
x_hat_cv = np.zeros((N_cv, 2))
P_cv = np.zeros((N_cv, 2, 2))
x_hat_cv[0] = [z_cv[0], 0.0]  # Init pos with first meas, vel=0
P_cv[0] = np.array([[R_cv, 0], [0, 1]])  # Init cov, larger vel uncertainty

# Run Kalman for CV (predict and update with shape fixes)
for k in range(1, N_cv):
    # Predict
    x_minus = F @ x_hat_cv[k - 1]
    P_minus = F @ P_cv[k - 1] @ F.T + Q_cv

    # Update
    Hx = H @ x_minus  # (1,)
    innovation = z_cv[k] - np.squeeze(Hx)  # Ensure scalar
    HPHT = H @ P_minus @ H.T  # (1,1)
    S = np.squeeze(HPHT) + R_cv  # Ensure scalar
    K = (P_minus @ H.T) / S  # (2,1)
    x_hat_cv[k] = x_minus + np.squeeze(K * innovation)  # Squeeze to (2,) for addition
    P_cv[k] = P_minus - K @ (H @ P_minus)  # K (2,1) for matrix mul, result (2,2)

# Plot CV results: Position and Velocity separately for clarity
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# Position plot
axs[0].plot(range(N_cv), truth_x[:, 0], 'k--', label='Truth position')
axs[0].plot(range(N_cv), z_cv, 'b-', label='Noisy measurements (connected)')
axs[0].plot(range(N_cv), x_hat_cv[:, 0], 'r-', label='Kalman position estimate')
axs[0].set_xlabel('Time step k')
axs[0].set_ylabel('Position (km)')
axs[0].set_title('Constant Velocity Model: Position Tracking')
axs[0].legend()
axs[0].grid(True)

# Velocity plot
axs[1].plot(range(N_cv), truth_x[:, 1], 'k--', label='Truth velocity (0.1 km/s)')
axs[1].plot(range(N_cv), x_hat_cv[:, 1], 'r-', label='Kalman velocity estimate')
axs[1].set_xlabel('Time step k')
axs[1].set_ylabel('Velocity (km/s)')
axs[1].set_title('Constant Velocity Model: Velocity Estimation')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Diagnostic prints for confirmation
print(f"Truth position at end: {truth_x[-1, 0]:.2f} km")
print(f"Final CV position estimate: {x_hat_cv[-1, 0]:.2f} km")
print(f"Final CV velocity estimate: {x_hat_cv[-1, 1]:.4f} km/s")