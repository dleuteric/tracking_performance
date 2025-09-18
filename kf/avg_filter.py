import numpy as np
import matplotlib.pyplot as plt

# Parameters: true value, noise std dev, number of measurements
mu = 100.0  # km, e.g., target altitude
sigma = 5.0  # km, measurement noise
N = 100     # number of measurements

# Generate noisy measurements
np.random.seed(42)  # for reproducibility
y = mu + sigma * np.random.randn(N)

# Plot noisy measurements as connected points (interpolated line)
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, N+1), y, 'b-', label='Noisy measurements (connected)')
# plt.xlabel('Measurement index k')
# plt.ylabel('Value (km)')
# plt.title('Noisy Measurements of Constant Value')
# plt.legend()
# plt.grid(True)
# #plt.show()

# Print sample mean for reference
print(f"Sample mean: {np.mean(y):.2f} km")

# Compute sample mean
y_mean = np.mean(y)

# Plot noisy measurements and overlay the numerical average
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, N+1), y, 'b-', label='Noisy measurements (connected)')
# plt.axhline(y=y_mean, color='g', linestyle='--', label=f'Numerical average ({y_mean:.2f} km)')
# plt.xlabel('Measurement index k')
# plt.ylabel('Value (km)')
# plt.title('Noisy Measurements with Numerical Average Overlay')
# plt.legend()
# plt.grid(True)
# plt.show()

# Compute recursive average filter estimates
mu_hat = np.zeros(N)
mu_hat[0] = y[0]  # Initialize with first measurement
for k in range(1, N):
    mu_hat[k] = mu_hat[k-1] + (1/(k+1)) * (y[k] - mu_hat[k-1])  # Recursive update

# Plot all overlays: noisy data, numerical average, recursive filter convergence
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, N+1), y, 'b-', label='Noisy measurements (connected)')
# plt.axhline(y=y_mean, color='g', linestyle='--', label=f'Numerical average ({y_mean:.2f} km)')
# plt.plot(range(1, N+1), mu_hat, 'r-', label='Recursive average filter')
# plt.xlabel('Measurement index k')
# plt.ylabel('Value (km)')
# plt.title('Noisy Measurements with Averages Overlay')
# plt.legend()
# plt.grid(True)
# plt.show()

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

# Update plot to overlay Kalman filter (should match recursive average)
plt.figure(figsize=(8, 4))
plt.plot(range(1, N + 1), y, 'b-', label='Noisy measurements (connected)')
plt.axhline(y=y_mean, color='g', linestyle='--', label=f'Numerical average ({y_mean:.2f} km)')
plt.plot(range(1, N + 1), mu_hat, 'r-', label='Recursive average filter')
plt.plot(range(1, N + 1), x_hat_kf, 'm--', label='Kalman filter (constant model)')
plt.xlabel('Measurement index k')
plt.ylabel('Value (km)')
plt.title('Noisy Measurements with Averages and Kalman Overlay')
plt.legend()
plt.grid(True)
plt.show()

