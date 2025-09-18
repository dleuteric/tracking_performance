import numpy as np
import matplotlib.pyplot as plt

# Target: movimento costante in 2D
dt = 1.0
n_steps = 50
true_positions = []
x0 = np.array([0, 0])
v = np.array([1, 0.5])
for i in range(n_steps):
    pos = x0 + v * i * dt
    true_positions.append(pos)

# Sensori: 3 sensori che misurano (x, y) con rumore
num_sensors = 3
measurement_noise_std = 3.0
np.random.seed(0)

def noisy_measurement(pos):
    return pos + np.random.normal(0, measurement_noise_std, size=(2,))

# Kalman Filter: stato = [x, y, vx, vy]
x = np.array([[0], [0], [0], [0]])  # stato iniziale
P = np.eye(4) * 500

F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

R = np.eye(2) * measurement_noise_std**2
Q = np.eye(4) * 0.1

# Logging
estimates = []

for i in range(n_steps):
    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    # Fondere 3 misure: media pesata (possono anche essere usate singolarmente con update iterati)
    z_total = np.zeros((2, 1))
    S_inv_total = np.zeros((2, 2))
    for _ in range(num_sensors):
        z = noisy_measurement(true_positions[i]).reshape((2, 1))
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

    estimates.append(x[:2].flatten())

# Convert to arrays
true_x = [p[0] for p in true_positions]
true_y = [p[1] for p in true_positions]
est_x = [e[0] for e in estimates]
est_y = [e[1] for e in estimates]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, label="True Position")
plt.plot(est_x, est_y, label="Kalman Estimate")
plt.title("2D Kalman Filter with 3 Sensors")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()