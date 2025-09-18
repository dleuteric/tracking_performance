import numpy as np
import matplotlib.pyplot as plt

# Parametri simulazione
dt = 1  # freq. di osservazione, Hz
n_steps = 50
true_position = [0 + i * dt for i in range(n_steps)]
true_velocity = 1.0
measurement_noise_std = 2.0

# Genera misure rumorose
np.random.seed(42)
measurements = [x + np.random.normal(0, measurement_noise_std) for x in true_position]

# Inizializza stato e matrici Kalman
x = np.array([[0], [0]])  # stato iniziale: posizione, velocità
P = np.eye(2) * 500        # incertezza iniziale grande

F = np.array([[1, dt],
              [0, 1]])     # modello di transizione
H = np.array([[1, 0]])     # osserviamo solo la posizione
R = measurement_noise_std**2
Q = np.array([[0.01, 0], [0, 0.01]])  # rumore di processo

# Logging
estimated_positions = []
estimated_velocities = []

# Loop di stima
for z in measurements:
    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    # Update
    y = np.array([[z]]) - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    estimated_positions.append(x[0, 0])
    estimated_velocities.append(x[1, 0])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(true_position, label="True Position")
plt.plot(measurements, label="Measurements", linestyle="dotted")
plt.plot(estimated_positions, label="Kalman Estimate")
plt.legend()
plt.title("1D Kalman Filter: Position Tracking")
plt.xlabel("Time Step")
plt.ylabel("Position [m]")
plt.grid()
plt.show()