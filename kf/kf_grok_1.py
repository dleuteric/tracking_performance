import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


def cv_kalman_filter_3d(dt, sigma_a, R_scalar):
    """Initialize 3D CV Kalman filter.

    Args:
        dt: Timestep (s).
        sigma_a: Process noise accel std dev (km/s^2).
        R_scalar: Measurement noise variance scalar (km^2), for diag(R).

    Returns:
        kf: Configured KalmanFilter object.
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x,vx,y,vy,z,vz]; Meas: [x,y,z]

    # Transition matrix F: block-diagonal for three axes
    f_1d = np.array([[1, dt], [0, 1]])
    kf.F = np.block([[f_1d, np.zeros((2, 2)), np.zeros((2, 2))],
                     [np.zeros((2, 2)), f_1d, np.zeros((2, 2))],
                     [np.zeros((2, 2)), np.zeros((2, 2)), f_1d]])

    # Process noise Q: block-diagonal, same q_1d for each axis
    q_1d = sigma_a ** 2 * np.array([[dt ** 3 / 3, dt ** 2 / 2],
                                    [dt ** 2 / 2, dt]])
    kf.Q = np.block([[q_1d, np.zeros((2, 2)), np.zeros((2, 2))],
                     [np.zeros((2, 2)), q_1d, np.zeros((2, 2))],
                     [np.zeros((2, 2)), np.zeros((2, 2)), q_1d]])

    # Measurement matrix H and noise R
    kf.H = np.array([[1, 0, 0, 0, 0, 0],  # Measure x_pos
                     [0, 0, 1, 0, 0, 0],  # Measure y_pos
                     [0, 0, 0, 0, 1, 0]])  # Measure z_pos
    kf.R = R_scalar * np.eye(3)  # Isotropic noise

    # Initial state and cov (wide prior)
    kf.x = np.zeros(6)  # Start at origin, zero vel
    kf.P *= 1000  # Large initial uncertainty
    return kf


def simulate_trajectory_3d(T, dt, v_true):
    """Generate 3D truth positions and velocities.

    Args:
        T: Total time (s).
        dt: Timestep (s).
        v_true: True velocity [vx, vy, vz] (km/s).

    Returns:
        times: Array of times.
        truths: (N,6) array of [x,vx,y,vy,z,vz] truths.
    """
    times = np.arange(0, T, dt)
    N = len(times)
    truths = np.zeros((N, 6))
    truths[:, 1] = v_true[0]  # Constant vx
    truths[:, 3] = v_true[1]  # vy
    truths[:, 5] = v_true[2]  # vz
    for i in range(1, N):
        truths[i, 0] = truths[i - 1, 0] + v_true[0] * dt  # Integrate x
        truths[i, 2] = truths[i - 1, 2] + v_true[1] * dt  # y
        truths[i, 4] = truths[i - 1, 4] + v_true[2] * dt  # z
    return times, truths


def run_monte_carlo_3d(num_runs, T, dt, sigma_a, R_scalar, v_true):
    """Run MC simulations, compute RMSE per axis.

    Args as above + num_runs.

    Returns:
        times: Times.
        rmse_xyz: (3, N array of mean RMSE for x,y,z positions.
    """
    times, truths = simulate_trajectory_3d(T, dt, v_true)
    N = len(times)
    errors = np.zeros((num_runs, len(times), 3))  # Pos errors per run, time, axis
    for run in range(num_runs):
        kf = cv_kalman_filter_3d(dt, sigma_a, R_scalar)

        # Noisy measurements: z_k] = H * truth + noise ~ N(0,R)
        noises = np.sqrt(R_scalar) * np.random.randn(len(times), 3)  # for [x,y,z]
        zs = truths[:, [0, 2, 4]] + noises

        # Initial state from first meas
        kf.x[[0, 2, 4]] = zs[0]

        estimates = np.zeros((len(times), 6))
        for k in range(len(times)):
            kf.predict()
            kf.update(zs[k])
            estimates[k] = kf.x

        # Pos errors per axis
        pos_err = estimates[:, [0, 2, 4]] - truths[:, [0, 2, 4]]
        errors[run] = pos_err  # Shape: (time,3)
    rmse_xyz = np.sqrt(np.mean(errors ** 2, axis=0)).T
    return times, rmse_xyz


# Parameters (tweak these)
T = 100.0 # Total sim time(s)
dt = 1.0  # Timestep (s)
sigma_a = 1e-3  # Process noise (km/s^2)
v_true = np.array([0.1, 0.1, 0.1])  # Diagonal vel for symmetry
num_runs = 100
R_scalar = 0.1  # km^2

# Run and plot
times, rmse_xyz = run_monte_carlo_3d(num_runs, T, dt, sigma_a, R_scalar, v_true)
plt.figure(figsize=(8, 5))
for i, label in enumerate(['x', 'y', 'z']):
    plt.plot(times, rmse_xyz[i], label=f'RMSE {label}')
plt.xlabel('Time (s)')
plt.ylabel('Per-Axis Position RMSE (km)')
plt.title('3D RMSE Symmetry Across Axes')
plt.legend()
plt.grid(True)
plt.show()