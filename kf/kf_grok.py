import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


def cv_kalman_filter(dt, sigma_a, R_scalar):
    """Initialize 2D CV Kalman filter.

    Args:
        dt: Timestep (s).
        sigma_a: Process noise accel std dev (km/s^2).
        R_scalar: Measurement noise variance scalar (km^2), for diag(R).

    Returns:
        kf: Configured KalmanFilter object.
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, vx, y, vy]; Meas: [x, y]

    # Transition matrix F from kinematics
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    # Process noise Q from white accel assumption
    q_1d = sigma_a ** 2 * np.array([[dt ** 3 / 3, dt ** 2 / 2],
                                    [dt ** 2 / 2, dt]])  # 1D block
    kf.Q = np.block([[q_1d, np.zeros((2, 2))],
                     [np.zeros((2, 2)), q_1d]])  # Block-diagonal for x/y

    # Measurement matrix H and noise R
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])
    kf.R = R_scalar * np.eye(2)  # Assume isotropic noise

    # Initial state and cov (wide prior)
    kf.x = np.array([0, 0, 0, 0])  # Start at origin, zero vel
    kf.P *= 1000  # Large initial uncertainty
    return kf


def simulate_trajectory(T, dt, v_true):
    """Generate truth positions and velocities.

    Args:
        T: Total time (s).
        dt: Timestep (s).
        v_true: True velocity [vx, vy] (km/s).

    Returns:
        times: Array of times.
        truths: (N,4) array of [x, vx, y, vy] truths.
    """
    times = np.arange(0, T, dt)
    N = len(times)
    truths = np.zeros((N, 4))
    truths[:, 1] = v_true[0]  # Constant vx
    truths[:, 3] = v_true[1]  # Constant vy
    for i in range(1, N):
        truths[i, 0] = truths[i - 1, 0] + v_true[0] * dt  # Integrate x
        truths[i, 2] = truths[i - 1, 2] + v_true[1] * dt  # Integrate y
    return times, truths


def run_monte_carlo(num_runs, T, dt, sigma_a, R_scalar, v_true):
    """Run MC simulations, compute position RMSE.

    Args as above + num_runs.

    Returns:
        times: Times.
        rmse_pos: (N,) mean position RMSE over runs.
    """
    times, truths = simulate_trajectory(T, dt, v_true)
    N = len(times)
    errors = np.zeros((num_runs, N))  # Position error norms
    for run in range(num_runs):
        kf = cv_kalman_filter(dt, sigma_a, R_scalar)

        # Generate noisy measurements: z_k = H * truth + noise ~ N(0,R)
        noises = np.sqrt(R_scalar) * np.random.randn(N, 2)  # For [x,y]
        zs = truths[:, [0, 2]] + noises  # Mimic direct observer measurements

        # Initial state from first noisy meas
        kf.x[:2] = [zs[0, 0], 0]  # Rough init x,vx
        kf.x[2:] = [zs[0, 1], 0]  # y,vy

        estimates = np.zeros((N, 4))
        for k in range(N):
            kf.predict()
            kf.update(zs[k])
            estimates[k] = kf.x

        # Position error: sqrt( (x_est - x_true)^2 + (y_est - y_true)^2 )
        pos_err = np.linalg.norm(estimates[:, [0, 2]] - truths[:, [0, 2]], axis=1)
        errors[run] = pos_err
    rmse_pos = np.mean(errors, axis=0)  # Avg over runs
    return times, rmse_pos


# Parameters (tweak these)
T = 100  # Total sim time (s)
dt = 1  # Timestep (s), measurements each second
sigma_a = 1e-3  # Process noise (km/s^2)
v_true = [0.1, 0.1]  # True vel [vx, vy] km/s
num_runs = 100  # For RMSE stats

# Run for different R scalars
R_values = [0.01, 0.1, 1.0]  # Measurement variance (km^2)
plt.figure()
for R_scalar in R_values:
    times, rmse = run_monte_carlo(num_runs, T, dt, sigma_a, R_scalar, v_true)
    plt.plot(times, rmse, label=f'R_scalar={R_scalar}')
plt.xlabel('Time (s)')
plt.ylabel('Position RMSE (km)')
plt.title('RMSE vs Time for Different Measurement Noises')
plt.legend()
plt.grid(True)
plt.show()