import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from module3 import triangulate, estimate_covariance  # Assume saved as module3.py


def cv_kalman_filter_3d(dt, sigma_a):
    """3D CV KF, R set per update."""
    kf = KalmanFilter(dim_x=6, dim_z=3)
    f_1d = np.array([[1, dt], [0, 1]])
    kf.F = np.block([[f_1d if i == j else np.zeros((2, 2)) for j in range(3)] for i in range(3)])
    q_1d = sigma_a ** 2 * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    kf.Q = np.block([[q_1d if i == j else np.zeros((2, 2)) for j in range(3)] for i in range(3)])
    kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])  # [x,y,z]
    kf.x = np.zeros(6)
    kf.P *= 1000
    return kf


def simulate_with_triangulation(T, dt, sigma_a, s1, s2, sigma_theta, v_true, num_runs):
    times = np.arange(0, T, dt)
    N = len(times)
    truths = np.zeros((N, 6))
    truths[0, [0, 2, 4]] = [250, 0, 100]  # Start pos
    truths[:, 1] = v_true[0]  # vx
    truths[:, 3] = v_true[1]  # vy
    truths[:, 5] = v_true[2]  # vz
    for i in range(1, N):
        truths[i, [0, 2, 4]] = truths[i - 1, [0, 2, 4]] + v_true * dt

    # Precompute R (assume fixed geometry, MC once at start pos)
    R, _ = estimate_covariance(s1, s2, truths[0, [0, 2, 4]], sigma_theta)

    rmse_xyz = np.zeros((3, N))
    innov_vars = np.zeros((N, 3))  # Track diag(S_k) avg
    for run in range(num_runs):
        kf = cv_kalman_filter_3d(dt, sigma_a)
        estimates = np.zeros((N, 6))
        for k in range(N):
            # Triangulate noisy z_k
            true_pos = truths[k, [0, 2, 4]]
            u1_nom = (true_pos - s1) / np.linalg.norm(true_pos - s1)
            u2_nom = (true_pos - s2) / np.linalg.norm(true_pos - s2)
            # Add noise (simplified, full perturb as in module3)
            pert1, pert2 = np.random.randn(2) * sigma_theta, np.random.randn(2) * sigma_theta
            # ... (copy perturb code from estimate_covariance for u1_pert, u2_pert)
            # For brevity, assume implemented
            z_k = triangulate(s1, s2, u1_pert, u2_pert)

            kf.predict()
            kf.R = R  # Set varying if geometry changes, fixed here
            kf.update(z_k)
            estimates[k] = kf.x

            if run == 0:  # Track innov var from first run
                S = kf.H @ kf.P @ kf.H.T + kf.R
                innov_vars[k] = np.diag(S)

        pos_err = estimates[:, [0, 2, 4]] - truths[:, [0, 2, 4]]
        rmse_xyz += np.abs(pos_err).T / num_runs  # Avg abs for approx RMSE
    return times, rmse_xyz, np.mean(innov_vars, axis=0)  # Wait, avg over time? No, innov_vars is (N,3)


# Params
T, dt, sigma_a = 100, 1, 1e-3
s1, s2 = np.array([0, 0, 1000]), np.array([500, 0, 1000])
sigma_theta = 1e-3
v_true = np.array([0.1, 0, 0])  # Move along X
num_runs = 100

times, rmse_xyz, innov_vars = simulate_with_triangulation(T, dt, sigma_a, s1, s2, sigma_theta, v_true, num_runs)

fig, axs = plt.subplots(2, 1)
for i, lab in enumerate(['x', 'y', 'z']):
    axs[0].plot(times, rmse_xyz[i], label=f'RMSE {lab}')
axs[0].set_title('Position RMSE')
axs[0].legend()
for i, lab in enumerate(['x', 'y', 'z']):
    axs[1].plot(times, innov_vars[:, i], label=f'Innov Var {lab}')
axs[1].set_title('Innovation Variance')
axs[1].legend()
plt.show()