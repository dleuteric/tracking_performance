import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# -----------------------------
# Simulation / dummy data setup
# -----------------------------
dt = 1.0  # seconds
n_steps = 50

# truth
x0_true = np.array([7000., 0., 0.])       # km ECI-ish
v_true  = np.array([0.5, -0.2, 0.1])      # km/s
t = np.arange(n_steps)*dt
r_true = x0_true + np.outer(t, v_true)    # (N,3)

# pretend geometry module: produce noisy r_tilde + R
def fake_geometry_measurement(r, k):
    # geometry quality oscillates with time: scale noise
    geom_scale = 1.0 + 2.0*np.abs(np.sin(0.1*k))
    # base 1-sigma errors [x,y,z] in km
    sigmas = np.array([0.1, 0.1, 0.3]) * geom_scale
    cov = np.diag(sigmas**2)
    meas = r + np.random.randn(3)*sigmas
    return meas, cov

meas_list = []
np.random.seed(0)
for k in range(n_steps):
    z, Rk = fake_geometry_measurement(r_true[k], k)
    meas_list.append((z, Rk))

# -----------------------------
# Build CV Kalman filter (6-state)
# -----------------------------
kf = KalmanFilter(dim_x=6, dim_z=3)

# F
F = np.eye(6)
F[0,3] = F[1,4] = F[2,5] = dt
kf.F = F

# H (position only)
H = np.zeros((3,6))
H[0,0] = H[1,1] = H[2,2] = 1.0
kf.H = H

# Q from white acceleration noise (choose accel std ~0.01 km/s^2)
sigma_a2 = (0.01)**2
Q = Q_discrete_white_noise(dim=2, dt=dt, var=sigma_a2, block_size=3, order_by_dim=True)
kf.Q = Q

# Initial state from first measurement:
z0, R0 = meas_list[0]
x_init = np.zeros(6)
x_init[0:3] = z0             # position
# velocities remain 0
kf.x = x_init

# Initial covariance: position from R0, velocity large uncertainty
P0 = np.zeros((6,6))
P0[0:3,0:3] = R0
vel_var = (10.0)**2          # (km/s)^2 large
P0[3:,3:] = np.eye(3)*vel_var
kf.P = P0

# -----------------------------
# Filtering loop
# -----------------------------
est_states = np.zeros((n_steps,6))
est_covs   = np.zeros((n_steps,6,6))

for k,(z,Rk) in enumerate(meas_list):
    kf.predict()             # propagate
    kf.update(z, R=Rk)       # measurement-specific covariance
    est_states[k] = kf.x
    est_covs[k]   = kf.P

# est_states[: ,0:3] are positions; [: ,3:6] are velocities
# est_covs gives full uncertainty; diag elements -> 1-sigma^2

pos_est = est_states[:,0:3]      # km
vel_est = est_states[:,3:6]      # km/s
pos_std = np.sqrt(est_covs[:, (0,1,2), (0,1,2)])  # 1σ per axis

rmse_pos = np.sqrt(np.mean(np.sum((pos_est - r_true)**2, axis=1)))
rmse_vel = np.sqrt(np.mean(np.sum((vel_est - v_true)**2, axis=1)))
print("RMSE pos [km]:", rmse_pos)
print("RMSE vel [km/s]:", rmse_vel)