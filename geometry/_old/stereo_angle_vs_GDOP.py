import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares  # For triangulation optimization


# Triangulation function from your first code (midpoint of shortest segment between skew lines)
def triangulate(s1, s2, u1, u2):
    """Compute midpoint of shortest segment between two skew lines.

    Args:
        s1, s2: (3,) sat positions (km).
        u1, u2: (3,) unit LOS vectors from s1,s2.

    Returns:
        r_mid: (3,) pseudo-position (km).
    """
    d = s2 - s1  # Baseline vector
    cross_u1_u2 = np.cross(u1, u2)
    denom = np.dot(cross_u1_u2, cross_u1_u2)
    if denom == 0:  # Parallel lines
        return (s1 + s2) / 2  # Fallback, though rare
    t1 = np.dot(np.cross(d, u2), cross_u1_u2) / denom
    t2 = np.dot(np.cross(d, u1), cross_u1_u2) / denom
    p1 = s1 + t1 * u1
    p2 = s2 + t2 * u2
    r_mid = (p1 + p2) / 2
    return r_mid


# MC covariance estimation from your first code (adapted for ballistic step k)
def estimate_cov_at_step(s1, s2, true_target, sigma_theta, num_mc=100):
    """Monte Carlo estimation of triangulation covariance R at a fixed point.

    Args:
        s1, s2: Sat positions.
        true_target: (3,) true target pos for nominal LOS.
        sigma_theta: Angular noise std dev (rad).
        num_mc: Number of MC trials.

    Returns:
        R: (3,3) covariance matrix (km^2).
        rs_mc: (num_mc,3) array of triangulated positions.
    """
    # Nominal unit LOS
    los1 = true_target - s1
    u1_nom = los1 / np.linalg.norm(los1)
    los2 = true_target - s2
    u2_nom = los2 / np.linalg.norm(los2)

    rs_mc = np.zeros((num_mc, 3))
    for i in range(num_mc):
        # Perturb directions: small angle approx in tangent plane
        pert1 = np.random.randn(2) * sigma_theta  # Two orthogonal dirs
        perp1_a = np.cross(u1_nom, [0, 0, 1])  # Arbitrary perp basis
        perp1_a /= np.linalg.norm(perp1_a) or 1
        perp1_b = np.cross(u1_nom, perp1_a)
        u1_pert = u1_nom + pert1[0] * perp1_a + pert1[1] * perp1_b
        u1_pert /= np.linalg.norm(u1_pert)

        pert2 = np.random.randn(2) * sigma_theta
        perp2_a = np.cross(u2_nom, [0, 0, 1])
        perp2_a /= np.linalg.norm(perp2_a) or 1
        perp2_b = np.cross(u2_nom, perp2_a)
        u2_pert = u2_nom + pert2[0] * perp2_a + pert2[1] * perp2_b
        u2_pert /= np.linalg.norm(u2_pert)

        rs_mc[i] = triangulate(s1, s2, u1_pert, u2_pert)

    # Covariance: mean-subtracted
    mean_r = np.mean(rs_mc, axis=0)
    diffs = rs_mc - mean_r
    R = np.dot(diffs.T, diffs) / (num_mc - 1)
    return R, rs_mc


# Parameters for ballistic trajectory (from our previous work)
dt = 1.0  # s
N = 700  # Number of steps to capture full arc
g = -0.0098  # km/s² (gravity in z)
initial_pos = np.array([0.0, 0.0, 0.0])  # [x, y, z] km at launch
initial_vel = np.array([7.0, 0.0, 3.0])  # [vx, vy, vz] km/s (horizontal boost + vertical climb)
sat_alt = 1000.0  # km (LEO altitude for sats)
angular_sigma = 0.001  # rad (~1 mrad angular noise std dev for LOS pointing)
num_mc = 100  # MC trials per subsample point for covariance
sub_interval = 50  # Adjust sampling frequency (lower = more points, longer run time)

# Propagate trajectory (as before)
target_pos = np.zeros((N, 3))
target_vel = np.zeros((N, 3))
target_pos[0] = initial_pos
target_vel[0] = initial_vel
impact_step = N
for k in range(1, N):
    accel_term = np.array([0.0, 0.0, g * dt])
    target_vel[k] = target_vel[k - 1] + accel_term
    next_pos = target_pos[k - 1] + target_vel[k - 1] * dt + 0.5 * accel_term * dt
    if next_pos[2] <= 0:
        impact_step = k
        break
    target_pos[k] = next_pos

# Trim to impact
target_pos = target_pos[:impact_step]
target_vel = target_vel[:impact_step]
N_actual = impact_step

# Updated sat positions (your preference)
sat1 = np.array([0.0, -300.0, sat_alt])  # Fixed for all steps (assume stationary)
sat2 = np.array([2000.0, 300.0, sat_alt+300])

# Subsample steps for MC and metrics (every sub_interval steps + impact)
sub_steps = list(range(0, N_actual, sub_interval)) + [N_actual - 1]
Rs = []
rs_mcs = []
stereo_angles = []
gdops = []
for k in sub_steps:
    true_target = target_pos[k]
    R, rs_mc = estimate_cov_at_step(sat1, sat2, true_target, angular_sigma, num_mc)
    Rs.append(R)
    rs_mcs.append(rs_mc)

    # Stereo angle: angle between nominal LOS vectors
    u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
    u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)
    cos_theta = np.dot(u1_nom, u2_nom)
    stereo_angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  # deg
    stereo_angles.append(stereo_angle)

    # Simplified GDOP for dual-sat (approx 1 / sin(theta/2) for position DOP)
    if stereo_angle > 0:
        gdop = 1 / np.sin(stereo_angle * np.pi / 360)  # Basic formula, adjust as needed
    else:
        gdop = np.inf
    gdops.append(gdop)

# 3D Plot with scattered MC points (dots) at subsampled points along trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'k-', label='Truth trajectory')
ax.scatter(sat1[0], sat1[1], sat1[2], c='b', label='Sat1')
ax.scatter(sat2[0], sat2[1], sat2[2], c='g', label='Sat2')
for i, k in enumerate(sub_steps):
    # Scatter MC points (dots) at this subsample
    ax.scatter(rs_mcs[i][:, 0], rs_mcs[i][:, 1], rs_mcs[i][:, 2], c='r', s=1, alpha=0.5,
               label='MC scattered points' if i == 0 else None)
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Ballistic Trajectory with MC Scattered Pseudo-Positions at Subsamples')
ax.legend()
plt.show()

# Separate plot for stereo angle and GDOP vs subsample steps
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(111)
ax1.plot(sub_steps, stereo_angles, 'b-', label='Stereo Angle (deg)')
ax1.set_xlabel('Step k')
ax1.set_ylabel('Stereo Angle (deg)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(sub_steps, gdops, 'r-', label='GDOP Approx')
ax2.set_ylabel('GDOP', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Stereo Angle and GDOP Evolution Along Trajectory')
fig.tight_layout()
plt.show()

# Diagnostic print for R, stereo, GDOP at peak subsample
peak_step = np.argmax(target_pos[:, 2])
closest_sub = sub_steps[np.argmin(np.abs(np.array(sub_steps) - peak_step))]
idx = sub_steps.index(closest_sub)
print(f"R at closest subsample to peak (step {closest_sub}):\n {Rs[idx]}")
print(f"Stereo angle at that step: {stereo_angles[idx]:.2f} deg")
print(f"GDOP at that step: {gdops[idx]:.2f}")