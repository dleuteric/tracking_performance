import numpy as np
import matplotlib.pyplot as plt


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


def estimate_covariance(s1, s2, true_target, sigma_theta, num_mc=1000):
    """Monte Carlo estimation of triangulation covariance R.

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


# Parameters (tweak these)
baseline = 500.0  # km between sats
s1 = np.array([0, 0, 1000])  # Sat 1 pos
s2 = np.array([baseline, 0, 1000])  # Sat 2
true_target = np.array([baseline / 2, 0, 100])  # Target at 100 km alt
sigma_theta = 1e-3  # 1 mrad angular noise
num_mc = 1000

# Run triangulation and cov estimate
u1_nom = (true_target - s1) / np.linalg.norm(true_target - s1)
u2_nom = (true_target - s2) / np.linalg.norm(true_target - s2)
r_nom = triangulate(s1, s2, u1_nom, u2_nom)  # Nominal (should match true)
R, rs_mc = estimate_covariance(s1, s2, true_target, sigma_theta, num_mc)

print("Nominal triangulated pos (km):", r_nom)
print("Covariance R (km^2):\n", R)

# Plot MC cloud for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rs_mc[:, 0], rs_mc[:, 1], rs_mc[:, 2], s=1, alpha=0.5)
ax.scatter(*true_target, color='r', label='True Target')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('MC Triangulation Cloud')
plt.legend()
plt.show()