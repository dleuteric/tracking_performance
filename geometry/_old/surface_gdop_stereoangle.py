import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to compute stereo angle and GDOP for given baseline and target position (fixed y/z for simplicity)
def compute_gdop(baseline, target_x, sat_alt=1000.0, target_y=0.0, target_z=100.0):
    sat1 = np.array([0.0, -300.0, sat_alt])
    sat2 = np.array([baseline, 300.0, sat_alt])
    true_target = np.array([target_x, target_y, target_z])

    u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
    u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)
    cos_theta = np.dot(u1_nom, u2_nom)
    stereo_angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  # deg

    if stereo_angle > 0:
        gdop = 1 / np.sin(stereo_angle * np.pi / 360)  # Simplified GDOP approx
    else:
        gdop = np.inf
    gdop = np.clip(gdop, 0, 20)  # Clip inf to max 20 for plotting
    return gdop, stereo_angle


# Grid for surface: baselines and target x
baselines = np.linspace(50, 500, 20)
target_xs = np.linspace(0, 4000, 50)
B, TX = np.meshgrid(baselines, target_xs)
GDOP = np.zeros_like(B)
Stereo = np.zeros_like(B)

# Compute over grid
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        GDOP[i, j], Stereo[i, j] = compute_gdop(B[i, j], TX[i, j])

# Surface plot for GDOP with contours
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(B, TX, GDOP, cmap='viridis', alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='GDOP')
ax.set_xlabel('Baseline (km)')
ax.set_ylabel('Target X (km)')
ax.set_zlabel('GDOP (clipped at 20)')
ax.set_title('GDOP Surface vs Baseline and Target X')

# Add 2D contour projection on xy plane for thresholds
cset = ax.contourf(B, TX, GDOP, zdir='z', offset=np.min(GDOP) - 1, levels=[0, 3, 5, 20],
                   colors=['green', 'yellow', 'red'], alpha=0.5)
fig.colorbar(cset, ax=ax, shrink=0.5, aspect=5, label='GDOP Thresholds (<3 green, 3-5 yellow, >5 red)')
plt.show()

# Diagnostic min GDOP (clipped)
min_gdop = np.min(GDOP)
print(f"Minimum GDOP in surface (clipped): {min_gdop:.2f}")