import numpy as np
import matplotlib.pyplot as plt

# Parameters for MCMC: target fixed at mid-trajectory, sat geometry
target_x = 2000.0  # km (mid-arc)
target_y = 0.0
target_z = 100.0
sat_alt = 1000.0  # km
mean_baseline = 800.0  # km (prior mean)
std_baseline = 100.0  # km (prior std)
num_samples = 10000  # MCMC chain length
burn_in = 1000  # Discard initial samples
gdop_threshold = 3.0  # GDOP <3 for "low error"


# Function to compute GDOP for given baseline (fixed target/sats y/z)
def compute_gdop(baseline):
    sat1 = np.array([0.0, -300.0, sat_alt])
    sat2 = np.array([baseline, 300.0, sat_alt])
    true_target = np.array([target_x, target_y, target_z])
    u1_nom = (true_target - sat1) / np.linalg.norm(true_target - sat1)
    u2_nom = (true_target - sat2) / np.linalg.norm(true_target - sat2)
    cos_theta = np.dot(u1_nom, u2_nom)
    stereo_angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  # deg
    if stereo_angle > 0:
        gdop = 1 / np.sin(stereo_angle * np.pi / 360)
    else:
        gdop = np.inf
    return gdop


# Metropolis-Hastings MCMC to sample baselines from prior, compute GDOP distribution
np.random.seed(42)
chain = np.zeros(num_samples)
chain[0] = mean_baseline  # Start at mean
accept_count = 0

for i in range(1, num_samples):
    current = chain[i - 1]
    proposal = current + np.random.normal(0, std_baseline / 10)  # Small step proposal
    if proposal <= 0:  # Reject negative baselines
        chain[i] = current
        continue

    # Prior ratio (normal prior, symmetric so 1 if proposal in range)
    prior_ratio = np.exp(-0.5 * ((proposal - mean_baseline) ** 2 - (current - mean_baseline) ** 2) / std_baseline ** 2)

    # Acceptance probability (since likelihood is uniform, it's prior-driven)
    accept_prob = min(1, prior_ratio)
    if np.random.rand() < accept_prob:
        chain[i] = proposal
        accept_count += 1
    else:
        chain[i] = current

# Burn-in and thin (every 10 for decorrelation)
chain = chain[burn_in::10]

# Compute GDOP for sampled baselines
gdops_sampled = np.array([compute_gdop(b) for b in chain])

# Estimate P(GDOP <3 | baseline >300 km)
mask = chain > 300
p_cond = np.mean(gdops_sampled[mask] < gdop_threshold) if np.any(mask) else 0
print(f"P(GDOP <{gdop_threshold} | baseline >300 km): {p_cond:.2f}")

# Histogram of sampled baselines and GDOP
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.hist(chain, bins=30, density=True, color='blue', alpha=0.6, label='Baseline Samples')
ax1.set_xlabel('Baseline (km)')
ax1.set_ylabel('Density', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.hist(gdops_sampled, bins=30, density=True, color='red', alpha=0.6, label='GDOP Samples')
ax2.set_ylabel('Density', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('MCMC Samples of Baseline and Corresponding GDOP')
fig.tight_layout()
plt.show()