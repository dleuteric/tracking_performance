""" Diverse eight‑trajectory demo ─ mirrors Zhang Fig. 1 (maneuver variety) """
import numpy as np, matplotlib.pyplot as plt; from mpl_toolkits.mplot3d import Axes3D  # noqa

# ------------- environment & aero -------------------------------------------------
Re, mu = 6378.137, 398600.4418
rho = lambda h: 1.225 * np.exp(-np.maximum(h, 0) / 7.2)
def aero(alpha): CL = 2 * alpha; return CL, 0.05 + 0.5 * CL**2

# ------------- longitudinal laws --------------------------------------------------
a_max, a_ld, v1, v2 = np.deg2rad(15), np.deg2rad(2.5), 6.0, 3.0
def alpha_jump(_, s):
    v = s[3]; mid = 0.5 * (a_max + a_ld); amp = 0.5 * (a_max - a_ld); vm = 0.5 * (v1 + v2)
    return a_max if v > v1 else a_ld if v < v2 else mid + amp * np.sin(np.pi * (v - vm) / (v1 - v2))

def alpha_bal(_, s, k=0.4):                        # θ̇ ≈ 0 ⇒ trim α (simple P‑controller)
    return np.clip(-k * s[4], -0.1, 0.1) + np.deg2rad(3)

# ------------- lateral bank functions ---------------------------------------------
bank_none  = lambda *_, **__: 0.0
bank_l30   = lambda *_, **__:  np.deg2rad(30)
bank_r30   = lambda *_, **__: -np.deg2rad(30)
bank_weave = lambda t, *_, per=120: np.deg2rad(25) if (int(t // per) % 2) == 0 else -np.deg2rad(25)

# ------------- VTC point‑mass EOM (Zhang Eq. 1) -----------------------------------
def f(t, y, p):
    r, lon, lat, v, th, sig = y; h = r - Re; g = mu / r**2
    bk = p['bank'](t, y); al = p['alpha'](t, y)
    CL, CD = aero(al); q = 0.5 * rho(h) * (v * 1e3) ** 2
    aL, aD = q * CL * p['S'] / p['m'] / 1e3, q * CD * p['S'] / p['m'] / 1e3
    ct, cl = np.cos(th), np.cos(lat) + 1e-9
    return np.array([v * np.sin(th),
                     v * ct * np.sin(sig) / (r * cl),
                     v * ct * np.cos(sig) / r,
                     -aD - g * np.sin(th),
                     aL * np.cos(bk) / v + v * ct / r - g * ct / v,
                     aL * np.sin(bk) / (v * ct + 1e-9) + v * ct * np.sin(sig) * np.tan(lat) / r])

def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p); k2 = fun(t + h/2, y + h/2 * k1, p)
    k3 = fun(t + h/2, y + h/2 * k2, p); k4 = fun(t + h, y + h * k3, p)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def propagate(y0, tf, dt, p):
    T, Y, t = [0], [y0.copy()], 0
    while t < tf and y0[0] > Re:
        y0 = rk4(f, t, y0, dt, p); t += dt; T.append(t); Y.append(y0.copy())
    return np.array(T), np.vstack(Y)

# ------------- define eight distinct cases ----------------------------------------
cases = [
    ("Trajectory one",   alpha_jump, bank_l30,   ( 0,   0, np.deg2rad(-5), np.deg2rad( 60))),
    ("Trajectory two",   alpha_jump, bank_r30,   ( 0,  10, np.deg2rad(-5), np.deg2rad(120))),
    ("Trajectory three", alpha_jump, bank_weave, ( 0, -10, np.deg2rad(-5), np.deg2rad( 45))),
    ("Trajectory four",  alpha_bal,  bank_none,  ( 0,  20, 0.0,             np.deg2rad( 90))), # flat & lateral
    ("Trajectory five",  alpha_bal,  bank_l30,   ( 5,   0, 0.0,             np.deg2rad( 80))),
    ("Trajectory six",   alpha_jump, bank_weave, (-5, -15, np.deg2rad(-5),  np.deg2rad(100))),
    ("Trajectory seven", alpha_bal,  bank_r30,   (10,   5, 0.0,             np.deg2rad(110))),
    ("Trajectory eight", alpha_jump, bank_weave, (-8,  15, np.deg2rad(-5),  np.deg2rad( 70))),
]

pars_common = dict(m=1_000.0, S=2.0)
tf, dt = 700, 0.5  # s

# ------------- run & plot ---------------------------------------------------------
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
for name, alpha_fn, bank_fn, (lat0, lon0, theta0, sigma0) in cases:
    pars = {**pars_common, 'alpha': alpha_fn, 'bank': bank_fn}
    # initial state: r, lon, lat, v, theta, sigma
    y0 = np.array([Re + 80, np.deg2rad(lon0), np.deg2rad(lat0), 7.0, theta0, sigma0])
    _, Y = propagate(y0, tf, dt, pars)
    ax.plot(np.rad2deg(Y[:,1]), np.rad2deg(Y[:,2]), Y[:,0]*1e3,
            marker='o', ms=3, lw=1, label=name)

ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)'); ax.set_zlabel('High (m)')
ax.set_title('Typical maneuver trajectories of HGV (demo)')
ax.legend(bbox_to_anchor=(1.04, 1)); plt.tight_layout(); plt.show()
