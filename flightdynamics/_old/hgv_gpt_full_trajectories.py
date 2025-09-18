import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── Constants (SI) ─────────────────────────────────────────────────────────
Re = 6.378e6       # Earth radius, m
mu = 3.986e14      # GM, m³ s⁻²
omega_e = 7.292e-5 # rad s⁻¹
m_nom = 907.0      # kg
S_nom = 0.4839     # m²

# ── Piece‑wise exponential atmosphere (U.S. Std 1976‑inspired) ────────────
def rho_atm(h):
    if h < 25e3:   Hs = 7.2e3
    elif h < 50e3: Hs = 6.6e3
    elif h < 75e3: Hs = 6.0e3
    else:          Hs = 5.6e3
    return 1.225 * np.exp(-h / Hs)

# ── Aero coefficients (unchanged toy model) ───────────────────────────────
def aero_coeffs(alpha):
    CL_alpha, CD0, k = 1.8, 0.05, 0.5
    CL = CL_alpha * alpha
    return CL, CD0 + k * CL**2

# ── Full VTC EOM with rotation ────────────────────────────────────────────
def hgv_dot(t, y, p):
    r, lon, lat, v, th, sig = y
    h = r - Re
    alpha, bank = p['alpha'](t, y), p['bank'](t, y)
    rho = rho_atm(h)
    CL, CD = aero_coeffs(alpha)
    q = 0.5 * rho * v**2
    aL = q * CL * p['S'] / p['m']
    aD = q * CD * p['S'] / p['m']
    g = mu / r**2

    rdot  = v * np.sin(th)
    vdot  = -aD - g*np.sin(th) + omega_e**2*r*np.cos(lat)*(np.sin(th)*np.cos(lat)
             - np.cos(th)*np.sin(sig)*np.sin(lat))
    thdot = (aL*np.cos(bank)/v) + (v/r - g/v)*np.cos(th) + 2*omega_e*np.cos(sig)*np.cos(lat) + \
            (omega_e**2*r/v)*np.cos(lat)*(np.cos(th)*np.cos(lat) +
            np.sin(th)*np.sin(sig)*np.sin(lat))
    londot = v*np.cos(th)*np.sin(sig)/(r*np.cos(lat) + 1e-9)
    latdot = v*np.cos(th)*np.cos(sig)/r
    sigdot = (aL*np.sin(bank)/(v*np.cos(th)+1e-9)) + (v/r)*np.cos(th)*np.sin(sig)*np.tan(lat) - \
             2*omega_e*(np.tan(th)*np.cos(lat)*np.sin(sig) - np.sin(lat)) + \
             (omega_e**2*r/(v*np.cos(th)+1e-9))*np.sin(sig)*np.cos(lat)*np.sin(lat)
    return np.array([rdot, londot, latdot, vdot, thdot, sigdot])

# ── RK4 + adaptive time‑step ──────────────────────────────────────────────
def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p)
    k2 = fun(t + 0.5*h, y + 0.5*h*k1, p)
    k3 = fun(t + 0.5*h, y + 0.5*h*k2, p)
    k4 = fun(t + h,     y + h*k3,     p)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6, k1  # return θ̇ in k1[4]

def propagate(t0, tf, dt0, y0, p):
    ts, ys = [t0], [y0.copy()]
    t, y, dt = t0, y0.copy(), dt0
    while t < tf and y[0] > Re:
        y_new, k1 = rk4(hgv_dot, t, y, dt, p)
        # adaptive halve dt if |θ̇| large
        if abs(k1[4]) > 0.05:       # rad s⁻¹ threshold
            dt *= 0.5
            continue                # redo step with smaller dt
        y, t = y_new, t + dt
        ts.append(t); ys.append(y.copy())
        dt = min(dt*1.2, dt0)       # slowly relax back toward nominal
    return np.array(ts), np.vstack(ys)

# ── Manoeuvre table (unchanged) ───────────────────────────────────────────
maneuvers = [
    {'name':'Bal-No',   'alpha':'bal','bank':'no'},
    {'name':'Bal-Left', 'alpha':'bal','bank':'left'},
    {'name':'Bal-Right','alpha':'bal','bank':'right'},
    {'name':'Bal-Weave','alpha':'bal','bank':'weave'},
    {'name':'Jump-No',  'alpha':'jump','bank':'no'},
    {'name':'Jump-Left','alpha':'jump','bank':'left'},
    {'name':'Jump-Right','alpha':'jump','bank':'right'},
    {'name':'Jump-Weave','alpha':'jump','bank':'weave'}
]

# ── Longitudinal controls (same equations) ───────────────────────────────
def alpha_bal(t, s): return np.deg2rad(8.0)
def alpha_jump(t, s):
    v_km = s[3] / 1e3
    a_max, a_ld, v1, v2 = 15.0, 2.5, 6.0, 3.0
    if v_km > v1: return np.deg2rad(a_max)
    if v_km < v2: return np.deg2rad(a_ld)
    a_mid = 0.5*(a_max+a_ld); a_amp = 0.5*(a_max-a_ld); v_mid = 0.5*(v1+v2)
    return np.deg2rad(a_mid + a_amp*np.sin(np.pi*(v_km - v_mid)/(v1-v2)))

# ── Lateral bank controls with sharper weave ─────────────────────────────
bank_no    = lambda t,s: 0.0
bank_left  = lambda t,s: np.deg2rad(-20.0)
bank_right = lambda t,s: np.deg2rad( 20.0)
bank_weave = lambda t,s: np.deg2rad(20.0*np.sin(t/120.0))  # period ~760 s

# ── Generate trajectories (σ₀ now ±60°) ───────────────────────────────────
colors = ['b','orange','y','purple','g','c','r','navy']
trajs=[]
for i,man in enumerate(maneuvers):
    p = dict(m=m_nom, S=S_nom,
             alpha=alpha_bal if man['alpha']=='bal' else alpha_jump,
             bank = bank_no if man['bank']=='no' else
                    bank_left if man['bank']=='left' else
                    bank_right if man['bank']=='right' else bank_weave)
    h0  = np.random.uniform(40e3,70e3)
    v0  = np.random.uniform(3e3,6e3)
    th0 = np.deg2rad(np.random.uniform(-0.1,0.0))
    sig0= np.deg2rad(np.random.uniform(-60,60))   # widened heading spread
    y0  = np.array([Re+h0,0,0,v0,th0,sig0])
    _, Y = propagate(0,1200,0.5,y0,p)
    trajs.append({'name':f"Traj {i+1} ({man['name']})",'Y':Y,'color':colors[i]})

# ── Plot (unchanged plotting block) ───────────────────────────────────────
fig=plt.figure(figsize=(12,6)); ax1=fig.add_subplot(1,2,1,projection='3d'); ax2=fig.add_subplot(1,2,2)
for tr in trajs:
    h = tr['Y'][:,0]-Re
    lon = np.rad2deg(tr['Y'][:,1]); lat = np.rad2deg(tr['Y'][:,2])
    ax1.plot(lon,lat,h,color=tr['color'],label=tr['name'],lw=0.8,ls='--')
    ax1.scatter(lon[::120],lat[::120],h[::120],color=tr['color'],s=8)
    ax2.plot(lon,lat,color=tr['color'],lw=0.8,ls='--',label=tr['name'])
    ax2.scatter(lon[::120],lat[::120],color=tr['color'],s=8)
ax1.set_xlabel('Lon (°)'); ax1.set_ylabel('Lat (°)'); ax1.set_zlabel('Height (m)')
ax1.set_title('3‑D Manoeuvre Trajectories'); ax1.legend(bbox_to_anchor=(1.04,1))
ax2.set_xlabel('Lon (°)'); ax2.set_ylabel('Lat (°)'); ax2.set_title('2‑D Ground‑tracks')
ax2.legend(bbox_to_anchor=(1.04,1)); ax2.grid(True); plt.tight_layout(); plt.show()
