import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import itertools

# ======== USER GATES ========
NUM_TRAJ                 = 1          # how many trajectories to output
R_MIN_KM, R_MAX_KM       = 800, 6500  # down-range window
LAT_MIN_DEG, LAT_MAX_DEG = -90, 90    # launch-latitude window
LON_MIN_DEG, LON_MAX_DEG = -180, 180    # launch-longitude window
#39.65,  9.65
#34.00, 18.00

OUTFILE = 'trajectory_set2.xlsx'
PROP_TIME = 8000.0 # seconds, increased max propagation time to ensure impact; was 1200.0
TIME_STEP = 1.0 # seconds, propagation time step
h0_min = 80e3
h0_max = 120e3
v0_min = 3e3
v0_max = 7e3
TRAJECTORY_SEED = 87

# Balancing knobs
HEADING_BALANCE = 0.5   # 0 → only N-S, 1 → only E-W, 0.5 → even mix
FLATTEN_RANGE   = True  # True => accept fewer long-range shots to flatten histogram

# ── CONSTANTS & ENVIRONMENT ------------------------------------------------
Re = 6.378e6; mu = 3.986e14; J2 = 1.08263e-3; omega_e = 7.292e-5
m_nom, S_nom = 907.0, 0.4839
def rho_atm(h):
    Hs = 7.2e3 if h < 25e3 else 6.6e3 if h < 50e3 else 6.0e3 if h < 75e3 else 5.6e3
    return 1.225*np.exp(-h/Hs)
def aero_coeffs(a): CL_a, CD0, k = 1.8, .05, .5; CL = CL_a*a; return CL, CD0 + k*CL**2

# ── DYNAMICS WITH J2 -------------------------------------------------------
def hgv_dot(t,y,p):
    r,lo,la,v,th,si=y; h=r-Re
    a,b=p['alpha'](t,y),p['bank'](t,y); rho=rho_atm(h); CL,CD=aero_coeffs(a)
    q=0.5*rho*v**2; aL=q*CL*p['S']/p['m']; aD=q*CD*p['S']/p['m']
    g0 = mu/r**2
    zsl = np.sin(la); aJ2_r = -1.5*J2*mu*Re**2/r**4*(1-3*zsl**2)
    aJ2_lat = -3*J2*mu*Re**2/r**4*zsl*np.cos(la)
    rdot = v*np.sin(th)
    vdot = -aD - g0*np.sin(th) + aJ2_r*np.sin(th) + \
           omega_e**2*r*np.cos(la)*(np.sin(th)*np.cos(la)-np.cos(th)*np.sin(si)*zsl)
    thdot = (aL*np.cos(b)/v) + (v/r-g0/v)*np.cos(th) + aJ2_lat*np.cos(th)/v + \
            2*omega_e*np.cos(si)*np.cos(la) + \
            (omega_e**2*r/v)*np.cos(la)*(np.cos(th)*np.cos(la)+np.sin(th)*np.sin(si)*zsl)
    londot = v*np.cos(th)*np.sin(si)/(r*np.cos(la)+1e-9)
    latdot = v*np.cos(th)*np.cos(si)/r
    sigdot = (aL*np.sin(b)/(v*np.cos(th)+1e-9)) + (v/r)*np.cos(th)*np.sin(si)*np.tan(la) - \
             2*omega_e*(np.tan(th)*np.cos(la)*np.sin(si)-zsl) + \
             (omega_e**2*r/(v*np.cos(th)+1e-9))*np.sin(si)*np.cos(la)*zsl
    return np.array([rdot,londot,latdot,vdot,thdot,sigdot])

def rk4(fun,t,y,h,p):
    k1=fun(t,y,p); k2=fun(t+.5*h,y+.5*h*k1,p)
    k3=fun(t+.5*h,y+.5*h*k2,p); k4=fun(t+h,y+h*k3,p)
    return y+h*(k1+2*k2+2*k3+k4)/6, k1[4]

def propagate(y0,tf=PROP_TIME,dt0=TIME_STEP,p=None):
    t,dt,y=0.,dt0,y0.copy(); T,Y=[0.],[y.copy()]
    while t<tf and y[0]>Re+1.0:                      # 1 m guard above surface
        y_new,thdot=rk4(hgv_dot,t,y,dt,p)
        if abs(thdot)>0.05: dt*=.5; continue
        y,t=y_new,t+dt; dt=min(dt*1.2,dt0)
        T.append(t); Y.append(y.copy())
    return np.array(T),np.vstack(Y)

# ── HELPERS ----------------------------------------------------------------
def downrange(lo1,la1,lo2,la2):
    c=np.arccos(np.clip(np.sin(la1)*np.sin(la2)+np.cos(la1)*np.cos(la2)*np.cos(lo2-lo1),-1,1))
    return Re*c/1e3
def ecef(r,lo,la):
    x=r*np.cos(la)*np.cos(lo); y=r*np.cos(la)*np.sin(lo); z=r*np.sin(la)
    return x,y,z

# ── CONTROL LAWS -----------------------------------------------------------
alpha_bal=lambda t,s:np.deg2rad(8)
def alpha_jump(t,s):
    v=s[3]/1e3; aM,aL,v1,v2=15,2.5,6,3
    if v>v1: return np.deg2rad(aM)
    if v<v2: return np.deg2rad(aL)
    return np.deg2rad(.5*(aM+aL)+.5*(aM-aL)*np.sin(np.pi*(v-.5*(v1+v2))/(v1-v2)))
bank_no=lambda t,s:0.; bank_left=lambda t,s:np.deg2rad(-20); bank_right=lambda t,s:np.deg2rad(20)
bank_weave=lambda t,s:np.deg2rad(20*np.sin(t/120))
man=[('Bal-No',alpha_bal,bank_no),('Bal-L',alpha_bal,bank_left),
     ('Bal-R',alpha_bal,bank_right),('Bal-W',alpha_bal,bank_weave),
     ('Jump-No',alpha_jump,bank_no),('Jump-L',alpha_jump,bank_left),
     ('Jump-R',alpha_jump,bank_right),('Jump-W',alpha_jump,bank_weave)]

# ── Build manoeuvre schedule to guarantee coverage ────────────────────────
np.random.seed(TRAJECTORY_SEED)
man_indices = np.arange(len(man))
np.random.shuffle(man_indices)
schedule = [man[i] for i in man_indices.tolist()]            # first 8 unique

while len(schedule) < NUM_TRAJ:                              # top-up randomly
    schedule.append(man[np.random.randint(0, len(man))])
sched_idx = 0                                                # pointer in schedule

colors = plt.cm.tab20(np.linspace(0, 1, NUM_TRAJ))

# ── TRAJECTORY SAMPLER -----------------------------------------------------
accepted = []
np.random.seed(TRAJECTORY_SEED)
tries_global = 0
while len(accepted) < NUM_TRAJ:
    lab, a_fn, b_fn = schedule[sched_idx]        # pull from schedule
    tries_global += 1
    lat0_deg = np.random.uniform(LAT_MIN_DEG, LAT_MAX_DEG)
    lon0_deg = np.random.uniform(LON_MIN_DEG, LON_MAX_DEG)
    h0 = np.random.uniform(h0_min, h0_max)
    v0 = np.random.uniform(v0_min, v0_max)
    th0 = np.deg2rad(np.random.uniform(-.1, 0))
    if np.random.rand() < HEADING_BALANCE:          # favour east-west ground-tracks
        # centre around ±5° so flight is mostly along latitude circles
        sig0 = np.deg2rad(np.random.uniform(-10, 10))
    else:                                           # classic north-south shot
        sig0 = np.deg2rad(np.random.uniform(80, 100)) * np.random.choice([-1, 1])
    y0 = np.array([Re + h0, np.deg2rad(lon0_deg), np.deg2rad(lat0_deg), v0, th0, sig0])
    T, Y = propagate(y0, p=dict(m=m_nom, S=S_nom, alpha=a_fn, bank=b_fn))
    if Y[-1, 0] > Re + 10:  # ensure it hit the ground (altitude < 10 m)
        continue
    R = downrange(np.deg2rad(lon0_deg), np.deg2rad(lat0_deg), Y[-1, 1], Y[-1, 2])
    if not (R_MIN_KM <= R <= R_MAX_KM):
        continue
    if FLATTEN_RANGE:
        keep_prob = (R_MAX_KM - R) / (R_MAX_KM - R_MIN_KM)  # higher R → lower prob
        if np.random.rand() > keep_prob:
            continue
    ToF = T[-1]                     # final propagation time (may be < PROP_TIME)
    # trajectory accepted -> move to next manoeuvre type
    sched_idx = (sched_idx + 1) % len(schedule)
    accepted.append((lab, T, Y, R, lat0_deg, lon0_deg, ToF))
    print(f"[{len(accepted)}/{NUM_TRAJ}] {lab}: φ0={lat0_deg:.1f}°, λ0={lon0_deg:.1f}°, "
          f"R={R:.0f} km (global tries {tries_global})")

# ── EXCEL LOGGING ----------------------------------------------------------
with pd.ExcelWriter(OUTFILE) as xl:
    for n,(lab,T,Y,R,lat0,lon0,ToF) in enumerate(accepted,1):
        x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2])
        vx=np.gradient(x,T); vy=np.gradient(y,T); vz=np.gradient(z,T)
        ax=np.gradient(vx,T); ay=np.gradient(vy,T); az=np.gradient(vz,T)
        pd.DataFrame({'t_s':T,'x_m':x,'y_m':y,'z_m':z,
                      'vx_mps':vx,'vy_mps':vy,'vz_mps':vz,
                      'ax_mps2':ax,'ay_mps2':ay,'az_mps2':az}) \
            .to_excel(xl,sheet_name=f'Traj{n}_{lab}',index=False)
print(f"Saved {len(accepted)} trajectories to {OUTFILE}")

# ── PLOTS ------------------------------------------------------------------
fig=plt.figure(figsize=(12,6)); ax3d=fig.add_subplot(1,2,1,projection='3d'); ax2d=fig.add_subplot(1,2,2)
for i,(lab,T,Y,R,lat0,lon0,ToF) in enumerate(accepted):
    h=Y[:,0]-Re; lon=np.rad2deg(Y[:,1]); lat=np.rad2deg(Y[:,2])
    lbl = f"{lab} (ToF={ToF:.0f}s, R={R:.0f} km, φ0={lat0:.0f}°, λ0={lon0:.0f}°)"
    ax3d.plot(lon,lat,h,color=colors[i],lw=.8,ls='--',label=lbl)
    ax3d.scatter(lon[::120],lat[::120],h[::120],color=colors[i],s=8)
    ax2d.plot(lon,lat,color=colors[i],lw=.8,ls='--',label=lbl)
    ax2d.scatter(lon[::120],lat[::120],color=colors[i],s=8)
ax3d.set_xlabel('Lon (°)'); ax3d.set_ylabel('Lat (°)'); ax3d.set_zlabel('Height (m)')
ax3d.set_title('3D Trajectories (J2 + Earth-guard)'); ax3d.legend(bbox_to_anchor=(1.04,1))
ax2d.set_xlabel('Lon (°)'); ax2d.set_ylabel('Lat (°)'); ax2d.set_title('2D Ground-tracks')
ax2d.legend(bbox_to_anchor=(1.04,1)); ax2d.grid(True); plt.tight_layout()

# ── ALTITUDE vs DOWNRANGE --------------------------------------------------
fig3, axAD = plt.subplots(figsize=(7,5))
for i,(lab,T,Y,R,lat0,lon0,ToF) in enumerate(accepted):
    # compute downrange for every epoch relative to launch point
    dr_km = [downrange(np.deg2rad(lon0), np.deg2rad(lat0), lon, lat)
             for lon, lat in zip(Y[:,1], Y[:,2])]
    alt_km = (Y[:,0] - Re) / 1e3
    axAD.plot(dr_km, alt_km, color=colors[i], lw=1, label=f"{lab}  ToF={ToF:.0f}s")

axAD.set_xlabel('Down-range [km]')
axAD.set_ylabel('Altitude [km]')
axAD.set_title('Altitude vs Down-range (ground impact check)')
axAD.grid(True); axAD.legend(fontsize=8)

plt.tight_layout()

# EXTRA 3D VIEW WITH EARTH SPHERE -----------------------------------------
# Improved Earth sphere rendering:
# - Increased grid resolution (u,v from 60,30 to 100,100) for smoother surface.
# - Reduced rstride/cstride to 1 for finer mesh detail.
# - Set edgecolor='none' to remove wireframe lines, relying on shading for depth.
# - Kept alpha=0.3 for semi-transparency to allow trajectories to stand out.
# - Added shade=True explicitly for better lighting effects.
# - Used a more appealing color 'lightblue' with low alpha to evoke Earth without texture.
# Note: Without external textures (no internet), this is as nice as it gets - smooth shaded sphere.
# For even better, could add simple lat/long grid lines if desired, but keeps it clean.
fig2=plt.figure(figsize=(8,8)); axE=fig2.add_subplot(111,projection='3d')
u=np.linspace(0,2*np.pi,100); v=np.linspace(0,np.pi,100)
xs=Re*np.outer(np.cos(u),np.sin(v)); ys=Re*np.outer(np.sin(u),np.sin(v)); zs=Re*np.outer(np.ones_like(u),np.cos(v))
axE.plot_surface(xs,ys,zs,rstride=1,cstride=1,color='lightblue',alpha=0.3,linewidth=0,antialiased=True,shade=True,edgecolor='none')
for i,(lab,_,Y,_,_,_,_) in enumerate(accepted):
    x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2])
    axE.plot(x,y,z,color=colors[i],lw=1.0,ls='-')
axE.set_title('Trajectory clearance vs Earth surface'); axE.set_box_aspect([1,1,1])
axE.set_xlim(-Re*1.2,Re*1.2); axE.set_ylim(-Re*1.2,Re*1.2); axE.set_zlim(-Re*1.2,Re*1.2)
plt.tight_layout(); plt.show()