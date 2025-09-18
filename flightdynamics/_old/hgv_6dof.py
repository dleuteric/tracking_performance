"""
hgv_6dof_batch_latlon.py
Generates one accepted 6-DoF trajectory per manoeuvre class
subject to user-defined windows on   down-range, launch latitude, launch longitude.
Exports each trajectory to its own Excel sheet and plots 3-D views.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ======== user gates =========
NUM_TRAJ = 3           # number of trajectories to generate
R_MIN_KM, R_MAX_KM         =  2500, 5000
LAT_MIN_DEG, LAT_MAX_DEG   = -20,   20
LON_MIN_DEG, LON_MAX_DEG   = -10,   10          # NEW  ⟵  launch-longitude window
OUTFILE = 'trajectory_set_6dof.xlsx'

# -------------------- constants & parameters ------------------------------
Re, mu, J2, ωE = 6.378e6, 3.986e14, 1.08263e-3, 7.292e-5
m, S, b, c_bar = 907.0, 0.4839, 2.0, 0.35
Ixx, Iyy, Izz  = 8.0e2, 1.2e3, 1.5e3
I = np.diag([Ixx, Iyy, Izz]); Iinv = np.linalg.inv(I)

# -------------------- atmosphere & aero -----------------------------------
def rho(h):
    h = np.clip(h, 0, 120e3)      # cap to 120 km to avoid overflow
    H = 7.2e3 if h < 25e3 else 6.6e3 if h < 50e3 else 6.0e3 if h < 75e3 else 5.6e3
    return 1.225*np.exp(-h/H)
def aero_coeff(a): CLα, CD0, k = 1.8, .05, .5; CL = CLα*a; return CL, CD0 + k*CL**2

# -------------------- quaternion algebra ----------------------------------
def quat_to_dcm(q):
    q0,q1,q2,q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])
def quat_dot(q, ωb):
    p,q_,r = ωb
    Ω = np.array([[0,-p,-q_,-r],[p,0,r,-q_],[q_,-r,0,p],[r,q_,-p,0]])
    return 0.5*Ω@q

# -------------------- simple PI moment laws -------------------------------
def roll_mom(bank_cmd, bank_meas, p, Kp=5, Kd=1): return Kp*(bank_cmd-bank_meas) - Kd*p
def pitch_mom(α_cmd, α_meas, q, Kp=3, Kd=0.8):    return Kp*(α_cmd-α_meas) - Kd*q
def yaw_damp(r, Kd=2):                             return -Kd*r

# -------------------- 6-DoF rigid-body EOM --------------------------------
def hgv6dof(t, s, cmd):
    r,lon,lat,V,γ,χ = s[:6]

    # ---- numeric sanity guard ---------------------------------------
    # if radius or speed explodes, return NaNs so caller skips sample
    if r > 10*Re or not np.isfinite(r):
        return np.full_like(s, np.nan)
    if V < 1 or V > 10_000:                    # discard corrupt states
        return np.full_like(s, np.nan)

    q = s[6:10]; p,q_b,r_b = s[10:]
    h = r - Re
    α = cmd['alpha'](t,s); ϕ_cmd = cmd['bank'](t,s)          # rad
    CL,CD = aero_coeff(α); qdyn = 0.5*rho(h)*V**2
    L_b, D_b = qdyn*S*CL, qdyn*S*CD
    Fb_b = np.array([-D_b, 0.0, -L_b])                       # -X, 0, -Z
    C_bi = quat_to_dcm(q).T; Fi_i = C_bi @ Fb_b

    if not np.isfinite(Fi_i).all():
        return np.full_like(s, np.nan)

    zsl = np.sin(lat); g0 = mu/r**2

    # protect against overflow in r**4 by capping input
    r4 = np.clip(r, Re, 2*Re)**4
    aJ2_r  = -1.5*J2*mu*Re**2/r4*(1-3*zsl**2)
    aJ2_lat= -3*J2*mu*Re**2/r4*zsl*np.cos(lat)

    # translational kinematics
    rdot = V*np.sin(γ)
    londot = V*np.cos(γ)*np.sin(χ)/(r*np.cos(lat)+1e-9)
    latdot = V*np.cos(γ)*np.cos(χ)/r
    Vi = np.array([V*np.cos(γ)*np.cos(χ), V*np.cos(γ)*np.sin(χ), V*np.sin(γ)])
    Vdot = (Fi_i @ Vi)/(m*V) - g0*np.sin(γ) + aJ2_r*np.sin(γ)
    γdot = (Fi_i[2]/m - D_b*np.sin(α)/m)/V + (V/r - g0/V)*np.cos(γ) + aJ2_lat*np.cos(γ)/V
    χdot = (Fi_i[1]/m + L_b*np.sin(ϕ_cmd)/m)/(V*np.cos(γ)) + \
           2*ωE*np.cos(lat) + aJ2_lat*np.sin(ϕ_cmd)/(V*np.cos(γ))

    # moments & rotational dynamics
    Lm = roll_mom(ϕ_cmd, 0.0, p); Mm = pitch_mom(α, α, q_b); Nm = yaw_damp(r_b)
    Mb = np.array([Lm,Mm,Nm])
    ω = np.array([p,q_b,r_b]); ωdot = Iinv @ (Mb - np.cross(ω, I@ω))
    qdot = quat_dot(q, ω)

    q_norm = np.linalg.norm(q)
    if q_norm == 0 or not np.isfinite(q_norm):
        return np.full_like(s, np.nan)
    q /= q_norm

    return np.concatenate([[rdot,londot,latdot,Vdot,γdot,χdot], qdot, ωdot])

# -------------------- propagate with RK-4 ---------------------------------
def propagate(s0, cmd):
    dt,t,S,T = 1.0,0.0,s0.copy(),[s0.copy()]
    while t<1200 and S[0]>Re+1:
        k1=hgv6dof(t,S,cmd); k2=hgv6dof(t+0.25*dt,S+0.25*dt*k1,cmd)
        k3=hgv6dof(t+0.25*dt,S+0.25*dt*k2,cmd); k4=hgv6dof(t+dt,S+dt*k3,cmd)
        S+=dt*(k1+2*k2+2*k3+k4)/6
        # abort trajectory if numerical overflow produced NaNs / infs
        if not np.isfinite(S).all():
            return np.array([[np.nan]*len(S)])
        t+=dt; T.append(S.copy())
    return np.array(T)

# -------------------- helpers ---------------------------------------------
def downrange(lon0,lat0,lon1,lat1):
    c=np.arccos(np.clip(np.sin(lat0)*np.sin(lat1)+np.cos(lat0)*np.cos(lat1)*np.cos(lon1-lon0),-1,1))
    return Re*c/1e3
def ecef(r,lo,la):
    return r*np.cos(la)*np.cos(lo), r*np.cos(la)*np.sin(lo), r*np.sin(la)

# -------------------- manoeuvre table -------------------------------------
alpha_bal=lambda t,s: np.deg2rad(8)
def alpha_jump(t,s):
    V=s[3]/1e3; aM,aL,v1,v2=15,2.5,6,3
    if V>v1:return np.deg2rad(aM)
    if V<v2:return np.deg2rad(aL)
    return np.deg2rad(.5*(aM+aL)+.5*(aM-aL)*np.sin(np.pi*(V-.5*(v1+v2))/(v1-v2)))
bank_no=lambda t,s:0.; bank_l=lambda t,s:np.deg2rad(-20); bank_r=lambda t,s:np.deg2rad(20)
bank_w=lambda t,s:np.deg2rad(20*np.sin(t/120))
MAN=[('Bal-No',alpha_bal,bank_no),('Bal-L',alpha_bal,bank_l),
     ('Bal-R',alpha_bal,bank_r), ('Bal-W',alpha_bal,bank_w),
     ('Jump-No',alpha_jump,bank_no),('Jump-L',alpha_jump,bank_l),
     ('Jump-R',alpha_jump,bank_r),('Jump-W',alpha_jump,bank_w)]

# -------------------- batch search ----------------------------------------
accepted = []
np.random.seed(0)

while len(accepted) < NUM_TRAJ:
    lab, a_fn, b_fn = MAN[np.random.randint(0, len(MAN))]     # random manoeuvre class
    for tries in range(1, 6001):
        lat0 = np.deg2rad(np.random.uniform(LAT_MIN_DEG, LAT_MAX_DEG))
        lon0 = np.deg2rad(np.random.uniform(LON_MIN_DEG, LON_MAX_DEG))
        h0 = np.random.uniform(40e3, 70e3); V0 = np.random.uniform(3e3, 6e3)
        γ0, χ0 = np.deg2rad(np.random.uniform(-0.1, 0)), np.deg2rad(np.random.uniform(-60, 60))
        q0 = np.array([1, 0, 0, 0]); ω0 = np.zeros(3)
        s0 = np.concatenate([[Re + h0, lon0, lat0, V0, γ0, χ0], q0, ω0])
        cmd = {'alpha': a_fn, 'bank': b_fn}
        Y = propagate(s0, cmd)
        if np.isnan(Y).any():
            continue   # skip trajectories that blew up numerically
        if Y[-1, 0] <= Re + 1:
            continue
        R = downrange(lon0, lat0, Y[-1, 1], Y[-1, 2])
        if R_MIN_KM <= R <= R_MAX_KM:
            accepted.append((lab, s0, Y, R))
            print(f"[{len(accepted)}/{NUM_TRAJ}] {lab}: φ0={np.rad2deg(lat0):.1f}°, "
                  f"λ0={np.rad2deg(lon0):.1f}°, R={R:.0f} km after {tries} draws")
            break
    else:
        print(f"WARNING: could not find trajectory for {lab} within limits after 6000 draws")
        break

# -------------------- Excel export ----------------------------------------
with pd.ExcelWriter(OUTFILE) as xl:
    for n,(lab,s0,Y,R) in enumerate(accepted,1):
        x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2])
        vx=np.gradient(x); vy=np.gradient(y); vz=np.gradient(z)
        ax=np.gradient(vx); ay=np.gradient(vy); az=np.gradient(vz)
        df=pd.DataFrame({'t_s':np.arange(len(Y))*0.5,'x':x,'y':y,'z':z,
                         'vx':vx,'vy':vy,'vz':vz,'ax':ax,'ay':ay,'az':az})
        df.to_excel(xl,sheet_name=f'Traj{n}',index=False)
print(f"Saved to {OUTFILE}")

# -------------------- plots ------------------------------------------------
colors = plt.cm.tab20(np.linspace(0, 1, NUM_TRAJ))
fig=plt.figure(figsize=(12,6)); ax3d=fig.add_subplot(1,2,1,projection='3d'); ax2d=fig.add_subplot(1,2,2)
for i,(lab,s0,Y,R) in enumerate(accepted):
    h=Y[:,0]-Re; lon=np.rad2deg(Y[:,1]); lat=np.rad2deg(Y[:,2])
    lbl=f"{lab} (R={R:.0f} km)"
    ax3d.plot(lon,lat,h,color=colors[i],lw=.8,ls='--',label=lbl)
    ax3d.scatter(lon[::120],lat[::120],h[::120],color=colors[i],s=8)
    ax2d.plot(lon,lat,color=colors[i],lw=.8,ls='--',label=lbl)
    ax2d.scatter(lon[::120],lat[::120],color=colors[i],s=8)
ax3d.set_xlabel('Lon (°)'); ax3d.set_ylabel('Lat (°)'); ax3d.set_zlabel('h (m)')
ax3d.legend(bbox_to_anchor=(1.04,1)); ax3d.set_title('6-DoF trajectories')
ax2d.set_xlabel('Lon (°)'); ax2d.set_ylabel('Lat (°)'); ax2d.set_title('Ground-tracks')
ax2d.legend(bbox_to_anchor=(1.04,1)); ax2d.grid(True); plt.tight_layout()

# Globe view
fig2=plt.figure(figsize=(6,6)); ax=fig2.add_subplot(111,projection='3d')
u,v=np.linspace(0,2*np.pi,60),np.linspace(0,np.pi,30)
ax.plot_surface(Re*np.outer(np.cos(u),np.sin(v)),
                Re*np.outer(np.sin(u),np.sin(v)),
                Re*np.outer(np.ones_like(u),np.cos(v)),
                alpha=.2,color='lightgrey',linewidth=0)
for i,(_,__,Y,___) in enumerate(accepted):
    x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2]); ax.plot(x,y,z,color=colors[i],lw=.8)
ax.set_box_aspect([1,1,1]); ax.set_title('Earth clearance'); plt.tight_layout(); plt.show()