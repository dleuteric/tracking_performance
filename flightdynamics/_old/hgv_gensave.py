"""
hgv_6dof_batch_latlon_impact.py
Generates NUM_TRAJ rigid-body HGV trajectories satisfying:
    • launch lat / lon window
    • down-range window
    • impact lat / lon window
Plots with start (green) and end (red) markers and exports to Excel.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ======== USER GATES ==========================================
NUM_TRAJ                      = 3
R_MIN_KM, R_MAX_KM            = 500, 6500
LAT_MIN_DEG, LAT_MAX_DEG      = 40, 55      # launch
LON_MIN_DEG, LON_MAX_DEG      = 60, 75
IMP_LAT_MIN_DEG, IMP_LAT_MAX_DEG = 40, 55   # impact
IMP_LON_MIN_DEG, IMP_LON_MAX_DEG = 00, 15
OUTFILE = 'trajectory_set_6dof.xlsx'

# -------- physics constants (UNCHANGED) -----------------------
Re, mu, J2, ωE = 6.378e6, 3.986e14, 1.08263e-3, 7.292e-5
m, S, b, c_bar = 907.0, 0.4839, 2.0, 0.35
Ixx, Iyy, Izz = 8e2, 1.2e3, 1.5e3
I = np.diag([Ixx, Iyy, Izz]); Iinv = np.linalg.inv(I)

def rho(h):
    h = np.clip(h, 0, 120e3)
    H = 7.2e3 if h < 25e3 else 6.6e3 if h < 50e3 else 6.0e3 if h < 75e3 else 5.6e3
    return 1.225*np.exp(-h/H)
def aero_coeff(a): CLα, CD0, k = 1.8, .05, .5; CL = CLα*a; return CL, CD0+k*CL**2

def quat_to_dcm(q):
    q0,q1,q2,q3=q
    return np.array([[1-2*(q2**2+q3**2),2*(q1*q2-q0*q3),2*(q1*q3+q0*q2)],
                     [2*(q1*q2+q0*q3),1-2*(q1**2+q3**2),2*(q2*q3-q0*q1)],
                     [2*(q1*q3-q0*q2),2*(q2*q3+q0*q1),1-2*(q1**2+q2**2)]])

def quat_dot(q, ω):
    p,q_,r=ω; Ω=np.array([[0,-p,-q_,-r],[p,0,r,-q_],[q_,-r,0,p],[r,q_,-p,0]])
    return 0.5*Ω@q

# ---------- simple control laws ------------------------------------------
def roll_mom(ϕ_cmd, ϕ_meas, p, Kp=5, Kd=1): return Kp*(ϕ_cmd-ϕ_meas)-Kd*p
def pitch_mom(a_cmd, a_meas, q, Kp=3, Kd=.8): return Kp*(a_cmd-a_meas)-Kd*q
def yaw_damp(r, Kd=2): return -Kd*r

# ---------- 6-DoF EOM -----------------------------------------------------
def hgv6dof(t,s,cmd):
    r,lon,lat,V,γ,χ=s[:6]; q=s[6:10]; p,qb,rb=s[10:]
    if r>10*Re or V<1 or V>1e4: return np.full_like(s,np.nan)
    α=cmd['alpha'](t,s); ϕ_cmd=cmd['bank'](t,s)
    CL,CD=aero_coeff(α); qdyn=.5*rho(r-Re)*V**2
    Fb_b=np.array([-qdyn*S*CD,0,-qdyn*S*CL])
    Fi = quat_to_dcm(q).T @ Fb_b
    zsl=np.sin(lat); g0=mu/r**2
    r4 = np.clip(r,Re,2*Re)**4
    aJ2_r=-1.5*J2*mu*Re**2/r4*(1-3*zsl**2)
    aJ2_lat=-3*J2*mu*Re**2/r4*zsl*np.cos(lat)
    rdot=V*np.sin(γ)
    londot=V*np.cos(γ)*np.sin(χ)/(r*np.cos(lat)+1e-9)
    latdot=V*np.cos(γ)*np.cos(χ)/r
    Vi=np.array([V*np.cos(γ)*np.cos(χ),V*np.cos(γ)*np.sin(χ),V*np.sin(γ)])
    Vdot=(Fi@Vi)/(m*V)-g0*np.sin(γ)+aJ2_r*np.sin(γ)
    γdot=(Fi[2]/m + aJ2_lat*np.cos(γ))/V + (V/r-g0/V)*np.cos(γ)
    χdot=(Fi[1]/m)/(V*np.cos(γ))+2*ωE*np.cos(lat)
    Lm=roll_mom(ϕ_cmd,0,p); Mm=pitch_mom(α,α,qb); Nm=yaw_damp(rb)
    ω=np.array([p,qb,rb]); ωdot=Iinv@(np.array([Lm,Mm,Nm])-np.cross(ω,I@ω))
    qdot=quat_dot(q,ω)
    return np.concatenate([[rdot,londot,latdot,Vdot,γdot,χdot],qdot,ωdot])

# ---------- propagate -----------------------------------------------------
def propagate(s0,cmd,dt=.5):
    t,S=0.,s0.copy(); out=[S.copy()]
    while t<1200 and S[0]>Re+1:
        k1=hgv6dof(t,S,cmd);
        if np.isnan(k1).any(): return np.array([[np.nan]*len(S)])
        k2=hgv6dof(t+.25*dt,S+.25*dt*k1,cmd)
        k3=hgv6dof(t+.25*dt,S+.25*dt*k2,cmd)
        k4=hgv6dof(t+dt,S+dt*k3,cmd)
        S+=dt*(k1+2*k2+2*k3+k4)/6; t+=dt
        out.append(S.copy())
    return np.array(out)

# ---------- helpers -------------------------------------------------------
def downrange(lo0,la0,lo1,la1):
    c=np.arccos(np.clip(np.sin(la0)*np.sin(la1)+np.cos(la0)*np.cos(la1)*np.cos(lo1-lo0),-1,1))
    return Re*c/1e3
def ecef(r,lo,la): return r*np.cos(la)*np.cos(lo),r*np.cos(la)*np.sin(lo),r*np.sin(la)

# ---------- manoeuvres ----------------------------------------------------
alpha_bal=lambda t,s:np.deg2rad(8)
def alpha_jump(t,s):
    V=s[3]/1e3; aM,aL,v1,v2=15,2.5,6,3
    if V>v1: return np.deg2rad(aM)
    if V<v2: return np.deg2rad(aL)
    return np.deg2rad(.5*(aM+aL)+.5*(aM-aL)*np.sin(np.pi*(V-.5*(v1+v2))/(v1-v2)))
bank_no=lambda t,s:0.; bank_l=lambda t,s:np.deg2rad(-20); bank_r=lambda t,s:np.deg2rad(20)
bank_w=lambda t,s:np.deg2rad(20*np.sin(t/120))
MAN=[('Bal-No',alpha_bal,bank_no),('Bal-L',alpha_bal,bank_l),
     ('Bal-R',alpha_bal,bank_r),('Bal-W',alpha_bal,bank_w),
     ('Jump-No',alpha_jump,bank_no),('Jump-L',alpha_jump,bank_l),
     ('Jump-R',alpha_jump,bank_r),('Jump-W',alpha_jump,bank_w)]

# ---------- sampling loop --------------------------------------------------
colors=plt.cm.tab20(np.linspace(0,1,NUM_TRAJ))
accepted=[]; np.random.seed(0); draws=0
while len(accepted)<NUM_TRAJ:
    lab,a_fn,b_fn=MAN[np.random.randint(0,len(MAN))]
    draws+=1
    lat0=np.deg2rad(np.random.uniform(LAT_MIN_DEG,LAT_MAX_DEG))
    lon0=np.deg2rad(np.random.uniform(LON_MIN_DEG,LON_MAX_DEG))
    h0=np.random.uniform(40e3,70e3); V0=np.random.uniform(3e3,6e3)
    γ0,χ0=np.deg2rad(np.random.uniform(-.1,0)),np.deg2rad(np.random.uniform(-60,60))
    q0=np.array([1,0,0,0]); ω0=np.zeros(3)
    s0=np.concatenate([[Re+h0,lon0,lat0,V0,γ0,χ0],q0,ω0])
    Y=propagate(s0,{'alpha':a_fn,'bank':b_fn})
    if np.isnan(Y).any(): continue
    if not (IMP_LAT_MIN_DEG<=np.rad2deg(Y[-1,2])<=IMP_LAT_MAX_DEG and
            IMP_LON_MIN_DEG<=np.rad2deg(Y[-1,1])<=IMP_LON_MAX_DEG):
        continue
    R=downrange(lon0,lat0,Y[-1,1],Y[-1,2])
    if not(R_MIN_KM<=R<=R_MAX_KM): continue
    accepted.append((lab,s0,Y,R))
    print(
        f"[{len(accepted)}/{NUM_TRAJ}] {lab}: "
        f"launch φ={np.rad2deg(lat0):.1f}°, "
        f"λ={np.rad2deg(lon0):.1f}°, "
        f"impact φ={np.rad2deg(Y[-1,2]):.1f}°, "
        f"λ={np.rad2deg(Y[-1,1]):.1f}°, "
        f"R={R:.0f} km after {draws} draws"
    )

# ---------- Excel export ---------------------------------------------------
with pd.ExcelWriter(OUTFILE) as xl:
    for i,(lab,s0,Y,R) in enumerate(accepted,1):
        x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2])
        vx=np.gradient(x); vy=np.gradient(y); vz=np.gradient(z)
        ax=np.gradient(vx); ay=np.gradient(vy); az=np.gradient(vz)
        pd.DataFrame({'t_s':np.arange(len(Y))*0.5,
                      'x':x,'y':y,'z':z,'vx':vx,'vy':vy,'vz':vz,
                      'ax':ax,'ay':ay,'az':az}).to_excel(xl,f'Traj{i}',index=False)
print(f"Saved {len(accepted)} sheets to {OUTFILE}")

# ---------- plots ----------------------------------------------------------
fig=plt.figure(figsize=(12,6)); ax3d=fig.add_subplot(1,2,1,projection='3d'); ax2d=fig.add_subplot(1,2,2)
for i,(lab,s0,Y,R) in enumerate(accepted):
    lon=np.rad2deg(Y[:,1]); lat=np.rad2deg(Y[:,2]); h=Y[:,0]-Re
    ax3d.plot(lon,lat,h,color=colors[i],lw=.8,ls='--',label=lab+f" (R={R:.0f} km)")
    ax3d.scatter(lon[0],lat[0],h[0],c='g',s=30,marker='o')      # start
    ax3d.scatter(lon[-1],lat[-1],h[-1],c='r',s=30,marker='x')   # end
    ax2d.plot(lon,lat,color=colors[i],lw=.8,ls='--',label=lab+f" (R={R:.0f} km)")
    ax2d.scatter(lon[0],lat[0],c='g',s=30,marker='o')
    ax2d.scatter(lon[-1],lat[-1],c='r',s=30,marker='x')
ax3d.set_xlabel('Lon°'); ax3d.set_ylabel('Lat°'); ax3d.set_zlabel('h m')
ax3d.legend(); ax3d.set_title('6-DoF trajectories (start ▪, impact ×)')
ax2d.set_xlabel('Lon°'); ax2d.set_ylabel('Lat°'); ax2d.set_title('Ground-tracks')
ax2d.legend(); ax2d.grid(True); plt.tight_layout(); plt.show()