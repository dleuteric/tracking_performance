# hgv_generate_and_save_v2.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── USER WINDOWS -----------------------------------------------------------
R_MIN_KM, R_MAX_KM   = 1500, 5000      # down‑range filter
LAT_MIN_DEG, LAT_MAX_DEG = 20, 80    # launch‑latitude filter
OUTFILE = 'trajectory_set2.xlsx'

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

def propagate(y0,tf=1200,dt0=.5,p=None):
    t,dt,y=0.,dt0,y0.copy(); T,Y=[0.],[y.copy()]
    while t<tf and y[0]>Re+1.0:                      # 1 m guard above surface
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
colors=['b','orange','y','purple','g','c','r','navy']

# ── TRAJECTORY SAMPLER -----------------------------------------------------
accepted=[]; max_tries=4000
for lab,a_fn,b_fn in man:
    for tries in range(1,max_tries+1):
        lat0_deg=np.random.uniform(LAT_MIN_DEG,LAT_MAX_DEG)
        h0=np.random.uniform(40e3,70e3); v0=np.random.uniform(3e3,6e3)
        th0=np.deg2rad(np.random.uniform(-.1,0)); sig0=np.deg2rad(np.random.uniform(-60,60))
        y0=np.array([Re+h0,0,np.deg2rad(lat0_deg),v0,th0,sig0])
        T,Y=propagate(y0,p=dict(m=m_nom,S=S_nom,alpha=a_fn,bank=b_fn))
        if Y[-1,0]<=Re+1.0:                                   # hit earth, discard
            continue
        R=downrange(0,np.deg2rad(lat0_deg),Y[-1,1],Y[-1,2])
        if R_MIN_KM<=R<=R_MAX_KM:
            accepted.append((lab,T,Y,R,lat0_deg))
            print(f"{lab}: φ0={lat0_deg:.1f}°, R={R:.0f} km (tries {tries})")
            break

# ── EXCEL LOGGING ----------------------------------------------------------
with pd.ExcelWriter(OUTFILE) as xl:
    for n,(lab,T,Y,R,lat0) in enumerate(accepted,1):
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
for i,(lab,T,Y,R,lat0) in enumerate(accepted):
    h=Y[:,0]-Re; lon=np.rad2deg(Y[:,1]); lat=np.rad2deg(Y[:,2])
    lbl=f"{lab} (R={R:.0f} km, φ0={lat0:.0f}°)"
    ax3d.plot(lon,lat,h,color=colors[i],lw=.8,ls='--',label=lbl)
    ax3d.scatter(lon[::120],lat[::120],h[::120],color=colors[i],s=8)
    ax2d.plot(lon,lat,color=colors[i],lw=.8,ls='--',label=lbl)
    ax2d.scatter(lon[::120],lat[::120],color=colors[i],s=8)
ax3d.set_xlabel('Lon (°)'); ax3d.set_ylabel('Lat (°)'); ax3d.set_zlabel('Height (m)')
ax3d.set_title('3‑D Trajectories (J₂ + Earth‑guard)'); ax3d.legend(bbox_to_anchor=(1.04,1))
ax2d.set_xlabel('Lon (°)'); ax2d.set_ylabel('Lat (°)'); ax2d.set_title('2‑D Ground‑tracks')
ax2d.legend(bbox_to_anchor=(1.04,1)); ax2d.grid(True); plt.tight_layout()

# EXTRA 3‑D VIEW WITH EARTH SPHERE -----------------------------------------
fig2=plt.figure(figsize=(6,6)); axE=fig2.add_subplot(111,projection='3d')
u=np.linspace(0,2*np.pi,60); v=np.linspace(0,np.pi,30)
xs=Re*np.outer(np.cos(u),np.sin(v)); ys=Re*np.outer(np.sin(u),np.sin(v)); zs=Re*np.outer(np.ones_like(u),np.cos(v))
axE.plot_surface(xs,ys,zs,rstride=3,cstride=3,color='lightgrey',alpha=.2,linewidth=0)
for i,(lab,_,Y,_,_) in enumerate(accepted):
    x,y,z=ecef(Y[:,0],Y[:,1],Y[:,2])
    axE.plot(x,y,z,color=colors[i],lw=.8,ls='--')
axE.set_title('Trajectory clearance vs Earth'); axE.set_box_aspect([1,1,1])
axE.set_xlim(-Re*1.2,Re*1.2); axE.set_ylim(-Re*1.2,Re*1.2); axE.set_zlim(-Re*1.2,Re*1.2)
plt.tight_layout(); plt.show()
