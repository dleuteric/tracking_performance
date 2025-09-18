# generate_range_filtered_trajs_lat.py
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── User‑defined windows ──────────────────────────────────────────────────
R_MIN_KM, R_MAX_KM   =  500, 5000   # down‑range gate
LAT_MIN_DEG, LAT_MAX_DEG = -20,  20 # launch‑latitude gate

# ── Constants & environment ───────────────────────────────────────────────
Re = 6.378e6; mu = 3.986e14; omega_e = 7.292e-5
m_nom, S_nom = 907.0, 0.4839
def rho_atm(h):
    Hs = 7.2e3 if h < 25e3 else 6.6e3 if h < 50e3 else 6.0e3 if h < 75e3 else 5.6e3
    return 1.225*np.exp(-h/Hs)
def aero_coeffs(a): CL_a, CD0, k = 1.8, 0.05, 0.5; CL = CL_a*a; return CL, CD0 + k*CL**2

# ── Six‑state EOM with rotation ───────────────────────────────────────────
def hgv_dot(t,y,p):
    r,lo,la,v,th,si=y; h=r-Re
    a,b=p['alpha'](t,y),p['bank'](t,y); rho=rho_atm(h); CL,CD=aero_coeffs(a)
    q=0.5*rho*v**2; aL=q*CL*p['S']/p['m']; aD=q*CD*p['S']/p['m']; g=mu/r**2
    rdot=v*np.sin(th)
    vdot=-aD-g*np.sin(th)+omega_e**2*r*np.cos(la)*(np.sin(th)*np.cos(la)-np.cos(th)*np.sin(si)*np.sin(la))
    thdot=(aL*np.cos(b)/v)+(v/r-g/v)*np.cos(th)+2*omega_e*np.cos(si)*np.cos(la)+ \
          (omega_e**2*r/v)*np.cos(la)*(np.cos(th)*np.cos(la)+np.sin(th)*np.sin(si)*np.sin(la))
    londot=v*np.cos(th)*np.sin(si)/(r*np.cos(la)+1e-9)
    latdot=v*np.cos(th)*np.cos(si)/r
    sigdot=(aL*np.sin(b)/(v*np.cos(th)+1e-9))+(v/r)*np.cos(th)*np.sin(si)*np.tan(la)- \
          2*omega_e*(np.tan(th)*np.cos(la)*np.sin(si)-np.sin(la))+ \
          (omega_e**2*r/(v*np.cos(th)+1e-9))*np.sin(si)*np.cos(la)*np.sin(la)
    return np.array([rdot,londot,latdot,vdot,thdot,sigdot])

def rk4(fun,t,y,h,p):
    k1=fun(t,y,p); k2=fun(t+.5*h,y+.5*h*k1,p)
    k3=fun(t+.5*h,y+.5*h*k2,p); k4=fun(t+h,y+h*k3,p)
    return y+h*(k1+2*k2+2*k3+k4)/6, k1[4]

def propagate(y0,tf=1200,dt0=0.5,p=None):
    t,dt,y=0.0,dt0,y0.copy(); T,Y=[0.0],[y.copy()]
    while t<tf and y[0]>Re:
        y_new,thdot=rk4(hgv_dot,t,y,dt,p)
        if abs(thdot)>0.05: dt*=0.5; continue
        y,t=y_new,t+dt; dt=min(dt*1.2,dt0); T.append(t); Y.append(y.copy())
    return np.array(T),np.vstack(Y)

# ── Range helper ───────────────────────────────────────────────────────────
def downrange(lo0,la0,lo1,la1):
    c=np.arccos(np.clip(np.sin(la0)*np.sin(la1)+np.cos(la0)*np.cos(la1)*np.cos(lo1-lo0),-1,1))
    return Re*c/1e3  # km

# ── Controls ───────────────────────────────────────────────────────────────
def alpha_bal(t,s): return np.deg2rad(8)
def alpha_jump(t,s):
    v_km=s[3]/1e3; a_max,a_ld,v1,v2=15,2.5,6,3
    if v_km>v1: return np.deg2rad(a_max)
    if v_km<v2: return np.deg2rad(a_ld)
    a_mid=.5*(a_max+a_ld); a_amp=.5*(a_max-a_ld); v_mid=.5*(v1+v2)
    return np.deg2rad(a_mid+a_amp*np.sin(np.pi*(v_km-v_mid)/(v1-v2)))
bank_no=lambda t,s:0.0
bank_left=lambda t,s:np.deg2rad(-20)
bank_right=lambda t,s:np.deg2rad(20)
bank_weave=lambda t,s:np.deg2rad(20*np.sin(t/120))

maneuvers=[('Bal-No',alpha_bal,bank_no),
           ('Bal-L', alpha_bal,bank_left),
           ('Bal-R', alpha_bal,bank_right),
           ('Bal-W', alpha_bal,bank_weave),
           ('Jump-No',alpha_jump,bank_no),
           ('Jump-L', alpha_jump,bank_left),
           ('Jump-R', alpha_jump,bank_right),
           ('Jump-W', alpha_jump,bank_weave)]

# ── Random‑draw engine with dual filters ──────────────────────────────────
accepted=[]; max_tries=3000; colors=['b','orange','y','purple','g','c','r','navy']
for name,alpha_fn,bank_fn in maneuvers:
    tries,found=0,False
    while tries<max_tries and not found:
        tries+=1
        lat0_deg=np.random.uniform(LAT_MIN_DEG,LAT_MAX_DEG)
        h0=np.random.uniform(40e3,70e3); v0=np.random.uniform(3e3,6e3)
        th0=np.deg2rad(np.random.uniform(-0.1,0)); sig0=np.deg2rad(np.random.uniform(-60,60))
        y0=np.array([Re+h0,0,np.deg2rad(lat0_deg),v0,th0,sig0])
        pars=dict(m=m_nom,S=S_nom,alpha=alpha_fn,bank=bank_fn)
        _,Y=propagate(y0,p=pars)
        R=downrange(0,np.deg2rad(lat0_deg),Y[-1,1],Y[-1,2])
        if R_MIN_KM<=R<=R_MAX_KM:
            accepted.append({'name':name,'Y':Y,'R':R,'lat0':lat0_deg})
            print(f"{name}: lat0={lat0_deg:.1f}°, R={R:.0f} km (found in {tries} draws)")
            found=True
    if not found:
        print(f"{name}: no trajectory found in {max_tries} draws")

# ── Plotting (unchanged style) ─────────────────────────────────────────────
fig=plt.figure(figsize=(12,6)); ax3d=fig.add_subplot(1,2,1,projection='3d'); ax2d=fig.add_subplot(1,2,2)
for i,tr in enumerate(accepted):
    Y=tr['Y']; h=Y[:,0]-Re; lon=np.rad2deg(Y[:,1]); lat=np.rad2deg(Y[:,2])
    label=f"{tr['name']} (R={tr['R']:.0f} km, φ0={tr['lat0']:.0f}°)"
    ax3d.plot(lon,lat,h,color=colors[i],lw=.8,ls='--',label=label)
    ax3d.scatter(lon[::120],lat[::120],h[::120],color=colors[i],s=8)
    ax2d.plot(lon,lat,color=colors[i],lw=.8,ls='--',label=label)
    ax2d.scatter(lon[::120],lat[::120],color=colors[i],s=8)
ax3d.set_xlabel('Lon (°)'); ax3d.set_ylabel('Lat (°)'); ax3d.set_zlabel('Height (m)')
ax3d.set_title(f'3‑D Trajectories (R {R_MIN_KM}-{R_MAX_KM} km, φ {LAT_MIN_DEG}°–{LAT_MAX_DEG}°)')
ax3d.legend(bbox_to_anchor=(1.04,1))
ax2d.set_xlabel('Lon (°)'); ax2d.set_ylabel('Lat (°)'); ax2d.set_title('2‑D Ground‑tracks')
ax2d.legend(bbox_to_anchor=(1.04,1)); ax2d.grid(True); plt.tight_layout(); plt.show()
