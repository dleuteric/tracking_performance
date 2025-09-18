# surrogate_builder_v2.py  – safe single‑thread version
import numpy as np, joblib, time
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from numba import njit

# ── constants & env
Re, mu = 6378.137, 398600.4418
rho = lambda h: 1.225*np.exp(-np.maximum(h,0)/7.2)
def aero(a): CL=2*a; return CL,0.05+0.5*CL**2

# ── Numba‑JIT dynamics (≤30 lines each)
@njit
def f(t,y,m,S,bdeg,tw):
    r,lo,la,v,th,si = y; h=r-Re; g=mu/r**2
    bank=np.deg2rad(bdeg)
    if tw<1e6 and np.floor(t/tw)%2==1: bank=-bank
    CL,CD=aero(np.deg2rad(15)); q=0.5*rho(h)*(v*1e3)**2
    aL,aD=q*CL*S/m/1e3,q*CD*S/m/1e3; ct,cl=np.cos(th),np.cos(la)+1e-9
    return np.array([v*np.sin(th),
                     v*ct*np.sin(si)/(r*cl),
                     v*ct*np.cos(si)/r,
                     -aD-g*np.sin(th),
                     aL*np.cos(bank)/v+v*ct/r-g*ct/v,
                     aL*np.sin(bank)/(v*ct+1e-9)+v*ct*np.sin(si)*np.tan(la)/r])

@njit
def rk4_step(t,y,h,m,S,bdeg,tw):
    k1=f(t,y,m,S,bdeg,tw); k2=f(t+h/2,y+h/2*k1,m,S,bdeg,tw)
    k3=f(t+h/2,y+h/2*k2,m,S,bdeg,tw); k4=f(t+h,y+h*k3,m,S,bdeg,tw)
    return y+h*(k1+2*k2+2*k3+k4)/6

@njit
def propagate_single(v0,bdeg,tw,m=1000.0,S=2.0,tf=600.0,dt=1.0):
    y=np.array([Re+80,0.0,0.0,v0,np.deg2rad(-5),np.deg2rad(90)])
    t=0.0
    while t<tf and y[0]>Re:
        y=rk4_step(t,y,dt,m,S,bdeg,tw); t+=dt
    return y[1],y[2]                # lon, lat

# ── great‑circle distance
def downrange(lo0,la0,lo1,la1):
    c=np.arccos(np.clip(np.sin(la0)*np.sin(la1)+np.cos(la0)*np.cos(la1)*np.cos(lo1-lo0),-1,1))
    return Re*c

# ── priors
def sample_batch(N):
    v0=np.random.uniform(3,7,N); E1=np.random.beta(2,5,N)  # E1 kept for future
    bank=np.random.normal(20,5,N)
    tw=np.exp(np.random.uniform(np.log(60),np.log(300),N))
    return np.column_stack([v0,E1,bank,tw])

# ── generate dataset
N=1000
X=sample_batch(N); yR=[]
t0=time.time()
for i,row in enumerate(X):
    v0,_,bk,tw=row
    lon,lat=propagate_single(v0,bk,tw)
    yR.append(downrange(0,0,lon,lat))
    if (i+1)%100==0: print(f"{i+1}/{N} trajectories done…")
yR=np.asarray(yR)
print(f"Sim time: {time.time()-t0:.1f}s")

# ── fit & cross‑validate surrogate
model=Pipeline([('sc',StandardScaler()),
                ('poly',PolynomialFeatures(3,include_bias=False)),
                ('ridge',Ridge(alpha=5e-3))])
cv=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,X,yR,cv=cv,scoring='r2')
print(f"5‑fold CV R²: {scores.mean():.3f} ±{scores.std():.3f}")
model.fit(X,yR); joblib.dump(model, 'surrogate_poly.pkl')
print("Surrogate saved to surrogate_poly.pkl")
