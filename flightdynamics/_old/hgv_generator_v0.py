"""
hgv_thermo_panels.py
  – Paper-faithful aerothermal model (stagnation + flat plates)

• Eq. 8  : stagnation-point heat flux (Tauber)
• Eq. 9/10 : flat-plate turbulent heat flux (> / ≤ 4 km/s)
• Radiative-adiabatic balance εσT⁴ = q  (iterative)
• Four-panel pyramid mesh (angles and x from the paper)
• Demo 600-s skip trajectory
"""

import numpy as np, matplotlib.pyplot as plt

# ── constants --------------------------------------------------------------
EPS   = 0.85                      # emissivity
RN    = 0.034                     # nose radius (m)
SIGMA = 5.670374419e-8            # Stefan–Boltzmann
RHO0, HS = 1.225, 7200.0          # ISA density scale

def rho_atm(h): return RHO0*np.exp(-h/HS)

# ── stagnation heat flux (Eq. 8) ------------------------------------------
def q_stag(rho, v, Tw):
    h0 = 0.5*v**2 + 2.3e5
    hw = 1_000*Tw
    term = np.maximum(1 - hw/h0, 1e-6)
    return 1.83e-4*np.sqrt(rho/RN)*np.sqrt(term)*v**3   # W m-2

# ── flat-plate heat flux (Eq. 9,10) ---------------------------------------
def q_flat(theta, x, rho, v, Tw):
    """
    theta [rad], x [m] may be arrays. rho, v, Tw broadcast.
    """
    cosh = np.cos(theta); sinh = np.sin(theta)
    h0   = 0.5*v**2 + 2.3e5
    hw   = 1_000*Tw
    common = np.maximum(1 - 1.11*hw/h0, 1e-6)
    if np.isscalar(v):
        high = v > 4_000
    else:
        high = v > 4_000
    q = np.where(high,
        2.2e-5 * (cosh**2.08)*(sinh**1.6) / x**0.2 * common**0.5  * rho**0.8 * v**3.7,
        3.89e-4 * (cosh**1.78)*(sinh**1.6) / x**0.2 * common**0.5 * (556/Tw)**0.25 *
                 rho**0.8 * v**3.37)
    return q  # W m-2

# ── iterative solver -------------------------------------------------------
def Tw_iter(q_fun, args, guess=1200.0, n=8):
    Tw = np.full_like(args[0], guess, dtype=float)
    for _ in range(n):
        q = q_fun(*args, Tw)
        Tw = (q/(EPS*SIGMA))**0.25
    return Tw, q

# ── panel mesh (θ [rad], x [m]) -------------------------------------------
PANELS = {"Top-L":   (np.deg2rad(30), 1.5),
          "Top-R":   (np.deg2rad(30), 1.5),
          "Lower":   (np.deg2rad(10), 2.0),
          "Base":    (np.deg2rad(70), 0.1)}

# ── demo trajectory --------------------------------------------------------
t = np.linspace(0,600,601)
alt = (60 - 20*np.sin(2*np.pi*t/600))*1e3      # m
vel = 5_500 - 1_000*np.sin(2*np.pi*t/600)      # m/s
rho = rho_atm(alt)

# ── stagnation Tw ----------------------------------------------------------
Tw_stag, q_stag_hist = Tw_iter(lambda rho,v,Tw: q_stag(rho,v,Tw),
                               (rho, vel), guess=1200.)

# ── panel temps ------------------------------------------------------------
Tw_panel, q_panel = {}, {}
for name,(th,x) in PANELS.items():
    Tw_panel[name], q_panel[name] = Tw_iter(
        lambda rho,v,Tw: q_flat(th, x, rho, v, Tw),
        (rho, vel), guess=800.)

# ── plots ------------------------------------------------------------------
fig, (axT, axQ) = plt.subplots(2,1, figsize=(8,6), sharex=True)
axT.plot(t, Tw_stag, label='Stag')
axQ.plot(t, q_stag_hist/1e6, label='Stag')

for name in PANELS:
    axT.plot(t, Tw_panel[name], label=name)
    axQ.plot(t, q_panel[name]/1e6, label=name)

axT.set_ylabel('T$_w$ [K]'); axT.set_title('Panel & stagnation temperatures')
axT.legend(); axT.grid(True)
axQ.set_ylabel('q [MW m$^{-2}$]'); axQ.set_xlabel('Time [s]')
axQ.set_title('Convective heat flux'); axQ.legend(); axQ.grid(True)
plt.tight_layout(); plt.show()

# ── peak table -------------------------------------------------------------
print("=== Peak heat/temperature ===")
print(f"{'Panel':8s} | Tw_peak [K] | q_peak [MW/m²]")
print("-"*35)
print(f"{'Stag':8s} | {Tw_stag.max():8.0f} | {q_stag_hist.max()/1e6:8.2f}")
for name in PANELS:
    print(f"{name:8s} | {Tw_panel[name].max():8.0f} | {q_panel[name].max()/1e6:8.2f}")