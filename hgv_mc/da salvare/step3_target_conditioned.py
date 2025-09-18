# step3_target_conditioned.py  (runnable draft)
import pickle, numpy as np, matplotlib.pyplot as plt
import footprint_evolution as fe
propagate, Re = fe.propagate, fe.Re
bank_left, bank_right = fe.bank_left, fe.bank_right
alpha_bal, m_nom, S_nom = fe.alpha_bal, fe.m_nom, fe.S_nom
downrange = fe.downrange

accepted = pickle.load(open("batch.pkl", "rb"))
snap_grid = np.arange(0, 800+1, 20)           # every 20 s
foot_r = np.zeros_like(snap_grid, dtype=float)
all_r = []
for traj_idx, (lab, T, Y, *_ ) in enumerate(accepted):
    lon_tgt, lat_tgt = Y[-1,1], Y[-1,2]       # its own impact point
    r_vs_t = []

    for t_snap in snap_grid:

        k = np.searchsorted(T, t_snap, 'right')-1
        if k < 0 or Y[k,0] <= Re:              # this traj already ended
            r_vs_t.append(0.0); continue
        ys = Y[k]

        # remaining specific energy (J/kg)
        E = 0.5*ys[3]**2 + 9.80665*(ys[0]-Re)
        if E < 1e6:                            # below manoeuvre usefulness
            r_vs_t.append(0.0); continue

        impacts = []
        for b_fun in (bank_left, bank_right):
            T2, Y2 = propagate(ys, p=dict(m=m_nom,S=S_nom,
                                           alpha=alpha_bal,bank=b_fun))
            # drop if future path violates range window
            dr_km = downrange(lon_tgt, lat_tgt, Y2[-1,1], Y2[-1,2])
            if 1500 <= dr_km <= 5000:
                impacts.append(dr_km)
        # If extreme‑bank continuations miss the 1 500–5 000 km gate,
        # use the straight‑on residual distance so the curve never sticks at 0.
        if impacts:
            r_vs_t.append(max(impacts))                 # max residual range
        else:
            # fallback: direct remaining down‑range along present heading
            dr_direct = downrange(lon_tgt, lat_tgt, ys[1], ys[2])
            r_vs_t.append(dr_direct)
    foot_r = np.maximum(foot_r, r_vs_t)        # union across trajectories
    all_r.append(r_vs_t)                       # store completed trace

# ---- plot ---------------------------------------------------------------
plt.figure(figsize=(7,4))
plt.plot(snap_grid, foot_r, marker='o')
plt.xlabel("Elapsed time [s]"); plt.ylabel("Footprint radius wrt target [km]")
plt.title("Target-conditioned footprint collapse")
plt.grid(True); plt.tight_layout(); plt.show()

# ---- enhanced visualisation ---------------------------------------------
import matplotlib.ticker as mtick

fig, (axR, axH) = plt.subplots(2, 1, figsize=(8,6), sharex=True,
                               gridspec_kw=dict(hspace=0.1))

# ── radius panel
for r in all_r:
    axR.plot(snap_grid, r, alpha=0.3, lw=1)
axR.plot(snap_grid, foot_r, color='k', lw=2.5, label='Envelope')
axR.set_ylabel("Footprint radius [km]")
axR.set_title("Target-conditioned footprint collapse")
axR.legend(loc="upper right")
axR.grid(True, which='both', ls=':')

# ── altitude panel
for (_, T, Y, *_), col in zip(accepted, plt.cm.tab10(np.linspace(0,1,len(accepted)))):
    axH.plot(T, (Y[:,0]-Re)/1e3, color=col, alpha=0.6)          # altitude km
axH.set_xlabel("Elapsed time [s]")
axH.set_ylabel("Altitude [km]")
axH.grid(True, which='both', ls=':')
axH.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

fig.tight_layout()
plt.savefig("footprint_vs_altitude.png", dpi=150)
plt.show()