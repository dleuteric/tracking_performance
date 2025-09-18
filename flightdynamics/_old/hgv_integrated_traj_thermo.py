# ══ PLOT TEMPERATURE HISTORY ══════════════════════════════════════════════
from matplotlib import pyplot as plt

figT, axT = plt.subplots(figsize=(8,5))
figQ, axQ = plt.subplots(figsize=(8,5))

for i, (lab, T, _, _, _, _, Tw_s, q_s, Tw_p, q_p) in enumerate(accepted):
    col = colors[i]
    axT.plot(T, Tw_s, color=col, lw=1.4, label=f'{lab}  Stag')
    axQ.plot(T, np.array(q_s)/1e6, color=col, lw=1.4, label=f'{lab}  Stag')

    # panel curves (dashed)
    for nm in PANELS:
        axT.plot(T, Tw_p[nm], color=col, ls='--', lw=0.9, label=f'{lab}  {nm}')
        axQ.plot(T, np.array(q_p[nm])/1e6, color=col, ls='--', lw=0.9, label=f'{lab}  {nm}')

axT.set_xlabel('Time [s]')
axT.set_ylabel('Wall temperature $T_w$ [K]')
axT.set_title('Panel & stagnation temperatures')
axT.grid(True)
axT.legend(fontsize=8, ncol=2)

axQ.set_xlabel('Time [s]')
axQ.set_ylabel('Heat flux $q$ [MW m$^{-2}$]')
axQ.set_title('Convective heat flux histories')
axQ.grid(True)
axQ.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()