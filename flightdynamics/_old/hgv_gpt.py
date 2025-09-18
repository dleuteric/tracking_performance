import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ── constants ─────────────────────────
Re, mu = 6378.137, 398600.4418                               # km, km³/s²

# ── simple environment & aero ─────────
rho = lambda h: 1.225 * np.exp(-np.maximum(h, 0) / 7.2)      # ρ(h) scale‑height
def aero(a):                                                 # CL=2α, CD=k+½CL²
    CL = 2 * a
    return CL, 0.05 + 0.5 * CL ** 2

# ── longitudinal & lateral controls (Zhang Eq.4–7) ─────────
a_max, a_ld, v1, v2 = np.deg2rad(15), np.deg2rad(2.5), 6.0, 3.0
alpha = lambda _, s: a_max if s[3] > v1 else a_ld if s[3] < v2 else \
    0.5 * (a_max + a_ld) + 0.5 * (a_max - a_ld) * np.sin(np.pi * (s[3] - (v1 + v2) / 2) / (v1 - v2))
bank  = lambda *_: np.deg2rad(20)                            # constant C‑turn

# ── point‑mass dynamics (Eq.1) ─────────────────────────────
def f(t, y, p):
    r, lon, lat, v, th, sig = y
    h = r - Re
    CL, CD = aero(alpha(t, y)); q = 0.5 * rho(h) * (v * 1e3) ** 2
    aL = q * CL * p['S'] / p['m'] / 1e3; aD = q * CD * p['S'] / p['m'] / 1e3
    g = mu / r ** 2; cth = np.cos(th);  ctl = np.cos(lat) + 1e-9
    return np.array([
        v * np.sin(th),
        v * cth * np.sin(sig) / (r * ctl),
        v * cth * np.cos(sig) / r,
        -aD - g * np.sin(th),
        aL * np.cos(bank()) / v + v * cth / r - g * cth / v,
        aL * np.sin(bank()) / (v * cth + 1e-9) + v * cth * np.sin(sig) * np.tan(lat) / r])

def rk4(fun, t, y, h, p):
    k1 = fun(t, y, p); k2 = fun(t + h/2, y + h/2 * k1, p)
    k3 = fun(t + h/2, y + h/2 * k2, p); k4 = fun(t + h, y + h * k3, p)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6

def propagate(y0, tf, dt, p):
    T, Y, t = [0], [y0], 0
    while t < tf and y0[0] > Re:
        y0 = rk4(f, t, y0, dt, p); t += dt; T.append(t); Y.append(y0)
    return np.array(T), np.vstack(Y)

# ── helpers for ECEF & Earth sphere ────────────────────────
def ecef(Y):
    r, lon, lat = Y[:, 0], Y[:, 1], Y[:, 2]
    return r*np.cos(lat)*np.cos(lon), r*np.cos(lat)*np.sin(lon), r*np.sin(lat)

def globe(ax, R=Re, n=40):
    u, v = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2)
    x = R*np.outer(np.cos(u), np.sin(v)); y = R*np.outer(np.sin(u), np.sin(v)); z = R*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgrey', alpha=.15, linewidth=0)

def downrange(Y):
    la1, lo1, la2, lo2 = Y[0, 2], Y[0, 1], Y[-1, 2], Y[-1, 1]
    c = np.arccos(np.clip(np.sin(la1)*np.sin(la2) + np.cos(la1)*np.cos(la2)*np.cos(lo2-lo1), -1, 1))
    return Re * c

# ── demo run ───────────────────────────────────────────────
if __name__ == '__main__':
    pars = dict(m=1_000.0, S=2.0)
    y0 = np.array([Re + 100, 0, 0, 7.0, np.deg2rad(-5), np.deg2rad(90)])
    T, Y = propagate(y0, tf=1_200, dt=0.5, p=pars); alt = Y[:, 0] - Re

    # altitude + AoA
    fig, ax1 = plt.subplots()
    ax1.plot(T, alt); ax1.set(xlabel='Time [s]', ylabel='Altitude [km]')
    ax2 = ax1.twinx(); ax2.plot(T, np.rad2deg([alpha(t, y) for t, y in zip(T, Y)]), '--'); ax2.set_ylabel('Attack‑angle [deg]')
    plt.title('Altitude & AoA'); plt.show()

    # 3‑D ECEF view
    x, y, z = ecef(Y)
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); globe(ax)
    ax.plot(x, y, z, color='navy'); ax.scatter(x[0], y[0], z[0], c='lime'); ax.scatter(x[-1], y[-1], z[-1], c='red', marker='x')
    ax.set_box_aspect([1, 1, 1]); plt.title('ECEF Skip‑Glide'); plt.show()

    print(f'Down‑range ≈ {downrange(Y):.0f} km | End h={alt[-1]:.1f} km v={Y[-1,3]:.3f} km/s')
