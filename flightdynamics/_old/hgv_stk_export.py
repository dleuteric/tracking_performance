from mpl_toolkits.mplot3d import Axes3D  # noqa
from datetime import datetime, timezone, timedelta

# USER GATES
OUTFILE = 'trajectory_set2.xlsx'   # Excel output for accepted trajectories
OUTFILE_OEM = 'trajectory_set2.oem'   # CCSDS OEM output for STK

def ecef(x, y, z):
    # Convert coordinates (example placeholder)
    return x, y, z
# ── CCSDS OEM EXPORT ------------------------------------------------------
def write_oem(trajs, filename=OUTFILE_OEM, base_epoch=None):
    """
    Write the accepted trajectories into a CCSDS‑OEM 2.0 file for STK import.

    Parameters
    ----------
    trajs : list of tuples
        Each element is (label, T, Y, R, lat0, lon0) as produced in
        the `accepted` list.
    filename : str
        Target file path.
    base_epoch : datetime, optional
        Epoch corresponding to t = 0 s (UTC). Defaults to now.
    """
    if base_epoch is None:
        base_epoch = datetime.now(timezone.utc).replace(microsecond=0)

    with open(filename, 'w') as f:
        # Global header ----------------------------------------------------
        f.write('CCSDS_OEM_VERS = 2.0\n')
        f.write(f'CREATION_DATE  = {datetime.now(timezone.utc).isoformat()}Z\n')
        f.write('ORIGINATOR     = ez-SMAD\n\n')

        # One segment per trajectory --------------------------------------
        for seg_id, (lab, T, Y, _, _, _) in enumerate(trajs, 1):
            # Cartesian states (m) → (km)
            x, y, z = ecef(Y[:, 0], Y[:, 1], Y[:, 2])
            vx = np.gradient(x, T); vy = np.gradient(y, T); vz = np.gradient(z, T)

            start_epoch = base_epoch + timedelta(seconds=float(T[0]))
            stop_epoch  = base_epoch + timedelta(seconds=float(T[-1]))

            # ---- META block ----------------------------------------------
            f.write('META_START\n')
            f.write(f'OBJECT_NAME    = {lab}\n')
            f.write(f'OBJECT_ID      = {seg_id}\n')
            f.write('CENTER_NAME    = EARTH\n')
            f.write('REF_FRAME      = ITRF\n')
            f.write('TIME_SYSTEM    = UTC\n')
            f.write(f'START_TIME     = {start_epoch.isoformat()}Z\n')
            f.write(f'STOP_TIME      = {stop_epoch.isoformat()}Z\n')
            f.write('META_STOP\n')

            # ---- ephemeris data ------------------------------------------
            for t, xi, yi, zi, vxi, vyi, vzi in zip(T, x, y, z, vx, vy, vz):
                epoch = base_epoch + timedelta(seconds=float(t))
                f.write(f'{epoch.isoformat()}Z '
                        f'{xi/1e3:.6f} {yi/1e3:.6f} {zi/1e3:.6f} '
                        f'{vxi/1e3:.6f} {vyi/1e3:.6f} {vzi/1e3:.6f}\n')
            f.write('\n')
    print(f"Saved CCSDS OEM to {filename}")


# -------------------------------------------------------------------------
# Final reporting and OEM export (executed only when run as a script)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print(f"Saved {len(accepted)} trajectories to {OUTFILE}")
        write_oem(accepted, OUTFILE_OEM)
    except NameError:
        # If 'accepted' wasn't produced (e.g., propagation failed),
        # give a graceful message instead of raising.
        print("No trajectories were generated; OEM not written.")
