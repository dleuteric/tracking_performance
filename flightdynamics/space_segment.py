# flightdynamics/space_segment.py
"""
Space Segment Module – multi-layer constellation init + propagation + ECEF ephemerides
- Reads configuration from config.py (EARTH_PARAMS, SPACE_SEGMENT)
- Initializes Walker layers
- Propagates with two-body + opzionale drift secolare J2 su RAAN/argomento del perigeo
- Exports per-satellite ephemerides in ECEF (o ECI) e un manifest della run
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# make parent importable and load config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


class SpaceSegment:
    """Keplerian initializer + light propagator for constellation ephemerides."""

    def __init__(self):
        # Config hooks
        self.earth = config.EARTH_PARAMS
        self.cfg = config.SPACE_SEGMENT

        # Satellite list: one dict per sat (keplerian state)
        self.satellites: list[dict] = []
        self._initialize_constellation()

    # --------------------------
    # Initialization (unchanged)
    # --------------------------
    def _initialize_constellation(self):
        """Create initial satellite states from config layers."""
        self.satellites.clear()

        for layer in self.cfg["layers"]:
            if not layer.get("enabled", True):
                continue

            a_km = self.earth["R_E"] + float(layer["altitude_km"])
            e = float(layer["eccentricity"])
            i_deg = float(layer["inclination_deg"])
            omega_deg = float(layer["arg_perigee_deg"])
            num_planes = int(layer["num_planes"])
            sats_per_plane = int(layer["sats_per_plane"])
            phase_off = float(layer.get("phase_offset_deg", 0.0))

            for plane_idx in range(num_planes):
                # RAAN equally spaced
                raan_deg = (360.0 / num_planes) * plane_idx

                for sat_in_plane in range(sats_per_plane):
                    # Mean anomaly with intra-plane spacing + optional inter-plane phasing
                    M_deg = (360.0 / sats_per_plane) * sat_in_plane + phase_off * plane_idx

                    # ---- ID format requested: Px_Sy (1-based indices) ----
                    sat_id = f"P{plane_idx+1}_S{sat_in_plane+1}"

                    layer_name = layer["name"]
                    regime = (layer.get("regime") or layer_name.split("_", 1)[0]).upper()

                    sat = {
                        "regime": regime,
                        "id": sat_id,
                        "layer": layer["name"],
                        "a_km": a_km,
                        "e": e,
                        "i_deg": i_deg,
                        "raan_deg": raan_deg,
                        "omega_deg": omega_deg,
                        "M_deg": M_deg % 360.0,
                    }

                    # also store initial Cartesian for quick inspection (ECI)
                    r0, v0 = self._keplerian_to_cartesian(
                        sat["a_km"], sat["e"], sat["i_deg"],
                        sat["raan_deg"], sat["omega_deg"], sat["M_deg"]
                    )
                    sat["r_eci"] = r0
                    sat["v_eci"] = v0

                    self.satellites.append(sat)

    # --------------------------
    # Keplerian -> Cartesian
    # --------------------------
    def _keplerian_to_cartesian(self, a_km, e, i_deg, raan_deg, omega_deg, M_deg):
        """
        Convert Kepler elements to Cartesian state in ECI.
        Units: km, km/s, degrees input angles
        """
        mu = self.earth["mu"]

        i = np.radians(i_deg)
        raan = np.radians(raan_deg)
        argp = np.radians(omega_deg)
        M = np.radians(M_deg)

        # Solve Kepler: M = E - e sin E (Newton)
        E = M
        for _ in range(10):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            E -= f / fp

        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                            np.sqrt(1 - e) * np.cos(E / 2))

        # Radius
        r = a_km * (1 - e * np.cos(E))

        # Perifocal position/velocity
        x_p = r * np.cos(nu)
        y_p = r * np.sin(nu)
        z_p = 0.0

        h = np.sqrt(mu * a_km * (1 - e**2))
        vx_p = -mu / h * np.sin(nu)
        vy_p =  mu / h * (e + np.cos(nu))
        vz_p = 0.0

        # Rotation PQW->ECI
        cO, sO = np.cos(raan), np.sin(raan)
        ci, si = np.cos(i), np.sin(i)
        cw, sw = np.cos(argp), np.sin(argp)

        R = np.array([
            [ cO*cw - sO*sw*ci,  -cO*sw - sO*cw*ci,  sO*si],
            [ sO*cw + cO*sw*ci,  -sO*sw + cO*cw*ci, -cO*si],
            [ sw*si,              cw*si,             ci   ]
        ])
        r_eci = R @ np.array([x_p, y_p, z_p])
        v_eci = R @ np.array([vx_p, vy_p, vz_p])
        return r_eci, v_eci

    # --------------------------
    # Frame conversion
    # --------------------------
    def _eci_to_ecef(self, r_eci, v_eci, theta_rad):
        """ECI->ECEF via rotation about Z and ω×r for velocity."""
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        Rz = np.array([[ c, -s, 0.0],
                       [ s,  c, 0.0],
                       [0.0, 0.0, 1.0]])
        r_ecef = Rz @ r_eci
        omega = np.array([0.0, 0.0, self.earth["omega_E"]])
        v_ecef = Rz @ v_eci - np.cross(omega, r_ecef)
        return r_ecef, v_ecef

    # --------------------------
    # Propagation + Outputs
    # --------------------------
    def propagate(self):
        """Propagate constellation and write ephemerides + manifest."""
        prop = self.cfg["propagation"]
        out = self.cfg["output"]

        dt = float(prop.get("timestep_sec", 1.0))
        Tsec = float(prop.get("duration_min", 30.0)) * 60.0
        include_J2 = bool(prop.get("include_J2", False))
        coord_frame = str(out.get("coordinate_frame", "ECEF")).upper()
        if coord_frame not in ("ECI", "ECEF"):
            coord_frame = "ECEF"

        # time grid + epoch
        t_grid = np.arange(0.0, Tsec + 1e-9, dt)
        t0_utc = datetime.utcnow()
        epochs = [t0_utc + timedelta(seconds=float(t)) for t in t_grid]

        # output folders
        base_dir = Path(out.get("ephemeris_dir", "exports/ephemeris/"))
        run_id = self._run_id(t0_utc)
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # per-satellite propagation
        for sat in self.satellites:
            a = float(sat["a_km"])
            e = float(sat["e"])
            i_deg = float(sat["i_deg"])
            i = np.radians(i_deg)
            raan0 = np.radians(float(sat["raan_deg"]))
            argp0 = np.radians(float(sat["omega_deg"]))
            M0 = np.radians(float(sat["M_deg"]))

            mu = self.earth["mu"]
            n = np.sqrt(mu / a**3)  # rad/s
            p = a * (1 - e**2)

            # simple J2 secular drifts (Ωdot, ωdot); Mdot left at n
            if include_J2:
                J2 = self.earth["J2"]
                Re = self.earth["R_E"]
                fac = (Re**2) / (p**2)
                O_dot = -1.5 * J2 * fac * n * np.cos(i)
                w_dot =  0.75 * J2 * fac * n * (5.0 * np.cos(i)**2 - 1.0)
            else:
                O_dot = 0.0
                w_dot = 0.0

            # allocate
            data = np.zeros((t_grid.size, 7), dtype=float)

            for k, t in enumerate(t_grid):
                M = M0 + n * t
                raan = raan0 + O_dot * t
                argp = argp0 + w_dot * t

                r_eci, v_eci = self._keplerian_to_cartesian(
                    a, e, i_deg,
                    np.degrees(raan), np.degrees(argp),
                    np.degrees(M % (2 * np.pi))
                )

                if coord_frame == "ECEF":
                    theta = self.earth["omega_E"] * t  # coarse GMST offset=0
                    r_xyz, v_xyz = self._eci_to_ecef(r_eci, v_eci, theta)
                else:
                    r_xyz, v_xyz = r_eci, v_eci

                data[k, 0] = t
                data[k, 1:4] = r_xyz
                data[k, 4:7] = v_xyz

            df = pd.DataFrame(
                data, columns=["t_sec", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
            )
            df.insert(1, "epoch_utc", [ts.isoformat() + "Z" for ts in epochs])

            reg = str(sat.get('regime', 'UNK')).upper()
            fname = f"{reg}_{sat['id']}.csv"
            fpath = run_dir / fname
            df.to_csv(fpath, index=False)
            files[sat["id"]] = str(Path(fname))

        # manifest
        manifest = {
            "run_id": run_id,
            "created_utc": t0_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "coordinate_frame": coord_frame,
            "propagation": {
                "include_J2": include_J2,
                "timestep_sec": dt,
                "duration_min": Tsec / 60.0,
                "method": prop.get("method", "RK4"),
            },
            "output": {
                "root": str(run_dir),
                "ephemeris_dir": str(base_dir),
                "file_format": out.get("file_format", "csv"),
            },
            "layers": [
                {
                    "name": layer["name"],
                    "enabled": layer.get("enabled", True),
                    "type": layer.get("type", "walker"),
                    "total_sats": layer["total_sats"],
                    "num_planes": layer["num_planes"],
                    "sats_per_plane": layer["sats_per_plane"],
                    "altitude_km": layer["altitude_km"],
                    "inclination_deg": layer["inclination_deg"],
                    "eccentricity": layer["eccentricity"],
                    "arg_perigee_deg": layer["arg_perigee_deg"],
                    "phase_offset_deg": layer.get("phase_offset_deg", 0.0),
                }
                for layer in self.cfg["layers"]
                if layer.get("enabled", True)
            ],
            "files": files,
        }

        man_path = run_dir / f"manifest_{run_id}.json"
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[OK] Ephemerides: {len(files)} files → {run_dir}")
        print(f"[OK] Manifest   : {man_path}")
        return {"run_id": run_id, "run_dir": str(run_dir), "manifest": str(man_path)}

    @staticmethod
    def _run_id(dt_utc: datetime) -> str:
        return dt_utc.strftime("%Y%m%dT%H%M%SZ")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    ss = SpaceSegment()
    ss.propagate()