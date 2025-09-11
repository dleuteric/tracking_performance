# geometry/test_fileReading.py
# Purpose: sanity-check that we can FIND the LOS + EPHEM files with robust paths.
# It does not triangulate; it only lists/validates what is on disk.
# Run: python -m geometry.test_fileReading


from __future__ import annotations
from pathlib import Path
import pandas as pd
import re

# Helper: Normalize STK-exported CSV headers
def normalize_stk_headers(cols: list[str]) -> list[str]:
    """Normalize STK CSV headers by lowercasing, stripping, normalizing whitespace,
    removing unit suffixes and mapping to canonical names.

    Rules:
    - Lowercase and strip each header.
    - Replace sequences of whitespace with a single space.
    - Remove unit suffixes in parentheses, mapping them to canonical suffixes:
      "time (utcg)" -> "time"
      "x (km)" -> "x_km", "y (km)" -> "y_km", "z (km)" -> "z_km"
      "vx (km/s)" -> "vx_kmps", "vy (km/s)" -> "vy_kmps", "vz (km/s)" -> "vz_kmps"
    - Normalize plain alternatives (e.g. "x" -> "x_km") only if any (km) variant is present.
    - Keep LOS columns ux, uy, uz and text columns observer, target, frame unchanged except lowercase/strip.
    """
    def clean_header(h: str) -> str:
        h = h.strip().lower()
        h = re.sub(r"\s+", " ", h)
        return h

    # First clean all headers by lowercasing, stripping, normalizing whitespace
    cleaned = [clean_header(c) for c in cols]

    # Detect if position columns with (km) units exist
    km_pos_cols = {"x (km)", "y (km)", "z (km)"}
    has_km_pos = any(c in cleaned for c in km_pos_cols)

    # Map from original cleaned header to canonical normalized header
    normalized = []
    for c in cleaned:
        # Remove unit suffixes and map
        # time (utcg) -> time
        if c == "time (utcg)":
            normalized.append("time")
            continue
        # position km
        if c == "x (km)":
            normalized.append("x_km")
            continue
        if c == "y (km)":
            normalized.append("y_km")
            continue
        if c == "z (km)":
            normalized.append("z_km")
            continue
        # velocity km/s
        if c == "vx (km/s)":
            normalized.append("vx_kmps")
            continue
        if c == "vy (km/s)":
            normalized.append("vy_kmps")
            continue
        if c == "vz (km/s)":
            normalized.append("vz_kmps")
            continue

        # Keep LOS columns and text columns unchanged (except lower/strip)
        if c in ("ux", "uy", "uz", "observer", "target", "frame"):
            normalized.append(c)
            continue

        # Normalize plain alternatives for position columns if km variants present
        if has_km_pos:
            if c == "x":
                normalized.append("x_km")
                continue
            if c == "y":
                normalized.append("y_km")
                continue
            if c == "z":
                normalized.append("z_km")
                continue
            if c == "vx":
                normalized.append("vx_kmps")
                continue
            if c == "vy":
                normalized.append("vy_kmps")
                continue
            if c == "vz":
                normalized.append("vz_kmps")
                continue

        # Otherwise leave as-is (cleaned)
        normalized.append(c)

    return normalized

# Helper: Read STK-exported CSV, skipping metadata lines starting with '#'
def read_stk_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read an STK-exported CSV, skipping metadata lines starting with '#'.
    Lowercase + strip headers. Optionally limit to nrows.
    """
    df = pd.read_csv(path, comment="#", nrows=nrows)
    raw_cols = list(df.columns)
    df.columns = normalize_stk_headers(raw_cols)
    print(f"[NORM] Header mapping: {dict(zip(raw_cols, df.columns))}")
    return df

def main():
    # Resolve project root = one level up from this file's folder: .../ez-SMAD_mk01/
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    print(f"[PATH] __file__          : {this_file}")
    print(f"[PATH] project_root      : {project_root}")

    # --- Configure here if needed ---
    target_id = "HGV_00001"  # <- adjust only this if you want a different target
    los_dir   = project_root / "exports" / "stk_exports" / "OUTPUT_LOS_VECTORS" / target_id
    eph_dir   = project_root / "exports" / "stk_exports" / "OUTPUT_EPHEM"

    print(f"[PATH] LOS_DIR           : {los_dir}")
    print(f"[PATH] EPH_DIR           : {eph_dir}")

    # Basic existence checks
    print(f"[CHK ] LOS_DIR exists    : {los_dir.exists()}  (is_dir={los_dir.is_dir()})")
    print(f"[CHK ] EPH_DIR exists    : {eph_dir.exists()}  (is_dir={eph_dir.is_dir()})")

    # List targets available (subfolders of OUTPUT_LOS_VECTORS)
    los_root = los_dir.parent  # .../OUTPUT_LOS_VECTORS/
    if los_root.exists():
        targets = sorted([p.name for p in los_root.iterdir() if p.is_dir()])
    else:
        targets = []
    print(f"[DISC] Targets under OUTPUT_LOS_VECTORS: {targets}")

    # Discover LOS files for selected target
    pattern = f"LOS_OBS_*_to_{target_id}_icrf.csv"
    los_files = sorted(los_dir.glob(pattern))
    print(f"[DISC] Searching LOS with pattern '{pattern}'")
    print(f"[DISC] Found {len(los_files)} LOS files")
    for p in los_files[:10]:
        print(f"  - {p.name}")

    if not los_files:
        # Show raw candidates if the pattern didn’t match
        any_los = sorted(los_dir.glob("LOS_OBS_*_to_*_icrf.csv"))
        candidates = sorted({q.name.split('_to_')[1].split('_icrf')[0] for q in any_los})
        print(f"[HINT] No LOS match. Candidates present here: {candidates}")
        print("[HINT] Double-check: target_id, case, and suffix (_icrf).")
        return

    # Infer observer IDs seen in filenames
    obs_ids = sorted({p.name.split("_")[2] for p in los_files})
    print(f"[DISC] Observer IDs in LOS: {obs_ids}")

    # Do a light CSV sniff (first file) and show columns & a couple of times
    sample_los = los_files[0]
    try:
        df_los = read_stk_csv(sample_los, nrows=5)
        print(f"[CSV ] Sample LOS '{sample_los.name}' columns: {list(df_los.columns)}")
        # Try to parse Time
        if "time" in df_los.columns:
            t = pd.to_datetime(df_los["time"], errors="coerce", utc=True)
            print(f"[CSV ] Parsed first 3 times: {list(t.head(3))}")
        else:
            print("[WARN] No 'time' column detected in LOS sample after comment-skip.")

        for c in ("ux","uy","uz"):
            if c not in df_los.columns:
                print(f"[WARN] Missing LOS column '{c}' after normalization")
        if all(c in df_los.columns for c in ("ux","uy","uz")):
            print("[CSV ] First 2 LOS unit vectors:")
            print(df_los[["ux","uy","uz"]].head(2).to_string(index=False))
    except Exception as e:
        print(f"[ERR ] Reading LOS CSV failed: {e}")

    # Check matching ephemeris files for each observer
    missing_eph = []
    for obs in obs_ids:
        eph_path = eph_dir / f"OBS_{obs}_ephem.csv"
        exists = eph_path.exists()
        print(f"[DISC] EPHEM for OBS {obs}: {eph_path.name} -> exists={exists}")
        if exists:
            try:
                df_e = read_stk_csv(eph_path, nrows=5)
                cols = list(df_e.columns)
                has_pos = all(c in cols for c in ("x_km","y_km","z_km"))
                print(f"[CSV ] Columns(OBS {obs}): {cols} | has position cols? {has_pos}")
                if "time" in df_e.columns:
                    t_e = pd.to_datetime(df_e.iloc[:3]["time"], errors="coerce", utc=True)
                    print(f"[CSV ] Sample EPHEM times(OBS {obs}): {list(t_e)}")
                else:
                    print(f"[WARN] No 'time' column in EPHEM for OBS {obs} after normalization.")
            except Exception as e:
                print(f"[ERR ] Reading EPHEM for OBS {obs} failed: {e}")
        else:
            missing_eph.append(obs)

    if missing_eph:
        print(f"[WARN] Missing ephemeris CSVs for observers: {missing_eph}")
        print(f"[HINT] Expected at: {eph_dir}/OBS_###_ephem.csv")

    print(f"[SUM ] LOS files seen: {len(los_files)} | Observers: {len(obs_ids)} | EPHEM missing: {len(missing_eph)}")
    print("\n[OK ] File reading smoke test finished.")

if __name__ == "__main__":
    main()