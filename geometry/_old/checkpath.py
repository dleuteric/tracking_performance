# check_paths.py
"""
Diagnostic script to check file structure
"""

from pathlib import Path
import os


def check_directory_structure():
    print("Current working directory:", os.getcwd())
    print("\nSearching for data directories...\n")

    # Check various possible locations
    possible_paths = [
        Path('..'),
        Path('exports_48'),
        Path('../../exports_48'),
        Path('../../exports_48'),
    ]

    for base in possible_paths:
        print(f"Checking: {base.absolute()}")
        if base.exists():
            print(f"  ✓ Path exists")

            # Look for subdirectories
            if base.is_dir():
                subdirs = [d for d in base.iterdir() if d.is_dir()]
                print(f"  Subdirectories: {[d.name for d in subdirs[:5]]}")

                # Check for expected structure
                stk_exports = base / 'stk_exports'
                target_exports = base / 'target_exports'

                if stk_exports.exists():
                    print(f"  ✓ Found stk_exports")
                    ephem_dir = stk_exports / 'OUTPUT_EPHEM'
                    los_dir = stk_exports / 'OUTPUT_LOS_VECTORS'

                    if ephem_dir.exists():
                        ephem_files = list(ephem_dir.glob('*.csv'))
                        print(f"    - OUTPUT_EPHEM has {len(ephem_files)} CSV files")
                        if ephem_files:
                            print(f"      First file: {ephem_files[0].name}")

                    if los_dir.exists():
                        los_subdirs = [d for d in los_dir.iterdir() if d.is_dir()]
                        print(f"    - OUTPUT_LOS_VECTORS has {len(los_subdirs)} subdirectories")
                        if los_subdirs:
                            print(f"      First subdir: {los_subdirs[0].name}")
                            los_files = list(los_subdirs[0].glob('*.csv'))
                            if los_files:
                                print(f"        Contains {len(los_files)} CSV files")

                if target_exports.exists():
                    print(f"  ✓ Found target_exports")
                    oem_dir = target_exports / 'OUTPUT_OEM'
                    if oem_dir.exists():
                        oem_files = list(oem_dir.glob('*.oem'))
                        print(f"    - OUTPUT_OEM has {len(oem_files)} OEM files")
                        if oem_files:
                            print(f"      First file: {oem_files[0].name}")

                print()
        else:
            print(f"  ✗ Path does not exist\n")

    # Also search recursively for specific file patterns
    print("\nSearching for data files from current directory...")
    cwd = Path('..')

    # Find any OEM files
    oem_files = list(cwd.rglob('*.oem'))
    if oem_files:
        print(f"Found {len(oem_files)} OEM files:")
        for f in oem_files[:3]:
            print(f"  - {f.relative_to(cwd)}")

    # Find ephemeris files
    eph_files = list(cwd.rglob('OBS_*.csv'))
    if eph_files:
        print(f"\nFound {len(eph_files)} ephemeris files:")
        for f in eph_files[:3]:
            print(f"  - {f.relative_to(cwd)}")

    # Find LOS files
    los_files = list(cwd.rglob('LOS_*.csv'))
    if los_files:
        print(f"\nFound {len(los_files)} LOS files:")
        for f in los_files[:3]:
            print(f"  - {f.relative_to(cwd)}")


if __name__ == "__main__":
    check_directory_structure()