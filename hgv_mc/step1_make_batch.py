"""
step1_make_batch.py
-----------------------------------------------------------
Generate a baseline Monte-Carlo batch using the *existing*
footprint_evolution.py (which runs on import) and pickle it.

The script:

1.  Sets a desired NUM_TRAJ before the module is executed.
2.  Imports (and thus runs) footprint_evolution -> `accepted` list appears.
3.  Saves that list to a pickle file for later steps.

Nothing inside footprint_evolution.py is modified on disk.
"""

from __future__ import annotations
import importlib
import importlib.util          # ensure .util is present even if shadowed
import types                   # needed later for placeholder module
import pickle, sys, time, pathlib, os

# --------------------------- CLI / defaults ---------------------------------
NUM_TRAJ   = int(sys.argv[1]) if len(sys.argv) > 1 else 5     # e.g. 200
OUT_PICKLE = sys.argv[2] if len(sys.argv) > 2 else "batch.pkl"  # e.g. batch.pkl
TRAJ_SEED  = 87                                              # same seed as baseline
# ---------------------------------------------------------------------------

print(f"\nStep 1: generating {NUM_TRAJ} trajectories "
      f"with TRAJ_SEED={TRAJ_SEED} …")

# Optional: render plots off-screen so the batch runs headless
import matplotlib
matplotlib.use("Agg")

 # --- Monkey-patch globals *before* the module executes ----------------------
fe_spec = importlib.util.find_spec("footprint_evolution")
if fe_spec is None:
    sys.exit("‼️  Cannot find footprint_evolution.py in PYTHONPATH.")

# Create a new module from the spec and inject the override globals
fe_mod = importlib.util.module_from_spec(fe_spec)
fe_mod.NUM_TRAJ          = NUM_TRAJ
fe_mod.TRAJECTORY_SEED   = TRAJ_SEED
sys.modules["footprint_evolution"] = fe_mod   # register so imports inside work

# Execute the module's code in its namespace
t0 = time.time()
fe_spec.loader.exec_module(fe_mod)
elapsed = time.time() - t0

accepted = getattr(fe_mod, "accepted", None)
if accepted is None:
    sys.exit("‼️  footprint_evolution did not export `accepted` list.")

print(f"✓  Accepted {len(accepted)} trajectories "
      f"({elapsed:.1f} s run time)")

# ----------------------------- pickle output --------------------------------
out_path = pathlib.Path(OUT_PICKLE).resolve()
with out_path.open("wb") as f:
    pickle.dump(accepted, f)
print(f"✅  Batch saved to {out_path}\n")