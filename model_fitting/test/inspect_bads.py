"""Quick script to inspect BADS object attributes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pybads import BADS
import numpy as np

def dummy_func(x):
    return np.sum(x**2)

# Create BADS instance
bads = BADS(
    dummy_func,
    np.array([1.0, 1.0]),
    np.array([0.0, 0.0]),
    np.array([10.0, 10.0]),
    np.array([0.5, 0.5]),
    np.array([5.0, 5.0]),
)

print("BADS object attributes:")
print("=" * 60)
for attr in dir(bads):
    if not attr.startswith('_'):
        try:
            val = getattr(bads, attr)
            if not callable(val):
                print(f"{attr}: {type(val).__name__}")
        except:
            pass

print("\n" + "=" * 60)
print("Running one optimization step to see if attributes change...")
print("=" * 60)

# Run a few iterations
result = bads.optimize()
print(f"\nResult keys: {result.keys() if hasattr(result, 'keys') else 'Not a dict'}")

# Check attributes after optimization
print("\nAfter optimization:")
for attr in ['iteration_history', 'iter', 'iteration', 'n_iter']:
    if hasattr(bads, attr):
        val = getattr(bads, attr)
        print(f"{attr}: {type(val).__name__} = {val}")

