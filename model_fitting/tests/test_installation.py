import sys, os
from pathlib import Path

# This script can be run from anywhere.
# model_fitting/ is the package root (contains _swig_fourbynine.so and fourbynine.py).
MODEL_FITTING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_FITTING_DIR))

# Check that the compiled .so exists before trying to import it, so we can
# give a clear error rather than a cryptic segfault or ImportError.
SO_PATH = MODEL_FITTING_DIR / "_swig_fourbynine.so"
if not SO_PATH.exists():
    print(
        f"❌ _swig_fourbynine.so not found at {SO_PATH}\n"
        "   Run autobuild.sh first to compile the C++ extension."
    )
    sys.exit(1)

# Sanity-check: warn if the running Python looks different from what compiled the .so.
# A mismatch in Python major/minor version almost always causes a segfault on import.
import importlib, struct
running_py = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"Using Python {running_py} at {sys.executable}")
print(f"Loading _swig_fourbynine.so from {SO_PATH}")

from fourbynine import fourbynine_board, fourbynine_pattern, fourbynine_move, Player_Player1, Player_Player2, bool_to_player, player_to_string

print("✅ Initial Python Installation Test Successful!")