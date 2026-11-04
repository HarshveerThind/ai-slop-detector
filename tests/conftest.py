import sys
from pathlib import Path

# the code in src/ imports its neighbours by plain module name
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
