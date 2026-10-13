"""Compatibility entry point for the non-destructive checkpoint audit."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, str(root / "reproduce.py"),
                                      "audit", *sys.argv[1:]], cwd=root))
