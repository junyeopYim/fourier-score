"""Legacy entry point of `python -m experiments gmm spectral` (ArgumentParser and flags in experiments/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main(["gmm", "spectral", *sys.argv[1:]])
