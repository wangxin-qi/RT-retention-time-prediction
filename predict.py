"""Stable command-line entry point for the bundled RouteRt runtime."""

from pathlib import Path
import sys
import warnings


warnings.filterwarnings("ignore")


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"


def main() -> None:
    if not RUNTIME_ROOT.is_dir():
        raise SystemExit(f"RouteRt runtime is missing: {RUNTIME_ROOT}")
    sys.path.insert(0, str(RUNTIME_ROOT))
    from core.predict.predict_v48_plus_mcmrt import main as runtime_main

    runtime_main()


if __name__ == "__main__":
    main()
